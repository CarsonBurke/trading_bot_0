#![feature(f16)]

mod results;

use std::time::Instant;
use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};
use trading_bot_0::torch::{
    constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT},
    model::{ModelVariant, TradingModel, TradingModelConfig},
    muon::{Muon, MuonConfig},
};

use results::{BenchmarkResult, BenchmarkRun, BenchmarkSuite};

fn sync_device(device: Device) {
    if let Device::Cuda(id) = device {
        tch::Cuda::synchronize(id as i64);
    }
}

/// Query CUDA memory used via nvidia-smi. Returns MiB, or None on CPU/error.
/// Reports *reserved* memory (caching allocator), so measurements should be
/// taken as deltas across a known-new allocation boundary.
fn gpu_mem_mib(device: Device) -> Option<f64> {
    let Device::Cuda(id) = device else { return None };
    let out = std::process::Command::new("nvidia-smi")
        .args([
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            &id.to_string(),
        ])
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    std::str::from_utf8(&out.stdout).ok()?.trim().parse().ok()
}

fn main() {
    let device = Device::cuda_if_available();
    if device == Device::Cpu {
        eprintln!("Warning: CUDA not detected, running on CPU (will be slow)");
    }
    let mut suite = BenchmarkSuite::new();

    println!("=== Trading Bot Benchmarks ===\n");

    run_optimizer_benchmarks(&mut suite, device);
    run_model_benchmarks(&mut suite, device);

    suite.save().expect("Failed to save benchmark results");
    println!("\n=== Benchmarks Complete ===");
}

fn run_model_benchmarks(suite: &mut BenchmarkSuite, device: Device) {
    println!("--- TradingModel Forward Benchmarks ---");
    let dtype = Kind::BFloat16;
    let raw_price_deltas_dim = (TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER) as i64;
    let static_obs_dim = STATIC_OBSERVATIONS as i64;

    for &variant in &[
        ModelVariant::Base,
        ModelVariant::Uniform256Stream,
        ModelVariant::AblationSmall,
    ] {
        let mut vs = nn::VarStore::new(device);
        let model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant,
                ..TradingModelConfig::default()
            },
        );
        vs.bfloat16();
        println!("  Variant: {}", variant.as_str());
        let price_deltas_dim = model.price_input_dim();
        let reported_seq_len = if variant == ModelVariant::Uniform256Stream {
            price_deltas_dim / TICKERS_COUNT
        } else {
            PRICE_DELTAS_PER_TICKER as i64
        };

        for &batch in &[1, 4, 8] {
            let price_deltas = if variant == ModelVariant::Uniform256Stream {
                let raw = Tensor::randn(&[batch, raw_price_deltas_dim], (dtype, device));
                model.uniform_stream_layout_from_raw_input(&raw)
            } else {
                Tensor::randn(&[batch, price_deltas_dim], (dtype, device))
            };
            let static_features = Tensor::randn(&[batch, static_obs_dim], (dtype, device));

            for _ in 0..10 {
                let _ = model.forward(&price_deltas, &static_features, false);
            }
            sync_device(device);

            let iters = 100;
            tch::no_grad(|| {
                let start = Instant::now();
                for _ in 0..iters {
                    let _ = model.forward(&price_deltas, &static_features, false);
                }
                sync_device(device);
                let fwd_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
                println!("    Forward (batch={}):       {:.3} ms/iter", batch, fwd_ms);
                suite.add(BenchmarkResult::new(
                    &format!("model_{}_forward_infer_b{}", variant.as_str(), batch),
                    fwd_ms,
                    BenchmarkRun {
                        batch,
                        seq_len: reported_seq_len,
                        dtype: format!("{:?}", dtype),
                    },
                ));
            });

            if variant == ModelVariant::Uniform256Stream {
                let raw_full_price = Tensor::randn(&[batch, raw_price_deltas_dim], (dtype, device));
                let full_price = model.uniform_stream_layout_from_raw_input(&raw_full_price);
                let static_features = Tensor::randn(&[batch, static_obs_dim], (dtype, device));
                let step_deltas = Tensor::randn(&[batch, TICKERS_COUNT], (dtype, device));
                let mut stream_state = model.init_stream_state_batched(batch);
                let _ = model.step_on_device(&full_price, &static_features, &mut stream_state);
                sync_device(device);

                let iters = 100;
                tch::no_grad(|| {
                    let start = Instant::now();
                    for _ in 0..iters {
                        let _ =
                            model.step_on_device(&step_deltas, &static_features, &mut stream_state);
                    }
                    sync_device(device);
                    let step_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
                    println!(
                        "    Stream Step (batch={}):    {:.3} ms/iter",
                        batch, step_ms
                    );
                    suite.add(BenchmarkResult::new(
                        &format!("model_{}_stream_step_b{}", variant.as_str(), batch),
                        step_ms,
                        BenchmarkRun {
                            batch,
                            seq_len: 1,
                            dtype: format!("{:?}", dtype),
                        },
                    ));
                });
            }

            let price_deltas = if variant == ModelVariant::Uniform256Stream {
                let raw = Tensor::randn(&[batch, raw_price_deltas_dim], (dtype, device));
                model.uniform_stream_layout_from_raw_input(&raw).detach()
            } else {
                Tensor::randn(&[batch, price_deltas_dim], (dtype, device))
            };
            let static_features = Tensor::randn(&[batch, static_obs_dim], (dtype, device));

            let start = Instant::now();
            for _ in 0..iters {
                let (values, action_mean, action_noise_std) =
                    model.forward(&price_deltas, &static_features, true);
                let loss = values.sum(Kind::Float)
                    + action_mean.sum(Kind::Float)
                    + action_noise_std.sum(Kind::Float);
                loss.backward();
            }
            sync_device(device);
            let fwd_bwd_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            println!(
                "    Fwd+Bwd (batch={}):       {:.3} ms/iter",
                batch, fwd_bwd_ms
            );
            suite.add(BenchmarkResult::new(
                &format!("model_{}_fwd_bwd_train_b{}", variant.as_str(), batch),
                fwd_bwd_ms,
                BenchmarkRun {
                    batch,
                    seq_len: reported_seq_len,
                    dtype: format!("{:?}", dtype),
                },
            ));
        }
    }
    println!();
}

fn build_bench_mlp(vs: &nn::Path, dim: i64, depth: usize) -> impl Module {
    let mut seq = nn::seq();
    for i in 0..depth {
        seq = seq
            .add(nn::linear(vs / format!("fc{}", i), dim, dim, Default::default()))
            .add_fn(|x| x.gelu("none"));
    }
    seq.add(nn::linear(vs / "head", dim, 1, Default::default()))
}

fn run_optimizer_benchmarks(suite: &mut BenchmarkSuite, device: Device) {
    println!("--- Optimizer Step Benchmarks (Muon vs AdamW) ---");
    let dim = 256i64;
    let depth = 6;
    let batch = 64i64;
    let iters = 200;
    let warmup = 20;

    // --- AdamW ---
    {
        let mut vs = nn::VarStore::new(device);
        let net = build_bench_mlp(&vs.root(), dim, depth);
        vs.bfloat16();
        let trainable = vs.trainable_variables();
        let x = Tensor::randn(&[batch, dim], (Kind::BFloat16, device));

        // Baseline: model + grads + one step's activations already allocated,
        // so the delta isolates optimizer state (not transient activations).
        let loss = net.forward(&x).sum(Kind::Float);
        loss.backward();
        sync_device(device);
        let mem_pre_opt = gpu_mem_mib(device);

        let mut opt = nn::AdamW::default()
            .build(&vs, 3e-3)
            .expect("adamw build");
        for _ in 0..warmup {
            let loss = net.forward(&x).sum(Kind::Float);
            opt.backward_step(&loss);
        }
        sync_device(device);
        // Measure after warmup — allocator has stabilized.
        let mem_post_opt = gpu_mem_mib(device);
        let opt_vram = mem_pre_opt.zip(mem_post_opt).map(|(a, b)| b - a);

        let start = Instant::now();
        for _ in 0..iters {
            let loss = net.forward(&x).sum(Kind::Float);
            opt.backward_step(&loss);
        }
        sync_device(device);
        let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        // AdamW keeps m + v per param, same dtype as params.
        let adamw_state_bytes: usize = trainable
            .iter()
            .map(|t| 2 * t.numel() * t.kind().elt_size_in_bytes())
            .sum();
        let analytic_mib = adamw_state_bytes as f64 / 1024.0 / 1024.0;
        let empirical = opt_vram
            .map(|v| format!(", nvidia-smi: {:+.1} MiB", v))
            .unwrap_or_default();
        println!(
            "  AdamW  fwd+bwd+step: {:.3} ms/iter  | state: {:.2} MiB{}",
            ms, analytic_mib, empirical
        );
        suite.add(BenchmarkResult::new(
            "optim_adamw_fwd_bwd_step",
            ms,
            BenchmarkRun { batch, seq_len: dim, dtype: "BFloat16".into() },
        ));
    }

    // --- Muon ---
    {
        let mut vs = nn::VarStore::new(device);
        let net = build_bench_mlp(&vs.root(), dim, depth);
        vs.bfloat16();
        let trainable = vs.trainable_variables();
        let x = Tensor::randn(&[batch, dim], (Kind::BFloat16, device));

        let loss = net.forward(&x).sum(Kind::Float);
        loss.backward();
        sync_device(device);
        let mem_pre_opt = gpu_mem_mib(device);

        let mut muon = Muon::new(&trainable, MuonConfig::default());
        for _ in 0..warmup {
            let loss = net.forward(&x).sum(Kind::Float);
            loss.backward();
            muon.step();
            muon.zero_grad();
        }
        sync_device(device);
        let mem_post_opt = gpu_mem_mib(device);
        let opt_vram = mem_pre_opt.zip(mem_post_opt).map(|(a, b)| b - a);

        let start = Instant::now();
        for _ in 0..iters {
            let loss = net.forward(&x).sum(Kind::Float);
            loss.backward();
            muon.step();
            muon.zero_grad();
        }
        sync_device(device);
        let ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        let analytic_mib = muon.state_bytes() as f64 / 1024.0 / 1024.0;
        let empirical = opt_vram
            .map(|v| format!(", nvidia-smi: {:+.1} MiB", v))
            .unwrap_or_default();
        println!(
            "  Muon   fwd+bwd+step: {:.3} ms/iter  | state: {:.2} MiB{}",
            ms, analytic_mib, empirical
        );
        suite.add(BenchmarkResult::new(
            "optim_muon_fwd_bwd_step",
            ms,
            BenchmarkRun { batch, seq_len: dim, dtype: "BFloat16".into() },
        ));
    }

    println!();
}
