#![feature(f16)]

mod results;

use std::time::Instant;
use tch::{nn, Device, Kind, Tensor};
use trading_bot_0::torch::{
    constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT},
    model::{ModelVariant, TradingModel, TradingModelConfig},
};

use results::{BenchmarkResult, BenchmarkRun, BenchmarkSuite};

fn sync_device(device: Device) {
    if let Device::Cuda(id) = device {
        tch::Cuda::synchronize(id as i64);
    }
}

fn main() {
    let device = Device::cuda_if_available();
    if device == Device::Cpu {
        eprintln!("Warning: CUDA not detected, running on CPU (will be slow)");
    }
    let mut suite = BenchmarkSuite::new();

    println!("=== Trading Bot Benchmarks ===\n");

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
