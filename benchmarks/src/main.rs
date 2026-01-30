#![feature(f16)]

mod results;

use std::time::Instant;
use tch::{nn, Device, Kind, Tensor};
use trading_bot_0::torch::{
    constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT},
    mamba_fused,
    model::TradingModel,
    ssm::{Mamba2, Mamba2Config},
};

use results::{BenchmarkResult, BenchmarkRun, BenchmarkSuite};

fn sync_device(device: Device) {
    if let Device::Cuda(id) = device {
        tch::Cuda::synchronize(id as i64);
    }
}

fn run_component_timing(mamba: &Mamba2, x: &Tensor, device: Device, iters: usize) {
    tch::no_grad(|| {
        let (batch, seqlen, d_model) = x.size3().unwrap();
        let dtype = x.kind();

        // Setup: create dummy tensors matching Mamba dimensions
        let config = Mamba2Config {
            d_model: 256,
            headdim: 64,
            d_state: 128,
            chunk_size: 256,
            ..Default::default()
        };
        let d_inner = config.d_inner();
        let d_ssm = d_inner;
        let ngroups = config.ngroups;
        let d_state = config.d_state;
        let nheads = d_ssm / config.headdim;
        let d_in_proj = 2 * d_inner + 2 * ngroups * d_state + nheads;

        // Time input projection (d_model -> d_in_proj)
        let w_in = Tensor::randn(&[d_in_proj, d_model], (dtype, device));
        let mut in_proj_time = 0.0;
        for _ in 0..iters {
            let start = Instant::now();
            let _ = x.matmul(&w_in.tr());
            sync_device(device);
            in_proj_time += start.elapsed().as_secs_f64();
        }
        in_proj_time = in_proj_time * 1000.0 / iters as f64;

        // Time Conv1D
        let conv_dim = d_ssm + 2 * ngroups * d_state;
        let xbc = Tensor::randn(&[batch, seqlen, conv_dim], (dtype, device));
        let xbc_t = xbc.transpose(1, 2);
        let conv_w = Tensor::randn(&[conv_dim, 1, config.d_conv], (dtype, device));
        let conv_b = Tensor::randn(&[conv_dim], (dtype, device));
        let mut conv_time = 0.0;
        for _ in 0..iters {
            let start = Instant::now();
            let _ = xbc_t.conv1d(&conv_w, Some(&conv_b), 1, config.d_conv - 1, 1, conv_dim);
            sync_device(device);
            conv_time += start.elapsed().as_secs_f64();
        }
        conv_time = conv_time * 1000.0 / iters as f64;

        // Time output projection (d_inner -> d_model)
        let y_dummy = Tensor::randn(&[batch, seqlen, d_inner], (dtype, device));
        let w_out = Tensor::randn(&[d_model, d_inner], (dtype, device));
        let mut out_proj_time = 0.0;
        for _ in 0..iters {
            let start = Instant::now();
            let _ = y_dummy.matmul(&w_out.tr());
            sync_device(device);
            out_proj_time += start.elapsed().as_secs_f64();
        }
        out_proj_time = out_proj_time * 1000.0 / iters as f64;

        // Time RMSNorm + gating
        let norm_w = Tensor::randn(&[d_ssm], (dtype, device));
        let y_norm = Tensor::randn(&[batch, seqlen, d_ssm], (dtype, device));
        let z_gate = Tensor::randn(&[batch, seqlen, d_ssm], (dtype, device));
        let mut norm_time = 0.0;
        for _ in 0..iters {
            let start = Instant::now();
            let y_f32 = y_norm.to_kind(tch::Kind::Float);
            let rms = (y_f32.pow_tensor_scalar(2).mean_dim(-1, true, tch::Kind::Float) + 1e-6).sqrt();
            let normed = (y_f32 / rms * &norm_w.to_kind(tch::Kind::Float)).to_kind(dtype);
            let _ = normed * z_gate.silu();
            sync_device(device);
            norm_time += start.elapsed().as_secs_f64();
        }
        norm_time = norm_time * 1000.0 / iters as f64;

        // Time the full forward pass
        let mut fused_time = 0.0;
        for _ in 0..iters {
            let start = Instant::now();
            let _ = mamba.forward_with_dt_scale(x, None);
            sync_device(device);
            fused_time += start.elapsed().as_secs_f64();
        }
        fused_time = fused_time * 1000.0 / iters as f64;

        // Estimate scan time (includes the CUDA kernel for chunk scan)
        let scan_time = fused_time - in_proj_time - conv_time - norm_time - out_proj_time;

        println!("  Input Projection:        {:.3} ms/iter ({:.1}%)",
                 in_proj_time, 100.0 * in_proj_time / fused_time);
        println!("  Conv1D:                  {:.3} ms/iter ({:.1}%)",
                 conv_time, 100.0 * conv_time / fused_time);
        println!("  Chunk Scan (CUDA):       {:.3} ms/iter ({:.1}%)",
                 scan_time, 100.0 * scan_time / fused_time);
        println!("  RMSNorm + Gate:          {:.3} ms/iter ({:.1}%)",
                 norm_time, 100.0 * norm_time / fused_time);
        println!("  Output Projection:       {:.3} ms/iter ({:.1}%)",
                 out_proj_time, 100.0 * out_proj_time / fused_time);
        println!("  Total:                   {:.3} ms/iter", fused_time);
    });
}

fn main() {
    let device = Device::cuda_if_available();
    if device == Device::Cpu {
        eprintln!("Warning: CUDA not detected, running on CPU (will be slow)");
    }
    let mut suite = BenchmarkSuite::new();

    println!("=== Trading Bot Benchmarks ===\n");

    run_mamba_benchmarks(&mut suite, device);
    run_model_benchmarks(&mut suite, device);

    suite.save().expect("Failed to save benchmark results");
    println!("\n=== Benchmarks Complete ===");
}

fn run_mamba_benchmarks(suite: &mut BenchmarkSuite, device: Device) {
    println!("--- Mamba2 SSM Benchmarks ---");
    let vs = nn::VarStore::new(device);
    let dtype = Kind::BFloat16;

    let config = Mamba2Config {
        d_model: 256,
        headdim: 64,
        d_state: 128,
        chunk_size: 256,
        ..Default::default()
    };
    let mamba = Mamba2::new(&vs.root(), config);
    for (_name, mut tensor) in vs.variables() {
        let _ = tensor.set_data(&tensor.to_kind(dtype));
    }

    let batch = 4;
    let seqlen = 4096;
    let x = Tensor::randn(&[batch, seqlen, 256], (dtype, device)).set_requires_grad(true);
    let mut state = mamba.init_state(batch, device);

    println!("  Batch: {}, SeqLen: {}, Dtype: {:?}", batch, seqlen, dtype);

    // Warmup
    for _ in 0..20 {
        let _ = mamba.forward_with_dt_scale(&x, None);
    }
    sync_device(device);

    // Component timing breakdown
    println!("\n  === Component Timing Breakdown ===");
    println!("  Note: For detailed CUDA kernel profiling, use CUDA_LAUNCH_BLOCKING=1 and nvprof");
    run_component_timing(&mamba, &x, device, 100);

    // Forward (Training)
    let iters = 200;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = mamba.forward_with_dt_scale(&x, None);
    }
    sync_device(device);
    let fwd_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("\n  Forward (Training):      {:.3} ms/iter", fwd_ms);
    suite.add(BenchmarkResult::new(
        "mamba2_forward_train",
        fwd_ms,
        BenchmarkRun { batch, seq_len: seqlen, dtype: format!("{:?}", dtype) },
    ));

    // Forward + Backward
    let start = Instant::now();
    for _ in 0..iters {
        let y = mamba.forward_with_dt_scale(&x, None);
        y.sum(Kind::Float).backward();
    }
    sync_device(device);
    let fwd_bwd_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("  Forward + Backward:      {:.3} ms/iter", fwd_bwd_ms);
    suite.add(BenchmarkResult::new(
        "mamba2_fwd_bwd_train",
        fwd_bwd_ms,
        BenchmarkRun { batch, seq_len: seqlen, dtype: format!("{:?}", dtype) },
    ));

    // Memory stats
    let stats = mamba_fused::cuda_memory_stats();
    let peak_mem_mb = stats.get(3).unwrap_or(&0) / (1024 * 1024);
    println!("  Peak Memory:             {} MB", peak_mem_mb);

    // Inference (Prefill)
    tch::no_grad(|| {
        let start = Instant::now();
        for _ in 0..iters {
            let _ = mamba.forward_with_state_dt_scale(&x, &mut state, None);
        }
        sync_device(device);
        let prefill_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("  Inference (Prefill):     {:.3} ms/iter", prefill_ms);
        suite.add(BenchmarkResult::new(
            "mamba2_infer_prefill",
            prefill_ms,
            BenchmarkRun { batch, seq_len: seqlen, dtype: format!("{:?}", dtype) },
        ));
    });

    // Inference (Step)
    let x_step = Tensor::randn(&[batch, 1, 256], (dtype, device));
    tch::no_grad(|| {
        let step_iters = 2000;
        let start = Instant::now();
        for _ in 0..step_iters {
            let _ = mamba.step(&x_step, &mut state);
        }
        sync_device(device);
        let step_ms = start.elapsed().as_secs_f64() * 1000.0 / step_iters as f64;
        println!("  Inference (Step):        {:.3} ms/iter", step_ms);
        suite.add(BenchmarkResult::new(
            "mamba2_infer_step",
            step_ms,
            BenchmarkRun { batch, seq_len: 1, dtype: format!("{:?}", dtype) },
        ));
    });

    println!();
}

fn run_model_benchmarks(suite: &mut BenchmarkSuite, device: Device) {
    println!("--- TradingModel Forward Benchmarks ---");
    let vs = nn::VarStore::new(device);

    // Cast varstore to BFloat16
    let model = TradingModel::new(&vs.root());
    let dtype = Kind::BFloat16;
    for (_name, mut tensor) in vs.variables() {
        let _ = tensor.set_data(&tensor.to_kind(dtype));
    }

    // Use actual model dimensions from shared constants
    let price_deltas_dim = (TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER) as i64;
    let static_obs_dim = STATIC_OBSERVATIONS as i64;

    for &batch in &[1, 4, 8] {
        let price_deltas = Tensor::randn(&[batch, price_deltas_dim], (dtype, device));
        let static_features = Tensor::randn(&[batch, static_obs_dim], (dtype, device));

        // Warmup
        for _ in 0..10 {
            let _ = model.forward(&price_deltas, &static_features, false);
        }
        sync_device(device);

        // Forward (Inference)
        let iters = 100;
        tch::no_grad(|| {
            let start = Instant::now();
            for _ in 0..iters {
                let _ = model.forward(&price_deltas, &static_features, false);
            }
            sync_device(device);
            let fwd_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
            println!("  Forward (batch={}):       {:.3} ms/iter", batch, fwd_ms);
            suite.add(BenchmarkResult::new(
                &format!("model_forward_infer_b{}", batch),
                fwd_ms,
                BenchmarkRun {
                    batch,
                    seq_len: PRICE_DELTAS_PER_TICKER as i64,
                    dtype: format!("{:?}", dtype),
                },
            ));
        });

        // Forward + Backward (Training) - need fresh tensors with grads
        let price_deltas = Tensor::randn(&[batch, price_deltas_dim], (dtype, device))
            .set_requires_grad(true);
        let static_features = Tensor::randn(&[batch, static_obs_dim], (dtype, device))
            .set_requires_grad(true);

        let start = Instant::now();
        for _ in 0..iters {
            let (values, critic_logits, (action_mean, sde_latent)) =
                model.forward(&price_deltas, &static_features, true);
            let loss = values.sum(Kind::Float)
                + critic_logits.sum(Kind::Float)
                + action_mean.sum(Kind::Float)
                + sde_latent.sum(Kind::Float);
            loss.backward();
        }
        sync_device(device);
        let fwd_bwd_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("  Fwd+Bwd (batch={}):       {:.3} ms/iter", batch, fwd_bwd_ms);
        suite.add(BenchmarkResult::new(
            &format!("model_fwd_bwd_train_b{}", batch),
            fwd_bwd_ms,
            BenchmarkRun {
                batch,
                seq_len: PRICE_DELTAS_PER_TICKER as i64,
                dtype: format!("{:?}", dtype),
            },
        ));
    }

    // Memory stats after model benchmarks
    let stats = mamba_fused::cuda_memory_stats();
    let peak_mem_mb = stats.get(3).unwrap_or(&0) / (1024 * 1024);
    println!("  Peak Memory:             {} MB", peak_mem_mb);
    println!();
}
