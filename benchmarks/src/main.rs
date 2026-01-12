#![feature(f16)]

mod results;

use std::time::Instant;
use tch::{nn, Device, Kind, Tensor};
use trading_bot_0::torch::{
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

    // Forward (Training)
    let iters = 200;
    let start = Instant::now();
    for _ in 0..iters {
        let _ = mamba.forward_with_dt_scale(&x, None);
    }
    sync_device(device);
    let fwd_ms = start.elapsed().as_secs_f64() * 1000.0 / iters as f64;
    println!("  Forward (Training):      {:.3} ms/iter", fwd_ms);
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

    // Use actual model dimensions from constants
    const TICKERS_COUNT: i64 = 1;
    const PRICE_DELTAS_PER_TICKER: i64 = 3400;
    const GLOBAL_STATIC_OBS: i64 = 7 + 8; // GLOBAL_MACRO_OBS = 8
    const PER_TICKER_STATIC_OBS: i64 = 19 + 12; // PER_TICKER_EARNINGS_OBS = 12
    const STATIC_OBS: i64 = GLOBAL_STATIC_OBS + TICKERS_COUNT * PER_TICKER_STATIC_OBS;
    const PRICE_DELTAS_DIM: i64 = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER;

    for &batch in &[1, 4, 8] {
        let price_deltas = Tensor::randn(&[batch, PRICE_DELTAS_DIM], (dtype, device));
        let static_features = Tensor::randn(&[batch, STATIC_OBS], (dtype, device));

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
                    seq_len: PRICE_DELTAS_PER_TICKER,
                    dtype: format!("{:?}", dtype),
                },
            ));
        });

        // Forward + Backward (Training) - need fresh tensors with grads
        let price_deltas = Tensor::randn(&[batch, PRICE_DELTAS_DIM], (dtype, device))
            .set_requires_grad(true);
        let static_features = Tensor::randn(&[batch, STATIC_OBS], (dtype, device))
            .set_requires_grad(true);

        let start = Instant::now();
        for _ in 0..iters {
            let (values, critic_logits, (action_mean, action_log_std), _attn_entropy) =
                model.forward(&price_deltas, &static_features, true);
            let loss = values.sum(Kind::Float)
                + critic_logits.sum(Kind::Float)
                + action_mean.sum(Kind::Float)
                + action_log_std.sum(Kind::Float);
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
                seq_len: PRICE_DELTAS_PER_TICKER,
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
