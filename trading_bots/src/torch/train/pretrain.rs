use anyhow::{anyhow, Context, Result};
use rand::seq::SliceRandom;
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::{Path, PathBuf},
};
use tch::{autocast, nn, nn::Module, Device, Kind, Reduction, Tensor};

use crate::data::universe::cached_eligible_training_universe;
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::cuda::cfg::configure_cuda;
use crate::torch::env::Env;
use crate::torch::load::load_var_store_partial;
use crate::torch::model::{ModelVariant, TradingModel, TradingModelConfig};
use crate::torch::optim::muon::{Muon, MuonConfig};
use shared::{
    paths::RUNS_PATH,
    report::{Report, ReportKind, ReportSeries, ScaleKind},
    run_dir::RunDir,
};

use super::config::{LEARNING_RATE, MAX_GRAD_NORM, MUON_LR, MUON_MOMENTUM_WARMUP_START, USE_MUON};
use super::optimizer_glue::{muon_momentum_for_step, named_trainable_variables};

const HORIZON_FEATURE_DIM: i64 = 7;

#[derive(Clone, Debug)]
pub struct PretrainArgs {
    pub weights: Option<String>,
    pub model_size: ModelVariant,
    pub run: Option<String>,
    pub epochs: usize,
    pub steps: Option<usize>,
    pub batch_size: usize,
    pub k_patches: usize,
    pub lambda_lat: f64,
    pub target_scale: f64,
    pub validation_batches: usize,
    pub validate_every: usize,
    pub checkpoint_every: usize,
    pub log_step_losses: bool,
}

struct PretrainHeads {
    forecast_queries: Tensor,
    horizon_pos_proj: nn::Linear,
    forecast_q_proj: nn::Linear,
    forecast_k_proj: nn::Linear,
    forecast_v_proj: nn::Linear,
    forecast_out_proj: nn::Linear,
    return_mean: nn::Linear,
    next_patch_embed: nn::Linear,
    latent_fc1: nn::Linear,
    latent_fc2: nn::Linear,
    horizon: i64,
    latent_dim: i64,
    forecast_heads: i64,
    dropout: f64,
}

struct PretrainBatch {
    obs: Tensor,
    static_obs: Tensor,
    next_obs: Tensor,
    next_static_obs: Tensor,
    future_patches: Tensor,
    next_patch: Tensor,
}

impl PretrainBatch {
    fn len(&self) -> i64 {
        self.obs.size()[0]
    }

    fn narrow(&self, start: i64, len: i64) -> Self {
        Self {
            obs: self.obs.narrow(0, start, len),
            static_obs: self.static_obs.narrow(0, start, len),
            next_obs: self.next_obs.narrow(0, start, len),
            next_static_obs: self.next_static_obs.narrow(0, start, len),
            future_patches: self.future_patches.narrow(0, start, len),
            next_patch: self.next_patch.narrow(0, start, len),
        }
    }
}

struct PretrainSampler {
    train_tickers: Vec<String>,
    val_tickers: Vec<String>,
    train_ticker_cursor: usize,
    k_patches: usize,
    patch_size: usize,
    target_scale: f64,
    train_offsets: Vec<usize>,
    train_epoch: Option<PretrainBatch>,
    train_cursor: usize,
    device: Device,
}

#[derive(Clone, Copy)]
enum SplitKind {
    Train,
    Validation,
}

impl PretrainHeads {
    fn new(
        p: &nn::Path,
        latent_dim: i64,
        _patch_token_count: i64,
        k_patches: i64,
        patch_size: i64,
    ) -> Self {
        let ff_dim = latent_dim * 2;
        let horizon = k_patches * patch_size;
        let forecast_heads = 4;
        assert_eq!(
            latent_dim % forecast_heads,
            0,
            "forecast attention heads must divide latent dim"
        );
        let forecast_queries = p.var(
            "forecast_queries",
            &[horizon, latent_dim],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        let mut horizon_pos_proj = nn::linear(
            p / "horizon_pos_proj",
            HORIZON_FEATURE_DIM,
            latent_dim,
            Default::default(),
        );
        tch::no_grad(|| {
            let init = Tensor::randn(
                horizon_pos_proj.ws.size(),
                (horizon_pos_proj.ws.kind(), horizon_pos_proj.ws.device()),
            ) * 0.01;
            horizon_pos_proj.ws.copy_(&init);
            if let Some(bias) = horizon_pos_proj.bs.as_mut() {
                let _ = bias.zero_();
            }
        });
        let forecast_q_proj = nn::linear(
            p / "forecast_q_proj",
            latent_dim,
            latent_dim,
            Default::default(),
        );
        let forecast_k_proj = nn::linear(
            p / "forecast_k_proj",
            latent_dim,
            latent_dim,
            Default::default(),
        );
        let forecast_v_proj = nn::linear(
            p / "forecast_v_proj",
            latent_dim,
            latent_dim,
            Default::default(),
        );
        let forecast_out_proj = nn::linear(
            p / "forecast_out_proj",
            latent_dim,
            latent_dim,
            Default::default(),
        );
        let mut return_mean = nn::linear(p / "return_mean", latent_dim, 1, Default::default());
        tch::no_grad(|| {
            let init = Tensor::randn(
                return_mean.ws.size(),
                (return_mean.ws.kind(), return_mean.ws.device()),
            ) * 0.01;
            return_mean.ws.copy_(&init);
            if let Some(bias) = return_mean.bs.as_mut() {
                let _ = bias.zero_();
            }
        });
        let next_patch_embed = nn::linear(
            p / "next_patch_embed",
            patch_size,
            latent_dim,
            Default::default(),
        );
        let latent_fc1 = nn::linear(p / "latent_fc1", latent_dim * 2, ff_dim, Default::default());
        let latent_fc2 = nn::linear(p / "latent_fc2", ff_dim, latent_dim, Default::default());
        Self {
            forecast_queries,
            horizon_pos_proj,
            forecast_q_proj,
            forecast_k_proj,
            forecast_v_proj,
            forecast_out_proj,
            return_mean,
            next_patch_embed,
            latent_fc1,
            latent_fc2,
            horizon,
            latent_dim,
            forecast_heads,
            dropout: 0.1,
        }
    }

    fn horizon_features(&self, device: Device, kind: Kind) -> Tensor {
        let denom = (self.horizon - 1).max(1) as f64;
        let x = (Tensor::arange(self.horizon, (Kind::Float, device)) / denom).unsqueeze(-1);
        let centered = &x * 2.0 - 1.0;
        let squared = x.pow_tensor_scalar(2.0);
        let angle1 = &x * std::f64::consts::TAU;
        let sin1 = angle1.sin();
        let cos1 = angle1.cos();
        let angle2 = &x * (std::f64::consts::TAU * 2.0);
        let sin2 = angle2.sin();
        let cos2 = angle2.cos();
        Tensor::cat(&[&x, &centered, &squared, &sin1, &cos1, &sin2, &cos2], -1).to_kind(kind)
    }

    fn forecast_tokens(&self, patch_tokens: &Tensor, train: bool) -> (Tensor, i64, i64) {
        let size = patch_tokens.size();
        let batch = size[0];
        let tickers = size[1];
        let source_len = size[2];
        let rows = batch * tickers;
        let source = patch_tokens.view([rows, source_len, self.latent_dim]);
        let horizon_features = self.horizon_features(source.device(), source.kind());
        let base_queries = self.forecast_queries.to_kind(source.kind())
            + self.horizon_pos_proj.forward(&horizon_features);
        let queries = base_queries
            .unsqueeze(0)
            .expand([rows, self.horizon, self.latent_dim], false);

        let head_dim = self.latent_dim / self.forecast_heads;
        let q = self
            .forecast_q_proj
            .forward(&queries)
            .view([rows, self.horizon, self.forecast_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let k = self
            .forecast_k_proj
            .forward(&source)
            .view([rows, source_len, self.forecast_heads, head_dim])
            .permute([0, 2, 1, 3]);
        let v = self
            .forecast_v_proj
            .forward(&source)
            .view([rows, source_len, self.forecast_heads, head_dim])
            .permute([0, 2, 1, 3]);

        let attn_scores = q.matmul(&k.transpose(-2, -1)) / (head_dim as f64).sqrt();
        let attn = attn_scores
            .softmax(-1, Kind::Float)
            .dropout(self.dropout, train)
            .to_kind(v.kind());
        let attended = attn.matmul(&v).permute([0, 2, 1, 3]).contiguous().view([
            rows,
            self.horizon,
            self.latent_dim,
        ]);
        let forecast_tokens = queries
            + self
                .forecast_out_proj
                .forward(&attended)
                .dropout(self.dropout, train);
        (forecast_tokens, batch, tickers)
    }

    fn forecast_readout(&self, forecast_tokens: &Tensor, train: bool) -> Tensor {
        forecast_tokens.dropout(self.dropout, train)
    }

    fn return_mean_from_readout(&self, readout: &Tensor, batch: i64, tickers: i64) -> Tensor {
        self.return_mean
            .forward(&readout)
            .view([batch, tickers, self.horizon])
    }

    fn predict_return_mean(&self, patch_tokens: &Tensor, train: bool) -> Tensor {
        let (forecast_tokens, batch, tickers) = self.forecast_tokens(patch_tokens, train);
        let readout = self.forecast_readout(&forecast_tokens, train);
        self.return_mean_from_readout(&readout, batch, tickers)
    }

    fn predict_next_latent(&self, latent: &Tensor, next_patch: &Tensor) -> Tensor {
        let next_patch_embed = self.next_patch_embed.forward(next_patch);
        let x = Tensor::cat(&[latent, &next_patch_embed], -1);
        let x = normalize_last_dim(&x);
        latent + self.latent_fc2.forward(&self.latent_fc1.forward(&x).relu())
    }
}

impl PretrainSampler {
    fn new(k_patches: usize, patch_size: usize, target_scale: f64, device: Device) -> Self {
        assert_eq!(
            TICKERS_COUNT, 1,
            "full-universe pretraining currently expects one ticker per observation"
        );
        let val_tickers = cached_eligible_training_universe().to_vec();
        let mut train_tickers = val_tickers.clone();
        train_tickers.shuffle(&mut rand::rng());
        assert!(
            !train_tickers.is_empty(),
            "not enough market history for pretraining: train_tickers={}",
            train_tickers.len()
        );
        Self {
            train_tickers,
            val_tickers,
            train_ticker_cursor: 0,
            k_patches,
            patch_size,
            target_scale,
            train_offsets: Vec::new(),
            train_epoch: None,
            train_cursor: 0,
            device,
        }
    }

    fn start_epoch(&mut self) {
        self.train_tickers.shuffle(&mut rand::rng());
        self.train_ticker_cursor = 0;
        self.train_offsets.clear();
        self.train_epoch = None;
        self.train_cursor = 0;
    }

    fn next_train_batch(&mut self, batch_size: usize) -> Option<PretrainBatch> {
        loop {
            if let Some(epoch) = self.train_epoch.as_ref() {
                let epoch_len = epoch.len() as usize;
                if self.train_cursor < epoch_len {
                    let end = (self.train_cursor + batch_size).min(epoch_len);
                    let batch =
                        epoch.narrow(self.train_cursor as i64, (end - self.train_cursor) as i64);
                    self.train_cursor = end;
                    return Some(batch);
                }
            }

            if !self.load_next_train_ticker_epoch() {
                return None;
            }
        }
    }

    fn load_next_train_ticker_epoch(&mut self) -> bool {
        while self.train_ticker_cursor < self.train_tickers.len() {
            let ticker = self.train_tickers[self.train_ticker_cursor].clone();
            self.train_ticker_cursor += 1;
            let mut env = Env::new_with_tickers_and_recording(vec![ticker], true, false, None);
            let mut offsets = build_split_offsets(
                env.price_deltas[0].len(),
                self.k_patches,
                self.patch_size,
                SplitKind::Train,
            );
            if offsets.is_empty() {
                continue;
            }
            offsets.shuffle(&mut rand::rng());
            self.train_epoch = Some(Self::batch_from_offsets(
                &mut env,
                &offsets,
                self.k_patches,
                self.patch_size,
                self.target_scale,
                self.device,
            ));
            self.train_offsets = offsets;
            self.train_cursor = 0;
            return true;
        }
        false
    }

    fn batch_from_offsets(
        env: &mut Env,
        offsets: &[usize],
        k_patches: usize,
        patch_size: usize,
        target_scale: f64,
        device: Device,
    ) -> PretrainBatch {
        let pd_dim = TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER;
        let so_dim = STATIC_OBSERVATIONS;
        let target_len = TICKERS_COUNT as usize * k_patches * patch_size;
        let next_patch_len = TICKERS_COUNT as usize * patch_size;

        let mut obs = Vec::with_capacity(offsets.len() * pd_dim);
        let mut static_obs = Vec::with_capacity(offsets.len() * so_dim);
        let mut next_obs = Vec::with_capacity(offsets.len() * pd_dim);
        let mut next_static_obs = Vec::with_capacity(offsets.len() * so_dim);
        let mut future_patches = Vec::with_capacity(offsets.len() * target_len);
        let mut next_patch = Vec::with_capacity(offsets.len() * next_patch_len);

        for &offset in offsets {
            let (obs_i, static_i) = env.reset_single_at_offset_for_pretrain(offset);
            let target_i =
                future_patches_for_current_perm(env, offset, k_patches, patch_size, target_scale);
            let next_patch_i = future_patches_for_current_perm(env, offset, 1, patch_size, 1.0);
            let (next_obs_i, next_static_i) =
                env.reset_single_at_offset_preserving_perm_for_pretrain(offset + patch_size);

            obs.extend(obs_i);
            static_obs.extend(static_i);
            future_patches.extend(target_i);
            next_patch.extend(next_patch_i);
            next_obs.extend(next_obs_i);
            next_static_obs.extend(next_static_i);
        }

        let batch = offsets.len() as i64;
        PretrainBatch {
            obs: Tensor::from_slice(&obs)
                .view([batch, pd_dim as i64])
                .to_device(device),
            static_obs: Tensor::from_slice(&static_obs)
                .view([batch, so_dim as i64])
                .to_device(device),
            next_obs: Tensor::from_slice(&next_obs)
                .view([batch, pd_dim as i64])
                .to_device(device),
            next_static_obs: Tensor::from_slice(&next_static_obs)
                .view([batch, so_dim as i64])
                .to_device(device),
            future_patches: Tensor::from_slice(&future_patches)
                .view([batch, TICKERS_COUNT, k_patches as i64, patch_size as i64])
                .to_device(device),
            next_patch: Tensor::from_slice(&next_patch)
                .view([batch, TICKERS_COUNT, patch_size as i64])
                .to_device(device),
        }
    }
}

fn build_split_offsets(
    data_len: usize,
    k_patches: usize,
    patch_size: usize,
    split_kind: SplitKind,
) -> Vec<usize> {
    let min_offset = PRICE_DELTAS_PER_TICKER;
    let horizon = k_patches * patch_size;
    let next_latent_advance = patch_size;
    let max_target_advance = horizon.max(next_latent_advance);
    let max_exclusive = data_len.saturating_sub(max_target_advance);
    if max_exclusive <= min_offset {
        return Vec::new();
    }
    let split_raw = min_offset + ((max_exclusive - min_offset) * 8 / 10).max(1);
    let split = align_up_to_step(split_raw, min_offset, patch_size);
    let train_max_exclusive = split.saturating_sub(max_target_advance);
    let (start, end) = match split_kind {
        SplitKind::Train => (min_offset, train_max_exclusive),
        SplitKind::Validation => (split, max_exclusive),
    };
    if start >= end {
        return Vec::new();
    }
    (start..end).step_by(patch_size).collect()
}

fn align_up_to_step(value: usize, origin: usize, step: usize) -> usize {
    let rem = (value - origin) % step;
    if rem == 0 {
        value
    } else {
        value + (step - rem)
    }
}

pub fn pretrain(args: PretrainArgs) -> Result<()> {
    assert_eq!(
        args.model_size,
        ModelVariant::UniformStream,
        "world-model pretraining currently supports --model-size uniform-stream only"
    );
    assert!(args.epochs > 0, "--epochs must be positive");
    if let Some(steps) = args.steps {
        assert!(steps > 0, "--steps must be positive when provided");
    }
    assert!(args.batch_size > 0, "--batch-size must be positive");
    assert!(args.k_patches > 0, "--k-patches must be positive");
    assert!(
        args.lambda_lat.is_finite() && args.lambda_lat >= 0.0,
        "--lambda-lat must be finite and non-negative"
    );
    assert!(
        args.target_scale.is_finite() && args.target_scale > 0.0,
        "--target-scale must be finite and positive"
    );
    configure_threads();
    let device = tch::Device::cuda_if_available();
    println!("device is cuda: {}", device.is_cuda());
    configure_cuda();

    let run_dir =
        RunDir::create_fresh(RUNS_PATH, args.run.as_deref()).expect("failed to create run dir");
    println!("Run dir: {}", run_dir.root.display());

    let mut model_vs = nn::VarStore::new(device);
    let model = TradingModel::new_with_config(
        &model_vs.root(),
        TradingModelConfig {
            variant: args.model_size,
            ..TradingModelConfig::default()
        },
    );
    let start_weights = args.weights.as_deref().map(PathBuf::from);
    if let Some(path) = &start_weights {
        println!("Loading pretrain start weights from {}", path.display());
        let load_summary =
            load_var_store_partial(&mut model_vs, path).map_err(|err| anyhow!("{err}"))?;
        load_summary
            .require_complete()
            .map_err(|err| anyhow!("{err}"))?;
    }

    let patch_size = model.pretrain_patch_size();
    assert_eq!(
        args.k_patches as i64 * patch_size,
        args.k_patches as i64 * model.pretrain_patch_size()
    );
    let mut sampler = PretrainSampler::new(
        args.k_patches,
        patch_size as usize,
        args.target_scale,
        device,
    );
    let mut head_vs = nn::VarStore::new(device);
    let heads = PretrainHeads::new(
        &head_vs.root(),
        model.pretrain_latent_dim(),
        model.pretrain_patch_token_count(),
        args.k_patches as i64,
        patch_size,
    );
    if let Some(path) = start_weights.as_deref() {
        load_matching_pretrain_heads(&mut head_vs, path)?;
    }

    let mut named_vars = named_trainable_variables(&model_vs);
    named_vars.extend(
        named_trainable_variables(&head_vs)
            .into_iter()
            .map(|(name, tensor)| (format!("pretrain_heads.{name}"), tensor)),
    );
    let mut opt = Muon::new_named(
        &named_vars,
        MuonConfig {
            lr: MUON_LR,
            use_muon_for_2d: USE_MUON,
            momentum: MUON_MOMENTUM_WARMUP_START,
            adamw_lr: LEARNING_RATE,
            adamw_betas: (0.9, 0.95),
            adamw_eps: 1e-8,
            weight_decay: 0.0,
            adamw_wd: 0.0,
            force_adamw_name_substrings: vec![
                "policy_concentration".to_string(),
                "value_proj".to_string(),
                "forecast_".to_string(),
                "horizon_pos_proj".to_string(),
                "return_mean".to_string(),
            ],
            ..MuonConfig::default()
        },
    );

    let mut optimizer_step = 0i64;
    let mut global_step = 0usize;
    let mut best_val = f64::INFINITY;
    let mut stop_requested = false;
    let final_path = run_dir.weights.join("pretrain_model.ot");
    let best_path = run_dir.weights.join("pretrain_model_best.ot");
    let final_heads_path = run_dir.weights.join("pretrain_heads.ot");
    let best_heads_path = run_dir.weights.join("pretrain_heads_best.ot");
    let mut train_epoch_log = BufWriter::new(File::create(
        run_dir.root.join("pretrain_train_epochs.csv"),
    )?);
    let mut validation_log =
        BufWriter::new(File::create(run_dir.root.join("pretrain_validation.csv"))?);
    writeln!(
        train_epoch_log,
        "epoch,global_step,total_loss,return_mse,return_mae,return_bias,pred_abs,target_abs,pred_std,target_std,terminal_mse,next_lat,samples,batches"
    )?;
    writeln!(
        validation_log,
        "epoch,global_step,total_loss,return_mse,return_mae,return_bias,pred_abs,target_abs,pred_std,target_std,terminal_mse,next_lat,zero_mse,samples,tickers,batches"
    )?;
    let mut step_log = if args.log_step_losses {
        let mut log = BufWriter::new(File::create(run_dir.root.join("pretrain_train_steps.csv"))?);
        writeln!(
            log,
            "global_step,epoch,total_loss,return_mse,return_mae,return_bias,pred_abs,target_abs,pred_std,target_std,terminal_mse,next_lat,samples"
        )?;
        Some(log)
    } else {
        None
    };

    'epoch_loop: for epoch in 1..=args.epochs {
        sampler.start_epoch();
        let mut train_epoch_loss = RunningLoss::new(device);
        println!(
            "pretrain epoch {epoch}/{} tickers={} batch_size={}",
            args.epochs,
            sampler.train_tickers.len(),
            args.batch_size
        );

        while let Some(batch) = sampler.next_train_batch(args.batch_size) {
            global_step += 1;
            let losses = pretrain_loss(&model, &heads, &batch, args.lambda_lat, true);
            let batch_samples = batch.len() as usize;
            train_epoch_loss.add(&losses, batch_samples);
            assert_finite_loss(&losses.total, global_step);
            opt.zero_grad();
            losses.total.backward();
            clip_all_grads(&named_vars, MAX_GRAD_NORM, device);
            opt.set_momentum(muon_momentum_for_step(optimizer_step));
            opt.step();
            optimizer_step += 1;

            let mut scalar_losses = None;
            if let Some(log) = step_log.as_mut() {
                let total_v = losses.total.double_value(&[]);
                let return_mse_v = losses.return_mse.double_value(&[]);
                let return_mae_v = losses.return_mae.double_value(&[]);
                let return_bias_v = losses.return_bias.double_value(&[]);
                let pred_abs_v = losses.pred_abs.double_value(&[]);
                let target_abs_v = losses.target_abs.double_value(&[]);
                let pred_std_v = losses.pred_std.double_value(&[]);
                let target_std_v = losses.target_std.double_value(&[]);
                let terminal_mse_v = losses.terminal_mse.double_value(&[]);
                let lat_v = losses.next_lat.double_value(&[]);
                writeln!(
                    log,
                    "{global_step},{epoch},{total_v:.9},{return_mse_v:.9},{return_mae_v:.9},{return_bias_v:.9},{pred_abs_v:.9},{target_abs_v:.9},{pred_std_v:.9},{target_std_v:.9},{terminal_mse_v:.9},{lat_v:.9},{batch_samples}"
                )?;
                scalar_losses = Some((
                    total_v,
                    return_mse_v,
                    return_mae_v,
                    return_bias_v,
                    pred_abs_v,
                    target_abs_v,
                    pred_std_v,
                    target_std_v,
                    terminal_mse_v,
                    lat_v,
                ));
            }

            if global_step == 1 || global_step % 20 == 0 {
                let (
                    total_v,
                    return_mse_v,
                    return_mae_v,
                    return_bias_v,
                    pred_abs_v,
                    target_abs_v,
                    pred_std_v,
                    target_std_v,
                    terminal_mse_v,
                    lat_v,
                ) = scalar_losses.unwrap_or_else(|| {
                    (
                        losses.total.double_value(&[]),
                        losses.return_mse.double_value(&[]),
                        losses.return_mae.double_value(&[]),
                        losses.return_bias.double_value(&[]),
                        losses.pred_abs.double_value(&[]),
                        losses.target_abs.double_value(&[]),
                        losses.pred_std.double_value(&[]),
                        losses.target_std.double_value(&[]),
                        losses.terminal_mse.double_value(&[]),
                        losses.next_lat.double_value(&[]),
                    )
                });
                println!(
                    "pretrain epoch {epoch} step {global_step} train total_loss={:.6} return_mse={:.6} return_mae={:.6} return_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} terminal_mse={:.6} next_lat={:.6}",
                    total_v,
                    return_mse_v,
                    return_mae_v,
                    return_bias_v,
                    pred_abs_v,
                    target_abs_v,
                    pred_std_v,
                    target_std_v,
                    terminal_mse_v,
                    lat_v
                );
            }

            if args.validate_every > 0 && global_step % args.validate_every == 0 {
                let val = validate_full(
                    &model,
                    &heads,
                    &mut sampler,
                    args.batch_size,
                    validation_batch_cap(args.validation_batches),
                    args.lambda_lat,
                    device,
                );
                println!(
                    "pretrain step {global_step} validation total_loss={:.6} return_mse={:.6} return_mae={:.6} return_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} terminal_mse={:.6} next_lat={:.6} zero_mse={:.6} samples={} tickers={} batches={}",
                    val.total,
                    val.return_mse,
                    val.return_mae,
                    val.return_bias,
                    val.pred_abs,
                    val.target_abs,
                    val.pred_std,
                    val.target_std,
                    val.terminal_mse,
                    val.next_lat,
                    val.zero_mse,
                    val.samples,
                    val.tickers,
                    val.batches
                );
                writeln!(
                    validation_log,
                    "step:{global_step},{global_step},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{}",
                    val.total,
                    val.return_mse,
                    val.return_mae,
                    val.return_bias,
                    val.pred_abs,
                    val.target_abs,
                    val.pred_std,
                    val.target_std,
                    val.terminal_mse,
                    val.next_lat,
                    val.zero_mse,
                    val.samples,
                    val.tickers,
                    val.batches
                )?;
                if val.total < best_val {
                    best_val = val.total;
                    model_vs.save(&best_path)?;
                    head_vs.save(&best_heads_path)?;
                    println!("Saved best pretrained model: {}", best_path.display());
                }
            }

            if args.checkpoint_every > 0 && global_step % args.checkpoint_every == 0 {
                let path = pretrain_step_model_path(&run_dir.weights, global_step);
                let heads_path = pretrain_step_heads_path(&run_dir.weights, global_step);
                model_vs.save(&path)?;
                head_vs.save(&heads_path)?;
                println!(
                    "Saved pretrained checkpoint: {} and {}",
                    path.display(),
                    heads_path.display()
                );
            }

            if args.steps.is_some_and(|max_steps| global_step >= max_steps) {
                stop_requested = true;
                break;
            }
        }

        let train = train_epoch_loss.finish();
        println!(
            "pretrain epoch {epoch} train_mean total_loss={:.6} return_mse={:.6} return_mae={:.6} return_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} terminal_mse={:.6} next_lat={:.6} samples={} batches={}",
            train.total,
            train.return_mse,
            train.return_mae,
            train.return_bias,
            train.pred_abs,
            train.target_abs,
            train.pred_std,
            train.target_std,
            train.terminal_mse,
            train.next_lat,
            train.samples,
            train.batches
        );
        writeln!(
            train_epoch_log,
            "{epoch},{global_step},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{}",
            train.total,
            train.return_mse,
            train.return_mae,
            train.return_bias,
            train.pred_abs,
            train.target_abs,
            train.pred_std,
            train.target_std,
            train.terminal_mse,
            train.next_lat,
            train.samples,
            train.batches
        )?;
        train_epoch_log.flush()?;
        if let Some(log) = step_log.as_mut() {
            log.flush()?;
        }

        let val = validate_full(
            &model,
            &heads,
            &mut sampler,
            args.batch_size,
            validation_batch_cap(args.validation_batches),
            args.lambda_lat,
            device,
        );
        println!(
            "pretrain epoch {epoch} validation total_loss={:.6} return_mse={:.6} return_mae={:.6} return_bias={:.6} pred_abs={:.6} target_abs={:.6} pred_std={:.6} target_std={:.6} terminal_mse={:.6} next_lat={:.6} zero_mse={:.6} samples={} tickers={} batches={}",
            val.total,
            val.return_mse,
            val.return_mae,
            val.return_bias,
            val.pred_abs,
            val.target_abs,
            val.pred_std,
            val.target_std,
            val.terminal_mse,
            val.next_lat,
            val.zero_mse,
            val.samples,
            val.tickers,
            val.batches
        );
        writeln!(
            validation_log,
            "{epoch},{global_step},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{}",
            val.total,
            val.return_mse,
            val.return_mae,
            val.return_bias,
            val.pred_abs,
            val.target_abs,
            val.pred_std,
            val.target_std,
            val.terminal_mse,
            val.next_lat,
            val.zero_mse,
            val.samples,
            val.tickers,
            val.batches
        )?;
        validation_log.flush()?;
        write_pretrain_diagnostics(
            &model,
            &heads,
            &mut sampler,
            args.batch_size,
            validation_batch_cap(args.validation_batches),
            epoch,
            global_step,
            &run_dir.gens,
            device,
        )?;
        if val.total < best_val {
            best_val = val.total;
            model_vs.save(&best_path)?;
            head_vs.save(&best_heads_path)?;
            println!("Saved best pretrained model: {}", best_path.display());
        }
        if stop_requested {
            break 'epoch_loop;
        }
    }

    if best_val == f64::INFINITY {
        let val = validate_full(
            &model,
            &heads,
            &mut sampler,
            args.batch_size,
            validation_batch_cap(args.validation_batches),
            args.lambda_lat,
            device,
        );
        best_val = val.total;
        writeln!(
            validation_log,
            "final,{global_step},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{:.9},{},{},{}",
            val.total,
            val.return_mse,
            val.return_mae,
            val.return_bias,
            val.pred_abs,
            val.target_abs,
            val.pred_std,
            val.target_std,
            val.terminal_mse,
            val.next_lat,
            val.zero_mse,
            val.samples,
            val.tickers,
            val.batches
        )?;
        validation_log.flush()?;
        model_vs.save(&best_path)?;
        head_vs.save(&best_heads_path)?;
        println!("Saved best pretrained model: {}", best_path.display());
    }

    if best_path.exists() {
        model_vs.load(&best_path)?;
    }
    if best_heads_path.exists() {
        head_vs.load(&best_heads_path)?;
    }
    model_vs.save(&final_path)?;
    head_vs.save(&final_heads_path)?;
    println!(
        "Saved final pretrained model: {} (best validation total_loss {:.6})",
        final_path.display(),
        best_val
    );
    Ok(())
}

fn load_matching_pretrain_heads(head_vs: &mut nn::VarStore, model_path: &Path) -> Result<()> {
    let Some(heads_path) = matching_pretrain_heads_path(model_path) else {
        return Ok(());
    };
    if !heads_path.exists() {
        return Err(anyhow!(
            "matching pretrain heads {} not found for model checkpoint {}",
            heads_path.display(),
            model_path.display()
        ));
    }
    let load_summary =
        load_var_store_partial(head_vs, &heads_path).map_err(|err| anyhow!("{err}"))?;
    load_summary
        .require_complete()
        .map_err(|err| anyhow!("failed to load complete pretrain heads: {err}"))?;
    println!("Loaded pretrain heads from {}", heads_path.display());
    Ok(())
}

fn matching_pretrain_heads_path(model_path: &Path) -> Option<PathBuf> {
    let parent = model_path.parent()?;
    let name = model_path.file_name()?.to_str()?;
    match name {
        "pretrain_model.ot" => Some(parent.join("pretrain_heads.ot")),
        "pretrain_model_best.ot" => Some(parent.join("pretrain_heads_best.ot")),
        _ => name
            .strip_prefix("pretrain_step")
            .and_then(|suffix| suffix.strip_suffix(".ot"))
            .map(|step| parent.join(format!("pretrain_heads_step{step}.ot"))),
    }
}

fn pretrain_step_model_path(weights_dir: &Path, global_step: usize) -> PathBuf {
    weights_dir.join(format!("pretrain_step{global_step}.ot"))
}

fn pretrain_step_heads_path(weights_dir: &Path, global_step: usize) -> PathBuf {
    weights_dir.join(format!("pretrain_heads_step{global_step}.ot"))
}

fn pretrain_loss(
    model: &TradingModel,
    heads: &PretrainHeads,
    batch: &PretrainBatch,
    lambda_lat: f64,
    train: bool,
) -> PretrainLoss {
    let batch_size = batch.obs.size()[0];
    let layout_len = model.pretrain_layout_len();
    let layouts = model
        .uniform_stream_layout_from_raw_input(&batch.obs)
        .view([batch_size * TICKERS_COUNT, layout_len]);

    let (patch_tokens, latent) = if lambda_lat == 0.0 {
        let patch_tokens = autocast(false, || {
            model.pretrain_patch_tokens(&layouts, &batch.static_obs, batch_size)
        });
        (patch_tokens, None)
    } else {
        let (patch_tokens, latent) = autocast(false, || {
            model.pretrain_patch_tokens_and_actor_latents(&layouts, &batch.static_obs, batch_size)
        });
        (patch_tokens, Some(latent))
    };
    let (forecast_tokens, forecast_batch, forecast_tickers) =
        heads.forecast_tokens(&patch_tokens, train);
    debug_assert_eq!(forecast_batch, batch_size);
    debug_assert_eq!(forecast_tickers, TICKERS_COUNT as i64);
    let forecast_readout = heads.forecast_readout(&forecast_tokens, train);
    let return_target = cumulative_future_returns(&batch.future_patches);
    let return_pred =
        heads.return_mean_from_readout(&forecast_readout, forecast_batch, forecast_tickers);
    let return_mse = return_pred.mse_loss(&return_target, Reduction::Mean);
    let return_err = &return_pred - &return_target;
    let return_mae = return_err.abs().mean(Kind::Float);
    let return_bias = return_err.mean(Kind::Float);
    let pred_abs = return_pred.abs().mean(Kind::Float);
    let target_abs = return_target.abs().mean(Kind::Float);
    let pred_std = return_pred.std(false);
    let target_std = return_target.std(false);
    let terminal_idx = heads.horizon - 1;
    let terminal_pred = return_pred.select(-1, terminal_idx);
    let terminal_target = return_target.select(-1, terminal_idx);
    let terminal_mse = terminal_pred.mse_loss(&terminal_target, Reduction::Mean);
    let base_loss = return_mse.shallow_clone();

    if lambda_lat == 0.0 {
        let next_lat = Tensor::zeros([], (Kind::Float, pred_abs.device()));
        return PretrainLoss {
            total: base_loss,
            return_mae,
            return_mse,
            pred_std,
            target_std,
            return_bias,
            pred_abs,
            target_abs,
            next_lat,
            terminal_mse,
        };
    }

    let latent = latent.expect("latent pretrain state should be computed when lambda_lat > 0");
    let next_layouts = model
        .uniform_stream_layout_from_raw_input(&batch.next_obs)
        .view([batch_size * TICKERS_COUNT, layout_len]);
    let next_latent = tch::no_grad(|| {
        autocast(false, || {
            model.pretrain_actor_latents(&next_layouts, &batch.next_static_obs, batch_size)
        })
    });
    let pred_next_latent = heads.predict_next_latent(&latent, &batch.next_patch);
    let latent_loss = pred_next_latent.smooth_l1_loss(&next_latent, Reduction::Mean, 1.0);
    let total = &base_loss + &latent_loss * lambda_lat;
    PretrainLoss {
        total,
        return_mae,
        return_mse,
        pred_std,
        target_std,
        return_bias,
        pred_abs,
        target_abs,
        next_lat: latent_loss,
        terminal_mse,
    }
}

fn cumulative_future_returns(future_patches: &Tensor) -> Tensor {
    let size = future_patches.size();
    future_patches
        .view([size[0], size[1], size[2] * size[3]])
        .cumsum(-1, Kind::Float)
}

fn predict_future_returns(
    model: &TradingModel,
    heads: &PretrainHeads,
    batch: &PretrainBatch,
) -> Tensor {
    let batch_size = batch.obs.size()[0];
    let layout_len = model.pretrain_layout_len();
    let layouts = model
        .uniform_stream_layout_from_raw_input(&batch.obs)
        .view([batch_size * TICKERS_COUNT, layout_len]);
    let patch_tokens = autocast(false, || {
        model.pretrain_patch_tokens(&layouts, &batch.static_obs, batch_size)
    });
    heads.predict_return_mean(&patch_tokens, false)
}

struct ValidationLoss {
    total: f64,
    return_mae: f64,
    return_mse: f64,
    pred_std: f64,
    target_std: f64,
    return_bias: f64,
    pred_abs: f64,
    target_abs: f64,
    next_lat: f64,
    terminal_mse: f64,
    zero_mse: f64,
    samples: usize,
    tickers: usize,
    batches: usize,
}

struct PretrainLoss {
    total: Tensor,
    return_mae: Tensor,
    return_mse: Tensor,
    pred_std: Tensor,
    target_std: Tensor,
    return_bias: Tensor,
    pred_abs: Tensor,
    target_abs: Tensor,
    next_lat: Tensor,
    terminal_mse: Tensor,
}

struct RunningLoss {
    total_sum: Tensor,
    return_mae_sum: Tensor,
    return_mse_sum: Tensor,
    pred_std_sum: Tensor,
    target_std_sum: Tensor,
    return_bias_sum: Tensor,
    pred_abs_sum: Tensor,
    target_abs_sum: Tensor,
    next_lat_sum: Tensor,
    terminal_mse_sum: Tensor,
    samples: usize,
    batches: usize,
}

impl RunningLoss {
    fn new(device: Device) -> Self {
        Self {
            total_sum: Tensor::zeros([], (Kind::Float, device)),
            return_mae_sum: Tensor::zeros([], (Kind::Float, device)),
            return_mse_sum: Tensor::zeros([], (Kind::Float, device)),
            pred_std_sum: Tensor::zeros([], (Kind::Float, device)),
            target_std_sum: Tensor::zeros([], (Kind::Float, device)),
            return_bias_sum: Tensor::zeros([], (Kind::Float, device)),
            pred_abs_sum: Tensor::zeros([], (Kind::Float, device)),
            target_abs_sum: Tensor::zeros([], (Kind::Float, device)),
            next_lat_sum: Tensor::zeros([], (Kind::Float, device)),
            terminal_mse_sum: Tensor::zeros([], (Kind::Float, device)),
            samples: 0,
            batches: 0,
        }
    }

    fn add(&mut self, losses: &PretrainLoss, samples: usize) {
        tch::no_grad(|| {
            let weight = samples as f64;
            self.total_sum += losses.total.detach() * weight;
            self.return_mae_sum += losses.return_mae.detach() * weight;
            self.return_mse_sum += losses.return_mse.detach() * weight;
            self.pred_std_sum += losses.pred_std.detach() * weight;
            self.target_std_sum += losses.target_std.detach() * weight;
            self.return_bias_sum += losses.return_bias.detach() * weight;
            self.pred_abs_sum += losses.pred_abs.detach() * weight;
            self.target_abs_sum += losses.target_abs.detach() * weight;
            self.next_lat_sum += losses.next_lat.detach() * weight;
            self.terminal_mse_sum += losses.terminal_mse.detach() * weight;
            self.samples += samples;
            self.batches += 1;
        });
    }

    fn finish(self) -> TrainEpochLoss {
        assert!(self.samples > 0, "train epoch is empty");
        let denom = self.samples as f64;
        TrainEpochLoss {
            total: self.total_sum.double_value(&[]) / denom,
            return_mae: self.return_mae_sum.double_value(&[]) / denom,
            return_mse: self.return_mse_sum.double_value(&[]) / denom,
            pred_std: self.pred_std_sum.double_value(&[]) / denom,
            target_std: self.target_std_sum.double_value(&[]) / denom,
            return_bias: self.return_bias_sum.double_value(&[]) / denom,
            pred_abs: self.pred_abs_sum.double_value(&[]) / denom,
            target_abs: self.target_abs_sum.double_value(&[]) / denom,
            next_lat: self.next_lat_sum.double_value(&[]) / denom,
            terminal_mse: self.terminal_mse_sum.double_value(&[]) / denom,
            samples: self.samples,
            batches: self.batches,
        }
    }
}

struct TrainEpochLoss {
    total: f64,
    return_mae: f64,
    return_mse: f64,
    pred_std: f64,
    target_std: f64,
    return_bias: f64,
    pred_abs: f64,
    target_abs: f64,
    next_lat: f64,
    terminal_mse: f64,
    samples: usize,
    batches: usize,
}

fn validation_batch_cap(validation_batches: usize) -> Option<usize> {
    (validation_batches > 0).then_some(validation_batches)
}

fn validate_full(
    model: &TradingModel,
    heads: &PretrainHeads,
    sampler: &mut PretrainSampler,
    batch_size: usize,
    max_batches: Option<usize>,
    lambda_lat: f64,
    device: Device,
) -> ValidationLoss {
    tch::no_grad(|| {
        let mut total_sum = 0.0;
        let mut return_mae_sum = 0.0;
        let mut return_mse_sum = 0.0;
        let mut pred_std_sum = 0.0;
        let mut target_std_sum = 0.0;
        let mut return_bias_sum = 0.0;
        let mut pred_abs_sum = 0.0;
        let mut target_abs_sum = 0.0;
        let mut next_lat_sum = 0.0;
        let mut terminal_mse_sum = 0.0;
        let mut zero_mse_sum = 0.0;
        let mut samples = 0usize;
        let mut tickers = 0usize;
        let mut batches = 0usize;

        for ticker in sampler.val_tickers.clone() {
            if max_batches.is_some_and(|limit| batches >= limit) {
                break;
            }

            let mut env = Env::new_with_tickers_and_recording(vec![ticker], false, false, None);
            let offsets = build_split_offsets(
                env.price_deltas[0].len(),
                sampler.k_patches,
                sampler.patch_size,
                SplitKind::Validation,
            );
            if offsets.is_empty() {
                continue;
            }
            tickers += 1;

            for chunk in offsets.chunks(batch_size) {
                if max_batches.is_some_and(|limit| batches >= limit) {
                    break;
                }

                let batch = PretrainSampler::batch_from_offsets(
                    &mut env,
                    chunk,
                    sampler.k_patches,
                    sampler.patch_size,
                    sampler.target_scale,
                    device,
                );
                let batch_samples = batch.len() as usize;
                let losses = pretrain_loss(model, heads, &batch, lambda_lat, false);
                let return_target = cumulative_future_returns(&batch.future_patches);
                let zero_mse_loss = return_target.pow_tensor_scalar(2.0).mean(Kind::Float);
                total_sum += losses.total.double_value(&[]) * batch_samples as f64;
                return_mae_sum += losses.return_mae.double_value(&[]) * batch_samples as f64;
                return_mse_sum += losses.return_mse.double_value(&[]) * batch_samples as f64;
                pred_std_sum += losses.pred_std.double_value(&[]) * batch_samples as f64;
                target_std_sum += losses.target_std.double_value(&[]) * batch_samples as f64;
                return_bias_sum += losses.return_bias.double_value(&[]) * batch_samples as f64;
                pred_abs_sum += losses.pred_abs.double_value(&[]) * batch_samples as f64;
                target_abs_sum += losses.target_abs.double_value(&[]) * batch_samples as f64;
                next_lat_sum += losses.next_lat.double_value(&[]) * batch_samples as f64;
                terminal_mse_sum += losses.terminal_mse.double_value(&[]) * batch_samples as f64;
                zero_mse_sum += zero_mse_loss.double_value(&[]) * batch_samples as f64;
                samples += batch_samples;
                batches += 1;
            }
        }

        assert!(samples > 0, "validation set is empty");
        ValidationLoss {
            total: total_sum / samples as f64,
            return_mae: return_mae_sum / samples as f64,
            return_mse: return_mse_sum / samples as f64,
            pred_std: pred_std_sum / samples as f64,
            target_std: target_std_sum / samples as f64,
            return_bias: return_bias_sum / samples as f64,
            pred_abs: pred_abs_sum / samples as f64,
            target_abs: target_abs_sum / samples as f64,
            next_lat: next_lat_sum / samples as f64,
            terminal_mse: terminal_mse_sum / samples as f64,
            zero_mse: zero_mse_sum / samples as f64,
            samples,
            tickers,
            batches,
        }
    })
}

struct DiagnosticTrace {
    label: String,
    loss: f64,
    actual: Vec<f32>,
    predicted: Vec<f32>,
}

fn write_pretrain_diagnostics(
    model: &TradingModel,
    heads: &PretrainHeads,
    sampler: &mut PretrainSampler,
    batch_size: usize,
    max_batches: Option<usize>,
    epoch: usize,
    global_step: usize,
    gens_dir: &Path,
    device: Device,
) -> Result<()> {
    const TRACE_COUNT: usize = 8;
    const WORST_COUNT: usize = 8;

    let epoch_dir = gens_dir.join(epoch.to_string());
    let samples_dir = epoch_dir.join("samples");
    fs::create_dir_all(&samples_dir)?;

    let horizon = sampler.k_patches * sampler.patch_size;
    let mut abs_sum = vec![0.0f64; horizon];
    let mut sq_sum = vec![0.0f64; horizon];
    let mut bias_sum = vec![0.0f64; horizon];
    let mut count = 0usize;
    let mut first_traces = Vec::new();
    let mut worst_traces: Vec<DiagnosticTrace> = Vec::new();

    tch::no_grad(|| -> Result<()> {
        let mut batches = 0usize;
        for ticker in sampler.val_tickers.clone() {
            if max_batches.is_some_and(|limit| batches >= limit) {
                break;
            }
            let mut env =
                Env::new_with_tickers_and_recording(vec![ticker.clone()], false, false, None);
            let offsets = build_split_offsets(
                env.price_deltas[0].len(),
                sampler.k_patches,
                sampler.patch_size,
                SplitKind::Validation,
            );
            if offsets.is_empty() {
                continue;
            }

            for chunk in offsets.chunks(batch_size) {
                if max_batches.is_some_and(|limit| batches >= limit) {
                    break;
                }
                let batch = PretrainSampler::batch_from_offsets(
                    &mut env,
                    chunk,
                    sampler.k_patches,
                    sampler.patch_size,
                    sampler.target_scale,
                    device,
                );
                let pred = predict_future_returns(model, heads, &batch);
                let actual_returns = cumulative_future_returns(&batch.future_patches);
                let actual = tensor_to_vec_f32(&actual_returns)?;
                let predicted = tensor_to_vec_f32(&pred)?;

                for (sample_idx, &offset) in chunk.iter().enumerate() {
                    let start = sample_idx * horizon;
                    let end = start + horizon;
                    let actual_sample = &actual[start..end];
                    let pred_sample = &predicted[start..end];
                    let mut sample_abs = 0.0;

                    for h in 0..horizon {
                        let err = pred_sample[h] as f64 - actual_sample[h] as f64;
                        abs_sum[h] += err.abs();
                        sq_sum[h] += err * err;
                        bias_sum[h] += err;
                        sample_abs += err.abs();
                    }
                    count += 1;
                    let loss = sample_abs / horizon as f64;
                    let trace = DiagnosticTrace {
                        label: format!("{}_offset_{}", ticker, offset),
                        loss,
                        actual: actual_sample.to_vec(),
                        predicted: pred_sample.to_vec(),
                    };

                    if first_traces.len() < TRACE_COUNT {
                        first_traces.push(DiagnosticTrace {
                            label: format!("sample_{:02}_{}", first_traces.len() + 1, trace.label),
                            loss,
                            actual: trace.actual.clone(),
                            predicted: trace.predicted.clone(),
                        });
                    }

                    worst_traces.push(trace);
                    worst_traces.sort_by(|a, b| {
                        b.loss
                            .partial_cmp(&a.loss)
                            .unwrap_or(std::cmp::Ordering::Equal)
                    });
                    worst_traces.truncate(WORST_COUNT);
                }
                batches += 1;
            }
        }
        Ok(())
    })?;

    assert!(count > 0, "pretrain diagnostics validation set is empty");
    let denom = count as f64;
    let mae = abs_sum
        .iter()
        .map(|v| (*v / denom) as f32)
        .collect::<Vec<_>>();
    let rmse = sq_sum
        .iter()
        .map(|v| (*v / denom).sqrt() as f32)
        .collect::<Vec<_>>();
    let bias = bias_sum
        .iter()
        .map(|v| (*v / denom) as f32)
        .collect::<Vec<_>>();

    write_report_file(
        &epoch_dir.join("pretrain_horizon_error.report.bin"),
        &Report {
            title: format!("Pretrain Horizon Error - epoch {epoch} step {global_step}"),
            x_label: Some("forecast step".to_string()),
            y_label: Some("target-scaled cumulative log return".to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine {
                series: vec![
                    ReportSeries {
                        label: "MAE".to_string(),
                        values: mae.clone(),
                    },
                    ReportSeries {
                        label: "RMSE".to_string(),
                        values: rmse.clone(),
                    },
                    ReportSeries {
                        label: "Bias".to_string(),
                        values: bias,
                    },
                ],
            },
        },
    )?;
    for (i, trace) in first_traces.iter().enumerate() {
        write_trace_reports(
            &samples_dir,
            &format!("sample_{:02}", i + 1),
            "Sample",
            epoch,
            global_step,
            trace,
        )?;
    }
    for (i, trace) in worst_traces.iter().enumerate() {
        write_trace_reports(
            &samples_dir,
            &format!("worst_{:02}", i + 1),
            "Worst",
            epoch,
            global_step,
            trace,
        )?;
    }

    Ok(())
}

fn write_trace_reports(
    dir: &Path,
    prefix: &str,
    group: &str,
    epoch: usize,
    global_step: usize,
    trace: &DiagnosticTrace,
) -> Result<()> {
    let error = trace
        .predicted
        .iter()
        .zip(trace.actual.iter())
        .map(|(pred, actual)| pred - actual)
        .collect::<Vec<_>>();
    write_report_file(
        &dir.join(format!("{prefix}_deltas.report.bin")),
        &Report {
            title: format!(
                "Pretrain {group} Returns - epoch {epoch} step {global_step} - {} - MAE {:.5}",
                trace.label, trace.loss
            ),
            x_label: Some("forecast step".to_string()),
            y_label: Some("target-scaled cumulative log return".to_string()),
            scale: ScaleKind::Linear,
            kind: ReportKind::MultiLine {
                series: vec![
                    ReportSeries {
                        label: "actual".to_string(),
                        values: trace.actual.clone(),
                    },
                    ReportSeries {
                        label: "predicted".to_string(),
                        values: trace.predicted.clone(),
                    },
                    ReportSeries {
                        label: "error".to_string(),
                        values: error,
                    },
                ],
            },
        },
    )?;
    Ok(())
}

fn tensor_to_vec_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    let tensor = tensor
        .to_device(Device::Cpu)
        .to_kind(Kind::Float)
        .contiguous()
        .view([-1]);
    let numel = tensor.numel();
    let mut values = vec![0.0f32; numel];
    tensor.copy_data(&mut values, numel);
    Ok(values)
}

fn write_report_file(path: &Path, report: &Report) -> Result<()> {
    let bytes = postcard::to_stdvec(report).context("failed to encode report")?;
    fs::write(path, bytes).with_context(|| format!("failed to write {}", path.display()))
}

fn future_patches_for_current_perm(
    env: &Env,
    offset: usize,
    k_patches: usize,
    patch_size: usize,
    target_scale: f64,
) -> Vec<f32> {
    let mut out = Vec::with_capacity(TICKERS_COUNT as usize * k_patches * patch_size);
    let first_future = offset + 1;
    for &real_idx in &env.ticker_perm {
        let deltas = &env.price_deltas[real_idx];
        for patch_i in 0..k_patches {
            let start = first_future + patch_i * patch_size;
            let end = start + patch_size;
            out.extend(
                deltas[start..end]
                    .iter()
                    .map(|&v| (v * target_scale) as f32),
            );
        }
    }
    out
}

fn normalize_last_dim(x: &Tensor) -> Tensor {
    let mean = x.mean_dim([-1].as_slice(), true, Kind::Float);
    let centered = x - &mean;
    let var = centered
        .pow_tensor_scalar(2.0)
        .mean_dim([-1].as_slice(), true, Kind::Float);
    centered / (var + 1e-5).sqrt()
}

fn clip_all_grads(named_vars: &[(String, Tensor)], max_grad_norm: f64, device: Device) {
    tch::no_grad(|| {
        let mut total_norm_sq = Tensor::zeros([], (Kind::Float, device));
        let mut grads = Vec::new();
        for (_, param) in named_vars {
            let grad = param.grad();
            if grad.defined() {
                total_norm_sq += grad.square().sum(Kind::Float);
                grads.push(grad);
            }
        }
        let total_norm = total_norm_sq.sqrt();
        let coef = (Tensor::from(max_grad_norm as f32).to_device(device) / (&total_norm + 1e-6))
            .clamp_max(1.0);
        for mut grad in grads {
            let coef = coef.to_kind(grad.kind());
            let _ = grad.g_mul_(&coef);
        }
    });
}

fn assert_finite_loss(loss: &Tensor, step: usize) {
    let loss_v = loss.double_value(&[]);
    assert!(
        loss_v.is_finite(),
        "non-finite pretrain loss at step {step}: {loss_v}"
    );
}

fn configure_threads() {
    if let Some(threads) = std::env::var("TORCH_NUM_THREADS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
    {
        tch::set_num_threads(threads);
    } else {
        tch::set_num_threads(1);
    }
    if let Some(threads) = std::env::var("TORCH_NUM_INTEROP_THREADS")
        .ok()
        .and_then(|v| v.parse::<i32>().ok())
    {
        tch::set_num_interop_threads(threads);
    } else {
        tch::set_num_interop_threads(1);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        build_split_offsets, cumulative_future_returns, future_patches_for_current_perm, SplitKind,
    };
    use crate::torch::{
        constants::PRICE_DELTAS_PER_TICKER,
        env::Env,
        model::{ModelVariant, TradingModel, TradingModelConfig},
    };
    use tch::nn;
    use tch::Tensor;

    #[test]
    fn cumulative_future_returns_flattens_patches_and_accumulates_horizon() {
        let future_patches =
            Tensor::from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]).view([1, 1, 2, 3]);
        let cumulative = cumulative_future_returns(&future_patches);
        let expected = Tensor::from_slice(&[1.0f32, 3.0, 6.0, 10.0, 15.0, 21.0]).view([1, 1, 6]);
        let max_diff = (cumulative - expected).abs().max().double_value(&[]);
        assert!(max_diff < 1e-6, "cumulative target mismatch: {max_diff}");
    }

    #[test]
    fn future_patches_follow_current_ticker_permutation() {
        let mut env = Env::new(false);
        let offset = crate::torch::constants::PRICE_DELTAS_PER_TICKER;
        let _ = env.reset_single_at_offset_for_pretrain(offset);
        let patches = future_patches_for_current_perm(&env, offset, 2, 3, 1.0);
        assert_eq!(
            patches.len(),
            crate::torch::constants::TICKERS_COUNT as usize * 2 * 3
        );
        let real_idx = env.ticker_perm[0];
        assert_eq!(patches[0], env.price_deltas[real_idx][offset + 1] as f32);
        assert_eq!(patches[3], env.price_deltas[real_idx][offset + 4] as f32);
    }

    #[test]
    fn uniform_stream_pretrain_patch_size_is_25() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant: ModelVariant::UniformStream,
                ..TradingModelConfig::default()
            },
        );
        assert_eq!(model.pretrain_patch_size(), 25);
        assert_eq!(model.pretrain_patch_token_count(), 240);
        assert_eq!(model.pretrain_layout_len(), PRICE_DELTAS_PER_TICKER as i64);
    }

    #[test]
    fn split_offsets_allow_last_future_safe_patch_aligned_anchor() {
        let data_len = PRICE_DELTAS_PER_TICKER + 801;
        let offsets = build_split_offsets(data_len, 16, 25, SplitKind::Validation);
        let last = *offsets
            .last()
            .expect("validation offsets should be non-empty");
        assert_eq!(last + 1 + 16 * 25, data_len);
    }

    #[test]
    fn train_split_keeps_forecast_targets_before_validation_contexts() {
        let data_len = PRICE_DELTAS_PER_TICKER + 10_000;
        let train = build_split_offsets(data_len, 16, 25, SplitKind::Train);
        let validation = build_split_offsets(data_len, 16, 25, SplitKind::Validation);
        let last_train = *train.last().expect("train offsets should be non-empty");
        let first_validation = *validation
            .first()
            .expect("validation offsets should be non-empty");
        assert!(last_train + 16 * 25 <= first_validation);
    }
}
