use std::cmp::Ordering;
use tch::{autocast, Kind, Tensor};

use crate::torch::model::TradingModel;

pub(crate) fn debug_tensor_stats(name: &str, t: &Tensor, episode: i64, step: usize) -> bool {
    let has_nan = t.isnan().any().int64_value(&[]) != 0;
    let has_inf = t.isinf().any().int64_value(&[]) != 0;
    if has_nan || has_inf {
        let mean = t.mean(Kind::Float).double_value(&[]);
        let min = t.min().double_value(&[]);
        let max = t.max().double_value(&[]);
        println!(
            "Non-finite in {} at ep {} step {} nan={} inf={} mean={:.6} min={:.6} max={:.6}",
            name, episode, step, has_nan, has_inf, mean, min, max
        );
        return false;
    }
    true
}

pub(crate) fn tensor_is_finite(t: &Tensor) -> bool {
    t.isfinite().all().int64_value(&[]) != 0
}

pub(crate) fn tensor_summary(name: &str, t: &Tensor) -> String {
    let mean = t.mean(Kind::Float).double_value(&[]);
    let min = t.min().double_value(&[]);
    let max = t.max().double_value(&[]);
    let abs_max = t.abs().max().double_value(&[]);
    let nan_count = t
        .isnan()
        .to_kind(Kind::Int64)
        .sum(Kind::Int64)
        .int64_value(&[]);
    let inf_count = t
        .isinf()
        .to_kind(Kind::Int64)
        .sum(Kind::Int64)
        .int64_value(&[]);
    format!(
        "{} shape={:?} kind={:?} mean={:.6} min={:.6} max={:.6} abs_max={:.6} nan={} inf={}",
        name,
        t.size(),
        t.kind(),
        mean,
        min,
        max,
        abs_max,
        nan_count,
        inf_count
    )
}

pub(crate) fn log_first_non_finite_tensor(
    logged: &mut bool,
    stage: &str,
    episode: usize,
    epoch: i64,
    chunk_i: usize,
    tensors: &[(&str, &Tensor)],
) -> bool {
    if *logged {
        return false;
    }
    for (name, tensor) in tensors {
        if !tensor_is_finite(tensor) {
            println!(
                "NUMERIC ROOT CAUSE: stage={} episode={} epoch={} chunk={} {}",
                stage,
                episode,
                epoch + 1,
                chunk_i,
                tensor_summary(name, tensor)
            );
            for (other_name, other_tensor) in tensors {
                println!("  {}", tensor_summary(other_name, other_tensor));
            }
            *logged = true;
            return true;
        }
    }
    false
}

pub(crate) fn log_first_non_finite_var(
    logged: &mut bool,
    stage: &str,
    episode: usize,
    epoch: i64,
    chunk_i: usize,
    vars: &[(String, Tensor)],
    use_grad: bool,
) -> bool {
    if *logged {
        return false;
    }
    for (idx, (name, var)) in vars.iter().enumerate() {
        let candidate = if use_grad {
            var.grad()
        } else {
            var.shallow_clone()
        };
        if !candidate.defined() || tensor_is_finite(&candidate) {
            continue;
        }
        println!(
            "NUMERIC ROOT CAUSE: stage={} episode={} epoch={} chunk={} param_idx={} param_name={} param_shape={:?}",
            stage,
            episode,
            epoch + 1,
            chunk_i,
            idx,
            name,
            var.size()
        );
        println!(
            "  {}",
            tensor_summary(if use_grad { "grad" } else { "param" }, &candidate)
        );
        println!("  {}", tensor_summary("param_snapshot", var));
        *logged = true;
        return true;
    }
    false
}

pub(crate) fn log_named_var_extremes(
    stage: &str,
    episode: usize,
    epoch: i64,
    chunk_i: usize,
    vars: &[(String, Tensor)],
    use_grad: bool,
    top_n: usize,
) {
    let mut rows = Vec::with_capacity(vars.len());
    let mut non_finite = 0usize;
    for (name, var) in vars {
        let candidate = if use_grad {
            var.grad()
        } else {
            var.shallow_clone()
        };
        if !candidate.defined() {
            continue;
        }
        if !tensor_is_finite(&candidate) {
            non_finite += 1;
        }
        let abs_max = candidate.abs().max().double_value(&[]);
        let sort_key = if abs_max.is_finite() {
            abs_max
        } else {
            f64::INFINITY
        };
        rows.push((sort_key, tensor_summary(name, &candidate)));
    }
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(Ordering::Equal));
    println!(
        "NUMERIC SNAPSHOT: stage={} episode={} epoch={} chunk={} tensor={} non_finite={} top_abs={}",
        stage,
        episode,
        epoch + 1,
        chunk_i,
        if use_grad { "grad" } else { "param" },
        non_finite,
        top_n.min(rows.len())
    );
    for (_, summary) in rows.into_iter().take(top_n) {
        println!("  {}", summary);
    }
}

/// Compute Beta policy concentration stats from a sample batch.
/// Returns [alpha_mean, action_mean, beta_mean, concentration_mean] where
/// action_mean = E[alpha/(alpha+beta)] and concentration = alpha+beta.
pub(crate) fn compute_beta_policy_stats(
    model: &TradingModel,
    price_deltas: &Tensor,
    static_obs: &Tensor,
) -> Tensor {
    tch::no_grad(|| {
        let (_, alpha, beta) = autocast(false, || {
            model.forward_on_device(price_deltas, static_obs, false)
        });
        let action_mean = (&alpha / (&alpha + &beta)).mean(Kind::Float);
        let concentration = (&alpha + &beta).mean(Kind::Float);
        Tensor::stack(
            &[
                alpha.mean(Kind::Float),
                action_mean,
                beta.mean(Kind::Float),
                concentration,
            ],
            0,
        )
    })
}

pub(crate) fn compute_value_diagnostics(rollout_values: &Tensor, returns: &Tensor) -> Tensor {
    tch::no_grad(|| {
        let values = rollout_values.to_kind(Kind::Float);
        let returns = returns.to_kind(Kind::Float);
        let residuals = &values - &returns;
        let value_mean = values.mean(Kind::Float);
        let return_mean = returns.mean(Kind::Float);
        let value_centered = &values - &value_mean;
        let return_centered = &returns - &return_mean;
        let value_std = value_centered.square().mean(Kind::Float).sqrt();
        let return_std = return_centered.square().mean(Kind::Float).sqrt();
        let residual_rmse = residuals.square().mean(Kind::Float).sqrt();
        let corr = (&value_centered * &return_centered).mean(Kind::Float)
            / ((&value_std * &return_std).clamp_min(1e-8));
        Tensor::stack(
            &[
                value_mean,
                value_std,
                return_mean,
                return_std,
                residual_rmse,
                corr,
            ],
            0,
        )
    })
}

/// Compute explained variance on a subset of the rollout.
/// EV using pre-training rollout values, matching CleanRL.
pub(crate) fn compute_explained_variance(rollout_values: &Tensor, returns: &Tensor) -> Tensor {
    tch::no_grad(|| {
        let residuals = rollout_values - returns;
        let mean_ret = returns.mean(Kind::Float);
        let var_ret = returns.square().mean(Kind::Float) - mean_ret.square();
        let var_resid = residuals.square().mean(Kind::Float);
        Tensor::from(1.0) - &var_resid / var_ret.clamp_min(1e-8)
    })
}
