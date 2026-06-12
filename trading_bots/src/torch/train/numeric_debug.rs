use std::cmp::Ordering;
use tch::{autocast, Kind, Tensor};

use crate::torch::model::TradingModel;
use crate::torch::value::hl_gauss::symlog_tensor;

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

fn value_diag_core(pred: &Tensor, target: &Tensor) -> Tensor {
    let residuals = pred - target;
    let value_mean = pred.mean(Kind::Float);
    let return_mean = target.mean(Kind::Float);
    let value_centered = pred - &value_mean;
    let return_centered = target - &return_mean;
    let residual_mean = residuals.mean(Kind::Float);
    let residual_centered = &residuals - &residual_mean;
    let value_std = value_centered.square().mean(Kind::Float).sqrt();
    let return_std = return_centered.square().mean(Kind::Float).sqrt();
    let residual_rmse = residuals.square().mean(Kind::Float).sqrt();
    let return_var = return_centered.square().mean(Kind::Float);
    let residual_var = residual_centered.square().mean(Kind::Float);
    let explained_var_raw = Tensor::from(1.0) - residual_var / return_var.clamp_min(1e-8);
    let explained_var = nan_if_zero_target_variance(&explained_var_raw, &return_var);
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
            explained_var,
        ],
        0,
    )
}

pub(crate) fn compute_value_diagnostics(rollout_values: &Tensor, returns: &Tensor) -> Tensor {
    tch::no_grad(|| {
        let pred = rollout_values.to_kind(Kind::Float);
        let target = returns.to_kind(Kind::Float);
        value_diag_core(&pred, &target)
    })
}

pub(crate) fn compute_value_diagnostics_symlog(
    rollout_values: &Tensor,
    returns: &Tensor,
) -> Tensor {
    tch::no_grad(|| {
        let pred = symlog_tensor(&rollout_values.to_kind(Kind::Float));
        let target = symlog_tensor(&returns.to_kind(Kind::Float));
        value_diag_core(&pred, &target)
    })
}

/// Compute explained variance on a subset of the rollout.
/// EV using pre-training rollout values, matching CleanRL.
pub(crate) fn compute_explained_variance(rollout_values: &Tensor, returns: &Tensor) -> Tensor {
    tch::no_grad(|| {
        let residuals = rollout_values - returns;
        let mean_ret = returns.mean(Kind::Float);
        let returns_centered = returns - &mean_ret;
        let var_ret = returns_centered.square().mean(Kind::Float);
        let mean_resid = residuals.mean(Kind::Float);
        let residuals_centered = residuals - &mean_resid;
        let var_resid = residuals_centered.square().mean(Kind::Float);
        let explained_var = Tensor::from(1.0) - &var_resid / var_ret.clamp_min(1e-8);
        nan_if_zero_target_variance(&explained_var, &var_ret)
    })
}

fn nan_if_zero_target_variance(explained_var: &Tensor, target_var: &Tensor) -> Tensor {
    let nan = Tensor::full_like(explained_var, f64::NAN);
    nan.where_self(&target_var.eq(0.0), explained_var)
}

#[cfg(test)]
mod tests {
    use super::{compute_explained_variance, compute_value_diagnostics};
    use tch::Tensor;

    #[test]
    fn explained_variance_matches_cleanrl_residual_variance() {
        let returns = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0]);
        let biased_values = &returns + 10.0;

        let ev = compute_explained_variance(&biased_values, &returns).double_value(&[]);

        assert!(
            (ev - 1.0).abs() < 1e-6,
            "expected CleanRL-style EV 1.0, got {ev}"
        );
    }

    #[test]
    fn value_diagnostics_use_same_explained_variance_formula() {
        let returns = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0]);
        let biased_values = &returns + 10.0;

        let diag = compute_value_diagnostics(&biased_values, &returns);
        let ev = diag.double_value(&[6]);

        assert!(
            (ev - 1.0).abs() < 1e-6,
            "expected diagnostic EV 1.0, got {ev}"
        );
    }

    #[test]
    fn explained_variance_is_nan_when_returns_have_zero_variance() {
        let returns = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0]);
        let values = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0]);

        let ev = compute_explained_variance(&values, &returns);

        assert_eq!(ev.isnan().int64_value(&[]), 1);
    }

    #[test]
    fn value_diagnostics_explained_variance_is_nan_when_returns_have_zero_variance() {
        let returns = Tensor::from_slice(&[1.0f32, 1.0, 1.0, 1.0]);
        let values = Tensor::from_slice(&[0.0f32, 1.0, 2.0, 3.0]);

        let diag = compute_value_diagnostics(&values, &returns);

        assert_eq!(diag.get(6).isnan().int64_value(&[]), 1);
    }

    #[test]
    fn explained_variance_reduces_over_all_dimensions_like_flattened_cleanrl_batch() {
        let returns = Tensor::from_slice(&[-1.0f32, 0.0, 1.0, 2.0]).view([2, 2]);
        let values = Tensor::from_slice(&[-1.0f32, 1.0, 1.0, 3.0]).view([2, 2]);

        let ev_2d = compute_explained_variance(&values, &returns).double_value(&[]);
        let ev_flat = compute_explained_variance(&values.reshape([-1]), &returns.reshape([-1]))
            .double_value(&[]);

        assert!((ev_2d - ev_flat).abs() < 1e-6);
    }
}
