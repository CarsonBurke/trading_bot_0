use tch::{Kind, Tensor};

use super::numeric_debug::{
    compute_beta_policy_stats, compute_explained_variance, compute_value_diagnostics,
    compute_value_diagnostics_symlog,
};
use super::trainer::{AdvantageData, Trainer, UpdateMetrics};

impl Trainer {
    pub(super) fn log_episode(
        &mut self,
        _episode: usize,
        adv_data: &AdvantageData,
        metrics: &UpdateMetrics,
    ) {
        let device = self.device;
        let max_param_norm = tch::no_grad(|| {
            let norms: Vec<Tensor> = self.trainable_vars.iter().map(|v| v.norm()).collect();
            if norms.is_empty() {
                0.0f64
            } else {
                Tensor::stack(&norms, 0).max().double_value(&[])
            }
        });
        if max_param_norm > 1000.0 {
            println!(
                "WARNING: Large parameter norm detected: {:.2}",
                max_param_norm
            );
        }

        // Compute all metrics on GPU, single transfer to CPU
        let (
            mean_policy_loss_t,
            mean_value_loss_t,
            mean_clip_gap_t,
            mean_actor_grad_norm_t,
            mean_critic_grad_norm_t,
            clip_fraction_t,
        ) = if metrics.total_sample_count > 0 {
            let n = metrics.total_sample_count as f64;
            let mean_policy = &metrics.total_policy_loss_weighted / n;
            let mean_value = &metrics.total_value_loss_weighted / n;
            let mean_clip_gap = &metrics.total_clip_gap_weighted / n;
            let (actor_grad_norm, critic_grad_norm) = if metrics.grad_norm_count > 0 {
                let d = metrics.grad_norm_count as f64;
                (
                    &metrics.actor_grad_norm_sum / d,
                    &metrics.critic_grad_norm_sum / d,
                )
            } else {
                (
                    Tensor::zeros([], (Kind::Float, device)),
                    Tensor::zeros([], (Kind::Float, device)),
                )
            };
            let clip_fraction = if metrics.total_ratio_samples > 0 {
                &metrics.total_clip_violations / (metrics.total_ratio_samples as f64)
            } else {
                Tensor::zeros([], (Kind::Float, device))
            };
            (
                mean_policy,
                mean_value,
                mean_clip_gap,
                actor_grad_norm,
                critic_grad_norm,
                clip_fraction,
            )
        } else {
            (
                Tensor::zeros([], (Kind::Float, device)),
                Tensor::zeros([], (Kind::Float, device)),
                Tensor::zeros([], (Kind::Float, device)),
                Tensor::zeros([], (Kind::Float, device)),
                Tensor::zeros([], (Kind::Float, device)),
                Tensor::zeros([], (Kind::Float, device)),
            )
        };

        let explained_var_t = if metrics.total_sample_count > 0 {
            compute_explained_variance(&self.s_values, &adv_data.returns)
        } else {
            Tensor::zeros([], (Kind::Float, device))
        };
        let value_diag_t = if metrics.total_sample_count > 0 {
            compute_value_diagnostics(&self.s_values, &adv_data.returns)
        } else {
            Tensor::zeros([7], (Kind::Float, device))
        };
        let value_diag_symlog_t = if metrics.total_sample_count > 0 {
            compute_value_diagnostics_symlog(&self.s_values, &adv_data.returns)
        } else {
            Tensor::zeros([7], (Kind::Float, device))
        };

        let entropy_mean_t = if metrics.total_sample_count > 0 {
            &metrics.total_entropy_weighted / metrics.total_sample_count as f64
        } else {
            Tensor::zeros([], (Kind::Float, device))
        };

        let beta_policy_stats = compute_beta_policy_stats(
            &self.trading_model,
            &self.s_chunk_start_layouts.narrow(0, 0, self.rollout.nprocs),
            &self
                .s_static_obs
                .narrow(0, 0, self.rollout.nprocs)
                .select(1, 0),
        );
        let return_range_stats = self.hl_gauss.range_stats(&adv_data.returns);
        // Compute all metrics on GPU, single transfer to CPU.
        let all_scalars = Tensor::cat(
            &[
                mean_policy_loss_t.view([1]),
                mean_value_loss_t.view([1]),
                mean_clip_gap_t.view([1]),
                explained_var_t.view([1]),
                mean_actor_grad_norm_t.view([1]),
                mean_critic_grad_norm_t.view([1]),
                clip_fraction_t.view([1]),
                adv_data.adv_stats.view([3]),
                adv_data.adv_stats_shaped.view([4]),
                beta_policy_stats.view([4]),
                entropy_mean_t.view([1]),
                metrics.entropy_min.view([1]),
                metrics.entropy_max.view([1]),
                return_range_stats.view([6]),
                value_diag_t.view([7]),
                value_diag_symlog_t.view([7]),
            ],
            0,
        );
        let total_scalar_len = all_scalars.size()[0] as usize;
        let all_scalars_vec: Vec<f64> = Vec::try_from(all_scalars.to_device(tch::Device::Cpu))
            .unwrap_or_else(|_| vec![0.0; total_scalar_len]);
        // Cursor unpack: consume named groups in the EXACT order they were cat'd above.
        let mut cur = 0usize;
        let mut take = |n: usize| {
            let s = all_scalars_vec[cur..cur + n].to_vec();
            cur += n;
            s
        };

        let mean_policy_loss = take(1)[0];
        let mean_value_loss = take(1)[0];
        let mean_clip_gap = take(1)[0];
        let explained_var = take(1)[0];
        let mean_actor_grad_norm = take(1)[0];
        let mean_critic_grad_norm = take(1)[0];
        let clip_fraction = take(1)[0];
        let adv_stats = take(3);
        let (adv_mean, adv_min, adv_max) = (adv_stats[0], adv_stats[1], adv_stats[2]);
        let adv_shaped = take(4);
        let (adv_shaped_mean, adv_shaped_std, adv_shaped_min, adv_shaped_max) =
            (adv_shaped[0], adv_shaped[1], adv_shaped[2], adv_shaped[3]);
        let beta_policy_stats_vec = take(4);
        let entropy_stats = take(3);
        let (entropy_mean, entropy_min_val, entropy_max_val) =
            (entropy_stats[0], entropy_stats[1], entropy_stats[2]);
        let return_range = take(6);
        let (
            return_min,
            return_max,
            support_min,
            support_max,
            below_support_frac,
            above_support_frac,
        ) = (
            return_range[0],
            return_range[1],
            return_range[2],
            return_range[3],
            return_range[4],
            return_range[5],
        );
        let value_diag = take(6);
        let (
            value_pred_mean,
            value_pred_std,
            value_target_mean,
            value_target_std,
            value_residual_rmse,
            value_return_corr,
        ) = (
            value_diag[0],
            value_diag[1],
            value_diag[2],
            value_diag[3],
            value_diag[4],
            value_diag[5],
        );
        let _ = take(1); // skip: redundant plain-space EV (see numeric_debug compute_explained_variance)
        let value_diag_symlog = take(7);
        let (
            value_pred_mean_symlog,
            value_pred_std_symlog,
            value_target_mean_symlog,
            value_target_std_symlog,
            value_residual_rmse_symlog,
            value_return_corr_symlog,
            value_explained_var_symlog,
        ) = (
            value_diag_symlog[0],
            value_diag_symlog[1],
            value_diag_symlog[2],
            value_diag_symlog[3],
            value_diag_symlog[4],
            value_diag_symlog[5],
            value_diag_symlog[6],
        );

        let mean_epoch_approx_kl = metrics.mean_epoch_approx_kl;
        let primary = self.env.primary_mut();
        primary
            .meta_history
            .record_advantage_stats(adv_mean, adv_min, adv_max);
        primary.meta_history.record_beta_policy_stats(
            beta_policy_stats_vec[0],
            beta_policy_stats_vec[1],
            beta_policy_stats_vec[2],
            beta_policy_stats_vec[3],
        );
        primary.meta_history.record_clip_fraction(clip_fraction);
        primary.meta_history.record_clip_gap(mean_clip_gap);
        primary.meta_history.record_policy_loss(mean_policy_loss);
        primary.meta_history.record_value_loss(mean_value_loss);
        primary.meta_history.record_explained_var(explained_var);
        primary
            .meta_history
            .record_grad_norm(mean_actor_grad_norm, mean_critic_grad_norm);
        primary
            .meta_history
            .record_policy_entropy(entropy_mean, entropy_min_val, entropy_max_val);
        primary.meta_history.record_approx_kl(mean_epoch_approx_kl);
        primary.meta_history.record_hl_gauss_range_stats(
            return_min,
            return_max,
            support_min,
            support_max,
            below_support_frac,
            above_support_frac,
        );

        println!(
            "  Policy: {:.4}, Value: {:.4} (EV: {:.3}), ClipGap: {:.4}, ActorGradNorm: {:.4}, CriticGradNorm: {:.4}",
            mean_policy_loss, mean_value_loss, explained_var, mean_clip_gap, mean_actor_grad_norm, mean_critic_grad_norm
        );
        println!(
            "  Adv raw: μ {:.4}, min {:.4}, max {:.4} | shaped: μ {:.4}, σ {:.4}, min {:.4}, max {:.4}",
            adv_mean,
            adv_min,
            adv_max,
            adv_shaped_mean,
            adv_shaped_std,
            adv_shaped_min,
            adv_shaped_max
        );
        println!(
            "  ValueDiag: pred μ/σ {:.3}/{:.3}, target μ/σ {:.3}/{:.3}, RMSE {:.3}, Corr {:.3}",
            value_pred_mean,
            value_pred_std,
            value_target_mean,
            value_target_std,
            value_residual_rmse,
            value_return_corr
        );
        println!(
            "  ValueDiag(symlog): pred μ/σ {:.3}/{:.3}, target μ/σ {:.3}/{:.3}, RMSE {:.3}, Corr {:.3}, EV {:.3}",
            value_pred_mean_symlog,
            value_pred_std_symlog,
            value_target_mean_symlog,
            value_target_std_symlog,
            value_residual_rmse_symlog,
            value_return_corr_symlog,
            value_explained_var_symlog
        );
    }
}
