use tch::{Kind, Tensor};

use super::numeric_debug::{
    compute_beta_policy_stats, compute_explained_variance, compute_value_diagnostics,
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
            mean_spo_penalty_t,
            mean_actor_grad_norm_t,
            mean_critic_grad_norm_t,
            spo_bound_fraction_t,
        ) = if metrics.total_sample_count > 0 {
            let n = metrics.total_sample_count as f64;
            let mean_policy = &metrics.total_policy_loss_weighted / n;
            let mean_value = &metrics.total_value_loss_weighted / n;
            let mean_spo_penalty = &metrics.total_spo_penalty_weighted / n;
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
            let spo_bound_fraction = if metrics.total_ratio_samples > 0 {
                &metrics.total_spo_bound_violations / (metrics.total_ratio_samples as f64)
            } else {
                Tensor::zeros([], (Kind::Float, device))
            };
            (
                mean_policy,
                mean_value,
                mean_spo_penalty,
                actor_grad_norm,
                critic_grad_norm,
                spo_bound_fraction,
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
            Tensor::zeros([6], (Kind::Float, device))
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
                mean_spo_penalty_t.view([1]),
                explained_var_t.view([1]),
                mean_actor_grad_norm_t.view([1]),
                mean_critic_grad_norm_t.view([1]),
                spo_bound_fraction_t.view([1]),
                adv_data.adv_stats.view([3]),
                beta_policy_stats.view([4]),
                entropy_mean_t.view([1]),
                metrics.entropy_min.view([1]),
                metrics.entropy_max.view([1]),
                return_range_stats.view([6]),
                value_diag_t.view([6]),
            ],
            0,
        );
        let all_scalars_vec: Vec<f64> = Vec::try_from(all_scalars.to_device(tch::Device::Cpu))
            .unwrap_or_else(|_| vec![0.0; 29]);
        let mean_policy_loss = all_scalars_vec[0];
        let mean_value_loss = all_scalars_vec[1];
        let mean_spo_penalty = all_scalars_vec[2];
        let explained_var = all_scalars_vec[3];
        let mean_actor_grad_norm = all_scalars_vec[4];
        let mean_critic_grad_norm = all_scalars_vec[5];
        let spo_bound_fraction = all_scalars_vec[6];
        let (adv_mean, adv_min, adv_max) =
            (all_scalars_vec[7], all_scalars_vec[8], all_scalars_vec[9]);
        let beta_policy_stats_vec = &all_scalars_vec[10..14];
        let (entropy_mean, entropy_min_val, entropy_max_val) = (
            all_scalars_vec[14],
            all_scalars_vec[15],
            all_scalars_vec[16],
        );
        let (
            return_min,
            return_max,
            support_min,
            support_max,
            below_support_frac,
            above_support_frac,
        ) = (
            all_scalars_vec[17],
            all_scalars_vec[18],
            all_scalars_vec[19],
            all_scalars_vec[20],
            all_scalars_vec[21],
            all_scalars_vec[22],
        );
        let (
            value_pred_mean,
            value_pred_std,
            value_target_mean,
            value_target_std,
            value_residual_rmse,
            value_return_corr,
        ) = (
            all_scalars_vec[23],
            all_scalars_vec[24],
            all_scalars_vec[25],
            all_scalars_vec[26],
            all_scalars_vec[27],
            all_scalars_vec[28],
        );

        let last_minibatch_approx_kl = metrics.last_minibatch_approx_kl;
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
        primary
            .meta_history
            .record_spo_bound_fraction(spo_bound_fraction);
        primary.meta_history.record_spo_penalty(mean_spo_penalty);
        primary.meta_history.record_policy_loss(mean_policy_loss);
        primary.meta_history.record_value_loss(mean_value_loss);
        primary.meta_history.record_explained_var(explained_var);
        primary
            .meta_history
            .record_grad_norm(mean_actor_grad_norm, mean_critic_grad_norm);
        primary
            .meta_history
            .record_policy_entropy(entropy_mean, entropy_min_val, entropy_max_val);
        primary
            .meta_history
            .record_approx_kl(last_minibatch_approx_kl);
        primary.meta_history.record_hl_gauss_range_stats(
            return_min,
            return_max,
            support_min,
            support_max,
            below_support_frac,
            above_support_frac,
        );

        println!(
            "  Policy: {:.4}, Value: {:.4} (EV: {:.3}), SPO penalty: {:.4}, ActorGradNorm: {:.4}, CriticGradNorm: {:.4}",
            mean_policy_loss, mean_value_loss, explained_var, mean_spo_penalty, mean_actor_grad_norm, mean_critic_grad_norm
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
    }
}
