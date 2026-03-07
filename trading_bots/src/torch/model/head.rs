use tch::{Kind, Tensor};

use super::{
    DebugMetrics, ModelOutput, TradingModel, SDE_EPS, NOISE_RANK,
};
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    pub(super) fn head_with_temporal_pool(
        &self,
        x_ssm: &Tensor,
        batch_size: i64,
        _debug: bool,
    ) -> (ModelOutput, Option<DebugMetrics>) {
        let temporal_len = x_ssm.size()[1];
        let x_time = x_ssm.view([batch_size, TICKERS_COUNT, temporal_len, self.model_dim]);
        let flat_dim_per = temporal_len * self.model_dim;
        let flat_dim_all = TICKERS_COUNT * flat_dim_per;

        let flat_per = x_time.reshape([batch_size * TICKERS_COUNT, flat_dim_per]);
        let flat_all = x_time.reshape([batch_size, flat_dim_all]);

        // Weightless RMSNorm on flattened vectors (computed in f32)
        let flat_per_normed = {
            let xf = flat_per.to_kind(Kind::Float);
            let rms = (xf.pow_tensor_scalar(2).mean_dim([-1].as_slice(), true, Kind::Float) + 1e-6).sqrt();
            (xf / rms).to_kind(flat_per.kind())
        };
        let flat_all_normed = {
            let xf = flat_all.to_kind(Kind::Float);
            let rms = (xf.pow_tensor_scalar(2).mean_dim([-1].as_slice(), true, Kind::Float) + 1e-6).sqrt();
            (xf / rms).to_kind(flat_all.kind())
        };

        // Actor: RMSNorm → projection (gain controls init scale directly)
        let action_mean = flat_per_normed
            .apply(&self.actor_proj)
            .reshape([batch_size, TICKERS_COUNT])
            * self.mean_scale.to_kind(flat_per_normed.kind());

        // Noise hidden features (silu activations, matching reference)
        let sde_in = flat_per_normed.apply(&self.sde_in_proj);
        let h = self.sde_norm.forward(&sde_in.apply(&self.sde_fc))
            .silu()
            .apply(&self.sde_fc2)
            .silu();

        // Per-direction amplitude with learnable range
        let raw = h.apply(&self.dir_head);
        let amp_floor = self.amp_floor.to_kind(raw.kind());
        let amp_scale = self.amp_scale.to_kind(raw.kind());
        let amp = amp_floor.softplus()
            + amp_scale.softplus() * raw.sigmoid();

        // State-dependent basis perturbation
        let perturb = h.apply(&self.perturb_head)
            .reshape([batch_size, NOISE_RANK, TICKERS_COUNT]) * 0.1;
        let effective_basis = self.noise_basis.to_kind(perturb.kind()) + perturb;

        // cov_factor: effective_basis^T scaled by amplitude
        let cov_factor = effective_basis.transpose(-2, -1) * amp.unsqueeze(1);

        // Floor diagonal variance
        let log_std_fl = self.log_std_floor.to_kind(cov_factor.kind());
        let cov_diag = log_std_fl.exp().pow_tensor_scalar(2)
            .expand([batch_size, TICKERS_COUNT], false);

        // Total std: sqrt(sum(cov_factor^2, dim=-1) + cov_diag)
        let factor_var = cov_factor.pow_tensor_scalar(2)
            .sum_dim_intlist([-1].as_slice(), false, cov_factor.kind());
        let action_noise_std = (&factor_var + &cov_diag + SDE_EPS).sqrt();

        // Critic: RMSNorm → projection
        let values = flat_all_normed
            .apply(&self.value_proj)
            .squeeze_dim(-1);

        let values = values.to_kind(Kind::Float);
        let action_mean = action_mean.to_kind(Kind::Float);
        let action_noise_std = action_noise_std.to_kind(Kind::Float);

        (
            (values, action_mean, action_noise_std),
            None,
        )
    }
}
