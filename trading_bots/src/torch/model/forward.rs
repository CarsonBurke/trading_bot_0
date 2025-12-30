use tch::{Kind, Tensor};

use super::{DebugMetrics, ModelOutput, StreamState, TradingModel, PATCH_SIZE};

impl TradingModel {
    pub fn forward(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _train: bool,
    ) -> ModelOutput {
        let price_deltas = self.cast_inputs(&price_deltas.to_device(self.device));
        let static_features = self.cast_inputs(&static_features.to_device(self.device));
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let x_stem =
            self.patch_embed_all_with_static(&price_deltas, &per_ticker_static, batch_size);

        let mut x_for_ssm = x_stem.permute([0, 2, 1]);
        let dt_scale = Tensor::full(
            &[1, x_for_ssm.size()[1], 1],
            PATCH_SIZE as f64,
            (Kind::Float, x_for_ssm.device()),
        );
        for (layer, norm) in self.ssm_layers.iter().zip(self.ssm_norms.iter()) {
            let normed = norm.forward(&x_for_ssm);
            let out = layer.forward_with_dt_scale(&normed, Some(&dt_scale));
            x_for_ssm = x_for_ssm + out;
        }
        let x_ssm = x_for_ssm.permute([0, 2, 1]);

        self.head_with_temporal_pool(
            &x_ssm,
            &global_static,
            &per_ticker_static,
            batch_size,
            false,
        )
        .0
    }

    pub fn forward_with_debug(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _train: bool,
    ) -> (ModelOutput, DebugMetrics) {
        let price_deltas = self.cast_inputs(&price_deltas.to_device(self.device));
        let static_features = self.cast_inputs(&static_features.to_device(self.device));
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let x_stem =
            self.patch_embed_all_with_static(&price_deltas, &per_ticker_static, batch_size);

        let mut x_for_ssm = x_stem.permute([0, 2, 1]);
        let dt_scale = Tensor::full(
            &[1, x_for_ssm.size()[1], 1],
            PATCH_SIZE as f64,
            (Kind::Float, x_for_ssm.device()),
        );
        for (layer, norm) in self.ssm_layers.iter().zip(self.ssm_norms.iter()) {
            let normed = norm.forward(&x_for_ssm);
            let out = layer.forward_with_dt_scale(&normed, Some(&dt_scale));
            x_for_ssm = x_for_ssm + out;
        }
        let x_ssm = x_for_ssm.permute([0, 2, 1]);

        let (out, debug) = self.head_with_temporal_pool(
            &x_ssm,
            &global_static,
            &per_ticker_static,
            batch_size,
            true,
        );
        (
            out,
            debug.unwrap_or(DebugMetrics {
                time_alpha_attn_mean: 0.0,
                time_alpha_mlp_mean: 0.0,
                cross_alpha_attn_mean: 0.0,
                cross_alpha_mlp_mean: 0.0,
                temporal_tau: 0.0,
                temporal_attn_entropy: 0.0,
                temporal_attn_max: 0.0,
                temporal_attn_eff_len: 0.0,
                temporal_attn_center: 0.0,
                temporal_attn_last_weight: 0.0,
                cross_ticker_embed_norm: 0.0,
            }),
        )
    }

    pub fn forward_sequence_with_state(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _state: &mut StreamState,
    ) -> ModelOutput {
        let seq_len = price_deltas.size()[0];
        let batch_size = price_deltas.size()[1];
        let total_samples = seq_len * batch_size;

        let price_deltas_flat = price_deltas.reshape([total_samples, -1]);
        let static_features_flat = static_features.reshape([total_samples, -1]);
        self.forward(&price_deltas_flat, &static_features_flat, true)
    }

    pub fn forward_with_state(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _state: &mut StreamState,
    ) -> ModelOutput {
        self.forward(price_deltas, static_features, false)
    }
}
