use std::env;
use tch::Tensor;

use super::{DebugMetrics, ModelOutput, StreamState, TradingModel};

impl TradingModel {
    pub fn forward(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _train: bool,
    ) -> ModelOutput {
        self.forward_with_seq_idx(price_deltas, static_features, None, _train)
    }

    pub fn forward_on_device(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _train: bool,
    ) -> ModelOutput {
        self.forward_with_seq_idx_on_device(price_deltas, static_features, None, _train)
    }

    pub fn forward_with_seq_idx(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        seq_idx: Option<&Tensor>,
        _train: bool,
    ) -> ModelOutput {
        self.forward_with_seq_idx_inner(price_deltas, static_features, seq_idx, true)
    }

    pub fn forward_with_seq_idx_on_device(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        seq_idx: Option<&Tensor>,
        _train: bool,
    ) -> ModelOutput {
        self.forward_with_seq_idx_inner_on_device(price_deltas, static_features, seq_idx, true)
    }

    pub fn forward_with_seq_idx_no_values(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        seq_idx: Option<&Tensor>,
        _train: bool,
    ) -> ModelOutput {
        self.forward_with_seq_idx_inner(price_deltas, static_features, seq_idx, false)
    }

    pub fn forward_with_seq_idx_no_values_on_device(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        seq_idx: Option<&Tensor>,
        _train: bool,
    ) -> ModelOutput {
        self.forward_with_seq_idx_inner_on_device(price_deltas, static_features, seq_idx, false)
    }

    fn forward_with_seq_idx_inner(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        seq_idx: Option<&Tensor>,
        compute_values: bool,
    ) -> ModelOutput {
        let price_deltas = self.cast_inputs(&price_deltas.to_device(self.device));
        let static_features = self.cast_inputs(&static_features.to_device(self.device));
        self.forward_with_seq_idx_inner_on_device(
            &price_deltas,
            &static_features,
            seq_idx,
            compute_values,
        )
    }

    fn forward_with_seq_idx_inner_on_device(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        seq_idx: Option<&Tensor>,
        compute_values: bool,
    ) -> ModelOutput {
        if price_deltas.device() != self.device || static_features.device() != self.device {
            panic!(
                "forward_with_seq_idx_on_device requires tensors on {:?}",
                self.device
            );
        }
        let price_deltas = self.cast_inputs(price_deltas);
        let static_features = self.cast_inputs(static_features);

        debug_fused("model_price_deltas", &price_deltas);
        debug_fused("model_static_features", &static_features);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let (x_stem, dt_scale, seq_idx) =
            self.patch_latent_stem_on_device(&price_deltas, batch_size, seq_idx);
        debug_fused("model_x_stem", &x_stem);
        debug_fused("model_dt_scale", &dt_scale);

        let exo_kv = self.build_exo_kv(&global_static, &per_ticker_static, batch_size);

        let mut x_for_ssm = x_stem;
        let seq_idx_ref = if seq_idx.numel() == 0 {
            None
        } else {
            Some(&seq_idx)
        };
        for (i, (layer, norm)) in self
            .ssm_layers
            .iter()
            .zip(self.ssm_norms.iter())
            .enumerate()
        {
            let out = layer.forward_with_pre_norm_seq_idx(
                &x_for_ssm,
                norm.weight(),
                norm.eps(),
                Some(&dt_scale),
                seq_idx_ref,
            );
            x_for_ssm = x_for_ssm + out;
            x_for_ssm = self.maybe_apply_exo_cross(&x_for_ssm, &exo_kv, i);
        }
        debug_fused("model_x_for_ssm", &x_for_ssm);

        self.head_with_temporal_pool(&x_for_ssm, batch_size, compute_values, false)
            .0
    }

    pub fn forward_with_debug(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _train: bool,
    ) -> (ModelOutput, DebugMetrics) {
        self.forward_with_debug_seq_idx(price_deltas, static_features, None, _train)
    }

    pub fn forward_with_debug_seq_idx(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        seq_idx: Option<&Tensor>,
        _train: bool,
    ) -> (ModelOutput, DebugMetrics) {
        let price_deltas = self.cast_inputs(&price_deltas.to_device(self.device));
        let static_features = self.cast_inputs(&static_features.to_device(self.device));
        debug_fused("model_price_deltas", &price_deltas);
        debug_fused("model_static_features", &static_features);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let (x_stem, dt_scale, seq_idx) =
            self.patch_latent_stem(&price_deltas, batch_size, seq_idx);
        debug_fused("model_x_stem", &x_stem);
        debug_fused("model_dt_scale", &dt_scale);

        let exo_kv = self.build_exo_kv(&global_static, &per_ticker_static, batch_size);

        let mut x_for_ssm = x_stem;
        let seq_idx_ref = if seq_idx.numel() == 0 {
            None
        } else {
            Some(&seq_idx)
        };
        for (layer_idx, (layer, norm)) in self
            .ssm_layers
            .iter()
            .zip(self.ssm_norms.iter())
            .enumerate()
        {
            debug_fused_layer("x_for_ssm_in", layer_idx, &x_for_ssm);
            let out = layer.forward_with_pre_norm_seq_idx(
                &x_for_ssm,
                norm.weight(),
                norm.eps(),
                Some(&dt_scale),
                seq_idx_ref,
            );
            debug_fused_layer("ssm_out", layer_idx, &out);
            x_for_ssm = x_for_ssm + out;
            x_for_ssm = self.maybe_apply_exo_cross(&x_for_ssm, &exo_kv, layer_idx);
            debug_fused_layer("x_for_ssm_out", layer_idx, &x_for_ssm);
        }
        debug_fused("model_x_for_ssm", &x_for_ssm);

        let (out, debug) = self.head_with_temporal_pool(&x_for_ssm, batch_size, true, true);
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

use std::sync::atomic::{AtomicBool, Ordering};

static DEBUG_ENABLED: AtomicBool = AtomicBool::new(false);
static DEBUG_INITIALIZED: AtomicBool = AtomicBool::new(false);

#[inline]
fn is_debug_enabled() -> bool {
    if !DEBUG_INITIALIZED.load(Ordering::Relaxed) {
        let enabled = crate::torch::ppo::DEBUG_NUMERICS
            || env::var("MAMBA_FUSED_DEBUG").ok().as_deref() == Some("1");
        DEBUG_ENABLED.store(enabled, Ordering::Relaxed);
        DEBUG_INITIALIZED.store(true, Ordering::Relaxed);
    }
    DEBUG_ENABLED.load(Ordering::Relaxed)
}

#[inline]
fn debug_fused(tag: &str, t: &Tensor) {
    if !is_debug_enabled() {
        return;
    }
    let has_nan = t.isnan().any().int64_value(&[]) != 0;
    let has_inf = t.isinf().any().int64_value(&[]) != 0;
    if has_nan || has_inf {
        eprintln!(
            "mamba_fused_debug {} nan={} inf={} shape={:?}",
            tag,
            has_nan,
            has_inf,
            t.size()
        );
    }
}

#[inline]
fn debug_fused_layer(tag: &str, layer_idx: usize, t: &Tensor) {
    if !is_debug_enabled() {
        return;
    }
    let has_nan = t.isnan().any().int64_value(&[]) != 0;
    let has_inf = t.isinf().any().int64_value(&[]) != 0;
    if has_nan || has_inf {
        eprintln!(
            "mamba_fused_debug {}_l{} nan={} inf={} shape={:?}",
            tag,
            layer_idx,
            has_nan,
            has_inf,
            t.size()
        );
    }
}
