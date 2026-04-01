use std::env;
use tch::Tensor;

use super::{DebugMetrics, ModelOutput, StreamState, TradingModel};

impl TradingModel {
    pub(super) fn tail_condition(
        &self,
        global_static: &Tensor,
        per_ticker_static: &Tensor,
        batch_size: i64,
        kind: tch::Kind,
    ) -> Tensor {
        let exo_kv = self.build_exo_kv(global_static, per_ticker_static, batch_size);
        exo_kv
            .mean_dim([1].as_slice(), false, tch::Kind::Float)
            .to_kind(kind)
            .unsqueeze(1)
    }

    pub(super) fn apply_tail_condition(
        &self,
        x: &Tensor,
        global_static: &Tensor,
        per_ticker_static: &Tensor,
        batch_size: i64,
    ) -> Tensor {
        let seq = x.size()[1];
        let tail = self.tail_condition(global_static, per_ticker_static, batch_size, x.kind());
        let prefix = x.narrow(1, 0, seq - super::NUM_HEAD_CLS_TOKENS);
        let suffix = x.narrow(
            1,
            seq - super::NUM_HEAD_CLS_TOKENS,
            super::NUM_HEAD_CLS_TOKENS,
        ) + &tail;
        Tensor::cat(&[&prefix, &suffix], 1)
    }

    pub fn forward(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _train: bool,
    ) -> ModelOutput {
        let price_deltas = self.cast_inputs(&price_deltas.to_device(self.device));
        let static_features = self.cast_inputs(&static_features.to_device(self.device));
        self.forward_on_device(&price_deltas, &static_features, _train)
    }

    pub fn forward_on_device(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _train: bool,
    ) -> ModelOutput {
        if price_deltas.device() != self.device || static_features.device() != self.device {
            panic!("forward_on_device requires tensors on {:?}", self.device);
        }
        let price_deltas = self.cast_inputs(price_deltas);
        let static_features = self.cast_inputs(static_features);

        debug_fused("model_price_deltas", &price_deltas);
        debug_fused("model_static_features", &static_features);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let x_stem = self.patch_latent_stem_on_device(&price_deltas, batch_size);
        let x_stem =
            self.apply_tail_condition(&x_stem, &global_static, &per_ticker_static, batch_size);
        debug_fused("model_x_stem", &x_stem);

        let mut x = x_stem;
        for (i, layer) in self.gqa_layers.iter().enumerate() {
            x = layer.forward(&x, &self.rope, true);
            x = self.maybe_apply_inter_ticker(&x, i);
        }
        debug_fused("model_x_gqa", &x);

        self.head_with_temporal_pool(&x, batch_size, false).0
    }

    pub fn forward_with_debug(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _train: bool,
    ) -> (ModelOutput, DebugMetrics) {
        let price_deltas = self.cast_inputs(&price_deltas.to_device(self.device));
        let static_features = self.cast_inputs(&static_features.to_device(self.device));
        debug_fused("model_price_deltas", &price_deltas);
        debug_fused("model_static_features", &static_features);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let x_stem = self.patch_latent_stem(&price_deltas, batch_size);
        let x_stem =
            self.apply_tail_condition(&x_stem, &global_static, &per_ticker_static, batch_size);
        debug_fused("model_x_stem", &x_stem);

        let mut x = x_stem;
        for (layer_idx, layer) in self.gqa_layers.iter().enumerate() {
            debug_fused_layer("x_gqa_in", layer_idx, &x);
            x = layer.forward(&x, &self.rope, true);
            debug_fused_layer("gqa_out", layer_idx, &x);
            x = self.maybe_apply_inter_ticker(&x, layer_idx);
            debug_fused_layer("x_gqa_out", layer_idx, &x);
        }
        debug_fused("model_x_gqa", &x);

        let (out, debug) = self.head_with_temporal_pool(&x, batch_size, true);
        (
            out,
            debug.unwrap_or(DebugMetrics {
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
            "debug {} nan={} inf={} shape={:?}",
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
            "debug {}_l{} nan={} inf={} shape={:?}",
            tag,
            layer_idx,
            has_nan,
            has_inf,
            t.size()
        );
    }
}
