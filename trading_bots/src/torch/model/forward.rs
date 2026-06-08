use std::env;
use tch::Tensor;

use super::trading_model::{DebugMetrics, ModelOutput, StreamState, TradingModel};

impl TradingModel {
    fn prepare_inputs(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        to_device: bool,
    ) -> (Tensor, Tensor) {
        let prep = |input: &Tensor| {
            if to_device {
                self.cast_inputs(&self.maybe_to_device(input, self.device))
            } else {
                self.cast_inputs(input)
            }
        };
        (prep(price_deltas), prep(static_features))
    }

    fn forward_prepared_on_device(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
    ) -> ModelOutput {
        debug_fused("model_price_deltas", price_deltas);
        debug_fused("model_static_features", static_features);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(static_features, batch_size);
        let exo_tokens = self.build_exo_tokens(&global_static, &per_ticker_static, batch_size);
        let x_stem = self.patch_latent_stem_on_device(price_deltas, batch_size);
        debug_fused("model_x_stem", &x_stem);

        let output = self.backbone_with_actor_critic_cls(&x_stem, &exo_tokens, batch_size);
        debug_fused("model_value_logits", &output.0);
        debug_fused("model_alpha", &output.1);
        debug_fused("model_beta", &output.2);
        output
    }

    pub fn forward(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _train: bool,
    ) -> ModelOutput {
        let (price_deltas, static_features) =
            self.prepare_inputs(price_deltas, static_features, true);
        self.forward_prepared_on_device(&price_deltas, &static_features)
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
        let (price_deltas, static_features) =
            self.prepare_inputs(price_deltas, static_features, false);
        self.forward_prepared_on_device(&price_deltas, &static_features)
    }

    pub fn forward_with_debug(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        _train: bool,
    ) -> (ModelOutput, DebugMetrics) {
        let (price_deltas, static_features) =
            self.prepare_inputs(price_deltas, static_features, true);
        debug_fused("model_price_deltas", &price_deltas);
        debug_fused("model_static_features", &static_features);
        let batch_size = price_deltas.size()[0];

        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let exo_tokens = self.build_exo_tokens(&global_static, &per_ticker_static, batch_size);
        let x_stem = self.patch_latent_stem_on_device(&price_deltas, batch_size);
        debug_fused("model_x_stem", &x_stem);

        let output = self.backbone_with_actor_critic_cls(&x_stem, &exo_tokens, batch_size);
        debug_fused("model_value_logits", &output.0);

        (
            output,
            DebugMetrics {
                temporal_tau: 0.0,
                temporal_attn_entropy: 0.0,
                temporal_attn_max: 0.0,
                temporal_attn_eff_len: 0.0,
                temporal_attn_center: 0.0,
                temporal_attn_last_weight: 0.0,
            },
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
        let enabled = crate::torch::train::config::DEBUG_NUMERICS
            || env::var("MODEL_FUSED_DEBUG").ok().as_deref() == Some("1");
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
