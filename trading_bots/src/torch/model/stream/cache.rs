use tch::Tensor;

use super::super::config::{ACTOR_CRITIC_CLS_COUNT, UNIFORM_STREAM_PATCH_COUNT};
use super::super::trading_model::{ModelOutput, StreamState, TradingModel};

impl TradingModel {
    pub(super) fn prefill_uniform_prefix_base_cache(&self, state: &mut StreamState) {
        let x = state
            .uniform_patch_tokens
            .narrow(1, 0, UNIFORM_STREAM_PATCH_COUNT - 1)
            .shallow_clone();
        let x0 = self.input_ln.forward(&x);
        let (x_next, k, v) = self.gqa_layers[0].forward_prefix_and_cache(&x0, &x0, &self.rope);
        state.uniform_prefix_x0 = x0;
        state.uniform_layer0_prefix_hidden = x_next;
        state.uniform_layer0_prefix_k = k;
        state.uniform_layer0_prefix_v = v;
        state.uniform_prefix_k.clear();
        state.uniform_prefix_v.clear();
        state.uniform_cached_static_features = None;
        state.uniform_cached_exo_tokens = None;
    }

    pub(super) fn prefill_uniform_prefix_base_cache_indexed(
        &self,
        state: &mut StreamState,
        row_idx: &Tensor,
    ) {
        let x = state.uniform_patch_tokens.index_select(0, row_idx).narrow(
            1,
            0,
            UNIFORM_STREAM_PATCH_COUNT - 1,
        );
        let x0 = self.input_ln.forward(&x);
        let (x_next, k, v) = self.gqa_layers[0].forward_prefix_and_cache(&x0, &x0, &self.rope);
        self.ensure_stream_cache_kind(state, &x0, &x_next, &k, &v);
        state.uniform_prefix_x0 = state.uniform_prefix_x0.index_copy(0, row_idx, &x0);
        state.uniform_layer0_prefix_hidden = state
            .uniform_layer0_prefix_hidden
            .index_copy(0, row_idx, &x_next);
        state.uniform_layer0_prefix_k = state.uniform_layer0_prefix_k.index_copy(0, row_idx, &k);
        state.uniform_layer0_prefix_v = state.uniform_layer0_prefix_v.index_copy(0, row_idx, &v);
        state.uniform_prefix_k.clear();
        state.uniform_prefix_v.clear();
        state.uniform_cached_static_features = None;
        state.uniform_cached_exo_tokens = None;
    }

    pub(super) fn conditioned_prefix_cache_is_fresh(
        &self,
        static_features: &Tensor,
        state: &StreamState,
    ) -> bool {
        state.uniform_prefix_k.len() == self.gqa_layers.len()
            && state.uniform_prefix_v.len() == self.gqa_layers.len()
            && state.uniform_cached_exo_tokens.is_some()
            && state
                .uniform_cached_static_features
                .as_ref()
                .map(|cached| cached.allclose(static_features, 0.0, 0.0, true))
                .unwrap_or(false)
    }

    pub(super) fn rebuild_uniform_conditioned_prefix_cache(
        &self,
        static_features: &Tensor,
        exo_tokens: &Tensor,
        state: &mut StreamState,
    ) {
        let x0 = &state.uniform_prefix_x0;
        let mut x = self
            .exogenous_ticker_block
            .forward(&state.uniform_layer0_prefix_hidden, exo_tokens);
        state.uniform_prefix_k = vec![state.uniform_layer0_prefix_k.shallow_clone()];
        state.uniform_prefix_v = vec![state.uniform_layer0_prefix_v.shallow_clone()];
        for layer in self.gqa_layers.iter().skip(1) {
            let (x_next, k, v) = layer.forward_prefix_and_cache(&x, x0, &self.rope);
            state.uniform_prefix_k.push(k);
            state.uniform_prefix_v.push(v);
            x = x_next;
        }
        state.uniform_cached_static_features = Some(static_features.shallow_clone());
        state.uniform_cached_exo_tokens = Some(exo_tokens.shallow_clone());
    }

    pub(super) fn head_from_cached_live_and_cls(
        &self,
        live_x0: &Tensor,
        prefix_k: &[Tensor],
        prefix_v: &[Tensor],
        exo_tokens: &Tensor,
        prefix_len: i64,
        batch_size: i64,
    ) -> ModelOutput {
        let cls_x0 = self.actor_critic_cls_from_live(live_x0);
        let suffix_x0 = Tensor::cat(&[live_x0, &cls_x0], 1);
        let mut suffix = suffix_x0.shallow_clone();

        for (layer_idx, layer) in self.gqa_layers.iter().enumerate() {
            suffix = layer.forward_causal_suffix_with_cache(
                &suffix,
                &suffix_x0,
                &prefix_k[layer_idx],
                &prefix_v[layer_idx],
                &self.rope,
                prefix_len,
            );

            if layer_idx == 0 {
                suffix = self.exogenous_ticker_block.forward(&suffix, exo_tokens);
            }
            let live = self.maybe_apply_endogenous_ticker(&suffix.narrow(1, 0, 1), layer_idx);
            let cls = suffix.narrow(1, 1, ACTOR_CRITIC_CLS_COUNT);
            suffix = Tensor::cat(&[&live, &cls], 1);
        }

        let suffix = self.final_ln.forward(&suffix);
        let actor = suffix.select(1, 1);
        let critic = suffix.select(1, 2);
        self.head_from_actor_critic_cls(&actor, &critic, batch_size)
    }
}
