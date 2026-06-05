use tch::Tensor;

use super::super::config::UNIFORM_STREAM_PATCH_COUNT;
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

    /// Live/replay readout. The heads consume the last patch's post-`final_ln`
    /// hidden state, which depends on every prior patch through the causal trunk.
    /// The prefix K/V cache only holds per-layer attention K/V, not the post-trunk
    /// hidden state, so reconstructing the finalized last-patch state from it would
    /// be as costly as a full forward. We therefore run the full stateless backbone
    /// over the cached patch tokens, which is correct by construction and
    /// bit-for-bit identical to `forward`/`windowed_replay_forward` (the
    /// streaming-vs-full parity gate).
    pub(super) fn readout_from_cached_patches(
        &self,
        exo_tokens: &Tensor,
        batch_size: i64,
        state: &StreamState,
    ) -> ModelOutput {
        let patch_hidden = self.input_ln.forward(&state.uniform_patch_tokens);
        self.backbone_with_actor_critic_cls(&patch_hidden, exo_tokens, batch_size)
    }
}
