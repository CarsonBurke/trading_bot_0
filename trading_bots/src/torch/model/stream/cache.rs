use tch::Tensor;

use super::super::trading_model::{ModelOutput, StreamState, TradingModel};

impl TradingModel {
    /// Live/replay readout. The bidirectional trunk makes every patch attend to
    /// every other (including future positions), so no causal prefix K/V cache is
    /// sound. We therefore run the full stateless backbone over the cached patch
    /// tokens, which is correct by construction and bit-for-bit identical to
    /// `forward`/`windowed_replay_forward` (the streaming-vs-full parity gate).
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
