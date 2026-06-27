use tch::{Kind, Tensor};

use super::config::{
    ModelVariant, UNIFORM_STREAM_LAYOUT_LEN, UNIFORM_STREAM_PATCH_COUNT, UNIFORM_STREAM_PATCH_SIZE,
};
use super::trading_model::TradingModel;
use crate::torch::constants::TICKERS_COUNT;

impl TradingModel {
    fn pretrain_encoded(
        &self,
        layouts: &Tensor,
        static_features: &Tensor,
        batch_size: i64,
    ) -> Tensor {
        assert_eq!(
            self.variant,
            ModelVariant::UniformStream,
            "pretraining is defined for UniformStream layouts"
        );
        let layouts = self.cast_inputs(layouts);
        let patch_tokens = self.patch_embed(&layouts);
        let patch_hidden = self.input_ln.forward(&patch_tokens);

        let static_features = self.cast_inputs(static_features);
        let static_features = self.ensure_batched(&static_features);
        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let exo_tokens = self.build_exo_tokens(&global_static, &per_ticker_static, batch_size);

        self.patch_trunk(&patch_hidden, &exo_tokens)
    }

    pub fn pretrain_latent_dim(&self) -> i64 {
        self.model_dim
    }

    pub fn pretrain_patch_size(&self) -> i64 {
        UNIFORM_STREAM_PATCH_SIZE
    }

    pub fn pretrain_layout_len(&self) -> i64 {
        UNIFORM_STREAM_LAYOUT_LEN
    }

    pub fn pretrain_patch_token_count(&self) -> i64 {
        UNIFORM_STREAM_PATCH_COUNT
    }

    /// Returns encoded per-ticker patch tokens before PMA/readout pooling:
    /// [batch, tickers, patches, model_dim].
    pub fn pretrain_patch_tokens(
        &self,
        layouts: &Tensor,
        static_features: &Tensor,
        batch_size: i64,
    ) -> Tensor {
        self.pretrain_encoded(layouts, static_features, batch_size)
            .view([
                batch_size,
                TICKERS_COUNT,
                UNIFORM_STREAM_PATCH_COUNT,
                self.model_dim,
            ])
            .to_kind(Kind::Float)
    }

    /// Returns the per-ticker actor/readout latent used immediately before the
    /// policy head: [batch, tickers, model_dim].
    pub fn pretrain_actor_latents(
        &self,
        layouts: &Tensor,
        static_features: &Tensor,
        batch_size: i64,
    ) -> Tensor {
        self.actor_latents_from_pretrain_encoded(
            &self.pretrain_encoded(layouts, static_features, batch_size),
            batch_size,
        )
    }

    pub fn pretrain_patch_tokens_and_actor_latents(
        &self,
        layouts: &Tensor,
        static_features: &Tensor,
        batch_size: i64,
    ) -> (Tensor, Tensor) {
        let encoded = self.pretrain_encoded(layouts, static_features, batch_size);
        let patch_tokens = encoded
            .view([
                batch_size,
                TICKERS_COUNT,
                UNIFORM_STREAM_PATCH_COUNT,
                self.model_dim,
            ])
            .to_kind(Kind::Float);
        let actor_latents = self.actor_latents_from_pretrain_encoded(&encoded, batch_size);
        (patch_tokens, actor_latents)
    }

    fn actor_latents_from_pretrain_encoded(&self, encoded: &Tensor, batch_size: i64) -> Tensor {
        let pooled = self.readout_ln.forward(&self.pma.forward(encoded));
        pooled
            .select(1, 0)
            .view([batch_size, TICKERS_COUNT, self.model_dim])
            .to_kind(Kind::Float)
    }
}
