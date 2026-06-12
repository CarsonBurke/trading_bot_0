use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use super::blocks::cross_attn::CrossAttnFfnBlock;
use super::blocks::endogenous::EndogenousTickerBlock;
use super::blocks::exogenous::ExoMLP;
use super::blocks::gqa::{GqaBlock, GQA_NUM_Q_HEADS};
use super::blocks::pma::PmaReadout;
use super::config::{
    compute_patch_totals, model_spec, ModelVariant, INTER_TICKER_AFTER, NUM_EXO_TOKENS,
    PATCH_SCALAR_FEATS, UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL, UNIFORM_STREAM_LAYOUT_LEN,
    UNIFORM_STREAM_PATCH_COUNT, UNIFORM_STREAM_PATCH_SIZE,
};
use super::init::{
    linear_orthogonal, linear_with_same_dtype, linear_zero, truncated_normal_init,
    xavier_normal_std,
};
use super::rmsnorm::RMSNorm;
use super::rope::{RotaryEmbedding, ROPE_DIMS};
use crate::torch::constants::{
    ACTION_COUNT, GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER, TICKERS_COUNT,
};
use crate::torch::value::hl_gauss::NUM_BINS;

/// (value_logits, action_alpha, action_beta)
pub type ModelOutput = (Tensor, Tensor, Tensor);

pub struct DebugMetrics {
    pub temporal_tau: f64,
    pub temporal_attn_entropy: f64,
    pub temporal_attn_max: f64,
    pub temporal_attn_eff_len: f64,
    pub temporal_attn_center: f64,
    pub temporal_attn_last_weight: f64,
}

#[derive(Clone, Copy)]
pub struct TradingModelConfig {
    pub variant: ModelVariant,
}

impl Default for TradingModelConfig {
    fn default() -> Self {
        Self {
            variant: ModelVariant::Base,
        }
    }
}

/// Streaming state for inference
/// - Ring buffer holds full delta history
/// - Patch buffer accumulates deltas until full patch ready
/// - No model state needed: the bidirectional trunk recomputes full
///   self-attention over the S patch window each step (no causal prefix cache),
///   so streamed/replay readouts are identical to the batched `forward`.
pub struct StreamState {
    /// Ring buffer: [TICKERS_COUNT, PRICE_DELTAS_PER_TICKER]
    pub delta_ring: Tensor,
    /// Write position in ring buffer
    pub ring_pos: i64,
    /// Patch accumulator: [TICKERS_COUNT, FINEST_PATCH_SIZE]
    pub patch_buf: Tensor,
    /// Position within current patch
    pub patch_pos: i64,
    /// Whether initialized with full sequence
    pub initialized: bool,
    /// Uniform stream bucket layout: [batch*TICKERS_COUNT, patch_count, patch_size]
    pub uniform_layout: Tensor,
    /// Cached patch tokens for uniform streamed rollout: [batch*TICKERS_COUNT, patch_count, model_dim]
    pub uniform_patch_tokens: Tensor,
    /// Live fill per env for the tail bucket: [batch]
    pub uniform_live_fill: Tensor,
    /// Host mirror of live fill to avoid per-step device syncs during streamed rollout.
    pub uniform_live_fill_host: Vec<i64>,
}

pub struct TradingModel {
    pub(in crate::torch::model) variant: ModelVariant,
    pub(in crate::torch::model) patch_configs: &'static [(i64, i64)],
    pub(in crate::torch::model) seq_len: i64,
    pub(in crate::torch::model) finest_patch_size: i64,
    pub(in crate::torch::model) model_dim: i64,
    pub(in crate::torch::model) ff_dim: i64,
    pub(in crate::torch::model) patch_embed_weight: Tensor,
    pub(in crate::torch::model) pma: PmaReadout,
    pub(in crate::torch::model) patch_config_ids: Tensor,
    pub(in crate::torch::model) patch_stream_proj: nn::Linear,
    pub(in crate::torch::model) input_ln: RMSNorm,
    pub(in crate::torch::model) final_ln: RMSNorm,
    pub(in crate::torch::model) readout_ln: RMSNorm,
    pub(in crate::torch::model) gqa_layers: Vec<GqaBlock>,
    pub(in crate::torch::model) exogenous_ticker_block: CrossAttnFfnBlock,
    pub(in crate::torch::model) exo_mlp: ExoMLP,
    pub(in crate::torch::model) rope: RotaryEmbedding,
    pub(in crate::torch::model) exo_feat_w: Tensor,
    pub(in crate::torch::model) exo_feat_b: Tensor,
    pub(in crate::torch::model) endogenous_ticker_block: EndogenousTickerBlock,
    pub(in crate::torch::model) policy_concentration: nn::Linear,
    pub(in crate::torch::model) value_proj: nn::Linear,
    pub(in crate::torch::model) device: tch::Device,
}

impl TradingModel {
    pub fn price_input_dim(&self) -> i64 {
        match self.variant {
            ModelVariant::UniformStream => TICKERS_COUNT * UNIFORM_STREAM_LAYOUT_LEN,
            _ => TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
        }
    }

    pub fn uniform_stream_bootstrap_live_fill(&self) -> i64 {
        UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL
    }

    pub fn input_kind(&self) -> Kind {
        self.patch_embed_weight.kind()
    }

    pub(in crate::torch::model) fn maybe_to_device(
        &self,
        input: &Tensor,
        device: tch::Device,
    ) -> Tensor {
        if input.device() == device {
            input.shallow_clone()
        } else {
            input.to_device(device)
        }
    }

    pub(in crate::torch::model) fn cast_inputs(&self, input: &Tensor) -> Tensor {
        let target_kind = self.activation_kind();
        if input.kind() == target_kind {
            input.shallow_clone()
        } else {
            input.to_kind(target_kind)
        }
    }

    pub(in crate::torch::model) fn activation_kind(&self) -> Kind {
        if self.device.is_cuda() {
            Kind::BFloat16
        } else {
            Kind::Float
        }
    }

    pub(in crate::torch::model) fn ensure_batched(&self, t: &Tensor) -> Tensor {
        if t.dim() == 1 {
            t.unsqueeze(0)
        } else {
            t.shallow_clone()
        }
    }

    pub(in crate::torch::model) fn is_full_obs(&self, new_deltas: &Tensor) -> bool {
        let raw_full_obs = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
        let layout_full_obs = self.price_input_dim();
        match new_deltas.dim() {
            1 => {
                let width = new_deltas.size()[0];
                width == raw_full_obs || width == layout_full_obs
            }
            2 => {
                let width = new_deltas.size()[1];
                width == raw_full_obs || width == layout_full_obs
            }
            _ => false,
        }
    }

    pub(in crate::torch::model) fn live_fill_from_layout(layout3d: &Tensor) -> Tensor {
        layout3d
            .select(1, UNIFORM_STREAM_PATCH_COUNT - 1)
            .isnan()
            .logical_not()
            .sum_dim_intlist([1].as_slice(), false, Kind::Int64)
    }

    pub(in crate::torch::model) fn advance_layout_and_reembed_inplace(
        &self,
        state: &mut StreamState,
        new_deltas: &Tensor,
    ) {
        let rows = new_deltas.size()[0] * TICKERS_COUNT;
        let row_deltas = new_deltas.reshape([rows, 1]);
        let history_len = PRICE_DELTAS_PER_TICKER as i64;
        let flat_layout = state.uniform_layout.view([rows, UNIFORM_STREAM_LAYOUT_LEN]);
        let mut shifted_valid = Tensor::zeros(
            [rows, history_len - 1],
            (flat_layout.kind(), flat_layout.device()),
        );
        let _ = shifted_valid.copy_(&flat_layout.narrow(1, 1, history_len - 1));
        let _ = flat_layout
            .narrow(1, 0, history_len - 1)
            .copy_(&shifted_valid);
        let _ = flat_layout.narrow(1, history_len - 1, 1).copy_(&row_deltas);
        state.uniform_patch_tokens = self.patch_embed(&flat_layout);
        state
            .uniform_live_fill_host
            .fill(UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL);
        let _ = state
            .uniform_live_fill
            .fill_(UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL);
        state.initialized = true;
    }

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn patch_seq_len(&self) -> i64 {
        self.seq_len
    }

    pub(in crate::torch::model) fn uniform_stream_layout_from_raw(
        &self,
        deltas: &Tensor,
    ) -> Tensor {
        let device = deltas.device();
        let batch = deltas.size()[0];
        let layout = Tensor::full(
            [batch, UNIFORM_STREAM_LAYOUT_LEN],
            f64::NAN,
            (Kind::Float, device),
        );
        let full_prefix =
            super::config::UNIFORM_STREAM_BOOTSTRAP_FULL_PATCHES * UNIFORM_STREAM_PATCH_SIZE;
        let _ = layout
            .narrow(1, 0, full_prefix)
            .copy_(&deltas.narrow(1, 0, full_prefix));
        let _ = layout
            .narrow(1, full_prefix, UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL)
            .copy_(&deltas.narrow(1, full_prefix, UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL));
        layout.to_kind(deltas.kind())
    }

    pub fn uniform_stream_layout_from_raw_input(&self, price_deltas: &Tensor) -> Tensor {
        assert_eq!(
            self.variant,
            ModelVariant::UniformStream,
            "uniform_stream_layout_from_raw_input is only valid for UniformStream",
        );
        let price = self.ensure_batched(price_deltas);
        let price = self.cast_inputs(&self.maybe_to_device(&price, self.device));
        let batch_size = price.size()[0];
        self.uniform_stream_layout_from_raw(
            &price
                .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
                .view([batch_size * TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64]),
        )
        .view([batch_size, TICKERS_COUNT * UNIFORM_STREAM_LAYOUT_LEN])
    }

    pub fn new(p: &nn::Path) -> Self {
        Self::new_with_config(p, TradingModelConfig::default())
    }

    pub fn new_with_config(p: &nn::Path, config: TradingModelConfig) -> Self {
        let spec = model_spec(config.variant);
        assert_eq!(
            spec.model_dim % GQA_NUM_Q_HEADS,
            0,
            "model_dim must divide evenly across GQA query heads"
        );
        assert!(
            super::blocks::gqa::GQA_NUM_KV_HEADS > 0,
            "GQA must have at least one KV head"
        );
        assert_eq!(
            GQA_NUM_Q_HEADS % super::blocks::gqa::GQA_NUM_KV_HEADS,
            0,
            "GQA query heads must divide evenly across KV heads"
        );
        assert_eq!(ROPE_DIMS % 2, 0, "RoPE dimensions must be even");
        let gqa_layers_count = spec.gqa_layers;
        let patch_configs = spec.patch_configs;
        let (total_days, seq_len) = compute_patch_totals(patch_configs);
        if config.variant == ModelVariant::UniformStream {
            assert_eq!(
                UNIFORM_STREAM_LAYOUT_LEN, PRICE_DELTAS_PER_TICKER as i64,
                "uniform stream layout must exactly match the raw observation history"
            );
            assert_eq!(
                UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL, UNIFORM_STREAM_PATCH_SIZE,
                "uniform stream bootstrap must fill the live patch exactly"
            );
            assert_eq!(
                total_days, UNIFORM_STREAM_LAYOUT_LEN,
                "uniform stream patch configs must sum to the layout length"
            );
        } else {
            assert!(
                total_days == PRICE_DELTAS_PER_TICKER as i64,
                "patch configs must sum to PRICE_DELTAS_PER_TICKER"
            );
        }
        let finest_patch_index = patch_configs.len() - 1;
        let finest_patch_size = patch_configs[finest_patch_index].1;
        let num_configs = patch_configs.len() as i64;
        let max_patch_size = patch_configs
            .iter()
            .map(|&(_, patch_size)| patch_size)
            .max()
            .unwrap_or(0);
        let max_input_dim = max_patch_size + PATCH_SCALAR_FEATS;
        let xavier_std = xavier_normal_std(max_input_dim, spec.model_dim);
        let patch_embed_weight = p.var(
            "patch_embed_weight",
            &[num_configs, max_input_dim, spec.model_dim],
            Init::Randn {
                mean: 0.0,
                stdev: xavier_std,
            },
        );
        let pma = PmaReadout::new(&(p / "pma_readout"), spec.model_dim, spec.ff_dim);
        let patch_stream_proj = nn::linear(
            p / "patch_stream_proj",
            UNIFORM_STREAM_PATCH_SIZE,
            spec.model_dim,
            nn::LinearConfig {
                ws_init: truncated_normal_init(UNIFORM_STREAM_PATCH_SIZE, spec.model_dim),
                bs_init: None,
                bias: false,
            },
        );
        let input_ln = RMSNorm::new(spec.model_dim, 1e-6);
        let final_ln = RMSNorm::new(spec.model_dim, 1e-6);
        let readout_ln = RMSNorm::new(spec.model_dim, 1e-6);
        let patch_config_ids = {
            let mut ids = Vec::with_capacity(seq_len as usize);
            for (cfg_idx, &(days, patch_size)) in patch_configs.iter().enumerate() {
                let n_patches = days / patch_size;
                for _ in 0..n_patches {
                    ids.push(cfg_idx as i64);
                }
            }
            Tensor::from_slice(&ids)
                .to_kind(Kind::Int64)
                .to_device(p.device())
        };

        let gqa_layers = (0..gqa_layers_count)
            .map(|i| GqaBlock::new(&(p / format!("gqa_{}", i)), spec.model_dim, spec.ff_dim, i))
            .collect::<Vec<_>>();
        let exogenous_ticker_block =
            CrossAttnFfnBlock::new(&(p / "cross_attn_0"), spec.model_dim, spec.ff_dim);
        let exo_mlp = ExoMLP::new(&(p / "exo_mlp"), spec.model_dim);
        let head_dim = spec.model_dim / GQA_NUM_Q_HEADS;
        // Bidirectional trunk runs over exactly S patch positions.
        let rope = RotaryEmbedding::new(seq_len, head_dim, ROPE_DIMS, p.device());
        let exo_feat_w = p.var(
            "exo_feat_w",
            &[NUM_EXO_TOKENS, spec.model_dim],
            Init::Randn {
                mean: 0.0,
                stdev: xavier_normal_std(1, spec.model_dim),
            },
        );
        let exo_feat_b = p.var(
            "exo_feat_b",
            &[NUM_EXO_TOKENS, spec.model_dim],
            Init::Const(0.0),
        );
        let endogenous_ticker_block =
            EndogenousTickerBlock::new(&(p / "inter_ticker_0"), spec.model_dim, spec.ff_dim);
        assert_eq!(
            ACTION_COUNT, TICKERS_COUNT,
            "per-ticker actor head requires one action per ticker"
        );
        let flat_all_tickers = TICKERS_COUNT * spec.model_dim;
        let policy_concentration =
            linear_orthogonal(p, "policy_concentration", spec.model_dim, 2, 0.01);
        let value_proj = linear_zero(p, "value_proj", flat_all_tickers, NUM_BINS);
        Self {
            variant: config.variant,
            patch_configs,
            seq_len,
            finest_patch_size,
            model_dim: spec.model_dim,
            ff_dim: spec.ff_dim,
            patch_embed_weight,
            pma,
            patch_config_ids,
            patch_stream_proj,
            input_ln,
            final_ln,
            readout_ln,
            gqa_layers,
            exogenous_ticker_block,
            exo_mlp,
            rope,
            exo_feat_w,
            exo_feat_b,
            endogenous_ticker_block,
            policy_concentration,
            value_proj,
            device: p.device(),
        }
    }

    pub(in crate::torch::model) fn parse_static(
        &self,
        static_features: &Tensor,
        batch_size: i64,
    ) -> (Tensor, Tensor) {
        let global = static_features.narrow(1, 0, GLOBAL_STATIC_OBS as i64);
        let per_ticker = static_features
            .narrow(
                1,
                GLOBAL_STATIC_OBS as i64,
                TICKERS_COUNT * PER_TICKER_STATIC_OBS as i64,
            )
            .reshape([batch_size, TICKERS_COUNT, PER_TICKER_STATIC_OBS as i64]);
        (global, per_ticker)
    }

    pub(in crate::torch::model) fn maybe_apply_endogenous_ticker(
        &self,
        x: &Tensor,
        layer_idx: usize,
    ) -> Tensor {
        if layer_idx != INTER_TICKER_AFTER || TICKERS_COUNT == 1 {
            return x.shallow_clone();
        }
        let bt = x.size()[0];
        let seq = x.size()[1];
        let batch_size = bt / TICKERS_COUNT;
        let x_4d = x.view([batch_size, TICKERS_COUNT, seq, self.model_dim]);
        // Mix the latest patch across tickers (no-op for TICKERS_COUNT == 1).
        let live_idx = seq - 1;
        let live = x_4d.narrow(2, live_idx, 1);
        let live_for_mix =
            live.permute([0, 2, 1, 3])
                .reshape([batch_size, TICKERS_COUNT, self.model_dim]);
        let enriched_live = self
            .endogenous_ticker_block
            .forward(&live_for_mix, self.model_dim, self.ff_dim)
            .reshape([batch_size, 1, TICKERS_COUNT, self.model_dim])
            .permute([0, 2, 1, 3]);
        if seq == 1 {
            enriched_live.reshape([bt, seq, self.model_dim])
        } else {
            let past = x_4d.narrow(2, 0, seq - 1);
            Tensor::cat(&[&past, &enriched_live], 2).reshape([bt, seq, self.model_dim])
        }
    }

    /// Build exogenous KV bank: [batch*tickers, NUM_EXO_TOKENS, MODEL_DIM]
    /// Each of the 46 static features gets its own token via per-feature learned projection
    pub(in crate::torch::model) fn build_exo_kv(
        &self,
        global_static: &Tensor,
        per_ticker_static: &Tensor,
        batch_size: i64,
    ) -> Tensor {
        let global_exp = global_static.unsqueeze(1).expand(
            &[batch_size, TICKERS_COUNT, GLOBAL_STATIC_OBS as i64],
            false,
        );
        let all_feats = Tensor::cat(&[global_exp, per_ticker_static.shallow_clone()], -1);
        let all_feats = all_feats.reshape([batch_size * TICKERS_COUNT, NUM_EXO_TOKENS]);
        let feats_expanded = all_feats.unsqueeze(-1);
        let exo_feat_w = self.exo_feat_w.to_kind(feats_expanded.kind());
        let exo_feat_b = self.exo_feat_b.to_kind(feats_expanded.kind());
        feats_expanded * &exo_feat_w + &exo_feat_b
    }

    /// Build exo tokens with MLP refinement: [batch*tickers, NUM_EXO_TOKENS, MODEL_DIM]
    pub(in crate::torch::model) fn build_exo_tokens(
        &self,
        global_static: &Tensor,
        per_ticker_static: &Tensor,
        batch_size: i64,
    ) -> Tensor {
        let exo_kv = self.build_exo_kv(global_static, per_ticker_static, batch_size);
        self.exo_mlp.forward(&exo_kv)
    }

    pub(in crate::torch::model) fn patch_latent_stem_on_device(
        &self,
        price_deltas: &Tensor,
        batch_size: i64,
    ) -> Tensor {
        let batch_tokens = batch_size * TICKERS_COUNT;
        let deltas = if self.variant == ModelVariant::UniformStream {
            let expected_layout = TICKERS_COUNT * UNIFORM_STREAM_LAYOUT_LEN;
            assert_eq!(
                price_deltas.size()[1],
                expected_layout,
                "UniformStream full forward expects anchored layout input"
            );
            price_deltas
                .view([batch_size, TICKERS_COUNT, UNIFORM_STREAM_LAYOUT_LEN])
                .view([batch_tokens, UNIFORM_STREAM_LAYOUT_LEN])
        } else {
            price_deltas
                .view([batch_size, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
                .view([batch_tokens, PRICE_DELTAS_PER_TICKER as i64])
        };

        let patch_tokens = self.patch_embed(&deltas);
        self.input_ln.forward(&patch_tokens)
    }

    /// Multi-scale BENCHMARK-only path; the default/live training path is
    /// `patch_embed_stream` (UniformStream). Per-config enrichment avoids expanding
    /// each patch to the full history width, then projects all tokens in one fused einsum.
    pub(in crate::torch::model) fn patch_embed(&self, deltas: &Tensor) -> Tensor {
        if self.variant == ModelVariant::UniformStream {
            return self.patch_embed_stream(deltas);
        }
        let device = deltas.device();
        let kind = deltas.kind();
        let batch = deltas.size()[0];
        let max_patch_size = self.patch_embed_weight.size()[1] - PATCH_SCALAR_FEATS;

        // Phase 1: per-config enrichment, zero-padded to max_input_dim, then cat
        let max_input_dim = max_patch_size + PATCH_SCALAR_FEATS;
        let mut enriched_parts = Vec::with_capacity(self.patch_configs.len());
        let mut delta_offset = 0i64;
        for &(days, patch_size) in self.patch_configs {
            let n_patches = days / patch_size;
            let patches = deltas
                .narrow(1, delta_offset, days)
                .view([batch, n_patches, patch_size])
                .to_kind(Kind::Float);
            let mean = patches.mean_dim([2].as_slice(), true, Kind::Float);
            let var = (&patches - &mean).pow_tensor_scalar(2.0).mean_dim(
                [2].as_slice(),
                true,
                Kind::Float,
            );
            let std = (var + 1e-5).sqrt();
            let first = patches.narrow(2, 0, 1);
            let last = patches.narrow(2, patch_size - 1, 1);
            let slope = &last - &first;
            let enriched = Tensor::cat(&[&patches, &mean, &std, &slope], 2);
            // Zero-pad to max_input_dim so all configs share the einsum
            let pad_cols = max_input_dim - (patch_size + PATCH_SCALAR_FEATS);
            let padded = if pad_cols > 0 {
                let pad = Tensor::zeros(&[batch, n_patches, pad_cols], (Kind::Float, device));
                Tensor::cat(&[&enriched, &pad], 2)
            } else {
                enriched
            };
            enriched_parts.push(padded);
            delta_offset += days;
        }
        let enriched = Tensor::cat(&enriched_parts.iter().collect::<Vec<_>>(), 1).to_kind(kind);

        // Phase 2: fused projection over all tokens.
        let weight_per_patch = self
            .patch_embed_weight
            .index_select(0, &self.patch_config_ids)
            .to_kind(kind);
        let out = Tensor::einsum(
            "blm,lmd->bld",
            &[&enriched, &weight_per_patch],
            None::<&[i64]>,
        );
        out
    }

    pub(in crate::torch::model) fn patch_embed_stream_batch(&self, patch_vals: &Tensor) -> Tensor {
        let target_kind = patch_vals.kind();
        let clean = patch_vals.to_kind(Kind::Float).nan_to_num(0.0, 0.0, 0.0);
        linear_with_same_dtype(&clean.to_kind(target_kind), &self.patch_stream_proj)
    }

    pub(in crate::torch::model) fn patch_embed_stream(&self, deltas: &Tensor) -> Tensor {
        let batch = deltas.size()[0];
        let patches = deltas.view([
            batch * UNIFORM_STREAM_PATCH_COUNT,
            UNIFORM_STREAM_PATCH_SIZE,
        ]);
        self.patch_embed_stream_batch(&patches).view([
            batch,
            UNIFORM_STREAM_PATCH_COUNT,
            self.model_dim,
        ])
    }

    /// Bidirectional trunk over all S patch embeddings: full all-to-all
    /// self-attention (no causal mask, no fork tokens), RoPE positions
    /// `arange(S)`. Portfolio cross-attention is injected after layer 0.
    /// Returns post-`final_ln` hidden states `[rows, S, model_dim]` — the PMA
    /// key/value source.
    pub(in crate::torch::model) fn patch_trunk(
        &self,
        patch_hidden: &Tensor,
        exo_tokens: &Tensor,
    ) -> Tensor {
        let num_patches = patch_hidden.size()[1];
        let x0 = patch_hidden.shallow_clone();
        let mut x = patch_hidden.shallow_clone();
        let rope_positions = Tensor::arange(num_patches, (Kind::Int64, self.device));
        for (layer_idx, layer) in self.gqa_layers.iter().enumerate() {
            x = layer.forward_bidirectional(&x, &x0, &self.rope, &rope_positions);
            if layer_idx == 0 {
                x = self.exogenous_ticker_block.forward(&x, exo_tokens);
            }
            x = self.maybe_apply_endogenous_ticker(&x, layer_idx);
        }
        self.final_ln.forward(&x)
    }

    pub(in crate::torch::model) fn backbone_with_actor_critic_cls(
        &self,
        patch_hidden: &Tensor,
        exo_tokens: &Tensor,
        batch_size: i64,
    ) -> ModelOutput {
        let encoded = self.patch_trunk(patch_hidden, exo_tokens);
        let pooled = self.readout_ln.forward(&self.pma.forward(&encoded));
        let actor_read = pooled.select(1, 0);
        let critic_read = pooled.select(1, 1);
        self.head_from_actor_critic_cls(&actor_read, &critic_read, batch_size)
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, path::PathBuf};

    use tch::{nn, Device, Kind, Tensor};

    use super::{ModelVariant, TradingModel, TradingModelConfig, UNIFORM_STREAM_PATCH_SIZE};
    use crate::torch::constants::TICKERS_COUNT;
    use crate::torch::load::load_var_store_partial;
    use crate::torch::value::hl_gauss::HlGaussBins;

    fn temp_checkpoint_path(name: &str) -> PathBuf {
        std::env::temp_dir().join(format!("trading_bot_0_{name}_{}.ot", uuid::Uuid::new_v4()))
    }

    fn assert_value_head_is_unbiased_zero(model: &TradingModel) {
        let weight_abs_max = model.value_proj.ws.abs().max().double_value(&[]);
        assert_eq!(
            weight_abs_max, 0.0,
            "legacy value weights must not overwrite zero HL-Gauss head"
        );
        assert!(
            model.value_proj.bs.is_none(),
            "HL-Gauss value projection should be bias-free"
        );
    }

    #[test]
    fn stream_patch_embed_treats_nan_padding_as_zero() {
        tch::manual_seed(20260612);

        let vs = nn::VarStore::new(Device::Cpu);
        let model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant: ModelVariant::UniformStream,
            },
        );

        let base = Tensor::arange(UNIFORM_STREAM_PATCH_SIZE, (Kind::Float, Device::Cpu)) * 0.001;
        let nan_padded = base.copy();
        let zero_padded = base.copy();
        let tail_start = 10;
        let tail_len = UNIFORM_STREAM_PATCH_SIZE - tail_start;
        let _ = nan_padded.narrow(0, tail_start, tail_len).fill_(f64::NAN);
        let _ = zero_padded.narrow(0, tail_start, tail_len).fill_(0.0);

        let out_nan = model.patch_embed_stream_batch(&nan_padded.unsqueeze(0));
        let out_zero = model.patch_embed_stream_batch(&zero_padded.unsqueeze(0));

        assert!(
            out_nan.isfinite().all().int64_value(&[]) == 1,
            "NaN padding must not leak into patch embeddings"
        );
        assert!(
            out_nan.allclose(&out_zero, 1e-6, 1e-6, false),
            "NaN-padded tail should embed identically to zero-padded tail"
        );
    }

    #[test]
    fn value_head_initializes_bias_free_to_zero_return() {
        tch::manual_seed(20260612);

        let vs = nn::VarStore::new(Device::Cpu);
        let model = TradingModel::new_with_config(&vs.root(), TradingModelConfig::default());

        let weight_abs_max = model.value_proj.ws.abs().max().double_value(&[]);
        assert_eq!(
            weight_abs_max, 0.0,
            "value head weights should start at zero"
        );

        assert!(
            model.value_proj.bs.is_none(),
            "value head should follow HLGaussLayer and omit bias"
        );

        let actor = Tensor::randn([TICKERS_COUNT, model.model_dim], (Kind::Float, Device::Cpu));
        let critic = Tensor::randn([TICKERS_COUNT, model.model_dim], (Kind::Float, Device::Cpu));
        let (value_logits, _, _) = model.head_from_actor_critic_cls(&actor, &critic, 1);
        let decoded = HlGaussBins::default_for(Device::Cpu).decode(&value_logits);
        let decoded_abs_max = decoded.abs().max().double_value(&[]);

        assert!(
            decoded_abs_max < 1e-6,
            "zero-logit value head should decode to zero, got abs max {decoded_abs_max}"
        );
    }

    #[test]
    fn current_model_checkpoint_roundtrips_through_partial_loader() {
        tch::manual_seed(20260612);

        let path = temp_checkpoint_path("model_roundtrip");
        let vs = nn::VarStore::new(Device::Cpu);
        let _model = TradingModel::new_with_config(&vs.root(), TradingModelConfig::default());
        vs.save(&path).expect("failed to save checkpoint");

        let mut loaded_vs = nn::VarStore::new(Device::Cpu);
        let _loaded_model =
            TradingModel::new_with_config(&loaded_vs.root(), TradingModelConfig::default());
        let summary =
            load_var_store_partial(&mut loaded_vs, &path).expect("failed to load checkpoint");
        let cleanup = fs::remove_file(&path);

        summary.require_complete().unwrap();
        assert!(summary.loaded > 0, "loader should copy checkpoint tensors");
        assert!(
            summary.migrated_legacy_value_head.is_empty(),
            "current checkpoints should not need value-head migration"
        );
        cleanup.expect("failed to remove temporary checkpoint");
    }

    #[test]
    fn biased_value_head_checkpoint_reports_dropped_bias() {
        tch::manual_seed(20260612);

        let path = temp_checkpoint_path("biased_current_value_head");
        let vs = nn::VarStore::new(Device::Cpu);
        let model = TradingModel::new_with_config(&vs.root(), TradingModelConfig::default());
        let stale_weight = Tensor::randn(model.value_proj.ws.size(), (Kind::Float, Device::Cpu));
        let mut named_tensors = vs
            .variables()
            .into_iter()
            .map(|(name, tensor)| {
                let tensor = match name.as_str() {
                    "value_proj.weight" => stale_weight.shallow_clone(),
                    _ => tensor,
                };
                (name, tensor)
            })
            .collect::<Vec<_>>();
        named_tensors.push((
            "value_proj.bias".to_string(),
            Tensor::randn([model.value_proj.ws.size()[0]], (Kind::Float, Device::Cpu)),
        ));
        let named_refs = named_tensors
            .iter()
            .map(|(name, tensor)| (name, tensor))
            .collect::<Vec<_>>();
        Tensor::save_multi(&named_refs, &path).expect("failed to save biased checkpoint");

        let mut loaded_vs = nn::VarStore::new(Device::Cpu);
        let loaded_model =
            TradingModel::new_with_config(&loaded_vs.root(), TradingModelConfig::default());
        let summary =
            load_var_store_partial(&mut loaded_vs, &path).expect("failed to load checkpoint");
        let cleanup = fs::remove_file(&path);

        summary.require_complete().unwrap();
        let mut migrated = summary.migrated_legacy_value_head.clone();
        migrated.sort();
        assert_eq!(
            migrated,
            vec![
                "value_proj.bias".to_string(),
                "value_proj.weight".to_string()
            ],
            "dropped biased-head checkpoints should be surfaced as value-head migration"
        );
        assert_value_head_is_unbiased_zero(&loaded_model);
        cleanup.expect("failed to remove temporary checkpoint");
    }

    #[test]
    fn partial_current_value_head_checkpoint_remains_strict() {
        tch::manual_seed(20260612);

        let path = temp_checkpoint_path("partial_current_value_head");
        let vs = nn::VarStore::new(Device::Cpu);
        let _model = TradingModel::new_with_config(&vs.root(), TradingModelConfig::default());
        let named_tensors = vs
            .variables()
            .into_iter()
            .filter(|(name, _)| name != "value_proj.weight")
            .collect::<Vec<_>>();
        let named_refs = named_tensors
            .iter()
            .map(|(name, tensor)| (name, tensor))
            .collect::<Vec<_>>();
        Tensor::save_multi(&named_refs, &path).expect("failed to save partial checkpoint");

        let mut loaded_vs = nn::VarStore::new(Device::Cpu);
        let _loaded_model =
            TradingModel::new_with_config(&loaded_vs.root(), TradingModelConfig::default());
        let summary =
            load_var_store_partial(&mut loaded_vs, &path).expect("failed to load checkpoint");
        let cleanup = fs::remove_file(&path);

        assert!(
            summary.require_complete().is_err(),
            "partial current value head checkpoints must remain strict"
        );
        assert!(
            summary
                .missing
                .iter()
                .any(|missing| missing == "value_proj.weight"),
            "missing current value_proj.weight should be reported"
        );
        assert!(
            summary.migrated_legacy_value_head.is_empty(),
            "current value_proj names alone must not trigger legacy migration"
        );
        cleanup.expect("failed to remove temporary checkpoint");
    }

    #[test]
    fn legacy_scalar_value_head_checkpoint_keeps_new_unbiased_zero_head() {
        tch::manual_seed(20260612);

        let path = temp_checkpoint_path("legacy_value_head");
        let vs = nn::VarStore::new(Device::Cpu);
        let model = TradingModel::new_with_config(&vs.root(), TradingModelConfig::default());
        let legacy_weight = Tensor::randn(
            [1, model.value_proj.ws.size()[1]],
            (Kind::Float, Device::Cpu),
        );
        let mut named_tensors = vs
            .variables()
            .into_iter()
            .map(|(name, tensor)| {
                let tensor = match name.as_str() {
                    "value_proj.weight" => legacy_weight.shallow_clone(),
                    _ => tensor,
                };
                (name, tensor)
            })
            .collect::<Vec<_>>();
        named_tensors.push((
            "value_proj.bias".to_string(),
            Tensor::randn([1], (Kind::Float, Device::Cpu)),
        ));
        let named_refs = named_tensors
            .iter()
            .map(|(name, tensor)| (name, tensor))
            .collect::<Vec<_>>();
        Tensor::save_multi(&named_refs, &path).expect("failed to save legacy checkpoint");

        let mut loaded_vs = nn::VarStore::new(Device::Cpu);
        let loaded_model =
            TradingModel::new_with_config(&loaded_vs.root(), TradingModelConfig::default());
        let summary =
            load_var_store_partial(&mut loaded_vs, &path).expect("failed to load checkpoint");
        let cleanup = fs::remove_file(&path);

        summary.require_complete().unwrap();
        let mut migrated = summary.migrated_legacy_value_head.clone();
        migrated.sort();
        assert_eq!(
            migrated,
            vec![
                "value_proj.bias".to_string(),
                "value_proj.weight".to_string()
            ]
        );
        assert_value_head_is_unbiased_zero(&loaded_model);
        cleanup.expect("failed to remove temporary checkpoint");
    }

    #[test]
    fn scalar_value_head_with_wrong_input_width_remains_strict() {
        tch::manual_seed(20260612);

        let path = temp_checkpoint_path("wrong_width_scalar_value_head");
        let vs = nn::VarStore::new(Device::Cpu);
        let model = TradingModel::new_with_config(&vs.root(), TradingModelConfig::default());
        let legacy_weight = Tensor::randn(
            [1, model.value_proj.ws.size()[1] - 1],
            (Kind::Float, Device::Cpu),
        );
        let mut named_tensors = vs
            .variables()
            .into_iter()
            .map(|(name, tensor)| {
                let tensor = match name.as_str() {
                    "value_proj.weight" => legacy_weight.shallow_clone(),
                    _ => tensor,
                };
                (name, tensor)
            })
            .collect::<Vec<_>>();
        named_tensors.push((
            "value_proj.bias".to_string(),
            Tensor::randn([1], (Kind::Float, Device::Cpu)),
        ));
        let named_refs = named_tensors
            .iter()
            .map(|(name, tensor)| (name, tensor))
            .collect::<Vec<_>>();
        Tensor::save_multi(&named_refs, &path).expect("failed to save legacy checkpoint");

        let mut loaded_vs = nn::VarStore::new(Device::Cpu);
        let _loaded_model =
            TradingModel::new_with_config(&loaded_vs.root(), TradingModelConfig::default());
        let summary =
            load_var_store_partial(&mut loaded_vs, &path).expect("failed to load checkpoint");
        let cleanup = fs::remove_file(&path);

        assert!(
            summary.require_complete().is_err(),
            "scalar value heads from a different critic width must remain strict"
        );
        assert!(
            summary
                .shape_mismatches
                .iter()
                .any(|mismatch| mismatch.name == "value_proj.weight"),
            "wrong-width scalar value_proj.weight should be reported"
        );
        assert!(
            !summary
                .migrated_legacy_value_head
                .iter()
                .any(|name| name == "value_proj.weight"),
            "wrong-width scalar value_proj.weight should not migrate"
        );
        cleanup.expect("failed to remove temporary checkpoint");
    }

    #[test]
    fn renamed_legacy_value_head_checkpoint_keeps_new_unbiased_zero_head() {
        tch::manual_seed(20260612);

        let path = temp_checkpoint_path("renamed_legacy_value_head");
        let vs = nn::VarStore::new(Device::Cpu);
        let model = TradingModel::new_with_config(&vs.root(), TradingModelConfig::default());
        let legacy_weight = Tensor::randn(
            [255, model.value_proj.ws.size()[1]],
            (Kind::Float, Device::Cpu),
        );
        let legacy_bias = Tensor::randn([255], (Kind::Float, Device::Cpu));
        let mut named_tensors = vs
            .variables()
            .into_iter()
            .filter(|(name, _)| name != "value_proj.weight")
            .collect::<Vec<_>>();
        named_tensors.push(("critic_out.weight".to_string(), legacy_weight));
        named_tensors.push(("critic_out.bias".to_string(), legacy_bias));
        let named_refs = named_tensors
            .iter()
            .map(|(name, tensor)| (name, tensor))
            .collect::<Vec<_>>();
        Tensor::save_multi(&named_refs, &path).expect("failed to save legacy checkpoint");

        let mut loaded_vs = nn::VarStore::new(Device::Cpu);
        let loaded_model =
            TradingModel::new_with_config(&loaded_vs.root(), TradingModelConfig::default());
        let summary =
            load_var_store_partial(&mut loaded_vs, &path).expect("failed to load checkpoint");
        let cleanup = fs::remove_file(&path);

        summary.require_complete().unwrap();
        let mut migrated = summary.migrated_legacy_value_head.clone();
        migrated.sort();
        assert_eq!(migrated, vec!["value_proj.weight".to_string()]);
        assert_value_head_is_unbiased_zero(&loaded_model);
        cleanup.expect("failed to remove temporary checkpoint");
    }

    #[test]
    fn nonscalar_value_head_shape_drift_remains_strict() {
        tch::manual_seed(20260612);

        let path = temp_checkpoint_path("mismatched_value_head");
        let vs = nn::VarStore::new(Device::Cpu);
        let model = TradingModel::new_with_config(&vs.root(), TradingModelConfig::default());
        let legacy_weight = Tensor::randn(
            [
                model.value_proj.ws.size()[0],
                model.value_proj.ws.size()[1] - 1,
            ],
            (Kind::Float, Device::Cpu),
        );
        let named_tensors = vs
            .variables()
            .into_iter()
            .map(|(name, tensor)| {
                let tensor = match name.as_str() {
                    "value_proj.weight" => legacy_weight.shallow_clone(),
                    _ => tensor,
                };
                (name, tensor)
            })
            .collect::<Vec<_>>();
        let named_refs = named_tensors
            .iter()
            .map(|(name, tensor)| (name, tensor))
            .collect::<Vec<_>>();
        Tensor::save_multi(&named_refs, &path).expect("failed to save legacy checkpoint");

        let mut loaded_vs = nn::VarStore::new(Device::Cpu);
        let _loaded_model =
            TradingModel::new_with_config(&loaded_vs.root(), TradingModelConfig::default());
        let summary =
            load_var_store_partial(&mut loaded_vs, &path).expect("failed to load checkpoint");
        let cleanup = fs::remove_file(&path);

        assert!(
            summary.require_complete().is_err(),
            "non-scalar value-head shape drift must remain strict"
        );
        assert!(
            summary.migrated_legacy_value_head.is_empty(),
            "non-scalar value-head shape drift should not use legacy migration"
        );
        assert!(
            summary
                .shape_mismatches
                .iter()
                .any(|mismatch| mismatch.name == "value_proj.weight"),
            "value_proj.weight shape drift should be reported"
        );
        cleanup.expect("failed to remove temporary checkpoint");
    }
}
