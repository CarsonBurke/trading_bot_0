use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use super::blocks::cross_attn::CrossAttnFfnBlock;
use super::blocks::endogenous::EndogenousTickerBlock;
use super::blocks::exogenous::ExoMLP;
use super::blocks::gqa::{GqaBlock, GQA_NUM_Q_HEADS};
use super::config::{
    compute_patch_totals, model_spec, ModelVariant, ACTOR_CRITIC_CLS_COUNT, INTER_TICKER_AFTER,
    NUM_EXO_TOKENS, PATCH_SCALAR_FEATS, UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL,
    UNIFORM_STREAM_LAYOUT_LEN, UNIFORM_STREAM_PATCH_COUNT, UNIFORM_STREAM_PATCH_SIZE,
};
use super::init::{
    linear_identity, linear_orthogonal, linear_with_same_dtype, residual_init_scale,
    truncated_normal_init, xavier_normal_std,
};
use super::rmsnorm::RMSNorm;
use super::rope::{RotaryEmbedding, ROPE_DIMS};
use crate::torch::constants::{
    ACTION_COUNT, GLOBAL_STATIC_OBS, PER_TICKER_STATIC_OBS, PRICE_DELTAS_PER_TICKER,
    TICKERS_COUNT,
};
use crate::torch::value::hl_gauss::NUM_BINS;

/// (value_logits, action_alpha, action_beta, action_std)
pub type ModelOutput = (Tensor, Tensor, Tensor, Tensor);

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
/// - No model state needed (GQA is stateless, uses full forward pass)
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
    /// Prefix hidden state after layer-0 self-attention/FFN, before exogenous cross-attention.
    pub uniform_layer0_prefix_hidden: Tensor,
    /// Layer-0 prefix K cache for uniform streamed rollout.
    pub uniform_layer0_prefix_k: Tensor,
    /// Layer-0 prefix V cache for uniform streamed rollout.
    pub uniform_layer0_prefix_v: Tensor,
    /// Per-layer cached prefix K for uniform streamed rollout.
    pub uniform_prefix_k: Vec<Tensor>,
    /// Per-layer cached prefix V for uniform streamed rollout.
    pub uniform_prefix_v: Vec<Tensor>,
    /// Prefix x0 embedding (post-input_ln) for x0 residual mixing.
    pub uniform_prefix_x0: Tensor,
    /// Static features associated with the currently conditioned prefix cache.
    pub uniform_cached_static_features: Option<Tensor>,
    /// Exogenous tokens associated with the currently conditioned prefix cache.
    pub uniform_cached_exo_tokens: Option<Tensor>,
}

pub struct TradingModel {
    pub(in crate::torch::model) variant: ModelVariant,
    pub(in crate::torch::model) patch_configs: &'static [(i64, i64)],
    pub(in crate::torch::model) seq_len: i64,
    pub(in crate::torch::model) finest_patch_size: i64,
    pub(in crate::torch::model) model_dim: i64,
    pub(in crate::torch::model) ff_dim: i64,
    pub(in crate::torch::model) patch_embed_weight: Tensor,
    pub(in crate::torch::model) patch_config_ids: Tensor,
    pub(in crate::torch::model) patch_stream_proj: nn::Linear,
    pub(in crate::torch::model) input_ln: RMSNorm,
    pub(in crate::torch::model) final_ln: RMSNorm,
    pub(in crate::torch::model) gqa_layers: Vec<GqaBlock>,
    pub(in crate::torch::model) exogenous_ticker_block: CrossAttnFfnBlock,
    pub(in crate::torch::model) exo_mlp: ExoMLP,
    pub(in crate::torch::model) exo_embed_ln: RMSNorm,
    pub(in crate::torch::model) rope: RotaryEmbedding,
    pub(in crate::torch::model) exo_feat_w: Tensor,
    pub(in crate::torch::model) exo_feat_b: Tensor,
    pub(in crate::torch::model) endogenous_ticker_block: EndogenousTickerBlock,
    pub(in crate::torch::model) actor_live_proj: nn::Linear,
    pub(in crate::torch::model) critic_live_proj: nn::Linear,
    pub(in crate::torch::model) policy_alpha_beta: nn::Linear,
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

    pub fn variant(&self) -> ModelVariant {
        self.variant
    }

    pub fn patch_seq_len(&self) -> i64 {
        self.seq_len
    }

    pub(in crate::torch::model) fn uniform_stream_layout_from_raw(&self, deltas: &Tensor) -> Tensor {
        let device = deltas.device();
        let batch = deltas.size()[0];
        let layout = Tensor::full(
            [batch, UNIFORM_STREAM_LAYOUT_LEN],
            f64::NAN,
            (Kind::Float, device),
        );
        let full_prefix = super::config::UNIFORM_STREAM_BOOTSTRAP_FULL_PATCHES
            * UNIFORM_STREAM_PATCH_SIZE;
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
        let price = if price_deltas.dim() == 1 {
            price_deltas.unsqueeze(0)
        } else {
            price_deltas.shallow_clone()
        };
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
        assert_eq!(ROPE_DIMS % 2, 0, "RoPE dimensions must be even");
        let gqa_layers_count = spec.gqa_layers;
        // SA + FFN per layer = 2 sublayers each, plus 1 CA sublayer after layer 0
        let num_residual_sublayers = gqa_layers_count * 2 + 1;
        let init_scale = residual_init_scale(num_residual_sublayers);
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
        let patch_stream_proj = nn::linear(
            p / "patch_stream_proj",
            UNIFORM_STREAM_PATCH_SIZE + 1, // patch values + fill_fraction
            spec.model_dim,
            nn::LinearConfig {
                ws_init: truncated_normal_init(UNIFORM_STREAM_PATCH_SIZE + 1, spec.model_dim),
                bs_init: None,
                bias: false,
            },
        );
        let input_ln = RMSNorm::new(&(p / "input_ln"), spec.model_dim, 1e-6);
        let final_ln = RMSNorm::new(&(p / "final_ln"), spec.model_dim, 1e-6);
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
            .map(|i| {
                GqaBlock::new(
                    &(p / format!("gqa_{}", i)),
                    spec.model_dim,
                    spec.ff_dim,
                    init_scale,
                    i,
                )
            })
            .collect::<Vec<_>>();
        let exogenous_ticker_block = CrossAttnFfnBlock::new(
            &(p / "cross_attn_0"),
            spec.model_dim,
            spec.ff_dim,
            init_scale,
        );
        let exo_mlp = ExoMLP::new(&(p / "exo_mlp"), spec.model_dim, init_scale);
        let exo_embed_ln = RMSNorm::new(&(p / "exo_embed_ln"), spec.model_dim, 1e-6);
        let head_dim = spec.model_dim / GQA_NUM_Q_HEADS;
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
        let endogenous_ticker_block = EndogenousTickerBlock::new(
            &(p / "inter_ticker_0"),
            spec.model_dim,
            spec.ff_dim,
            init_scale,
        );
        let actor_live_proj = linear_identity(p, "actor_live_proj", spec.model_dim);
        let critic_live_proj = linear_identity(p, "critic_live_proj", spec.model_dim);
        assert_eq!(
            ACTION_COUNT, TICKERS_COUNT,
            "per-ticker actor head requires one action per ticker"
        );
        let flat_all_tickers = TICKERS_COUNT * spec.model_dim;
        let policy_alpha_beta = nn::linear(
            p / "policy_alpha_beta",
            spec.model_dim,
            2,
            nn::LinearConfig {
                ws_init: Init::Randn {
                    mean: 0.0,
                    stdev: 0.01,
                },
                bs_init: None,
                bias: false,
            },
        );
        let value_proj = linear_orthogonal(p, "value_proj", flat_all_tickers, NUM_BINS, 0.1);
        Self {
            variant: config.variant,
            patch_configs,
            seq_len,
            finest_patch_size,
            model_dim: spec.model_dim,
            ff_dim: spec.ff_dim,
            patch_embed_weight,
            patch_config_ids,
            patch_stream_proj,
            input_ln,
            final_ln,
            gqa_layers,
            exogenous_ticker_block,
            exo_mlp,
            exo_embed_ln,
            rope,
            exo_feat_w,
            exo_feat_b,
            endogenous_ticker_block,
            actor_live_proj,
            critic_live_proj,
            policy_alpha_beta,
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
        let live_idx = if seq > self.seq_len {
            self.seq_len - 1
        } else {
            seq - 1
        };
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
        } else if live_idx + 1 == seq {
            let past = x_4d.narrow(2, 0, seq - 1);
            Tensor::cat(&[&past, &enriched_live], 2).reshape([bt, seq, self.model_dim])
        } else {
            let before = x_4d.narrow(2, 0, live_idx);
            let after = x_4d.narrow(2, live_idx + 1, seq - live_idx - 1);
            Tensor::cat(&[&before, &enriched_live, &after], 2).reshape([bt, seq, self.model_dim])
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
        self.exo_mlp.forward(&self.exo_embed_ln.forward(&exo_kv))
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

    /// Per-config enrichment avoids expanding each patch to the full history width,
    /// then projects all tokens in one fused einsum.
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

    pub(in crate::torch::model) fn patch_embed_stream_batch(
        &self,
        patch_vals: &Tensor,
        fill_counts: &Tensor,
    ) -> Tensor {
        let target_kind = patch_vals.kind();
        let patch_size = UNIFORM_STREAM_PATCH_SIZE;
        // Build position mask from fill counts — don't read NaN positions
        let positions = Tensor::arange(patch_size, (Kind::Int64, patch_vals.device()));
        let mask = positions
            .unsqueeze(0)
            .less_tensor(&fill_counts.unsqueeze(-1)); // [batch, patch_size]
        let patch_vals_float = patch_vals.to_kind(Kind::Float);
        let clean = patch_vals_float * mask.to_kind(Kind::Float);
        let fill_fraction = fill_counts.to_kind(Kind::Float).unsqueeze(-1) / patch_size as f64;
        let input = Tensor::cat(
            &[
                &clean.to_kind(target_kind),
                &fill_fraction.to_kind(target_kind),
            ],
            -1,
        ); // [batch, patch_size + 1]
        linear_with_same_dtype(&input, &self.patch_stream_proj)
    }

    pub(in crate::torch::model) fn patch_embed_stream(&self, deltas: &Tensor) -> Tensor {
        let batch = deltas.size()[0];
        let patches = deltas.view([
            batch * UNIFORM_STREAM_PATCH_COUNT,
            UNIFORM_STREAM_PATCH_SIZE,
        ]);
        // Compute fill counts per patch from valid (non-NaN) positions
        let fill_counts = patches
            .isnan()
            .logical_not()
            .to_kind(Kind::Int64)
            .sum_dim_intlist([1].as_slice(), false, Kind::Int64);
        self.patch_embed_stream_batch(&patches, &fill_counts).view([
            batch,
            UNIFORM_STREAM_PATCH_COUNT,
            self.model_dim,
        ])
    }

    pub(in crate::torch::model) fn actor_critic_rope_positions(&self, total_seq_len: i64) -> Tensor {
        let patch_seq_len = total_seq_len - ACTOR_CRITIC_CLS_COUNT;
        let patch_positions = Tensor::arange(patch_seq_len, (Kind::Int64, self.device));
        let cls_positions = Tensor::full(
            [ACTOR_CRITIC_CLS_COUNT],
            patch_seq_len - 1,
            (Kind::Int64, self.device),
        );
        Tensor::cat(&[&patch_positions, &cls_positions], 0)
    }

    pub(in crate::torch::model) fn actor_critic_cls_from_live(&self, live: &Tensor) -> Tensor {
        let live = if live.dim() == 2 {
            live.unsqueeze(1)
        } else {
            live.shallow_clone()
        };
        let actor = linear_with_same_dtype(&live, &self.actor_live_proj);
        let critic = linear_with_same_dtype(&live, &self.critic_live_proj);
        Tensor::cat(&[&actor, &critic], 1)
    }

    pub(in crate::torch::model) fn append_actor_critic_cls(&self, x: &Tensor) -> Tensor {
        let live = x.narrow(1, x.size()[1] - 1, 1);
        let cls = self.actor_critic_cls_from_live(&live);
        Tensor::cat(&[x, &cls], 1)
    }

    pub(in crate::torch::model) fn backbone_with_actor_critic_cls(
        &self,
        patch_hidden: &Tensor,
        exo_tokens: &Tensor,
        batch_size: i64,
    ) -> ModelOutput {
        let x0 = self.append_actor_critic_cls(patch_hidden);
        let mut x = x0.shallow_clone();
        let rope_positions = self.actor_critic_rope_positions(x0.size()[1]);
        for (layer_idx, layer) in self.gqa_layers.iter().enumerate() {
            x = layer.forward_with_rope_positions(&x, &x0, &self.rope, &rope_positions, true);
            if layer_idx == 0 {
                x = self.exogenous_ticker_block.forward(&x, exo_tokens);
            }
            x = self.maybe_apply_endogenous_ticker(&x, layer_idx);
        }
        let x = self.final_ln.forward(&x);
        let seq = x.size()[1];
        let actor = x.select(1, seq - 2);
        let critic = x.select(1, seq - 1);
        self.head_from_actor_critic_cls(&actor, &critic, batch_size)
    }
}
