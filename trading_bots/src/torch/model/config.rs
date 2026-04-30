use clap::ValueEnum;

use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS};

pub(in crate::torch::model) const BASE_MODEL_DIM: i64 = 256;
pub(in crate::torch::model) const BASE_FF_DIM: i64 = 512;
pub(in crate::torch::model) const BASE_GQA_LAYERS: usize = 3;
pub(in crate::torch::model) const ABLATION_SMALL_MODEL_DIM: i64 = 96;
pub(in crate::torch::model) const ABLATION_SMALL_FF_DIM: i64 = 192;
pub(in crate::torch::model) const ABLATION_SMALL_GQA_LAYERS: usize = 1;
pub(in crate::torch::model) const ACTOR_CRITIC_CLS_COUNT: i64 = 2;
pub(in crate::torch::model) const UNIFORM_STREAM_PATCH_COUNT: i64 = 120;
pub(in crate::torch::model) const UNIFORM_STREAM_PATCH_SIZE: i64 = 50;
pub(in crate::torch::model) const UNIFORM_STREAM_LAYOUT_LEN: i64 =
    UNIFORM_STREAM_PATCH_COUNT * UNIFORM_STREAM_PATCH_SIZE;
pub(in crate::torch::model) const UNIFORM_STREAM_BOOTSTRAP_FULL_PATCHES: i64 =
    UNIFORM_STREAM_PATCH_COUNT - 1;
pub(in crate::torch::model) const UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL: i64 =
    PRICE_DELTAS_PER_TICKER as i64
        - UNIFORM_STREAM_BOOTSTRAP_FULL_PATCHES * UNIFORM_STREAM_PATCH_SIZE;
pub(in crate::torch::model) const INTER_TICKER_AFTER: usize = 1;
pub(in crate::torch::model) const NUM_EXO_TOKENS: i64 = STATIC_OBSERVATIONS as i64;
pub(in crate::torch::model) const PATCH_SCALAR_FEATS: i64 = 3;

pub(in crate::torch::model) const BASE_PATCH_CONFIGS: &[(i64, i64)] = &[
    (3072, 128),
    (1536, 64),
    (768, 32),
    (384, 16),
    (128, 8),
    (64, 4),
    (46, 2),
    (2, 1),
];

const fn uniform_stream_patch_configs() -> [(i64, i64); UNIFORM_STREAM_PATCH_COUNT as usize] {
    let mut configs = [(0i64, 0i64); UNIFORM_STREAM_PATCH_COUNT as usize];
    let mut idx = 0usize;
    while idx < UNIFORM_STREAM_PATCH_COUNT as usize {
        configs[idx] = (UNIFORM_STREAM_PATCH_SIZE, UNIFORM_STREAM_PATCH_SIZE);
        idx += 1;
    }
    configs
}

pub(in crate::torch::model) const UNIFORM_STREAM_PATCH_CONFIGS: &[(i64, i64)] =
    &uniform_stream_patch_configs();

pub(in crate::torch::model) const ABLATION_SMALL_PATCH_CONFIGS: &[(i64, i64)] =
    &[(5632, 256), (360, 8), (8, 1)];

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
#[value(rename_all = "kebab-case")]
pub enum ModelVariant {
    Base,
    #[value(
        name = "uniform-stream",
        alias = "uniform-256-stream",
        alias = "uniform256-stream"
    )]
    UniformStream,
    AblationSmall,
}

impl ModelVariant {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Base => "base",
            Self::UniformStream => "uniform-stream",
            Self::AblationSmall => "ablation-small",
        }
    }
}

#[derive(Clone, Copy)]
pub(in crate::torch::model) struct ModelSpec {
    pub(in crate::torch::model) model_dim: i64,
    pub(in crate::torch::model) ff_dim: i64,
    pub(in crate::torch::model) gqa_layers: usize,
    pub(in crate::torch::model) patch_configs: &'static [(i64, i64)],
}

pub(in crate::torch::model) fn model_spec(variant: ModelVariant) -> ModelSpec {
    match variant {
        ModelVariant::Base => ModelSpec {
            model_dim: BASE_MODEL_DIM,
            ff_dim: BASE_FF_DIM,
            gqa_layers: BASE_GQA_LAYERS,
            patch_configs: BASE_PATCH_CONFIGS,
        },
        ModelVariant::UniformStream => ModelSpec {
            model_dim: BASE_MODEL_DIM,
            ff_dim: BASE_FF_DIM,
            gqa_layers: BASE_GQA_LAYERS,
            patch_configs: UNIFORM_STREAM_PATCH_CONFIGS,
        },
        ModelVariant::AblationSmall => ModelSpec {
            model_dim: ABLATION_SMALL_MODEL_DIM,
            ff_dim: ABLATION_SMALL_FF_DIM,
            gqa_layers: ABLATION_SMALL_GQA_LAYERS,
            patch_configs: ABLATION_SMALL_PATCH_CONFIGS,
        },
    }
}

pub(in crate::torch::model) fn compute_patch_totals(patch_configs: &[(i64, i64)]) -> (i64, i64) {
    let mut total_days = 0i64;
    let mut total_tokens = 0i64;
    for &(days, patch_size) in patch_configs {
        assert!(
            days % patch_size == 0,
            "days must be divisible by patch_size"
        );
        total_days += days;
        total_tokens += days / patch_size;
    }
    (total_days, total_tokens)
}

pub fn patch_seq_len_for_variant(variant: ModelVariant) -> i64 {
    compute_patch_totals(model_spec(variant).patch_configs).1
}

pub fn patch_ends_for_variant(variant: ModelVariant) -> Vec<i64> {
    let patch_configs = model_spec(variant).patch_configs;
    let seq_len = patch_seq_len_for_variant(variant);
    let mut ends = Vec::with_capacity(seq_len as usize);
    let mut total = 0i64;
    for &(days, patch_size) in patch_configs {
        let num = days / patch_size;
        for _ in 0..num {
            total += patch_size;
            ends.push(total);
        }
    }
    ends
}
