use std::env;

use crate::torch::constants::EPISODE_TRANSITIONS;

/// NorMuon LR for 2D weight matrices (NS5 + per-row second-moment updates).
/// 5e-3 is the offline grid optimum on a transformer LM (benchmarks/optim-grid,
/// SDPA + paper-matching NorMuon/AdamW routing); the MLP-tuned 3e-3 under-shoots
/// on a real transformer. ~4x below the NorMuon reference's 0.02, which overshoots
/// at our scale. Watch policy KL on the first RL run; RL may tolerate less.
pub(crate) const MUON_LR: f64 = 5e-3;
/// AdamW LR for 1D params (biases, norms) and the standalone rho scalar.
pub(crate) const LEARNING_RATE: f64 = 3e-4;
/// Warmup endpoint; reference NorMuon default. The grid showed 0.99 is the worst
/// beta1 (over-smooths, caps usable LR); 0.95 matches the reference and is
/// grid-competitive with 0.90.
pub(crate) const MUON_MOMENTUM: f64 = 0.95;
pub(crate) const MUON_MOMENTUM_WARMUP_START: f64 = 0.92;
pub(crate) const MUON_MOMENTUM_WARMUP_STEPS: i64 = 50;
pub(crate) const USE_MUON: bool = true;
pub const DEFAULT_NPROCS: i64 = 16;
pub(crate) const DEFAULT_SEQ_LEN: i64 = EPISODE_TRANSITIONS as i64;
pub(crate) const DEFAULT_PPO_CHUNK_LEN: i64 = 60;
pub(crate) const DEFAULT_PPO_MINIBATCH_RATIO: f64 = 1.0 / 16.0;
pub(crate) const OPTIM_EPOCHS: i64 = 3;
pub(crate) const SPO_EPS_LOW: f64 = 0.40;
pub(crate) const SPO_EPS_HIGH: f64 = 0.56;
pub(crate) const TARGET_KL: f64 = 0.03;
pub(crate) const KL_STOP_MULTIPLIER: f64 = 1.5;
pub(crate) const VALUE_LOSS_COEF: f64 = 1.0;
pub(crate) const ENTROPY_COEF: f64 = 0.0;
pub(crate) const MAX_GRAD_NORM: f64 = 0.5;
pub(crate) const DEBUG_NUMERICS: bool = false;

pub(crate) fn parse_positive_i64_env(name: &str) -> Option<i64> {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|&v| v > 0)
}
