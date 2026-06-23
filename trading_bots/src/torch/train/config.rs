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
pub(crate) const CLIP_EPS_LOW: f64 = 0.20;
pub(crate) const CLIP_EPS_HIGH: f64 = 0.28;
pub(crate) const KL_LR_TARGET: f64 = 0.035;
pub(crate) const KL_LR_EMA_HALF_LIFE: f64 = 50.0;
pub(crate) const KL_LR_MIN_SCALE: f64 = 0.01;
pub(crate) const KL_LR_MAX_SCALE: f64 = 10.0;
pub(crate) const TARGET_KL: f64 = KL_LR_TARGET;
pub(crate) const KL_STOP_MULTIPLIER: f64 = 1.0;
pub(crate) const VALUE_LOSS_COEF: f64 = 1.0;
/// Critic-only pretraining horizon. For the first `CRITIC_PRETRAIN_EPISODES`
/// episodes only the value loss is backpropagated: the shared trunk learns a
/// representation driven purely by value error and the policy head stays frozen
/// (zero actor gradient, no entropy bonus). This warms up a usable critic before
/// the actor begins, so early advantages are not estimated against noise. The
/// CUDA update graph is suppressed during this phase and captured fresh once the
/// actor turns on, so the persisted graph always reflects full actor+critic.
pub(crate) const CRITIC_PRETRAIN_EPISODES: usize = 100;
/// Policy-optimization objective, selected at compile time.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PolicyObjective {
    Ppo,
    Pmpo,
}
/// PMPO (sign-based weighted-MLE + closed-form reverse-KL trust region) is the
/// default. PMPO uses RAW GAE advantages (no per-minibatch percentile norm).
pub(crate) const POLICY_OBJECTIVE: PolicyObjective = PolicyObjective::Pmpo;
pub(crate) const PMPO_POS_TO_NEG_WEIGHT: f64 = 0.5;
pub(crate) const PMPO_KL_COEF: f64 = 0.3;
/// Per-minibatch percentile return-norm ("mbpercnorm"): scale advantages by
/// S = max(FLOOR, P_HI - P_LO) of THIS minibatch's raw GAE returns, recomputed
/// fresh per minibatch (no EMA). Divide-only (no mean subtraction). Matches the
/// CleanRL reference's `ret_percnorm` with `ret_perc_scope="minibatch"`.
pub(crate) const RET_PERC_LO: f64 = 0.05;
pub(crate) const RET_PERC_HI: f64 = 0.95;
pub(crate) const RET_PERC_FLOOR: f64 = 1.0;
/// Our beta distribution with log variance explores very well, and better without entropy regulation.
pub(crate) const ENTROPY_COEF: f64 = 0.0;
pub(crate) const MAX_GRAD_NORM: f64 = 0.5;
pub(crate) const DEBUG_NUMERICS: bool = false;

pub(crate) fn parse_positive_i64_env(name: &str) -> Option<i64> {
    env::var(name)
        .ok()
        .and_then(|v| v.parse::<i64>().ok())
        .filter(|&v| v > 0)
}
