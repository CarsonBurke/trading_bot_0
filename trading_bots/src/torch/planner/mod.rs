use tch::nn::{Init, Module};
use tch::{nn, Tensor};

use crate::torch::action_space::beta_concentration;
use crate::torch::value::hl_gauss::NUM_BINS;
use crate::torch::world_model::BAR_MODEL_DIM;

pub mod checkpoint;
pub mod data;
pub mod gae;
pub mod losses;
pub mod portfolio;
pub mod reports;
pub mod rollout;
pub mod runner;

pub use data::PlannerDataSplit;
pub use runner::{
    infer_planner, train_planner, InferPlannerArgs, PlannerInferenceEpisode,
    PlannerInferenceSummary, TrainPlannerArgs,
};

pub const PLANNER_MODEL_DIM: i64 = 256;
pub const PLANNER_LAYERS: usize = 3;
pub const PLANNER_HEADS: i64 = 4;
/// Per-horizon-step forecast width: the mean AND the standard deviation, across
/// the imagined samples, of the world model's belief at that step.
pub const PLANNER_LATENT_DIM: i64 = 2 * BAR_MODEL_DIM;
pub const PLANNER_BELIEF_DIM: i64 = BAR_MODEL_DIM;
pub const PLANNER_PORTFOLIO_DIM: i64 = 4;
/// Cumulative-log-return quantiles carried per horizon step: p10, p50, p90.
pub const PLANNER_RETURN_QUANTILES: i64 = 3;

const PLANNER_FF_DIM: i64 = PLANNER_MODEL_DIM * 4;
const NORM_EPS: f64 = 1e-6;
const RESIDUAL_GAIN: f64 = 0.2;
const LATENT_LOG_RMS_LIMIT: (f64, f64) = (-6.0, 3.0);
const READOUT_SEEDS: i64 = 1;

/// One complete frozen-world-model forecast presented to the planner.
///
/// `latent` summarizes the sample axis of an ancestral rollout as a mean and a
/// standard deviation per horizon step; `return_quantiles` carries the p10/p50/p90
/// of the cumulative log return implied by the same samples at that step. The
/// belief spread is a latent-space uncertainty proxy, the return quantiles are the
/// risk the policy actually trades against, and both are summarized once at
/// collection time so the stored transition stays small and the PPO ratio exact.
pub struct PlannerForecast {
    /// `[batch, horizon, PLANNER_LATENT_DIM]`
    pub latent: Tensor,
    /// `[batch, horizon, 1]`
    pub relative_horizon: Tensor,
    /// `[batch, horizon, PLANNER_RETURN_QUANTILES]`
    pub return_quantiles: Tensor,
}

/// A forecast, the decision-time world-model belief, and the pre-action
/// portfolio state.
///
/// The belief is the normalized autoregressive summary of the real context.
/// Portfolio columns are stock weight, cash weight, previous target holding,
/// and recent turnover, in that order.
pub struct WorldModelPlannerInput {
    pub forecast: PlannerForecast,
    pub belief: Tensor,
    pub portfolio_state: Tensor,
}

pub struct WorldModelPlannerOutput {
    pub value_logits: Tensor,
    pub alpha: Tensor,
    pub beta: Tensor,
    pub next_return: Tensor,
}

pub(crate) struct PlannerCriticOutput {
    pub value_logits: Tensor,
    pub next_return: Tensor,
}

pub struct WorldModelPlanner {
    policy: PlannerBranch,
    critic: PlannerBranch,
    policy_concentration: nn::Linear,
    value_projection: nn::Linear,
    next_return_head: nn::Linear,
}

struct PlannerBranch {
    latent_projection: nn::Linear,
    latent_scale_projection: nn::Linear,
    belief_projection: nn::Linear,
    horizon_projection: nn::Linear,
    quantile_projection: nn::Linear,
    trunk: Vec<BidirectionalBlock>,
    trunk_norm: RmsNorm,
    portfolio_projection: nn::Linear,
    pma: PortfolioConditionedPma,
    readout_norm: RmsNorm,
}

impl WorldModelPlanner {
    pub fn new(p: &nn::Path) -> Self {
        let policy = PlannerBranch::new(&(p / "policy"));
        let critic = PlannerBranch::new(&(p / "critic"));
        let policy_concentration = linear_orthogonal(
            &(p / "policy" / "concentration"),
            PLANNER_MODEL_DIM,
            2,
            0.01,
        );
        let value_projection = linear_zero(
            &(p / "critic" / "value_projection"),
            PLANNER_MODEL_DIM,
            NUM_BINS,
        );
        let next_return_head =
            linear_zero(&(p / "critic" / "next_return_head"), PLANNER_MODEL_DIM, 1);
        Self {
            policy,
            critic,
            policy_concentration,
            value_projection,
            next_return_head,
        }
    }

    pub fn forward(&self, input: &WorldModelPlannerInput) -> WorldModelPlannerOutput {
        let (value_logits, raw_concentration, next_return) = self.forward_raw(input);
        mixed_precision_output(value_logits, raw_concentration, next_return)
    }

    fn forward_raw(&self, input: &WorldModelPlannerInput) -> (Tensor, Tensor, Tensor) {
        validate_input(input);
        let policy_encoded = self.policy.encode(&input.forecast, &input.belief);
        let critic_encoded = self.critic.encode(&input.forecast, &input.belief);
        let policy = self.policy.readout(&policy_encoded, &input.portfolio_state);
        let critic = self.critic.readout(&critic_encoded, &input.portfolio_state);
        let raw_concentration = self.policy_concentration.forward(&policy);
        let value_logits = self.value_projection.forward(&critic);
        let next_return = self.next_return_head.forward(&critic_encoded.select(1, 1));
        (value_logits, raw_concentration, next_return)
    }

    pub fn forward_mixed_precision(
        &self,
        input: &WorldModelPlannerInput,
    ) -> WorldModelPlannerOutput {
        let use_cuda_autocast = input.forecast.latent.device().is_cuda();
        let (value_logits, raw_concentration, next_return) =
            tch::autocast(use_cuda_autocast, || self.forward_raw(input));
        mixed_precision_output(value_logits, raw_concentration, next_return)
    }

    pub(crate) fn forward_critic_mixed_precision(
        &self,
        input: &WorldModelPlannerInput,
    ) -> PlannerCriticOutput {
        validate_input(input);
        let use_cuda_autocast = input.forecast.latent.device().is_cuda();
        let (value_logits, next_return) = tch::autocast(use_cuda_autocast, || {
            let encoded = self.critic.encode(&input.forecast, &input.belief);
            let critic = self.critic.readout(&encoded, &input.portfolio_state);
            (
                self.value_projection.forward(&critic),
                self.next_return_head.forward(&encoded.select(1, 1)),
            )
        });
        PlannerCriticOutput {
            value_logits: value_logits.to_kind(tch::Kind::Float),
            next_return: next_return.to_kind(tch::Kind::Float),
        }
    }

    #[cfg(test)]
    pub(crate) fn encode_policy_forecast_mixed_precision(
        &self,
        forecast: &PlannerForecast,
        belief: &Tensor,
    ) -> Tensor {
        validate_forecast(forecast);
        validate_belief(belief, forecast.latent.size()[0]);
        let use_cuda_autocast = forecast.latent.device().is_cuda();
        tch::autocast(use_cuda_autocast, || self.policy.encode(forecast, belief))
    }

    pub(crate) fn forward_with_policy_encoding_mixed_precision(
        &self,
        input: &WorldModelPlannerInput,
    ) -> (Tensor, WorldModelPlannerOutput) {
        validate_input(input);
        let use_cuda_autocast = input.forecast.latent.device().is_cuda();
        let (policy_encoded, value_logits, raw_concentration, next_return) =
            tch::autocast(use_cuda_autocast, || {
                let policy_encoded = self.policy.encode(&input.forecast, &input.belief);
                let critic_encoded = self.critic.encode(&input.forecast, &input.belief);
                let policy = self.policy.readout(&policy_encoded, &input.portfolio_state);
                let critic = self.critic.readout(&critic_encoded, &input.portfolio_state);
                let raw_concentration = self.policy_concentration.forward(&policy);
                let value_logits = self.value_projection.forward(&critic);
                let next_return = self.next_return_head.forward(&critic_encoded.select(1, 1));
                (policy_encoded, value_logits, raw_concentration, next_return)
            });
        (
            policy_encoded,
            mixed_precision_output(value_logits, raw_concentration, next_return),
        )
    }

    pub(crate) fn readout_policy_encoded_mixed_precision(
        &self,
        encoded_forecast: &Tensor,
        portfolio_state: &Tensor,
    ) -> (Tensor, Tensor) {
        validate_encoded_readout(encoded_forecast, portfolio_state);
        let use_cuda_autocast = encoded_forecast.device().is_cuda();
        let raw = tch::autocast(use_cuda_autocast, || {
            let policy = self.policy.readout(encoded_forecast, portfolio_state);
            self.policy_concentration.forward(&policy)
        })
        .to_kind(tch::Kind::Float);
        (
            beta_concentration(&raw.narrow(-1, 0, 1)),
            beta_concentration(&raw.narrow(-1, 1, 1)),
        )
    }

    #[cfg(test)]
    fn encode_policy_forecast(&self, forecast: &PlannerForecast, belief: &Tensor) -> Tensor {
        self.policy.encode(forecast, belief)
    }

    #[cfg(test)]
    fn policy_pooled_readout(&self, encoded: &Tensor, portfolio_state: &Tensor) -> Tensor {
        self.policy.readout(encoded, portfolio_state).unsqueeze(1)
    }
}

impl PlannerBranch {
    fn new(p: &nn::Path) -> Self {
        let latent_projection = linear_orthogonal(
            &(p / "latent_projection"),
            PLANNER_LATENT_DIM,
            PLANNER_MODEL_DIM,
            1.0,
        );
        let latent_scale_projection =
            linear_orthogonal(&(p / "latent_scale_projection"), 1, PLANNER_MODEL_DIM, 1.0);
        let belief_projection = linear_orthogonal(
            &(p / "belief_projection"),
            PLANNER_BELIEF_DIM,
            PLANNER_MODEL_DIM,
            1.0,
        );
        let horizon_projection =
            linear_orthogonal(&(p / "horizon_projection"), 1, PLANNER_MODEL_DIM, 1.0);
        let quantile_projection = linear_orthogonal(
            &(p / "quantile_projection"),
            PLANNER_RETURN_QUANTILES,
            PLANNER_MODEL_DIM,
            1.0,
        );
        let trunk = (0..PLANNER_LAYERS)
            .map(|layer| BidirectionalBlock::new(&(p / format!("trunk_{layer}"))))
            .collect();
        let portfolio_projection = linear_orthogonal(
            &(p / "portfolio_projection"),
            PLANNER_PORTFOLIO_DIM,
            PLANNER_MODEL_DIM,
            1.0,
        );
        let pma = PortfolioConditionedPma::new(&(p / "pma"));
        Self {
            latent_projection,
            latent_scale_projection,
            belief_projection,
            horizon_projection,
            quantile_projection,
            trunk,
            trunk_norm: RmsNorm::new(PLANNER_MODEL_DIM),
            portfolio_projection,
            pma,
            readout_norm: RmsNorm::new(PLANNER_MODEL_DIM),
        }
    }

    fn readout(&self, encoded_forecast: &Tensor, portfolio_state: &Tensor) -> Tensor {
        validate_encoded_readout(encoded_forecast, portfolio_state);
        let portfolio = self.portfolio_projection.forward(portfolio_state);
        let pooled = self.pma.forward(encoded_forecast, &portfolio);
        let pooled = self.readout_norm.forward(&pooled);
        pooled.select(1, 0)
    }

    fn encode(&self, forecast: &PlannerForecast, belief: &Tensor) -> Tensor {
        let (latent, latent_log_rms) = normalized_forecast_modalities(forecast);
        let forecast_tokens = self.latent_projection.forward(&latent)
            + self.latent_scale_projection.forward(&latent_log_rms)
            + self.horizon_projection.forward(&forecast.relative_horizon)
            + self
                .quantile_projection
                .forward(&forecast.return_quantiles);
        let belief_token = self.belief_projection.forward(belief).unsqueeze(1);
        let mut x = Tensor::cat(&[belief_token, forecast_tokens], 1);
        for block in &self.trunk {
            x = block.forward(&x);
        }
        self.trunk_norm.forward(&x)
    }
}

fn mixed_precision_output(
    value_logits: Tensor,
    raw_concentration: Tensor,
    next_return: Tensor,
) -> WorldModelPlannerOutput {
    let raw_concentration = raw_concentration.to_kind(tch::Kind::Float);
    WorldModelPlannerOutput {
        value_logits: value_logits.to_kind(tch::Kind::Float),
        alpha: beta_concentration(&raw_concentration.narrow(-1, 0, 1)),
        beta: beta_concentration(&raw_concentration.narrow(-1, 1, 1)),
        next_return: next_return.to_kind(tch::Kind::Float),
    }
}

/// RMS-normalizes each latent to unit scale and returns the pre-normalization
/// `log(rms)` as a per-step feature. The forecast latents are zero-noise
/// conditional means whose RMS shrinks with predictive uncertainty, so this scale
/// carries information the normalization would otherwise destroy.
fn normalized_forecast_modalities(forecast: &PlannerForecast) -> (Tensor, Tensor) {
    let latent_rms =
        (forecast
            .latent
            .square()
            .mean_dim([-1i64].as_slice(), true, tch::Kind::Float)
            + NORM_EPS)
            .sqrt();
    let latent = &forecast.latent / &latent_rms;
    let latent_log_rms = latent_rms
        .log()
        .clamp(LATENT_LOG_RMS_LIMIT.0, LATENT_LOG_RMS_LIMIT.1);
    (latent, latent_log_rms)
}

struct BidirectionalBlock {
    attention_norm: RmsNorm,
    qkv: nn::Linear,
    attention_output: nn::Linear,
    attention_scale: Tensor,
    feed_forward: FeedForward,
}

impl BidirectionalBlock {
    fn new(p: &nn::Path) -> Self {
        Self {
            attention_norm: RmsNorm::new(PLANNER_MODEL_DIM),
            qkv: linear_orthogonal(&(p / "qkv"), PLANNER_MODEL_DIM, 3 * PLANNER_MODEL_DIM, 1.0),
            attention_output: linear_orthogonal(
                &(p / "attention_output"),
                PLANNER_MODEL_DIM,
                PLANNER_MODEL_DIM,
                RESIDUAL_GAIN,
            ),
            attention_scale: p.var("attention_scale", &[PLANNER_MODEL_DIM], Init::Const(1.0)),
            feed_forward: FeedForward::new(&(p / "feed_forward")),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let (batch, horizon, _) = x.size3().expect("planner trunk input must be rank 3");
        let qkv = self.qkv.forward(&self.attention_norm.forward(x));
        let parts = qkv.chunk(3, -1);
        let head_dim = PLANNER_MODEL_DIM / PLANNER_HEADS;
        let reshape = |tensor: &Tensor| {
            tensor
                .reshape([batch, horizon, PLANNER_HEADS, head_dim])
                .permute([0, 2, 1, 3])
        };
        let q = reshape(&parts[0]);
        let k = reshape(&parts[1]);
        let v = reshape(&parts[2]);
        let attended = Tensor::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            None::<&Tensor>,
            0.0,
            false,
            None,
            false,
        )
        .permute([0, 2, 1, 3])
        .contiguous()
        .reshape([batch, horizon, PLANNER_MODEL_DIM]);
        let residual = self.attention_output.forward(&attended);
        let residual = residual * self.attention_scale.view([1, 1, PLANNER_MODEL_DIM]);
        self.feed_forward.forward(&(x + residual))
    }
}

struct PortfolioConditionedPma {
    seeds: Tensor,
    query_norm: RmsNorm,
    source_norm: RmsNorm,
    query: nn::Linear,
    key: nn::Linear,
    value: nn::Linear,
    output: nn::Linear,
    attention_scale: Tensor,
    feed_forward: FeedForward,
}

impl PortfolioConditionedPma {
    fn new(p: &nn::Path) -> Self {
        Self {
            seeds: p.var(
                "seeds",
                &[READOUT_SEEDS, PLANNER_MODEL_DIM],
                Init::Randn {
                    mean: 0.0,
                    stdev: 0.02,
                },
            ),
            query_norm: RmsNorm::new(PLANNER_MODEL_DIM),
            source_norm: RmsNorm::new(PLANNER_MODEL_DIM),
            query: linear_orthogonal(&(p / "query"), PLANNER_MODEL_DIM, PLANNER_MODEL_DIM, 1.0),
            key: linear_orthogonal(&(p / "key"), PLANNER_MODEL_DIM, PLANNER_MODEL_DIM, 1.0),
            value: linear_orthogonal(&(p / "value"), PLANNER_MODEL_DIM, PLANNER_MODEL_DIM, 1.0),
            output: linear_orthogonal(&(p / "output"), PLANNER_MODEL_DIM, PLANNER_MODEL_DIM, 1.0),
            attention_scale: p.var("attention_scale", &[PLANNER_MODEL_DIM], Init::Const(1.0)),
            feed_forward: FeedForward::new(&(p / "feed_forward")),
        }
    }

    fn forward(&self, encoded: &Tensor, portfolio: &Tensor) -> Tensor {
        let (batch, horizon, _) = encoded.size3().expect("PMA encoded input must be rank 3");
        let conditioned_seeds = self
            .seeds
            .unsqueeze(0)
            .expand([batch, READOUT_SEEDS, PLANNER_MODEL_DIM], false)
            + portfolio.unsqueeze(1);
        let source = self.source_norm.forward(encoded);
        let queries = self
            .query
            .forward(&self.query_norm.forward(&conditioned_seeds));
        let keys = self.key.forward(&source);
        let values = self.value.forward(&source);
        let head_dim = PLANNER_MODEL_DIM / PLANNER_HEADS;
        let query = queries
            .reshape([batch, READOUT_SEEDS, PLANNER_HEADS, head_dim])
            .permute([0, 2, 1, 3]);
        let key = keys
            .reshape([batch, horizon, PLANNER_HEADS, head_dim])
            .permute([0, 2, 1, 3]);
        let value = values
            .reshape([batch, horizon, PLANNER_HEADS, head_dim])
            .permute([0, 2, 1, 3]);
        let attended = Tensor::scaled_dot_product_attention(
            &query,
            &key,
            &value,
            None::<&Tensor>,
            0.0,
            false,
            None,
            false,
        )
        .permute([0, 2, 1, 3])
        .contiguous()
        .reshape([batch, READOUT_SEEDS, PLANNER_MODEL_DIM]);
        let residual = self.output.forward(&attended);
        let residual = residual * self.attention_scale.view([1, 1, PLANNER_MODEL_DIM]);
        self.feed_forward.forward(&(conditioned_seeds + residual))
    }
}

struct FeedForward {
    norm: RmsNorm,
    input: nn::Linear,
    output: nn::Linear,
    scale: Tensor,
}

impl FeedForward {
    fn new(p: &nn::Path) -> Self {
        Self {
            norm: RmsNorm::new(PLANNER_MODEL_DIM),
            input: linear_orthogonal(&(p / "input"), PLANNER_MODEL_DIM, PLANNER_FF_DIM, 1.0),
            output: linear_orthogonal(
                &(p / "output"),
                PLANNER_FF_DIM,
                PLANNER_MODEL_DIM,
                RESIDUAL_GAIN,
            ),
            scale: p.var("scale", &[PLANNER_MODEL_DIM], Init::Const(1.0)),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let hidden = self.input.forward(&self.norm.forward(x)).relu().square();
        let residual = self.output.forward(&hidden) * self.scale.view([1, 1, PLANNER_MODEL_DIM]);
        x + residual
    }
}

struct RmsNorm {
    dim: i64,
}

impl RmsNorm {
    fn new(dim: i64) -> Self {
        Self { dim }
    }

    fn forward(&self, input: &Tensor) -> Tensor {
        input
            .internal_fused_rms_norm([self.dim], None::<&Tensor>, Some(NORM_EPS))
            .0
    }
}

fn validate_input(input: &WorldModelPlannerInput) {
    validate_forecast(&input.forecast);
    let batch = input.forecast.latent.size()[0];
    validate_belief(&input.belief, batch);
    assert_eq!(
        input.portfolio_state.size(),
        [batch, PLANNER_PORTFOLIO_DIM],
        "portfolio_state must have shape [batch, 4]"
    );
}

fn validate_forecast(forecast: &PlannerForecast) {
    let latent_shape = forecast.latent.size();
    assert_eq!(
        latent_shape.len(),
        3,
        "latent must have shape [batch, horizon, {PLANNER_LATENT_DIM}]"
    );
    let batch = latent_shape[0];
    let horizon = latent_shape[1];
    assert!(batch > 0, "planner batch must not be empty");
    assert!(horizon > 0, "planner horizon must not be empty");
    assert_eq!(
        latent_shape[2], PLANNER_LATENT_DIM,
        "latent width must be {PLANNER_LATENT_DIM}"
    );
    assert_eq!(
        forecast.relative_horizon.size(),
        [batch, horizon, 1],
        "relative_horizon must have shape [batch, horizon, 1]"
    );
    assert_eq!(
        forecast.return_quantiles.size(),
        [batch, horizon, PLANNER_RETURN_QUANTILES],
        "return_quantiles must have shape [batch, horizon, {PLANNER_RETURN_QUANTILES}]"
    );
}

fn validate_belief(belief: &Tensor, batch: i64) {
    assert_eq!(
        belief.size(),
        [batch, PLANNER_BELIEF_DIM],
        "belief must have shape [batch, {PLANNER_BELIEF_DIM}]"
    );
}

fn validate_encoded_readout(encoded_forecast: &Tensor, portfolio_state: &Tensor) {
    let shape = encoded_forecast.size();
    assert_eq!(
        shape.len(),
        3,
        "encoded forecast must have shape [batch, horizon, model_dim]"
    );
    assert!(shape[0] > 0, "planner batch must not be empty");
    assert!(shape[1] > 0, "planner horizon must not be empty");
    assert_eq!(
        shape[2], PLANNER_MODEL_DIM,
        "encoded forecast width must match planner model dim"
    );
    assert_eq!(
        portfolio_state.size(),
        [shape[0], PLANNER_PORTFOLIO_DIM],
        "portfolio_state must have shape [batch, 4]"
    );
}

fn linear_orthogonal(p: &nn::Path, input: i64, output: i64, gain: f64) -> nn::Linear {
    nn::linear(
        p,
        input,
        output,
        nn::LinearConfig {
            ws_init: Init::Orthogonal { gain },
            bs_init: None,
            bias: false,
        },
    )
}

fn linear_zero(p: &nn::Path, input: i64, output: i64) -> nn::Linear {
    nn::linear(
        p,
        input,
        output,
        nn::LinearConfig {
            ws_init: Init::Const(0.0),
            bs_init: None,
            bias: false,
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::torch::test_rng;
    use tch::{Device, IndexOp, Kind};

    fn forecast(batch: i64, horizon: i64) -> PlannerForecast {
        let relative_horizon = (Tensor::arange(horizon, (Kind::Float, Device::Cpu)) + 1.0)
            .view([1, horizon, 1])
            .expand([batch, horizon, 1], false)
            / horizon as f64;
        // Monotone in the horizon and ordered p10 < p50 < p90, like a real
        // cumulative-return quantile fan.
        let spread = Tensor::from_slice(&[-1.0f32, 0.0, 1.0]).view([1, 1, 3]);
        let return_quantiles = &relative_horizon * spread * 0.05;
        PlannerForecast {
            latent: Tensor::randn(
                [batch, horizon, PLANNER_LATENT_DIM],
                (Kind::Float, Device::Cpu),
            ),
            relative_horizon,
            return_quantiles,
        }
    }

    fn belief(batch: i64) -> Tensor {
        Tensor::randn([batch, PLANNER_BELIEF_DIM], (Kind::Float, Device::Cpu))
    }

    fn input(batch: i64, horizon: i64) -> WorldModelPlannerInput {
        WorldModelPlannerInput {
            forecast: forecast(batch, horizon),
            belief: belief(batch),
            portfolio_state: Tensor::randn(
                [batch, PLANNER_PORTFOLIO_DIM],
                (Kind::Float, Device::Cpu),
            ),
        }
    }

    fn planner() -> (nn::VarStore, WorldModelPlanner) {
        tch::manual_seed(7);
        let vs = nn::VarStore::new(Device::Cpu);
        let planner = WorldModelPlanner::new(&vs.root());
        (vs, planner)
    }

    #[test]
    fn output_contract_has_expected_shapes_and_finite_values() {
        let _torch_rng_guard = test_rng::exclusive();
        let (_vs, planner) = planner();
        let output = planner.forward(&input(2, 9));
        assert_eq!(output.value_logits.size(), [2, NUM_BINS]);
        assert_eq!(output.alpha.size(), [2, 1]);
        assert_eq!(output.beta.size(), [2, 1]);
        assert!(output.value_logits.isfinite().all().int64_value(&[]) != 0);
        assert!(output.alpha.isfinite().all().int64_value(&[]) != 0);
        assert!(output.beta.isfinite().all().int64_value(&[]) != 0);
        assert!(output.alpha.ge(1.0).all().int64_value(&[]) != 0);
        assert!(output.beta.ge(1.0).all().int64_value(&[]) != 0);
        assert_eq!(output.next_return.size(), [2, 1]);
        assert!(output.next_return.isfinite().all().int64_value(&[]) != 0);
    }

    #[test]
    fn mixed_precision_boundary_returns_fp32_outputs_and_gradients() {
        let _torch_rng_guard = test_rng::exclusive();
        let (vs, planner) = planner();
        let output = planner.forward_mixed_precision(&input(2, 5));
        for tensor in [
            &output.value_logits,
            &output.alpha,
            &output.beta,
            &output.next_return,
        ] {
            assert_eq!(tensor.kind(), Kind::Float);
        }
        (output.value_logits.sum(Kind::Float)
            + output.alpha.sum(Kind::Float)
            + output.beta.sum(Kind::Float)
            + output.next_return.sum(Kind::Float))
        .backward();
        assert!(vs
            .trainable_variables()
            .iter()
            .any(|parameter| parameter.grad().defined()));
    }

    #[test]
    fn staged_policy_encoding_matches_the_policy_forward() {
        let _torch_rng_guard = test_rng::exclusive();
        let (_vs, planner) = planner();
        let input = input(2, 7);
        let direct = planner.forward_mixed_precision(&input);
        let encoded =
            planner.encode_policy_forecast_mixed_precision(&input.forecast, &input.belief);
        assert_eq!(encoded.size(), [2, 8, PLANNER_MODEL_DIM]);
        let (alpha, beta) =
            planner.readout_policy_encoded_mixed_precision(&encoded, &input.portfolio_state);

        assert!(direct.alpha.allclose(&alpha, 1e-6, 1e-6, false));
        assert!(direct.beta.allclose(&beta, 1e-6, 1e-6, false));
    }

    #[test]
    fn staged_policy_encoding_and_readout_preserve_gradients() {
        let _torch_rng_guard = test_rng::exclusive();
        let (vs, planner) = planner();
        let mut input = input(2, 5);
        input.forecast.latent = input.forecast.latent.set_requires_grad(true);
        input.belief = input.belief.set_requires_grad(true);
        input.portfolio_state = input.portfolio_state.set_requires_grad(true);
        let encoded =
            planner.encode_policy_forecast_mixed_precision(&input.forecast, &input.belief);
        let (alpha, beta) =
            planner.readout_policy_encoded_mixed_precision(&encoded, &input.portfolio_state);
        (alpha.sum(Kind::Float) + beta.sum(Kind::Float)).backward();

        assert!(input.forecast.latent.grad().defined());
        assert!(input.belief.grad().defined());
        assert!(input.portfolio_state.grad().defined());
        assert!(vs
            .trainable_variables()
            .iter()
            .any(|parameter| parameter.grad().defined()));
    }

    #[test]
    fn cuda_flash_only_forward_uses_supported_autocast_dtype() {
        let _torch_rng_guard = test_rng::shared();
        let device = Device::cuda_if_available();
        if !device.is_cuda() {
            return;
        }
        crate::torch::cuda::cfg::configure_cuda();
        let vs = nn::VarStore::new(device);
        let planner = WorldModelPlanner::new(&vs.root());
        let cpu = input(2, 9);
        let cuda_input = WorldModelPlannerInput {
            forecast: PlannerForecast {
                latent: cpu.forecast.latent.to_device(device),
                relative_horizon: cpu.forecast.relative_horizon.to_device(device),
                return_quantiles: cpu.forecast.return_quantiles.to_device(device),
            },
            belief: cpu.belief.to_device(device),
            portfolio_state: cpu.portfolio_state.to_device(device),
        };
        let output = tch::no_grad(|| planner.forward_mixed_precision(&cuda_input));
        assert_eq!(output.alpha.kind(), Kind::Float);
        assert!(output.alpha.isfinite().all().int64_value(&[]) != 0);
        assert!(output.value_logits.isfinite().all().int64_value(&[]) != 0);
    }

    #[test]
    fn gradients_reach_forecast_portfolio_and_parameters() {
        let _torch_rng_guard = test_rng::exclusive();
        let (vs, planner) = planner();
        let mut input = input(2, 6);
        input.forecast.latent = input.forecast.latent.set_requires_grad(true);
        input.portfolio_state = input.portfolio_state.set_requires_grad(true);
        let output = planner.forward(&input);
        let loss = output.alpha.sum(Kind::Float)
            + output.beta.sum(Kind::Float)
            + output.value_logits.square().sum(Kind::Float);
        loss.backward();

        let latent_grad = input.forecast.latent.grad();
        let portfolio_grad = input.portfolio_state.grad();
        assert!(latent_grad.defined());
        assert!(portfolio_grad.defined());
        assert!(latent_grad.isfinite().all().int64_value(&[]) != 0);
        assert!(portfolio_grad.isfinite().all().int64_value(&[]) != 0);
        assert!(latent_grad.abs().sum(Kind::Float).double_value(&[]) > 0.0);
        assert!(portfolio_grad.abs().sum(Kind::Float).double_value(&[]) > 0.0);
        assert!(vs
            .trainable_variables()
            .iter()
            .any(|parameter| parameter.grad().defined()));
    }

    #[test]
    fn first_token_depends_on_a_future_token() {
        let _torch_rng_guard = test_rng::exclusive();
        let (_vs, planner) = planner();
        let original = forecast(1, 7);
        let belief = belief(1);
        let changed = PlannerForecast {
            latent: Tensor::cat(
                &[
                    original.latent.narrow(1, 0, 6),
                    original.latent.narrow(1, 6, 1) + 10.0,
                ],
                1,
            ),
            relative_horizon: original.relative_horizon.shallow_clone(),
            return_quantiles: original.return_quantiles.shallow_clone(),
        };
        let before = planner.encode_policy_forecast(&original, &belief).i((0, 0));
        let after = planner.encode_policy_forecast(&changed, &belief).i((0, 0));
        let difference = (before - after).abs().max().double_value(&[]);
        assert!(difference > 1e-6, "future token did not affect first token");
    }

    #[test]
    fn all_pma_seeds_depend_on_portfolio_state() {
        let _torch_rng_guard = test_rng::exclusive();
        let (_vs, planner) = planner();
        let encoded = planner.encode_policy_forecast(&forecast(1, 5), &belief(1));
        let portfolio_a = Tensor::zeros([1, PLANNER_PORTFOLIO_DIM], (Kind::Float, Device::Cpu));
        let portfolio_b = Tensor::ones([1, PLANNER_PORTFOLIO_DIM], (Kind::Float, Device::Cpu));
        let pooled_a = planner.policy_pooled_readout(&encoded, &portfolio_a);
        let pooled_b = planner.policy_pooled_readout(&encoded, &portfolio_b);
        let per_seed =
            (pooled_a - pooled_b)
                .abs()
                .sum_dim_intlist([-1].as_slice(), false, Kind::Float);
        assert!(per_seed.double_value(&[0, 0]) > 1e-6);
    }

    #[test]
    fn policy_and_critic_have_disjoint_parameter_names() {
        let _torch_rng_guard = test_rng::exclusive();
        let (vs, _planner) = planner();
        let variables = vs.variables();
        let policy = variables
            .keys()
            .filter(|name| name.starts_with("policy."))
            .count();
        let critic = variables
            .keys()
            .filter(|name| name.starts_with("critic."))
            .count();
        assert!(policy > 0);
        assert!(critic > 0);
        assert_eq!(policy + critic, variables.len());
    }

    #[test]
    fn forecast_modalities_normalize_latents_and_expose_bounded_log_scale() {
        let _torch_rng_guard = test_rng::shared();
        let scaled = Tensor::randn([1, 2, PLANNER_LATENT_DIM], (Kind::Float, Device::Cpu)) * 7.0;
        let forecast = PlannerForecast {
            latent: scaled.shallow_clone(),
            relative_horizon: Tensor::ones([1, 2, 1], (Kind::Float, Device::Cpu)),
            return_quantiles: Tensor::zeros(
                [1, 2, PLANNER_RETURN_QUANTILES],
                (Kind::Float, Device::Cpu),
            ),
        };
        let (latent, latent_log_rms) = normalized_forecast_modalities(&forecast);
        let latent_rms = latent
            .square()
            .mean_dim([-1].as_slice(), false, Kind::Float)
            .sqrt();
        assert!((latent_rms - 1.0).abs().max().double_value(&[]) < 1e-5);
        assert_eq!(latent_log_rms.size(), [1, 2, 1]);
        assert!(
            latent_log_rms
                .ge(LATENT_LOG_RMS_LIMIT.0)
                .logical_and(&latent_log_rms.le(LATENT_LOG_RMS_LIMIT.1))
                .all()
                .int64_value(&[])
                != 0
        );
        let expected_log_rms = (scaled
            .square()
            .mean_dim([-1i64].as_slice(), true, Kind::Float)
            + NORM_EPS)
            .sqrt()
            .log();
        assert!(
            (latent_log_rms - expected_log_rms)
                .abs()
                .max()
                .double_value(&[])
                < 1e-5
        );
    }
}
