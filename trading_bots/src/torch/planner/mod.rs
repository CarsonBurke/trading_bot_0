use tch::nn::Init;
use tch::{nn, Kind, Tensor};

use crate::torch::action_space::beta_concentration;
use crate::torch::value::hl_gauss::NUM_BINS;

pub mod checkpoint;
pub mod data;
pub mod gae;
pub mod losses;
pub mod portfolio;
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
pub const PLANNER_LATENT_DIM: i64 = 256;
pub const PLANNER_OHLC_DIM: i64 = 16;
pub const PLANNER_PORTFOLIO_DIM: i64 = 4;

const PLANNER_FF_DIM: i64 = PLANNER_MODEL_DIM * 4;
const FORECAST_TOKEN_DIM: i64 = PLANNER_LATENT_DIM + 2 * PLANNER_OHLC_DIM + 1;
const NORM_EPS: f64 = 1e-6;
const RESIDUAL_GAIN: f64 = 0.2;

/// One complete frozen-world-model forecast presented to the planner.
pub struct PlannerForecast {
    pub latent: Tensor,
    pub ohlc_mean: Tensor,
    pub ohlc_log_variance: Tensor,
    pub relative_horizon: Tensor,
}

/// A forecast plus the pre-action portfolio state.
///
/// Portfolio columns are stock weight, cash weight, previous target holding,
/// and recent turnover, in that order.
pub struct WorldModelPlannerInput {
    pub forecast: PlannerForecast,
    pub portfolio_state: Tensor,
}

pub struct WorldModelPlannerOutput {
    pub value_logits: Tensor,
    pub alpha: Tensor,
    pub beta: Tensor,
}

pub struct WorldModelPlanner {
    token_projection: nn::Linear,
    trunk: Vec<BidirectionalBlock>,
    trunk_norm: RmsNorm,
    portfolio_projection: nn::Linear,
    pma: PortfolioConditionedPma,
    readout_norm: RmsNorm,
    policy_concentration: nn::Linear,
    value_projection: nn::Linear,
}

impl WorldModelPlanner {
    pub fn new(p: &nn::Path) -> Self {
        let token_projection = linear_orthogonal(
            &(p / "token_projection"),
            FORECAST_TOKEN_DIM,
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
        let policy_concentration =
            linear_orthogonal(&(p / "policy_concentration"), PLANNER_MODEL_DIM, 2, 0.01);
        let value_projection = linear_zero(&(p / "value_projection"), PLANNER_MODEL_DIM, NUM_BINS);
        Self {
            token_projection,
            trunk,
            trunk_norm: RmsNorm::new(PLANNER_MODEL_DIM),
            portfolio_projection,
            pma,
            readout_norm: RmsNorm::new(PLANNER_MODEL_DIM),
            policy_concentration,
            value_projection,
        }
    }

    pub fn forward(&self, input: &WorldModelPlannerInput) -> WorldModelPlannerOutput {
        validate_input(input);
        let encoded = self.encode_forecast(&input.forecast);
        let portfolio = align_to(&input.portfolio_state, &encoded);
        let portfolio = linear_same_dtype(&portfolio, &self.portfolio_projection);
        let pooled = self.pma.forward(&encoded, &portfolio);
        let pooled = self.readout_norm.forward(&pooled);
        let actor = pooled.select(1, 0);
        let critic = pooled.select(1, 1);

        let raw_concentration =
            linear_same_dtype(&actor, &self.policy_concentration).to_kind(Kind::Float);
        let alpha = beta_concentration(&raw_concentration.narrow(-1, 0, 1));
        let beta = beta_concentration(&raw_concentration.narrow(-1, 1, 1));
        let value_logits = linear_same_dtype(&critic, &self.value_projection).to_kind(Kind::Float);

        WorldModelPlannerOutput {
            value_logits,
            alpha,
            beta,
        }
    }

    /// Encodes every predicted step with full, non-causal attention.
    pub fn encode_forecast(&self, forecast: &PlannerForecast) -> Tensor {
        validate_forecast(forecast);
        let latent = &forecast.latent;
        let features = Tensor::cat(
            &[
                latent.shallow_clone(),
                align_to(&forecast.ohlc_mean, latent),
                align_to(&forecast.ohlc_log_variance, latent),
                align_to(&forecast.relative_horizon, latent),
            ],
            -1,
        );
        let mut x = linear_same_dtype(&features, &self.token_projection);
        for block in &self.trunk {
            x = block.forward(&x);
        }
        self.trunk_norm.forward(&x)
    }

    #[cfg(test)]
    fn pooled_readout(&self, encoded: &Tensor, portfolio_state: &Tensor) -> Tensor {
        let portfolio = linear_same_dtype(
            &align_to(portfolio_state, encoded),
            &self.portfolio_projection,
        );
        self.pma.forward(encoded, &portfolio)
    }
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
        let qkv = linear_same_dtype(&self.attention_norm.forward(x), &self.qkv);
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
        let residual = linear_same_dtype(&attended, &self.attention_output);
        let residual = residual
            * self
                .attention_scale
                .to_kind(x.kind())
                .view([1, 1, PLANNER_MODEL_DIM]);
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
                &[2, PLANNER_MODEL_DIM],
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
            .to_device(encoded.device())
            .to_kind(encoded.kind())
            .unsqueeze(0)
            .expand([batch, 2, PLANNER_MODEL_DIM], false)
            + portfolio.unsqueeze(1);
        let source = self.source_norm.forward(encoded);
        let queries = linear_same_dtype(&self.query_norm.forward(&conditioned_seeds), &self.query);
        let keys = linear_same_dtype(&source, &self.key);
        let values = linear_same_dtype(&source, &self.value);
        let head_dim = PLANNER_MODEL_DIM / PLANNER_HEADS;
        let query = queries
            .reshape([batch, 2, PLANNER_HEADS, head_dim])
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
        .reshape([batch, 2, PLANNER_MODEL_DIM]);
        let residual = linear_same_dtype(&attended, &self.output);
        let residual = residual
            * self
                .attention_scale
                .to_kind(encoded.kind())
                .view([1, 1, PLANNER_MODEL_DIM]);
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
        let hidden = linear_same_dtype(&self.norm.forward(x), &self.input)
            .relu()
            .square();
        let residual = linear_same_dtype(&hidden, &self.output)
            * self.scale.to_kind(x.kind()).view([1, 1, PLANNER_MODEL_DIM]);
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
        "latent must have shape [batch, horizon, 256]"
    );
    let batch = latent_shape[0];
    let horizon = latent_shape[1];
    assert!(batch > 0, "planner batch must not be empty");
    assert!(horizon > 0, "planner horizon must not be empty");
    assert_eq!(
        latent_shape[2], PLANNER_LATENT_DIM,
        "latent width must be 256"
    );
    assert_eq!(
        forecast.ohlc_mean.size(),
        [batch, horizon, PLANNER_OHLC_DIM],
        "ohlc_mean must have shape [batch, horizon, 16]"
    );
    assert_eq!(
        forecast.ohlc_log_variance.size(),
        [batch, horizon, PLANNER_OHLC_DIM],
        "ohlc_log_variance must have shape [batch, horizon, 16]"
    );
    assert_eq!(
        forecast.relative_horizon.size(),
        [batch, horizon, 1],
        "relative_horizon must have shape [batch, horizon, 1]"
    );
}

fn align_to(input: &Tensor, reference: &Tensor) -> Tensor {
    input
        .to_device(reference.device())
        .to_kind(reference.kind())
}

fn linear_same_dtype(input: &Tensor, linear: &nn::Linear) -> Tensor {
    let weight = linear.ws.to_kind(input.kind());
    let bias = linear.bs.as_ref().map(|bias| bias.to_kind(input.kind()));
    input.linear(&weight, bias.as_ref())
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
    use tch::{Device, IndexOp};

    fn forecast(batch: i64, horizon: i64) -> PlannerForecast {
        PlannerForecast {
            latent: Tensor::randn(
                [batch, horizon, PLANNER_LATENT_DIM],
                (Kind::Float, Device::Cpu),
            ),
            ohlc_mean: Tensor::randn(
                [batch, horizon, PLANNER_OHLC_DIM],
                (Kind::Float, Device::Cpu),
            ),
            ohlc_log_variance: Tensor::randn(
                [batch, horizon, PLANNER_OHLC_DIM],
                (Kind::Float, Device::Cpu),
            ),
            relative_horizon: Tensor::arange(horizon, (Kind::Float, Device::Cpu))
                .view([1, horizon, 1])
                .expand([batch, horizon, 1], false)
                / horizon as f64,
        }
    }

    fn input(batch: i64, horizon: i64) -> WorldModelPlannerInput {
        WorldModelPlannerInput {
            forecast: forecast(batch, horizon),
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
    }

    #[test]
    fn gradients_reach_forecast_portfolio_and_parameters() {
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
        let (_vs, planner) = planner();
        let original = forecast(1, 7);
        let changed = PlannerForecast {
            latent: Tensor::cat(
                &[
                    original.latent.narrow(1, 0, 6),
                    original.latent.narrow(1, 6, 1) + 10.0,
                ],
                1,
            ),
            ohlc_mean: original.ohlc_mean.shallow_clone(),
            ohlc_log_variance: original.ohlc_log_variance.shallow_clone(),
            relative_horizon: original.relative_horizon.shallow_clone(),
        };
        let before = planner.encode_forecast(&original).i((0, 0));
        let after = planner.encode_forecast(&changed).i((0, 0));
        let difference = (before - after).abs().max().double_value(&[]);
        assert!(difference > 1e-6, "future token did not affect first token");
    }

    #[test]
    fn both_pma_seeds_depend_on_portfolio_state() {
        let (_vs, planner) = planner();
        let encoded = planner.encode_forecast(&forecast(1, 5));
        let portfolio_a = Tensor::zeros([1, PLANNER_PORTFOLIO_DIM], (Kind::Float, Device::Cpu));
        let portfolio_b = Tensor::ones([1, PLANNER_PORTFOLIO_DIM], (Kind::Float, Device::Cpu));
        let pooled_a = planner.pooled_readout(&encoded, &portfolio_a);
        let pooled_b = planner.pooled_readout(&encoded, &portfolio_b);
        let per_seed =
            (pooled_a - pooled_b)
                .abs()
                .sum_dim_intlist([-1].as_slice(), false, Kind::Float);
        assert!(per_seed.double_value(&[0, 0]) > 1e-6);
        assert!(per_seed.double_value(&[0, 1]) > 1e-6);
    }
}
