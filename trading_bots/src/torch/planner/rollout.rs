use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use tch::{Device, Tensor};

use super::{
    PLANNER_BELIEF_DIM, PLANNER_LATENT_DIM, PLANNER_PORTFOLIO_DIM, PLANNER_RETURN_QUANTILES,
};

#[derive(Debug)]
pub struct PlannerObservation {
    /// `[horizon, PLANNER_LATENT_DIM]`
    pub forecast_latent: Tensor,
    /// `[horizon, 1]`
    pub relative_horizon: Tensor,
    /// `[horizon, PLANNER_RETURN_QUANTILES]`
    pub return_quantiles: Tensor,
    pub belief: Tensor,
    pub portfolio_state: Tensor,
}

#[derive(Debug)]
pub struct PlannerTransition {
    pub observation: PlannerObservation,
    pub environment_id: usize,
    pub decision_index: usize,
    pub action: Tensor,
    pub old_alpha: Tensor,
    pub old_beta: Tensor,
    pub value: Tensor,
    pub next_value: Option<Tensor>,
    pub reward: f32,
    pub next_log_return: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub commission: f64,
    pub turnover: f64,
    pub executed_stock_weight: f64,
    pub assets_before: f64,
    pub assets_after: f64,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PlannerRolloutError {
    ZeroCapacity,
    Full,
    Incomplete { actual: usize, expected: usize },
    InvalidMinibatchSize(usize),
    EmptyWorldModelHash,
    InconsistentForecastHorizon,
    InvalidTensorShape(&'static str),
    NonFiniteValue(&'static str),
    OutOfRangeValue(&'static str),
}

impl Display for PlannerRolloutError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCapacity => write!(formatter, "rollout capacity must be positive"),
            Self::Full => write!(formatter, "rollout is full"),
            Self::Incomplete { actual, expected } => {
                write!(
                    formatter,
                    "rollout requires {expected} samples, got {actual}"
                )
            }
            Self::InvalidMinibatchSize(size) => write!(
                formatter,
                "minibatch size must be positive and divide the rollout; got {size}"
            ),
            Self::EmptyWorldModelHash => write!(formatter, "world-model checkpoint hash is empty"),
            Self::InconsistentForecastHorizon => {
                write!(
                    formatter,
                    "all transitions must use the same forecast horizon"
                )
            }
            Self::InvalidTensorShape(name) => write!(formatter, "invalid tensor shape for {name}"),
            Self::NonFiniteValue(name) => write!(formatter, "non-finite value in {name}"),
            Self::OutOfRangeValue(name) => write!(formatter, "out-of-range value in {name}"),
        }
    }
}

impl Error for PlannerRolloutError {}

pub struct PlannerRollout {
    expected_samples: usize,
    transitions: Vec<PlannerTransition>,
    world_model_hash: String,
}

impl PlannerRollout {
    pub fn new(
        expected_samples: usize,
        world_model_hash: String,
    ) -> Result<Self, PlannerRolloutError> {
        if expected_samples == 0 {
            return Err(PlannerRolloutError::ZeroCapacity);
        }
        if world_model_hash.trim().is_empty() {
            return Err(PlannerRolloutError::EmptyWorldModelHash);
        }
        Ok(Self {
            expected_samples,
            transitions: Vec::with_capacity(expected_samples),
            world_model_hash,
        })
    }

    pub fn world_model_hash(&self) -> &str {
        &self.world_model_hash
    }

    pub fn push(&mut self, transition: PlannerTransition) -> Result<(), PlannerRolloutError> {
        validate_transition(&transition)?;
        if self.transitions.len() == self.expected_samples {
            return Err(PlannerRolloutError::Full);
        }
        self.transitions.push(transition);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn transitions(&self) -> &[PlannerTransition] {
        &self.transitions
    }

    pub fn validate_complete(&self) -> Result<(), PlannerRolloutError> {
        if self.transitions.len() != self.expected_samples {
            return Err(PlannerRolloutError::Incomplete {
                actual: self.transitions.len(),
                expected: self.expected_samples,
            });
        }
        Ok(())
    }

    pub fn minibatch_indices(
        &self,
        minibatch_size: usize,
        seed: u64,
    ) -> Result<Vec<Vec<i64>>, PlannerRolloutError> {
        self.validate_complete()?;
        if minibatch_size == 0 || self.len() % minibatch_size != 0 {
            return Err(PlannerRolloutError::InvalidMinibatchSize(minibatch_size));
        }
        let mut indices = (0..self.len() as i64).collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(seed);
        indices.shuffle(&mut rng);
        Ok(indices
            .chunks(minibatch_size)
            .map(<[i64]>::to_vec)
            .collect())
    }

    pub fn to_batch(&self, device: Device) -> Result<PlannerBatch, PlannerRolloutError> {
        self.validate_complete()?;
        PlannerBatch::from_validated_transitions(&self.transitions, device)
    }

    pub fn metrics(&self) -> RolloutMetrics {
        RolloutMetrics::from_transitions(&self.transitions)
    }
}

pub struct PlannerBatch {
    pub forecast_latent: Tensor,
    pub relative_horizon: Tensor,
    pub return_quantiles: Tensor,
    pub belief: Tensor,
    pub portfolio_state: Tensor,
    pub actions: Tensor,
    pub old_alpha: Tensor,
    pub old_beta: Tensor,
    pub values: Tensor,
    pub next_values: Tensor,
    pub rewards: Tensor,
    pub next_log_returns: Tensor,
    pub terminated: Tensor,
    pub truncated: Tensor,
}

impl PlannerBatch {
    pub fn from_transitions(
        transitions: &[PlannerTransition],
        device: Device,
    ) -> Result<Self, PlannerRolloutError> {
        if transitions.is_empty() {
            return Err(PlannerRolloutError::ZeroCapacity);
        }
        for transition in transitions {
            validate_transition(transition)?;
        }
        Self::from_validated_transitions(transitions, device)
    }

    fn from_validated_transitions(
        transitions: &[PlannerTransition],
        device: Device,
    ) -> Result<Self, PlannerRolloutError> {
        if transitions.is_empty() {
            return Err(PlannerRolloutError::ZeroCapacity);
        }
        let horizon = transitions[0].observation.forecast_latent.size()[0];
        if transitions
            .iter()
            .any(|transition| transition.observation.forecast_latent.size()[0] != horizon)
        {
            return Err(PlannerRolloutError::InconsistentForecastHorizon);
        }

        Ok(Self {
            forecast_latent: stack_to_device(
                transitions.iter().map(|t| &t.observation.forecast_latent),
                device,
            ),
            relative_horizon: stack_to_device(
                transitions.iter().map(|t| &t.observation.relative_horizon),
                device,
            ),
            return_quantiles: stack_to_device(
                transitions.iter().map(|t| &t.observation.return_quantiles),
                device,
            ),
            belief: stack_to_device(transitions.iter().map(|t| &t.observation.belief), device),
            portfolio_state: stack_to_device(
                transitions.iter().map(|t| &t.observation.portfolio_state),
                device,
            ),
            actions: stack_flat_to_device(transitions.iter().map(|t| &t.action), device),
            old_alpha: stack_flat_to_device(transitions.iter().map(|t| &t.old_alpha), device),
            old_beta: stack_flat_to_device(transitions.iter().map(|t| &t.old_beta), device),
            values: cat_scalars(transitions.iter().map(|t| &t.value), device),
            next_values: cat_scalars(
                transitions.iter().map(|t| {
                    t.next_value
                        .as_ref()
                        .expect("next_value filled before batching")
                }),
                device,
            ),
            rewards: Tensor::from_slice(&transitions.iter().map(|t| t.reward).collect::<Vec<_>>())
                .to_device(device),
            next_log_returns: Tensor::from_slice(
                &transitions
                    .iter()
                    .map(|t| t.next_log_return)
                    .collect::<Vec<_>>(),
            )
            .to_device(device),
            terminated: bool_tensor(transitions.iter().map(|t| t.terminated), device),
            truncated: bool_tensor(transitions.iter().map(|t| t.truncated), device),
        })
    }

    pub fn len(&self) -> i64 {
        self.rewards.size()[0]
    }

    pub fn select(&self, indices: &[i64]) -> Self {
        let index = Tensor::from_slice(indices).to_device(self.rewards.device());
        Self {
            forecast_latent: self.forecast_latent.index_select(0, &index),
            relative_horizon: self.relative_horizon.index_select(0, &index),
            return_quantiles: self.return_quantiles.index_select(0, &index),
            belief: self.belief.index_select(0, &index),
            portfolio_state: self.portfolio_state.index_select(0, &index),
            actions: self.actions.index_select(0, &index),
            old_alpha: self.old_alpha.index_select(0, &index),
            old_beta: self.old_beta.index_select(0, &index),
            values: self.values.index_select(0, &index),
            next_values: self.next_values.index_select(0, &index),
            rewards: self.rewards.index_select(0, &index),
            next_log_returns: self.next_log_returns.index_select(0, &index),
            terminated: self.terminated.index_select(0, &index),
            truncated: self.truncated.index_select(0, &index),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RolloutMetrics {
    pub samples: usize,
    pub reward_sum: f64,
    pub reward_mean: f64,
    pub commissions: f64,
    pub turnover_mean: f64,
    pub requested_target_weight_mean: f64,
    pub executed_stock_weight_mean: f64,
    pub action_boundary_fraction: f64,
    pub mean_environment_wealth_ratio: f64,
}

impl RolloutMetrics {
    fn from_transitions(transitions: &[PlannerTransition]) -> Self {
        if transitions.is_empty() {
            return Self::default();
        }
        let samples = transitions.len();
        let reward_sum = transitions.iter().map(|t| t.reward as f64).sum::<f64>();
        let commissions = transitions.iter().map(|t| t.commission).sum::<f64>();
        let turnover_sum = transitions.iter().map(|t| t.turnover).sum::<f64>();
        let executed_stock_weight_sum = transitions
            .iter()
            .map(|t| t.executed_stock_weight)
            .sum::<f64>();
        let mut requested_target_weight_sum = 0.0;
        let mut boundary_count = 0usize;
        for transition in transitions {
            let action = transition.action.flatten(0, -1).double_value(&[0]);
            requested_target_weight_sum += action;
            if !(0.01..=0.99).contains(&action) {
                boundary_count += 1;
            }
        }
        let mut environments: HashMap<usize, (usize, f64, usize, f64)> = HashMap::new();
        for transition in transitions {
            environments
                .entry(transition.environment_id)
                .and_modify(|entry| {
                    if transition.decision_index < entry.0 {
                        entry.0 = transition.decision_index;
                        entry.1 = transition.assets_before;
                    }
                    if transition.decision_index >= entry.2 {
                        entry.2 = transition.decision_index;
                        entry.3 = transition.assets_after;
                    }
                })
                .or_insert((
                    transition.decision_index,
                    transition.assets_before,
                    transition.decision_index,
                    transition.assets_after,
                ));
        }
        let wealth_ratio_sum = environments
            .values()
            .map(|(_, initial, _, final_assets)| final_assets / initial)
            .sum::<f64>();
        Self {
            samples,
            reward_sum,
            reward_mean: reward_sum / samples as f64,
            commissions,
            turnover_mean: turnover_sum / samples as f64,
            requested_target_weight_mean: requested_target_weight_sum / samples as f64,
            executed_stock_weight_mean: executed_stock_weight_sum / samples as f64,
            action_boundary_fraction: boundary_count as f64 / samples as f64,
            mean_environment_wealth_ratio: wealth_ratio_sum / environments.len() as f64,
        }
    }
}

fn validate_transition(transition: &PlannerTransition) -> Result<(), PlannerRolloutError> {
    let latent = transition.observation.forecast_latent.size();
    if latent.len() != 2 || latent[0] <= 0 || latent[1] != PLANNER_LATENT_DIM {
        return Err(PlannerRolloutError::InvalidTensorShape("forecast_latent"));
    }
    if transition.observation.relative_horizon.size() != [latent[0], 1] {
        return Err(PlannerRolloutError::InvalidTensorShape("relative_horizon"));
    }
    if transition.observation.return_quantiles.size() != [latent[0], PLANNER_RETURN_QUANTILES] {
        return Err(PlannerRolloutError::InvalidTensorShape("return_quantiles"));
    }
    if transition.observation.belief.size() != [PLANNER_BELIEF_DIM] {
        return Err(PlannerRolloutError::InvalidTensorShape("belief"));
    }
    if transition.observation.portfolio_state.size() != [PLANNER_PORTFOLIO_DIM] {
        return Err(PlannerRolloutError::InvalidTensorShape("portfolio_state"));
    }
    let next_value = transition
        .next_value
        .as_ref()
        .ok_or(PlannerRolloutError::InvalidTensorShape("next_value"))?;
    if transition.action.numel() != 1
        || transition.old_alpha.numel() != 1
        || transition.old_beta.numel() != 1
        || transition.value.numel() != 1
        || next_value.numel() != 1
    {
        return Err(PlannerRolloutError::InvalidTensorShape(
            "scalar rollout fields",
        ));
    }
    for (name, tensor) in [
        ("forecast_latent", &transition.observation.forecast_latent),
        ("relative_horizon", &transition.observation.relative_horizon),
        ("return_quantiles", &transition.observation.return_quantiles),
        ("belief", &transition.observation.belief),
        ("portfolio_state", &transition.observation.portfolio_state),
        ("action", &transition.action),
        ("old_alpha", &transition.old_alpha),
        ("old_beta", &transition.old_beta),
        ("value", &transition.value),
        ("next_value", next_value),
    ] {
        if tensor.isfinite().all().int64_value(&[]) == 0 {
            return Err(PlannerRolloutError::NonFiniteValue(name));
        }
    }
    let requested_target_weight = transition.action.flatten(0, -1).double_value(&[0]);
    if !(0.0..=1.0).contains(&requested_target_weight) {
        return Err(PlannerRolloutError::OutOfRangeValue("action"));
    }
    for (name, finite) in [
        ("reward", transition.reward.is_finite()),
        ("next_log_return", transition.next_log_return.is_finite()),
        ("commission", transition.commission.is_finite()),
        ("turnover", transition.turnover.is_finite()),
        (
            "executed_stock_weight",
            transition.executed_stock_weight.is_finite(),
        ),
        ("assets_before", transition.assets_before.is_finite()),
        ("assets_after", transition.assets_after.is_finite()),
    ] {
        if !finite {
            return Err(PlannerRolloutError::NonFiniteValue(name));
        }
    }
    if !(0.0..=1.0).contains(&transition.executed_stock_weight) {
        return Err(PlannerRolloutError::OutOfRangeValue(
            "executed_stock_weight",
        ));
    }
    Ok(())
}

fn stack_to_device<'a>(tensors: impl Iterator<Item = &'a Tensor>, device: Device) -> Tensor {
    Tensor::stack(&tensors.collect::<Vec<_>>(), 0).to_device(device)
}

fn stack_flat_to_device<'a>(tensors: impl Iterator<Item = &'a Tensor>, device: Device) -> Tensor {
    Tensor::stack(
        &tensors
            .map(|tensor| tensor.flatten(0, -1))
            .collect::<Vec<_>>(),
        0,
    )
    .to_device(device)
}

fn cat_scalars<'a>(tensors: impl Iterator<Item = &'a Tensor>, device: Device) -> Tensor {
    Tensor::cat(
        &tensors
            .map(|tensor| tensor.flatten(0, -1))
            .collect::<Vec<_>>(),
        0,
    )
    .to_device(device)
}

fn bool_tensor(values: impl Iterator<Item = bool>, device: Device) -> Tensor {
    Tensor::from_slice(
        &values
            .map(|value| if value { 1.0f32 } else { 0.0f32 })
            .collect::<Vec<_>>(),
    )
    .to_device(device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use tch::Kind;

    fn transition(environment_id: usize, decision_index: usize) -> PlannerTransition {
        PlannerTransition {
            observation: PlannerObservation {
                forecast_latent: Tensor::zeros([3, PLANNER_LATENT_DIM], (Kind::Float, Device::Cpu)),
                relative_horizon: Tensor::arange(3, (Kind::Float, Device::Cpu)).view([3, 1]),
                return_quantiles: Tensor::zeros(
                    [3, PLANNER_RETURN_QUANTILES],
                    (Kind::Float, Device::Cpu),
                ),
                belief: Tensor::zeros([PLANNER_BELIEF_DIM], (Kind::Float, Device::Cpu)),
                portfolio_state: Tensor::from_slice(&[0.5f32, 0.5, 0.25, 0.0]),
            },
            environment_id,
            decision_index,
            action: Tensor::from_slice(&[0.25f32]),
            old_alpha: Tensor::from_slice(&[2.0f32]),
            old_beta: Tensor::from_slice(&[2.0f32]),
            value: Tensor::from_slice(&[0.2f32]),
            next_value: Some(Tensor::from_slice(&[0.3f32])),
            reward: 0.4,
            next_log_return: 0.01,
            terminated: false,
            truncated: decision_index == 1,
            commission: 0.01,
            turnover: 0.1,
            executed_stock_weight: 0.2,
            assets_before: 100.0 + decision_index as f64,
            assets_after: 101.0 + decision_index as f64,
        }
    }

    #[test]
    fn rollout_requires_exact_sample_count() {
        let mut rollout = PlannerRollout::new(2, "wm-test".to_owned()).unwrap();
        rollout.push(transition(0, 0)).unwrap();
        assert_eq!(
            rollout.validate_complete(),
            Err(PlannerRolloutError::Incomplete {
                actual: 1,
                expected: 2
            })
        );
        rollout.push(transition(0, 1)).unwrap();
        rollout.validate_complete().unwrap();
        assert_eq!(
            rollout.push(transition(0, 2)),
            Err(PlannerRolloutError::Full)
        );
    }

    #[test]
    fn minibatches_are_deterministic_complete_partitions() {
        let mut rollout = PlannerRollout::new(8, "wm-test".to_owned()).unwrap();
        for i in 0..8 {
            rollout.push(transition(i / 4, i % 4)).unwrap();
        }
        let first = rollout.minibatch_indices(4, 7).unwrap();
        let second = rollout.minibatch_indices(4, 7).unwrap();
        assert_eq!(first, second);
        let mut indices = first.into_iter().flatten().collect::<Vec<_>>();
        indices.sort_unstable();
        assert_eq!(indices, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn tensor_batch_preserves_model_contract_shapes_without_source_labels() {
        let batch =
            PlannerBatch::from_transitions(&[transition(0, 0), transition(1, 0)], Device::Cpu)
                .unwrap();
        assert_eq!(batch.forecast_latent.size(), [2, 3, PLANNER_LATENT_DIM]);
        assert_eq!(batch.belief.size(), [2, PLANNER_BELIEF_DIM]);
        assert_eq!(batch.portfolio_state.size(), [2, PLANNER_PORTFOLIO_DIM]);
        assert_eq!(batch.actions.size(), [2, 1]);
        assert_eq!(batch.values.size(), [2]);
        assert_eq!(batch.next_log_returns.size(), [2]);
    }

    #[test]
    fn metrics_aggregate_real_environments() {
        let mut first = transition(0, 0);
        first.reward = 1.0;
        first.assets_before = 100.0;
        first.assets_after = 101.0;
        let mut last = transition(0, 1);
        last.reward = -0.5;
        last.assets_before = 101.0;
        last.assets_after = 105.0;
        let mut rollout = PlannerRollout::new(2, "wm-test".to_owned()).unwrap();
        rollout.push(first).unwrap();
        rollout.push(last).unwrap();
        let metrics = rollout.metrics();
        assert_eq!(metrics.reward_sum, 0.5);
        assert_eq!(metrics.mean_environment_wealth_ratio, 1.05);
        assert_eq!(metrics.requested_target_weight_mean, 0.25);
        assert_eq!(metrics.executed_stock_weight_mean, 0.2);
    }

    #[test]
    fn rejects_invalid_shapes_non_finite_values_and_empty_hash() {
        let mut invalid = transition(0, 0);
        invalid.observation.forecast_latent =
            Tensor::zeros([3, PLANNER_LATENT_DIM - 1], (Kind::Float, Device::Cpu));
        assert!(matches!(
            PlannerBatch::from_transitions(&[invalid], Device::Cpu),
            Err(PlannerRolloutError::InvalidTensorShape("forecast_latent"))
        ));
        let mut non_finite = transition(0, 0);
        non_finite.reward = f32::NAN;
        assert!(matches!(
            PlannerBatch::from_transitions(&[non_finite], Device::Cpu),
            Err(PlannerRolloutError::NonFiniteValue("reward"))
        ));
        let mut out_of_range = transition(0, 0);
        out_of_range.action = Tensor::from_slice(&[1.01f32]);
        assert!(matches!(
            PlannerBatch::from_transitions(&[out_of_range], Device::Cpu),
            Err(PlannerRolloutError::OutOfRangeValue("action"))
        ));
        let mut out_of_range = transition(0, 0);
        out_of_range.executed_stock_weight = -0.01;
        assert!(matches!(
            PlannerBatch::from_transitions(&[out_of_range], Device::Cpu),
            Err(PlannerRolloutError::OutOfRangeValue(
                "executed_stock_weight"
            ))
        ));
        assert_eq!(
            PlannerRollout::new(1, String::new()).err(),
            Some(PlannerRolloutError::EmptyWorldModelHash)
        );
    }
}
