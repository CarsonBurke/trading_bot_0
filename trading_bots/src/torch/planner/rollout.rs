use std::collections::HashMap;
use std::error::Error;
use std::fmt::{Display, Formatter};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use tch::{Device, Tensor};

use super::{PLANNER_LATENT_DIM, PLANNER_OHLC_DIM, PLANNER_PORTFOLIO_DIM};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[repr(i64)]
pub enum RolloutSource {
    Real = 0,
    Fantasy = 1,
}

#[derive(Debug)]
pub struct PlannerObservation {
    pub forecast_latent: Tensor,
    pub forecast_mean: Tensor,
    pub forecast_logvar: Tensor,
    pub relative_horizon: Tensor,
    pub portfolio_state: Tensor,
}

#[derive(Debug)]
pub struct PlannerTransition {
    pub observation: PlannerObservation,
    pub source: RolloutSource,
    pub environment_id: usize,
    pub decision_index: usize,
    pub action: Tensor,
    pub old_alpha: Tensor,
    pub old_beta: Tensor,
    pub old_log_prob: Tensor,
    pub value: Tensor,
    /// Bootstrap value of the successor decision. Left `None` while the
    /// transition is still pending and filled the moment the next decision's
    /// value is known; always `Some` before the transition enters a rollout.
    pub next_value: Option<Tensor>,
    pub reward: f32,
    pub terminated: bool,
    pub truncated: bool,
    pub commission: f64,
    pub turnover: f64,
    pub assets_before: f64,
    pub assets_after: f64,
    pub fantasy_clamped: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum MixedRolloutError {
    ZeroCapacity,
    SourceFull(RolloutSource),
    Unbalanced {
        real: usize,
        fantasy: usize,
        expected: usize,
    },
    InvalidMinibatchSize(usize),
    EmptyWorldModelHash,
    InconsistentForecastHorizon,
    InvalidTensorShape(&'static str),
    NonFiniteValue(&'static str),
}

impl Display for MixedRolloutError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroCapacity => write!(formatter, "rollout capacity must be positive"),
            Self::SourceFull(source) => write!(formatter, "{source:?} rollout partition is full"),
            Self::Unbalanced { real, fantasy, expected } => write!(
                formatter,
                "rollout must contain exactly {expected} samples per source, got real={real}, fantasy={fantasy}"
            ),
            Self::InvalidMinibatchSize(size) => write!(
                formatter,
                "balanced minibatch size must be positive, even, and divide the rollout; got {size}"
            ),
            Self::EmptyWorldModelHash => write!(formatter, "world-model checkpoint hash is empty"),
            Self::InconsistentForecastHorizon => {
                write!(formatter, "all transitions must use the same forecast horizon")
            }
            Self::InvalidTensorShape(name) => write!(formatter, "invalid tensor shape for {name}"),
            Self::NonFiniteValue(name) => write!(formatter, "non-finite value in {name}"),
        }
    }
}

impl Error for MixedRolloutError {}

pub struct MixedRollout {
    expected_per_source: usize,
    transitions: Vec<PlannerTransition>,
    real_count: usize,
    fantasy_count: usize,
    world_model_hash: String,
}

impl MixedRollout {
    pub fn new(
        expected_per_source: usize,
        world_model_hash: String,
    ) -> Result<Self, MixedRolloutError> {
        if expected_per_source == 0 {
            return Err(MixedRolloutError::ZeroCapacity);
        }
        if world_model_hash.trim().is_empty() {
            return Err(MixedRolloutError::EmptyWorldModelHash);
        }
        Ok(Self {
            expected_per_source,
            transitions: Vec::with_capacity(expected_per_source * 2),
            real_count: 0,
            fantasy_count: 0,
            world_model_hash,
        })
    }

    pub fn world_model_hash(&self) -> &str {
        &self.world_model_hash
    }

    pub fn push(&mut self, transition: PlannerTransition) -> Result<(), MixedRolloutError> {
        validate_transition(&transition)?;
        let count = match transition.source {
            RolloutSource::Real => &mut self.real_count,
            RolloutSource::Fantasy => &mut self.fantasy_count,
        };
        if *count == self.expected_per_source {
            return Err(MixedRolloutError::SourceFull(transition.source));
        }
        *count += 1;
        self.transitions.push(transition);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.transitions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.transitions.is_empty()
    }

    pub fn source_counts(&self) -> (usize, usize) {
        (self.real_count, self.fantasy_count)
    }

    pub fn transitions(&self) -> &[PlannerTransition] {
        &self.transitions
    }

    pub fn validate_complete(&self) -> Result<(), MixedRolloutError> {
        if self.real_count != self.expected_per_source
            || self.fantasy_count != self.expected_per_source
        {
            return Err(MixedRolloutError::Unbalanced {
                real: self.real_count,
                fantasy: self.fantasy_count,
                expected: self.expected_per_source,
            });
        }
        Ok(())
    }

    pub fn balanced_minibatch_indices(
        &self,
        minibatch_size: usize,
        seed: u64,
    ) -> Result<Vec<Vec<i64>>, MixedRolloutError> {
        self.validate_complete()?;
        if minibatch_size == 0 || minibatch_size % 2 != 0 || self.len() % minibatch_size != 0 {
            return Err(MixedRolloutError::InvalidMinibatchSize(minibatch_size));
        }
        let per_source = minibatch_size / 2;
        if self.expected_per_source % per_source != 0 {
            return Err(MixedRolloutError::InvalidMinibatchSize(minibatch_size));
        }

        let mut real = Vec::with_capacity(self.expected_per_source);
        let mut fantasy = Vec::with_capacity(self.expected_per_source);
        for (index, transition) in self.transitions.iter().enumerate() {
            match transition.source {
                RolloutSource::Real => real.push(index as i64),
                RolloutSource::Fantasy => fantasy.push(index as i64),
            }
        }
        let mut rng = StdRng::seed_from_u64(seed);
        real.shuffle(&mut rng);
        fantasy.shuffle(&mut rng);

        let mut batches = Vec::with_capacity(self.len() / minibatch_size);
        for (real_chunk, fantasy_chunk) in real.chunks(per_source).zip(fantasy.chunks(per_source)) {
            let mut batch = Vec::with_capacity(minibatch_size);
            batch.extend_from_slice(real_chunk);
            batch.extend_from_slice(fantasy_chunk);
            batch.shuffle(&mut rng);
            batches.push(batch);
        }
        Ok(batches)
    }

    pub fn to_batch(&self, device: Device) -> Result<PlannerBatch, MixedRolloutError> {
        self.validate_complete()?;
        // Every transition was validated at push time; only the cross-transition
        // horizon invariant still needs checking before stacking.
        PlannerBatch::from_validated_transitions(&self.transitions, device)
    }

    pub fn metrics(&self) -> RolloutMetrics {
        RolloutMetrics {
            real: SourceMetrics::from_transitions(&self.transitions, RolloutSource::Real),
            fantasy: SourceMetrics::from_transitions(&self.transitions, RolloutSource::Fantasy),
        }
    }
}

pub struct PlannerBatch {
    pub forecast_latent: Tensor,
    pub forecast_mean: Tensor,
    pub forecast_logvar: Tensor,
    pub relative_horizon: Tensor,
    pub portfolio_state: Tensor,
    pub actions: Tensor,
    pub old_alpha: Tensor,
    pub old_beta: Tensor,
    pub old_log_probs: Tensor,
    pub values: Tensor,
    pub next_values: Tensor,
    pub rewards: Tensor,
    pub terminated: Tensor,
    pub truncated: Tensor,
    pub sources: Tensor,
}

impl PlannerBatch {
    pub fn from_transitions(
        transitions: &[PlannerTransition],
        device: Device,
    ) -> Result<Self, MixedRolloutError> {
        if transitions.is_empty() {
            return Err(MixedRolloutError::ZeroCapacity);
        }
        for transition in transitions {
            validate_transition(transition)?;
        }
        Self::from_validated_transitions(transitions, device)
    }

    fn from_validated_transitions(
        transitions: &[PlannerTransition],
        device: Device,
    ) -> Result<Self, MixedRolloutError> {
        if transitions.is_empty() {
            return Err(MixedRolloutError::ZeroCapacity);
        }
        let horizon = transitions[0].observation.forecast_latent.size()[0];
        if transitions
            .iter()
            .any(|transition| transition.observation.forecast_latent.size()[0] != horizon)
        {
            return Err(MixedRolloutError::InconsistentForecastHorizon);
        }

        let forecast_latent = stack_to_device(
            transitions.iter().map(|t| &t.observation.forecast_latent),
            device,
        );
        let forecast_mean = stack_to_device(
            transitions.iter().map(|t| &t.observation.forecast_mean),
            device,
        );
        let forecast_logvar = stack_to_device(
            transitions.iter().map(|t| &t.observation.forecast_logvar),
            device,
        );
        let relative_horizon = stack_to_device(
            transitions.iter().map(|t| &t.observation.relative_horizon),
            device,
        );
        let portfolio_state = stack_to_device(
            transitions.iter().map(|t| &t.observation.portfolio_state),
            device,
        );
        let actions = stack_flat_to_device(transitions.iter().map(|t| &t.action), device);
        let old_alpha = stack_flat_to_device(transitions.iter().map(|t| &t.old_alpha), device);
        let old_beta = stack_flat_to_device(transitions.iter().map(|t| &t.old_beta), device);
        let old_log_probs = cat_scalars(transitions.iter().map(|t| &t.old_log_prob), device);
        let values = cat_scalars(transitions.iter().map(|t| &t.value), device);
        let next_values = cat_scalars(
            transitions.iter().map(|t| {
                t.next_value
                    .as_ref()
                    .expect("next_value filled before batching")
            }),
            device,
        );
        let rewards = Tensor::from_slice(&transitions.iter().map(|t| t.reward).collect::<Vec<_>>())
            .to_device(device);
        let terminated = bool_tensor(transitions.iter().map(|t| t.terminated), device);
        let truncated = bool_tensor(transitions.iter().map(|t| t.truncated), device);
        let sources = Tensor::from_slice(
            &transitions
                .iter()
                .map(|t| t.source as i64)
                .collect::<Vec<_>>(),
        )
        .to_device(device);

        Ok(Self {
            forecast_latent,
            forecast_mean,
            forecast_logvar,
            relative_horizon,
            portfolio_state,
            actions,
            old_alpha,
            old_beta,
            old_log_probs,
            values,
            next_values,
            rewards,
            terminated,
            truncated,
            sources,
        })
    }

    pub fn len(&self) -> i64 {
        self.rewards.size()[0]
    }

    pub fn select(&self, indices: &[i64]) -> Self {
        let index = Tensor::from_slice(indices).to_device(self.rewards.device());
        Self {
            forecast_latent: self.forecast_latent.index_select(0, &index),
            forecast_mean: self.forecast_mean.index_select(0, &index),
            forecast_logvar: self.forecast_logvar.index_select(0, &index),
            relative_horizon: self.relative_horizon.index_select(0, &index),
            portfolio_state: self.portfolio_state.index_select(0, &index),
            actions: self.actions.index_select(0, &index),
            old_alpha: self.old_alpha.index_select(0, &index),
            old_beta: self.old_beta.index_select(0, &index),
            old_log_probs: self.old_log_probs.index_select(0, &index),
            values: self.values.index_select(0, &index),
            next_values: self.next_values.index_select(0, &index),
            rewards: self.rewards.index_select(0, &index),
            terminated: self.terminated.index_select(0, &index),
            truncated: self.truncated.index_select(0, &index),
            sources: self.sources.index_select(0, &index),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SourceMetrics {
    pub samples: usize,
    pub reward_sum: f64,
    pub reward_mean: f64,
    pub commissions: f64,
    pub turnover_mean: f64,
    pub action_mean: f64,
    pub action_boundary_fraction: f64,
    pub mean_environment_wealth_ratio: f64,
    pub fantasy_clamp_fraction: f64,
}

impl SourceMetrics {
    fn from_transitions(transitions: &[PlannerTransition], source: RolloutSource) -> Self {
        let selected: Vec<_> = transitions.iter().filter(|t| t.source == source).collect();
        if selected.is_empty() {
            return Self::default();
        }

        let samples = selected.len();
        let reward_sum = selected.iter().map(|t| t.reward as f64).sum::<f64>();
        let commissions = selected.iter().map(|t| t.commission).sum::<f64>();
        let turnover_sum = selected.iter().map(|t| t.turnover).sum::<f64>();
        let fantasy_clamps = selected.iter().filter(|t| t.fantasy_clamped).count();
        let mut action_sum = 0.0;
        let mut boundary_count = 0usize;
        for transition in &selected {
            let action = transition.action.flatten(0, -1).double_value(&[0]);
            action_sum += action;
            if !(0.01..=0.99).contains(&action) {
                boundary_count += 1;
            }
        }

        let mut environments: HashMap<usize, (usize, f64, usize, f64)> = HashMap::new();
        for transition in &selected {
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
            .map(|(_, initial, _, final_assets)| {
                if *initial > 0.0 {
                    final_assets / initial
                } else {
                    1.0
                }
            })
            .sum::<f64>();

        Self {
            samples,
            reward_sum,
            reward_mean: reward_sum / samples as f64,
            commissions,
            turnover_mean: turnover_sum / samples as f64,
            action_mean: action_sum / samples as f64,
            action_boundary_fraction: boundary_count as f64 / samples as f64,
            mean_environment_wealth_ratio: wealth_ratio_sum / environments.len() as f64,
            fantasy_clamp_fraction: fantasy_clamps as f64 / samples as f64,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct RolloutMetrics {
    pub real: SourceMetrics,
    pub fantasy: SourceMetrics,
}

fn validate_transition(transition: &PlannerTransition) -> Result<(), MixedRolloutError> {
    let latent = transition.observation.forecast_latent.size();
    let mean = transition.observation.forecast_mean.size();
    let logvar = transition.observation.forecast_logvar.size();
    if latent.len() != 2 || latent[0] <= 0 || latent[1] != PLANNER_LATENT_DIM {
        return Err(MixedRolloutError::InvalidTensorShape("forecast_latent"));
    }
    if mean != [latent[0], PLANNER_OHLC_DIM] {
        return Err(MixedRolloutError::InvalidTensorShape("forecast_mean"));
    }
    if logvar != mean {
        return Err(MixedRolloutError::InvalidTensorShape("forecast_logvar"));
    }
    if transition.observation.relative_horizon.size() != [latent[0], 1] {
        return Err(MixedRolloutError::InvalidTensorShape("relative_horizon"));
    }
    if transition.observation.portfolio_state.size() != [PLANNER_PORTFOLIO_DIM] {
        return Err(MixedRolloutError::InvalidTensorShape("portfolio_state"));
    }
    let next_value = transition
        .next_value
        .as_ref()
        .ok_or(MixedRolloutError::InvalidTensorShape("next_value"))?;
    let action_count = transition.action.numel();
    if action_count != 1
        || transition.old_alpha.numel() != action_count
        || transition.old_beta.numel() != action_count
    {
        return Err(MixedRolloutError::InvalidTensorShape("Beta action fields"));
    }
    if transition.old_log_prob.numel() != 1
        || transition.value.numel() != 1
        || next_value.numel() != 1
    {
        return Err(MixedRolloutError::InvalidTensorShape(
            "scalar rollout fields",
        ));
    }
    for (name, tensor) in [
        ("forecast_latent", &transition.observation.forecast_latent),
        ("forecast_mean", &transition.observation.forecast_mean),
        ("forecast_logvar", &transition.observation.forecast_logvar),
        ("relative_horizon", &transition.observation.relative_horizon),
        ("portfolio_state", &transition.observation.portfolio_state),
        ("action", &transition.action),
        ("old_alpha", &transition.old_alpha),
        ("old_beta", &transition.old_beta),
        ("old_log_prob", &transition.old_log_prob),
        ("value", &transition.value),
        ("next_value", next_value),
    ] {
        if tensor.isfinite().all().int64_value(&[]) == 0 {
            return Err(MixedRolloutError::NonFiniteValue(name));
        }
    }
    for (name, finite) in [
        ("reward", transition.reward.is_finite()),
        ("commission", transition.commission.is_finite()),
        ("turnover", transition.turnover.is_finite()),
        ("assets_before", transition.assets_before.is_finite()),
        ("assets_after", transition.assets_after.is_finite()),
    ] {
        if !finite {
            return Err(MixedRolloutError::NonFiniteValue(name));
        }
    }
    Ok(())
}

fn stack_to_device<'a>(tensors: impl Iterator<Item = &'a Tensor>, device: Device) -> Tensor {
    let tensors = tensors.collect::<Vec<_>>();
    Tensor::stack(&tensors, 0).to_device(device)
}

fn stack_flat_to_device<'a>(tensors: impl Iterator<Item = &'a Tensor>, device: Device) -> Tensor {
    let tensors = tensors
        .map(|tensor| tensor.flatten(0, -1))
        .collect::<Vec<_>>();
    Tensor::stack(&tensors, 0).to_device(device)
}

fn cat_scalars<'a>(tensors: impl Iterator<Item = &'a Tensor>, device: Device) -> Tensor {
    let tensors = tensors
        .map(|tensor| tensor.flatten(0, -1))
        .collect::<Vec<_>>();
    Tensor::cat(&tensors, 0).to_device(device)
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

    fn transition(
        source: RolloutSource,
        environment_id: usize,
        decision_index: usize,
    ) -> PlannerTransition {
        let action = match source {
            RolloutSource::Real => 0.25,
            RolloutSource::Fantasy => 0.75,
        };
        PlannerTransition {
            observation: PlannerObservation {
                forecast_latent: Tensor::zeros([3, PLANNER_LATENT_DIM], (Kind::Float, Device::Cpu)),
                forecast_mean: Tensor::zeros([3, PLANNER_OHLC_DIM], (Kind::Float, Device::Cpu)),
                forecast_logvar: Tensor::zeros([3, PLANNER_OHLC_DIM], (Kind::Float, Device::Cpu)),
                relative_horizon: Tensor::arange(3, (Kind::Float, Device::Cpu)).view([3, 1]),
                portfolio_state: Tensor::from_slice(&[0.5f32, 0.5, 0.25, 0.0]),
            },
            source,
            environment_id,
            decision_index,
            action: Tensor::from_slice(&[action as f32]),
            old_alpha: Tensor::from_slice(&[2.0f32]),
            old_beta: Tensor::from_slice(&[2.0f32]),
            old_log_prob: Tensor::from_slice(&[0.1f32]),
            value: Tensor::from_slice(&[0.2f32]),
            next_value: Some(Tensor::from_slice(&[0.3f32])),
            reward: 0.4,
            terminated: false,
            truncated: decision_index == 1,
            commission: 0.01,
            turnover: 0.1,
            assets_before: 100.0 + decision_index as f64,
            assets_after: 101.0 + decision_index as f64,
            fantasy_clamped: false,
        }
    }

    #[test]
    fn rollout_requires_exactly_equal_source_counts() {
        let mut rollout = MixedRollout::new(2, "wm-test".to_string()).unwrap();
        rollout.push(transition(RolloutSource::Real, 0, 0)).unwrap();
        rollout
            .push(transition(RolloutSource::Fantasy, 1, 0))
            .unwrap();
        assert_eq!(
            rollout.validate_complete(),
            Err(MixedRolloutError::Unbalanced {
                real: 1,
                fantasy: 1,
                expected: 2
            })
        );
        rollout.push(transition(RolloutSource::Real, 0, 1)).unwrap();
        rollout
            .push(transition(RolloutSource::Fantasy, 1, 1))
            .unwrap();
        rollout.validate_complete().unwrap();
    }

    #[test]
    fn every_minibatch_is_source_balanced() {
        let mut rollout = MixedRollout::new(4, "wm-test".to_string()).unwrap();
        for i in 0..4 {
            rollout.push(transition(RolloutSource::Real, 0, i)).unwrap();
            rollout
                .push(transition(RolloutSource::Fantasy, 1, i))
                .unwrap();
        }
        let batches = rollout.balanced_minibatch_indices(4, 7).unwrap();
        assert_eq!(batches.len(), 2);
        for batch in batches {
            let real = batch
                .iter()
                .filter(|&&index| rollout.transitions[index as usize].source == RolloutSource::Real)
                .count();
            assert_eq!(real, 2);
        }
    }

    #[test]
    fn tensor_batch_preserves_model_contract_shapes() {
        let transitions = vec![
            transition(RolloutSource::Real, 0, 0),
            transition(RolloutSource::Fantasy, 1, 0),
        ];
        let batch = PlannerBatch::from_transitions(&transitions, Device::Cpu).unwrap();
        assert_eq!(batch.forecast_latent.size(), [2, 3, PLANNER_LATENT_DIM]);
        assert_eq!(batch.forecast_mean.size(), [2, 3, PLANNER_OHLC_DIM]);
        assert_eq!(batch.forecast_logvar.size(), [2, 3, PLANNER_OHLC_DIM]);
        assert_eq!(batch.relative_horizon.size(), [2, 3, 1]);
        assert_eq!(batch.portfolio_state.size(), [2, 4]);
        assert_eq!(batch.actions.size(), [2, 1]);
        assert_eq!(batch.values.size(), [2]);
        assert_eq!(batch.sources.size(), [2]);
    }

    #[test]
    fn rejects_observations_that_do_not_match_planner_widths() {
        let mut invalid = transition(RolloutSource::Real, 0, 0);
        invalid.observation.forecast_latent =
            Tensor::zeros([3, PLANNER_LATENT_DIM - 1], (Kind::Float, Device::Cpu));
        assert!(matches!(
            PlannerBatch::from_transitions(&[invalid], Device::Cpu),
            Err(MixedRolloutError::InvalidTensorShape("forecast_latent"))
        ));
    }

    #[test]
    fn rejects_mixed_horizons_before_stacking() {
        let first = transition(RolloutSource::Real, 0, 0);
        let mut second = transition(RolloutSource::Fantasy, 1, 0);
        second.observation.forecast_latent =
            Tensor::zeros([4, PLANNER_LATENT_DIM], (Kind::Float, Device::Cpu));
        second.observation.forecast_mean =
            Tensor::zeros([4, PLANNER_OHLC_DIM], (Kind::Float, Device::Cpu));
        second.observation.forecast_logvar =
            Tensor::zeros([4, PLANNER_OHLC_DIM], (Kind::Float, Device::Cpu));
        second.observation.relative_horizon =
            Tensor::arange(4, (Kind::Float, Device::Cpu)).view([4, 1]);
        assert!(matches!(
            PlannerBatch::from_transitions(&[first, second], Device::Cpu),
            Err(MixedRolloutError::InconsistentForecastHorizon)
        ));
    }

    #[test]
    fn rejects_non_finite_values() {
        let mut non_finite = transition(RolloutSource::Real, 0, 0);
        non_finite.reward = f32::NAN;
        assert!(matches!(
            PlannerBatch::from_transitions(&[non_finite], Device::Cpu),
            Err(MixedRolloutError::NonFiniteValue("reward"))
        ));
    }

    #[test]
    fn rollout_requires_non_empty_world_model_hash() {
        assert_eq!(
            MixedRollout::new(1, String::new()).err(),
            Some(MixedRolloutError::EmptyWorldModelHash)
        );
        MixedRollout::new(1, "wm-test".to_string()).unwrap();
    }

    #[test]
    fn metrics_remain_separate_by_source() {
        let mut real = transition(RolloutSource::Real, 0, 0);
        real.reward = 1.0;
        let mut fantasy = transition(RolloutSource::Fantasy, 1, 0);
        fantasy.reward = -2.0;
        let metrics = MixedRollout {
            expected_per_source: 1,
            transitions: vec![real, fantasy],
            real_count: 1,
            fantasy_count: 1,
            world_model_hash: "wm-test".to_string(),
        }
        .metrics();
        assert_eq!(metrics.real.reward_sum, 1.0);
        assert_eq!(metrics.fantasy.reward_sum, -2.0);
        assert_eq!(metrics.real.action_mean, 0.25);
        assert_eq!(metrics.fantasy.action_mean, 0.75);
    }
}
