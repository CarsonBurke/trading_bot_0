use super::super::single::{Env, EnvMarketSnapshot};
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::model::ModelVariant;
use tch::{Device, Tensor};

#[derive(Clone)]
pub(super) struct EnvGroupEpisode {
    pub(super) market: EnvMarketSnapshot,
    pub(super) start_offset: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct EnvGroupKey {
    pub(super) tickers: Vec<String>,
    pub(super) start_offset: usize,
}

impl EnvGroupEpisode {
    pub(super) fn key(&self) -> EnvGroupKey {
        EnvGroupKey {
            tickers: self.market.tickers.clone(),
            start_offset: self.start_offset,
        }
    }
}

pub(super) struct RingStepResult {
    pub(super) reward_per_ticker: [f32; TICKERS_COUNT as usize],
    pub(super) is_done: f32,
    pub(super) step_deltas: [f32; TICKERS_COUNT as usize],
    pub(super) static_obs: [f32; STATIC_OBSERVATIONS],
}

pub struct VecEnv {
    pub(super) nprocs: usize,
    pub envs: Vec<Env>,
    pub(super) done_mask: Vec<bool>,
    pub(super) last_static_obs: Vec<f32>,
    pub(super) last_step_deltas: Vec<f32>,
    pub(super) step_deltas_buf: Vec<f32>,
    pub(super) reward_buf: Vec<f32>,
    pub(super) reward_per_ticker_buf: Vec<f32>,
    pub(super) is_done_buf: Vec<f32>,
    pub(super) price_deltas_buf: Vec<f32>,
    pub(super) static_obs_buf: Vec<f32>,
}

impl VecEnv {
    pub(super) fn nprocs_i64(&self) -> i64 {
        self.nprocs as i64
    }

    pub(super) fn tensor_from_f32(&self, data: &[f32], size: &[i64]) -> Tensor {
        unsafe {
            Tensor::from_blob(
                data.as_ptr() as *const u8,
                size,
                &[],
                tch::Kind::Float,
                Device::Cpu,
            )
        }
    }

    pub(super) fn owned_tensor_from_f32(&self, data: &[f32], size: &[i64]) -> Tensor {
        Tensor::from_slice(data).view(size)
    }

    pub fn new(
        random_start: bool,
        _model_variant: ModelVariant,
        gens_path: String,
        nprocs: usize,
    ) -> Self {
        let env_group_count = reset_group_count_for_nprocs(nprocs);
        assert_eq!(
            nprocs % env_group_count,
            0,
            "PPO_NPROCS={} must divide evenly into {} env reset groups",
            nprocs,
            env_group_count
        );
        eprintln!(
            "env reset groups: groups={} group_size={}",
            env_group_count,
            nprocs / env_group_count
        );
        let mut envs = Vec::with_capacity(nprocs);
        envs.push(Env::new_with_recording(random_start, true, Some(gens_path)));
        eprintln!("first env");
        for i in 1..nprocs {
            envs.push(Env::new_with_recording(random_start, false, None));
            eprintln!("env {}", i);
        }
        for (i, env) in envs.iter_mut().enumerate() {
            env.env_id = i;
        }
        let price_deltas_dim = nprocs * TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER;
        let static_obs_dim = nprocs * STATIC_OBSERVATIONS;
        let done_mask = vec![false; nprocs];
        let last_static_obs = vec![0.0; nprocs * STATIC_OBSERVATIONS];
        let last_step_deltas = vec![0.0; nprocs * TICKERS_COUNT as usize];
        let step_deltas_buf = vec![0.0; nprocs * TICKERS_COUNT as usize];
        Self {
            nprocs,
            envs,
            done_mask,
            last_static_obs,
            last_step_deltas,
            step_deltas_buf,
            reward_buf: vec![0.0; nprocs],
            reward_per_ticker_buf: vec![0.0; nprocs * TICKERS_COUNT as usize],
            is_done_buf: vec![0.0; nprocs],
            price_deltas_buf: vec![0.0; price_deltas_dim],
            static_obs_buf: vec![0.0; static_obs_dim],
        }
    }

    pub(super) fn env_group_count(&self) -> usize {
        reset_group_count_for_nprocs(self.nprocs)
    }

    pub(super) fn env_group_size(&self) -> usize {
        self.nprocs / self.env_group_count()
    }

    pub(super) fn group_bounds(&self, group_idx: usize) -> (usize, usize) {
        let group_size = self.env_group_size();
        let start = group_idx * group_size;
        (start, start + group_size)
    }

    pub(super) fn current_group_episode(&self, env_idx: usize) -> EnvGroupEpisode {
        EnvGroupEpisode {
            market: self.envs[env_idx].market_snapshot(),
            start_offset: self.envs[env_idx].episode_start_offset,
        }
    }

    pub(super) fn current_group_key(&self, env_idx: usize) -> EnvGroupKey {
        EnvGroupKey {
            tickers: self.envs[env_idx].tickers.clone(),
            start_offset: self.envs[env_idx].episode_start_offset,
        }
    }

    pub(super) fn current_group_keys(&self) -> Vec<EnvGroupKey> {
        (0..self.env_group_count())
            .map(|group_idx| {
                let (group_start, _) = self.group_bounds(group_idx);
                self.current_group_key(group_start)
            })
            .collect()
    }

    pub(super) fn has_used_market_episode_key(
        used_keys: &[EnvGroupKey],
        key: &EnvGroupKey,
    ) -> bool {
        used_keys
            .iter()
            .any(|used| used.tickers == key.tickers && used.start_offset == key.start_offset)
    }

    pub fn max_step(&self) -> usize {
        let first = self.envs[0].max_step;
        for (i, env) in self.envs.iter().enumerate().skip(1) {
            assert_eq!(
                env.max_step, first,
                "VecEnv desync: env[{}].max_step={} != env[0].max_step={}",
                i, env.max_step, first
            );
        }
        first
    }

    pub fn set_episode(&mut self, episode: usize) {
        for env in &mut self.envs {
            env.episode = episode;
        }
    }

    pub fn set_step(&mut self, step: usize) {
        for env in &mut self.envs {
            env.step = step;
        }
    }

    pub fn primary(&self) -> &Env {
        &self.envs[0]
    }

    pub fn primary_mut(&mut self) -> &mut Env {
        &mut self.envs[0]
    }

    pub fn tickers(&self) -> &Vec<String> {
        &self.envs[0].tickers
    }

    pub fn prices(&self) -> &Vec<Vec<f64>> {
        &self.envs[0].prices
    }
}

fn reset_group_count_for_nprocs(nprocs: usize) -> usize {
    nprocs.max(1)
}

#[cfg(test)]
mod tests {
    use super::{reset_group_count_for_nprocs, EnvGroupKey, VecEnv};

    #[test]
    fn reset_groups_match_positive_nprocs() {
        for nprocs in [1, 2, 8, 16, 31] {
            assert_eq!(reset_group_count_for_nprocs(nprocs), nprocs);
        }
    }

    #[test]
    fn market_episode_keys_reject_active_duplicate() {
        let used_keys = vec![EnvGroupKey {
            tickers: vec!["AAPL".to_string(), "MSFT".to_string()],
            start_offset: 128,
        }];

        assert!(VecEnv::has_used_market_episode_key(
            &used_keys,
            &EnvGroupKey {
                tickers: vec!["AAPL".to_string(), "MSFT".to_string()],
                start_offset: 128,
            }
        ));
        assert!(!VecEnv::has_used_market_episode_key(
            &used_keys,
            &EnvGroupKey {
                tickers: vec!["AAPL".to_string(), "MSFT".to_string()],
                start_offset: 256,
            }
        ));
    }
}
