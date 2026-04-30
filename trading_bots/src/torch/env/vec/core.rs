use super::super::single::{Env, EnvMarketSnapshot};
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use crate::torch::model::ModelVariant;
use tch::{Device, Tensor};

pub(super) const ENV_RESET_GROUPS: usize = 2;

#[derive(Clone)]
pub(super) struct EnvGroupEpisode {
    pub(super) market: EnvMarketSnapshot,
    pub(super) start_offset: usize,
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
        let env_group_count = ENV_RESET_GROUPS.min(nprocs).max(1);
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
        ENV_RESET_GROUPS.min(self.nprocs).max(1)
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

    pub(super) fn has_used_market_episode(
        used_specs: &[EnvGroupEpisode],
        spec: &EnvGroupEpisode,
    ) -> bool {
        used_specs.iter().any(|used| {
            used.market.tickers == spec.market.tickers && used.start_offset == spec.start_offset
        })
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
