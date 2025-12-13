use tch::Tensor;
use super::env::{Env, Step};
use crate::torch::ppo::NPROCS;
use crate::torch::constants::{TICKERS_COUNT, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS};

pub struct VecEnv {
    pub envs: Vec<Env>,
}

impl VecEnv {
    pub fn new(random_start: bool) -> Self {
        let envs = (0..NPROCS).map(|_| Env::new(random_start)).collect();
        Self { envs }
    }

    pub fn reset(&mut self) -> (Tensor, Tensor) {
        let mut all_price_deltas = Vec::new();
        let mut all_static_obs = Vec::new();

        for env in &mut self.envs {
            let (pd, so) = env.reset_single();
            all_price_deltas.extend(pd);
            all_static_obs.extend(so);
        }

        let price_deltas = Tensor::from_slice(&all_price_deltas)
            .view([NPROCS, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs = Tensor::from_slice(&all_static_obs)
            .view([NPROCS, STATIC_OBSERVATIONS as i64]);

        (price_deltas, static_obs)
    }

    pub fn step(&mut self, all_actions: Vec<Vec<f64>>) -> Step {
        let mut rewards = Vec::with_capacity(NPROCS as usize);
        let mut is_dones = Vec::with_capacity(NPROCS as usize);
        let mut all_price_deltas = Vec::new();
        let mut all_static_obs = Vec::new();

        for (env, actions) in self.envs.iter_mut().zip(all_actions.iter()) {
            let step = env.step_single(actions.clone());
            rewards.push(step.reward);
            is_dones.push(step.is_done);
            all_price_deltas.extend(step.price_deltas);
            all_static_obs.extend(step.static_obs);
        }

        Step {
            reward: Tensor::from_slice(&rewards),
            is_done: Tensor::from_slice(&is_dones),
            price_deltas: Tensor::from_slice(&all_price_deltas)
                .view([NPROCS, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]),
            static_obs: Tensor::from_slice(&all_static_obs)
                .view([NPROCS, STATIC_OBSERVATIONS as i64]),
        }
    }

    pub fn max_step(&self) -> usize {
        self.envs[0].max_step
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

    pub fn apply_retroactive_rewards(&mut self, rewards: &Tensor) {
        for (i, env) in self.envs.iter_mut().enumerate() {
            env.apply_retroactive_rewards(&rewards.select(1, i as i64));
        }
    }
}
