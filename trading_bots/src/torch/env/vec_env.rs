use tch::{Device, Tensor};
use rand::seq::SliceRandom;
use super::env::{Env, Step};
use crate::torch::ppo::NPROCS;
use crate::torch::constants::{ACTION_COUNT, TICKERS_COUNT, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS};

pub struct VecEnv {
    pub envs: Vec<Env>,
}

/// Available tickers for random selection
const AVAILABLE_TICKERS: &[&str] = &["TSLA", "AAPL", "MSFT", "NVDA", "INTC", "AMD"];

impl VecEnv {
    pub fn new(random_start: bool) -> Self {
        // Select random tickers ONCE, share across all envs
        let count = TICKERS_COUNT as usize;
        let mut available: Vec<String> = AVAILABLE_TICKERS.iter().map(|s| s.to_string()).collect();
        available.shuffle(&mut rand::rng());
        let tickers: Vec<String> = available.into_iter().take(count).collect();
        eprintln!("VecEnv using tickers: {:?}", tickers);

        let mut envs = Vec::with_capacity(NPROCS as usize);
        envs.push(Env::new_with_tickers_and_recording(tickers.clone(), random_start, true));
        eprintln!("first env");
        for i in 1..(NPROCS as usize) {
            envs.push(Env::new_with_tickers_and_recording(tickers.clone(), random_start, false));
            eprintln!("env {}", i);
        }
        for (i, env) in envs.iter_mut().enumerate() {
            env.env_id = i;
        }
        Self { envs }
    }

    pub fn reset(&mut self) -> (Tensor, Tensor) {
        assert_eq!(
            self.envs.len(),
            NPROCS as usize,
            "VecEnv desync: envs.len={} != NPROCS={}",
            self.envs.len(),
            NPROCS
        );
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

    pub fn reset_step(&mut self) -> (Tensor, Tensor) {
        assert_eq!(
            self.envs.len(),
            NPROCS as usize,
            "VecEnv desync: envs.len={} != NPROCS={}",
            self.envs.len(),
            NPROCS
        );
        let mut all_step_deltas = Vec::new();
        let mut all_static_obs = Vec::new();

        for env in &mut self.envs {
            let (d, so) = env.reset_step_single();
            all_step_deltas.extend(d);
            all_static_obs.extend(so);
        }

        let step_deltas = Tensor::from_slice(&all_step_deltas).view([NPROCS, TICKERS_COUNT]);
        let static_obs = Tensor::from_slice(&all_static_obs)
            .view([NPROCS, STATIC_OBSERVATIONS as i64]);

        (step_deltas, static_obs)
    }

    pub fn reset_incremental(&mut self) -> (Vec<Vec<f32>>, Tensor) {
        assert_eq!(self.envs.len(), NPROCS as usize);
        let mut all_deltas_per_env: Vec<Vec<f32>> = Vec::with_capacity(NPROCS as usize);
        let mut all_static_obs = Vec::new();

        for env in &mut self.envs {
            let (pd, so) = env.reset_single();
            all_deltas_per_env.push(pd);
            all_static_obs.extend(so);
        }

        let static_obs = Tensor::from_slice(&all_static_obs)
            .view([NPROCS, STATIC_OBSERVATIONS as i64]);

        (all_deltas_per_env, static_obs)
    }

    pub fn step(&mut self, all_actions: Vec<Vec<f64>>) -> Step {
        assert_eq!(
            all_actions.len(),
            self.envs.len(),
            "VecEnv: actions.len={} != envs.len={}",
            all_actions.len(),
            self.envs.len()
        );
        let mut rewards = Vec::with_capacity(NPROCS as usize);
        let mut rewards_per_ticker = Vec::with_capacity(NPROCS as usize * TICKERS_COUNT as usize);
        let mut is_dones = Vec::with_capacity(NPROCS as usize);
        let mut all_price_deltas = Vec::new();
        let mut all_static_obs = Vec::new();

        for (i, env) in self.envs.iter_mut().enumerate() {
            let step = env.step_single(all_actions[i].clone());
            rewards.push(step.reward);
            rewards_per_ticker.extend(step.reward_per_ticker);
            is_dones.push(step.is_done);
            all_price_deltas.extend(step.price_deltas);
            all_static_obs.extend(step.static_obs);
        }

        Step {
            reward: Tensor::from_slice(&rewards),
            reward_per_ticker: Tensor::from_slice(&rewards_per_ticker)
                .view([NPROCS, TICKERS_COUNT]),
            is_done: Tensor::from_slice(&is_dones),
            price_deltas: Tensor::from_slice(&all_price_deltas)
                .view([NPROCS, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]),
            static_obs: Tensor::from_slice(&all_static_obs)
                .view([NPROCS, STATIC_OBSERVATIONS as i64]),
        }
    }

    /// Step all envs and write results directly into pre-allocated GPU tensors.
    /// Uses a single batched CPU→GPU copy instead of per-env copies.
    pub fn step_into(
        &mut self,
        all_actions: &[Vec<f64>],
        out_price_deltas: &mut Tensor, // [NPROCS, price_deltas_dim] on device
        out_static_obs: &mut Tensor,   // [NPROCS, static_obs_dim] on device
    ) -> (Tensor, Tensor) {
        debug_assert_eq!(all_actions.len(), self.envs.len());

        let price_deltas_dim = (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64) as usize;
        let static_obs_dim = STATIC_OBSERVATIONS;

        let mut rewards = [0f32; NPROCS as usize];
        let mut is_dones = [0f32; NPROCS as usize];
        let mut all_price_deltas = Vec::with_capacity(NPROCS as usize * price_deltas_dim);
        let mut all_static_obs = Vec::with_capacity(NPROCS as usize * static_obs_dim);

        for (i, env) in self.envs.iter_mut().enumerate() {
            let step = env.step_single(all_actions[i].clone());
            rewards[i] = step.reward as f32;
            is_dones[i] = step.is_done;
            all_price_deltas.extend(step.price_deltas);
            all_static_obs.extend(step.static_obs);
        }

        // Single batched copy to GPU
        let pd_cpu = Tensor::from_slice(&all_price_deltas)
            .view([NPROCS, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        out_price_deltas.copy_(&pd_cpu);

        let so_cpu = Tensor::from_slice(&all_static_obs)
            .view([NPROCS, STATIC_OBSERVATIONS as i64]);
        out_static_obs.copy_(&so_cpu);

        // Return small tensors - these go to GPU via arithmetic ops later
        let reward = Tensor::from_slice(&rewards);
        let is_done = Tensor::from_slice(&is_dones);

        (reward, is_done)
    }

    pub fn step_into_step(
        &mut self,
        all_actions: &[Vec<f64>],
        out_step_deltas: &mut Tensor, // [NPROCS, TICKERS_COUNT] on device
        out_static_obs: &mut Tensor,  // [NPROCS, static_obs_dim] on device
    ) -> (Tensor, Tensor) {
        debug_assert_eq!(all_actions.len(), self.envs.len());

        let static_obs_dim = STATIC_OBSERVATIONS;

        let mut rewards = [0f32; NPROCS as usize];
        let mut is_dones = [0f32; NPROCS as usize];
        let mut all_step_deltas = Vec::with_capacity(NPROCS as usize * TICKERS_COUNT as usize);
        let mut all_static_obs = Vec::with_capacity(NPROCS as usize * static_obs_dim);

        for (i, env) in self.envs.iter_mut().enumerate() {
            let step = env.step_step_single(all_actions[i].clone());
            rewards[i] = step.reward as f32;
            is_dones[i] = step.is_done;
            all_step_deltas.extend(step.step_deltas);
            all_static_obs.extend(step.static_obs);
        }

        let deltas_cpu = Tensor::from_slice(&all_step_deltas).view([NPROCS, TICKERS_COUNT]);
        out_step_deltas.copy_(&deltas_cpu);

        let so_cpu = Tensor::from_slice(&all_static_obs)
            .view([NPROCS, STATIC_OBSERVATIONS as i64]);
        out_static_obs.copy_(&so_cpu);

        let reward = Tensor::from_slice(&rewards);
        let is_done = Tensor::from_slice(&is_dones);

        (reward, is_done)
    }

    pub fn step_incremental(
        &mut self,
        all_actions: &[Vec<f64>],
        out_static_obs: &mut Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        debug_assert_eq!(all_actions.len(), self.envs.len());

        let mut rewards = [0f32; NPROCS as usize];
        let mut rewards_per_ticker = Vec::with_capacity(NPROCS as usize * TICKERS_COUNT as usize);
        let mut is_dones = [0f32; NPROCS as usize];
        let mut all_step_deltas = Vec::with_capacity(NPROCS as usize * TICKERS_COUNT as usize);
        let mut all_static_obs = Vec::with_capacity(NPROCS as usize * STATIC_OBSERVATIONS);

        for (i, env) in self.envs.iter_mut().enumerate() {
            let step = env.step_step_single(all_actions[i].clone());
            rewards[i] = step.reward as f32;
            rewards_per_ticker.extend(step.reward_per_ticker);
            is_dones[i] = step.is_done;
            all_step_deltas.extend(step.step_deltas);
            all_static_obs.extend(step.static_obs);
        }

        let so_cpu = Tensor::from_slice(&all_static_obs)
            .view([NPROCS, STATIC_OBSERVATIONS as i64]);
        out_static_obs.copy_(&so_cpu);

        let reward = Tensor::from_slice(&rewards);
        let reward_per_ticker =
            Tensor::from_slice(&rewards_per_ticker).view([NPROCS, TICKERS_COUNT]);
        let is_done = Tensor::from_slice(&is_dones);
        let step_deltas = Tensor::from_slice(&all_step_deltas).view([NPROCS, TICKERS_COUNT]);

        (reward, reward_per_ticker, is_done, step_deltas)
    }

    pub fn step_incremental_tensor(
        &mut self,
        actions: &Tensor,
        out_static_obs: &mut Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let actions_cpu = actions.to_device(Device::Cpu);
        let actions_flat = Vec::<f64>::try_from(actions_cpu.flatten(0, -1)).unwrap();
        let all_actions: Vec<Vec<f64>> = actions_flat
            .chunks(ACTION_COUNT as usize)
            .map(|chunk| chunk.to_vec())
            .collect();
        self.step_incremental(&all_actions, out_static_obs)
    }

    pub fn max_step(&self) -> usize {
        let first = self.envs[0].max_step;
        for (i, env) in self.envs.iter().enumerate().skip(1) {
            assert_eq!(env.max_step, first, "VecEnv desync: env[{}].max_step={} != env[0].max_step={}", i, env.max_step, first);
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
