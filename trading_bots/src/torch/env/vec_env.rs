use super::env::Env;
use crate::torch::constants::{
    ACTION_COUNT, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT,
};
use crate::torch::model::ModelVariant;
use rand::seq::SliceRandom;
use rayon::prelude::*;
use tch::{Device, Tensor};

pub struct CpuStepBatch {
    pub actions_f32: Vec<f32>,
    actions_f64: Vec<f64>,
    pub reset_indices: Vec<usize>,
    pub reset_price_deltas: Vec<f32>,
}

impl CpuStepBatch {
    pub fn new(nprocs: usize, action_dim: usize, pd_dim: usize) -> Self {
        Self {
            actions_f32: vec![0.0; nprocs * action_dim],
            actions_f64: vec![0.0; nprocs * action_dim],
            reset_indices: Vec::with_capacity(nprocs),
            reset_price_deltas: Vec::with_capacity(nprocs * pd_dim),
        }
    }
}

pub struct VecEnv {
    nprocs: usize,
    pub envs: Vec<Env>,
    done_mask: Vec<bool>,
    last_static_obs: Vec<f32>,
    last_step_deltas: Vec<f32>,
    step_deltas_buf: Vec<f32>,
    reward_buf: Vec<f32>,
    reward_per_ticker_buf: Vec<f32>,
    is_done_buf: Vec<f32>,
    price_deltas_buf: Vec<f32>,
    static_obs_buf: Vec<f32>,
}

/// Available tickers for random selection
const AVAILABLE_TICKERS: &[&str] = &["TSLA", "AAPL", "MSFT", "NVDA", "INTC", "AMD"];

impl VecEnv {
    fn nprocs_i64(&self) -> i64 {
        self.nprocs as i64
    }

    fn tensor_from_f32(&self, data: &[f32], size: &[i64]) -> Tensor {
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

    fn owned_tensor_from_f32(&self, data: &[f32], size: &[i64]) -> Tensor {
        Tensor::from_slice(data).view(size)
    }

    pub fn new(
        random_start: bool,
        _model_variant: ModelVariant,
        gens_path: String,
        nprocs: usize,
    ) -> Self {
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

    pub fn reset(&mut self) -> (Tensor, Tensor) {
        assert_eq!(
            self.envs.len(),
            self.nprocs,
            "VecEnv desync: envs.len={} != nprocs={}",
            self.envs.len(),
            self.nprocs
        );
        let mut all_price_deltas = Vec::new();
        let mut all_static_obs = Vec::new();

        for env in &mut self.envs {
            let (pd, so) = env.reset_single();
            all_price_deltas.extend(pd);
            all_static_obs.extend(so);
        }

        self.price_deltas_buf.copy_from_slice(&all_price_deltas);
        self.static_obs_buf.copy_from_slice(&all_static_obs);

        let price_deltas = Tensor::from_slice(&all_price_deltas).view([
            self.nprocs_i64(),
            TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
        ]);
        let static_obs = Tensor::from_slice(&all_static_obs)
            .view([self.nprocs_i64(), STATIC_OBSERVATIONS as i64]);

        self.done_mask.fill(false);
        self.last_static_obs.clone_from(&all_static_obs);

        (price_deltas, static_obs)
    }

    pub fn reset_step(&mut self) -> (Tensor, Tensor) {
        assert_eq!(
            self.envs.len(),
            self.nprocs,
            "VecEnv desync: envs.len={} != nprocs={}",
            self.envs.len(),
            self.nprocs
        );
        let mut all_step_deltas = Vec::new();
        let mut all_static_obs = Vec::new();

        for env in &mut self.envs {
            let (d, so) = env.reset_step_single();
            all_step_deltas.extend(d);
            all_static_obs.extend(so);
        }

        let step_deltas =
            Tensor::from_slice(&all_step_deltas).view([self.nprocs_i64(), TICKERS_COUNT]);
        let static_obs = Tensor::from_slice(&all_static_obs)
            .view([self.nprocs_i64(), STATIC_OBSERVATIONS as i64]);

        self.done_mask.fill(false);
        self.last_static_obs.clone_from(&all_static_obs);
        self.last_step_deltas.fill(0.0);

        (step_deltas, static_obs)
    }

    pub fn reset_incremental(&mut self) -> (Vec<Vec<f32>>, Tensor) {
        assert_eq!(self.envs.len(), self.nprocs);
        let mut all_deltas_per_env: Vec<Vec<f32>> = Vec::with_capacity(self.nprocs);
        let mut all_static_obs = Vec::new();

        for env in &mut self.envs {
            let (pd, so) = env.reset_single();
            all_deltas_per_env.push(pd);
            all_static_obs.extend(so);
        }

        let static_obs = Tensor::from_slice(&all_static_obs)
            .view([self.nprocs_i64(), STATIC_OBSERVATIONS as i64]);

        let mut deltas_flat = Vec::with_capacity(self.nprocs * TICKERS_COUNT as usize);
        for deltas in &all_deltas_per_env {
            deltas_flat.extend_from_slice(deltas);
        }
        self.done_mask.fill(false);
        self.last_static_obs.clone_from(&all_static_obs);
        self.last_step_deltas.clone_from(&deltas_flat);

        (all_deltas_per_env, static_obs)
    }

    /// Step all envs and write results directly into pre-allocated GPU tensors.
    /// Uses a single batched CPU→GPU copy instead of per-env copies.
    pub fn step_into(
        &mut self,
        all_actions: &[Vec<f64>],
        out_price_deltas: &mut Tensor, // [nprocs, price_deltas_dim] on device
        out_static_obs: &mut Tensor,   // [nprocs, static_obs_dim] on device
    ) -> (Tensor, Tensor) {
        debug_assert_eq!(all_actions.len(), self.envs.len());

        let price_deltas_dim = (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64) as usize;
        let static_obs_dim = STATIC_OBSERVATIONS;

        for (i, env) in self.envs.iter_mut().enumerate() {
            let step = env.step_single(&all_actions[i]);
            self.reward_buf[i] = step.reward as f32;
            self.is_done_buf[i] = step.is_done;
            let pd_offset = i * price_deltas_dim;
            let so_offset = i * static_obs_dim;
            if step.is_done == 1.0 {
                let (price_deltas, static_obs) = env.reset_single();
                self.price_deltas_buf[pd_offset..pd_offset + price_deltas_dim]
                    .copy_from_slice(&price_deltas);
                self.static_obs_buf[so_offset..so_offset + static_obs_dim]
                    .copy_from_slice(&static_obs);
            } else {
                self.price_deltas_buf[pd_offset..pd_offset + price_deltas_dim]
                    .copy_from_slice(&step.price_deltas);
                self.static_obs_buf[so_offset..so_offset + static_obs_dim]
                    .copy_from_slice(&step.static_obs);
            }
        }

        // Single batched copy to GPU
        let pd_cpu = self.tensor_from_f32(
            &self.price_deltas_buf,
            &[
                self.nprocs_i64(),
                TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
            ],
        );
        out_price_deltas.copy_(&pd_cpu);

        let so_cpu = self.tensor_from_f32(
            &self.static_obs_buf,
            &[self.nprocs_i64(), STATIC_OBSERVATIONS as i64],
        );
        out_static_obs.copy_(&so_cpu);

        // Return small tensors - these go to GPU via arithmetic ops later
        let reward = self.tensor_from_f32(&self.reward_buf, &[self.nprocs_i64()]);
        let is_done = self.tensor_from_f32(&self.is_done_buf, &[self.nprocs_i64()]);

        (reward, is_done)
    }

    pub fn step_into_full(
        &mut self,
        all_actions: &[Vec<f64>],
        out_price_deltas: &mut Tensor,
        out_static_obs: &mut Tensor,
        out_reward_per_ticker: &mut Tensor,
        out_is_done: &mut Tensor,
    ) {
        debug_assert_eq!(all_actions.len(), self.envs.len());

        let price_deltas_dim = (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64) as usize;
        let static_obs_dim = STATIC_OBSERVATIONS;

        for (i, env) in self.envs.iter_mut().enumerate() {
            let step = env.step_single(&all_actions[i]);
            let reward_start = i * TICKERS_COUNT as usize;
            self.reward_per_ticker_buf[reward_start..reward_start + TICKERS_COUNT as usize]
                .copy_from_slice(&step.reward_per_ticker);
            self.is_done_buf[i] = step.is_done;
            let pd_offset = i * price_deltas_dim;
            let so_offset = i * static_obs_dim;
            if step.is_done == 1.0 {
                let (price_deltas, static_obs) = env.reset_single();
                self.price_deltas_buf[pd_offset..pd_offset + price_deltas_dim]
                    .copy_from_slice(&price_deltas);
                self.static_obs_buf[so_offset..so_offset + static_obs_dim]
                    .copy_from_slice(&static_obs);
            } else {
                self.price_deltas_buf[pd_offset..pd_offset + price_deltas_dim]
                    .copy_from_slice(&step.price_deltas);
                self.static_obs_buf[so_offset..so_offset + static_obs_dim]
                    .copy_from_slice(&step.static_obs);
            }
        }

        let pd_cpu = self.tensor_from_f32(
            &self.price_deltas_buf,
            &[
                self.nprocs_i64(),
                TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
            ],
        );
        out_price_deltas.copy_(&pd_cpu);

        let so_cpu = self.tensor_from_f32(
            &self.static_obs_buf,
            &[self.nprocs_i64(), STATIC_OBSERVATIONS as i64],
        );
        out_static_obs.copy_(&so_cpu);

        let rpt_cpu = self.tensor_from_f32(
            &self.reward_per_ticker_buf,
            &[self.nprocs_i64(), TICKERS_COUNT],
        );
        out_reward_per_ticker.copy_(&rpt_cpu);
        out_is_done.copy_(&self.tensor_from_f32(&self.is_done_buf, &[self.nprocs_i64()]));
    }

    pub fn step_into_step(
        &mut self,
        all_actions: &[Vec<f64>],
        out_step_deltas: &mut Tensor, // [nprocs, TICKERS_COUNT] on device
        out_static_obs: &mut Tensor,  // [nprocs, static_obs_dim] on device
    ) -> (Tensor, Tensor) {
        debug_assert_eq!(all_actions.len(), self.envs.len());

        let static_obs_dim = STATIC_OBSERVATIONS;

        for (i, env) in self.envs.iter_mut().enumerate() {
            let step = env.step_step_single(&all_actions[i]);
            self.reward_buf[i] = step.reward as f32;
            self.is_done_buf[i] = step.is_done;
            let step_offset = i * TICKERS_COUNT as usize;
            self.step_deltas_buf[step_offset..step_offset + TICKERS_COUNT as usize]
                .copy_from_slice(&step.step_deltas);
            let static_offset = i * static_obs_dim;
            self.static_obs_buf[static_offset..static_offset + static_obs_dim]
                .copy_from_slice(&step.static_obs);
        }

        let deltas_cpu =
            self.owned_tensor_from_f32(&self.step_deltas_buf, &[self.nprocs_i64(), TICKERS_COUNT]);
        out_step_deltas.copy_(&deltas_cpu);

        let so_cpu = self.owned_tensor_from_f32(
            &self.static_obs_buf,
            &[self.nprocs_i64(), STATIC_OBSERVATIONS as i64],
        );
        out_static_obs.copy_(&so_cpu);

        let reward = self.owned_tensor_from_f32(&self.reward_buf, &[self.nprocs_i64()]);
        let is_done = self.owned_tensor_from_f32(&self.is_done_buf, &[self.nprocs_i64()]);

        (reward, is_done)
    }

    pub fn step_incremental(
        &mut self,
        actions_flat: &[f64],
        out_static_obs: &mut Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let action_dim = ACTION_COUNT as usize;
        let envs_len = self.envs.len();
        debug_assert_eq!(actions_flat.len(), envs_len * action_dim);

        let static_obs_dim = STATIC_OBSERVATIONS;

        for (i, env) in self.envs.iter_mut().enumerate() {
            let action_start = i * action_dim;
            let action_slice = &actions_flat[action_start..action_start + action_dim];
            let step_start = i * TICKERS_COUNT as usize;
            let static_start = i * static_obs_dim;

            if self.done_mask[i] {
                self.reward_buf[i] = 0.0;
                self.is_done_buf[i] = 1.0;
                self.reward_per_ticker_buf[step_start..step_start + TICKERS_COUNT as usize]
                    .fill(0.0);
                self.step_deltas_buf[step_start..step_start + TICKERS_COUNT as usize]
                    .copy_from_slice(
                        &self.last_step_deltas[step_start..step_start + TICKERS_COUNT as usize],
                    );
                self.static_obs_buf[static_start..static_start + static_obs_dim].copy_from_slice(
                    &self.last_static_obs[static_start..static_start + static_obs_dim],
                );
                continue;
            }

            let step = env.step_step_single(action_slice);
            self.reward_buf[i] = step.reward as f32;
            let reward_start = i * TICKERS_COUNT as usize;
            self.reward_per_ticker_buf[reward_start..reward_start + TICKERS_COUNT as usize]
                .copy_from_slice(&step.reward_per_ticker);
            self.is_done_buf[i] = step.is_done;
            self.step_deltas_buf[step_start..step_start + TICKERS_COUNT as usize]
                .copy_from_slice(&step.step_deltas);
            self.static_obs_buf[static_start..static_start + static_obs_dim]
                .copy_from_slice(&step.static_obs);

            self.last_step_deltas[step_start..step_start + TICKERS_COUNT as usize]
                .copy_from_slice(&step.step_deltas);
            self.last_static_obs[static_start..static_start + static_obs_dim]
                .copy_from_slice(&step.static_obs);
            if step.is_done == 1.0 {
                self.done_mask[i] = true;
            }
        }

        let so_cpu = self.owned_tensor_from_f32(
            &self.static_obs_buf,
            &[self.nprocs_i64(), STATIC_OBSERVATIONS as i64],
        );
        out_static_obs.copy_(&so_cpu);

        let reward = self.owned_tensor_from_f32(&self.reward_buf, &[self.nprocs_i64()]);
        let reward_per_ticker = self.owned_tensor_from_f32(
            &self.reward_per_ticker_buf,
            &[self.nprocs_i64(), TICKERS_COUNT],
        );
        let is_done = self.owned_tensor_from_f32(&self.is_done_buf, &[self.nprocs_i64()]);
        let step_deltas =
            self.owned_tensor_from_f32(&self.step_deltas_buf, &[self.nprocs_i64(), TICKERS_COUNT]);

        (reward, reward_per_ticker, is_done, step_deltas)
    }

    pub fn step_incremental_tensor_into(
        &mut self,
        actions: &Tensor,
        out_static_obs: &mut Tensor,
        out_reward: &mut Tensor,
        out_reward_per_ticker: &mut Tensor,
        out_is_done: &mut Tensor,
        out_step_deltas: &mut Tensor,
    ) {
        let actions_cpu = actions.to_device(Device::Cpu);
        let actions_flat = Vec::<f64>::try_from(actions_cpu.flatten(0, -1)).unwrap();
        let (reward, reward_per_ticker, is_done, step_deltas) =
            self.step_incremental(&actions_flat, out_static_obs);
        out_reward.copy_(&reward);
        out_reward_per_ticker.copy_(&reward_per_ticker);
        out_is_done.copy_(&is_done);
        out_step_deltas.copy_(&step_deltas);
    }

    pub fn step_incremental_tensor(
        &mut self,
        actions: &Tensor,
        out_static_obs: &mut Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor) {
        let actions_cpu = actions.to_device(Device::Cpu);
        let actions_flat = Vec::<f64>::try_from(actions_cpu.flatten(0, -1)).unwrap();
        self.step_incremental(&actions_flat, out_static_obs)
    }

    fn step_into_ring_cpu(&mut self, all_actions: &[f64]) -> (Vec<usize>, Vec<f32>) {
        let action_dim = ACTION_COUNT as usize;
        debug_assert_eq!(all_actions.len(), self.envs.len() * action_dim);

        let static_obs_dim = STATIC_OBSERVATIONS;
        let step_deltas_dim = TICKERS_COUNT as usize;
        let price_deltas_dim = TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER;

        let mut reset_indices = Vec::new();
        let mut reset_price_deltas = Vec::new();
        let step_results = self
            .envs
            .par_iter_mut()
            .enumerate()
            .map(|(i, env)| {
                let action_start = i * action_dim;
                let action_slice = &all_actions[action_start..action_start + action_dim];
                let step = env.step_step_single(action_slice);
                if step.is_done == 1.0 {
                    let (price_deltas, static_obs) = env.reset_single();
                    (
                        i,
                        step.reward_per_ticker,
                        step.is_done,
                        None,
                        static_obs.try_into().unwrap(),
                        Some(price_deltas),
                    )
                } else {
                    (
                        i,
                        step.reward_per_ticker,
                        step.is_done,
                        Some(step.step_deltas),
                        step.static_obs,
                        None,
                    )
                }
            })
            .collect::<Vec<_>>();

        let tail_base = PRICE_DELTAS_PER_TICKER - 1;
        for (i, reward_per_ticker, is_done, step_deltas, static_obs, reset_price) in step_results {
            let reward_start = i * TICKERS_COUNT as usize;
            self.reward_per_ticker_buf[reward_start..reward_start + TICKERS_COUNT as usize]
                .copy_from_slice(&reward_per_ticker);
            self.is_done_buf[i] = is_done;

            let step_offset = i * step_deltas_dim;
            let so_offset = i * static_obs_dim;
            if let Some(price_deltas) = reset_price {
                for t in 0..TICKERS_COUNT as usize {
                    let idx = t * PRICE_DELTAS_PER_TICKER + tail_base;
                    self.step_deltas_buf[step_offset + t] = price_deltas[idx];
                }
                self.static_obs_buf[so_offset..so_offset + static_obs_dim]
                    .copy_from_slice(&static_obs);
                reset_indices.push(i);
                reset_price_deltas.extend(price_deltas);
            } else if let Some(step_deltas) = step_deltas {
                self.step_deltas_buf[step_offset..step_offset + step_deltas_dim]
                    .copy_from_slice(&step_deltas);
                self.static_obs_buf[so_offset..so_offset + static_obs_dim]
                    .copy_from_slice(&static_obs);
            }
        }

        (reset_indices, reset_price_deltas)
    }

    pub fn step_into_ring_flat(
        &mut self,
        all_actions: &[f64],
        out_step_deltas: &mut Tensor,
        out_static_obs: &mut Tensor,
        out_reward_per_ticker: &mut Tensor,
        out_is_done: &mut Tensor,
    ) -> (Vec<usize>, Vec<f32>) {
        let (reset_indices, reset_price_deltas) = self.step_into_ring_cpu(all_actions);

        let deltas_cpu =
            self.tensor_from_f32(&self.step_deltas_buf, &[self.nprocs_i64(), TICKERS_COUNT]);
        out_step_deltas.copy_(&deltas_cpu);

        let so_cpu = self.tensor_from_f32(
            &self.static_obs_buf,
            &[self.nprocs_i64(), STATIC_OBSERVATIONS as i64],
        );
        out_static_obs.copy_(&so_cpu);

        let rpt_cpu = self.tensor_from_f32(
            &self.reward_per_ticker_buf,
            &[self.nprocs_i64(), TICKERS_COUNT],
        );
        out_reward_per_ticker.copy_(&rpt_cpu);
        out_is_done.copy_(&self.tensor_from_f32(&self.is_done_buf, &[self.nprocs_i64()]));

        (reset_indices, reset_price_deltas)
    }

    pub fn step_from_actions_f32_into(
        &mut self,
        batch: &mut CpuStepBatch,
        out_step_deltas: &mut Tensor,
        out_static_obs: &mut Tensor,
        out_reward_per_ticker: &mut Tensor,
        out_is_done: &mut Tensor,
    ) {
        debug_assert_eq!(
            batch.actions_f32.len(),
            self.envs.len() * ACTION_COUNT as usize
        );
        for (dst, src) in batch.actions_f64.iter_mut().zip(batch.actions_f32.iter()) {
            *dst = *src as f64;
        }
        batch.reset_indices.clear();
        batch.reset_price_deltas.clear();
        let (reset_indices, reset_price_deltas) = self.step_into_ring_flat(
            &batch.actions_f64,
            out_step_deltas,
            out_static_obs,
            out_reward_per_ticker,
            out_is_done,
        );
        batch.reset_indices = reset_indices;
        batch.reset_price_deltas = reset_price_deltas;
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
