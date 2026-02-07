use super::env::Env;
use crate::torch::constants::{
    ACTION_COUNT, PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT,
};
use crate::torch::model::{patch_ends_for_variant, patch_seq_len_for_variant, ModelVariant};
use crate::torch::ppo::NPROCS;
use rand::seq::SliceRandom;
use tch::{Device, Tensor};

pub struct VecEnv {
    pub envs: Vec<Env>,
    done_mask: Vec<bool>,
    last_static_obs: Vec<f32>,
    last_step_deltas: Vec<f32>,
    step_deltas_buf: Vec<f32>,
    actions_buf: Vec<f64>,
    reward_buf: Vec<f32>,
    reward_per_ticker_buf: Vec<f32>,
    cash_reward_buf: Vec<f32>,
    is_done_buf: Vec<f32>,
    price_deltas_buf: Vec<f32>,
    static_obs_buf: Vec<f32>,
    seq_idx_buf: Vec<i64>,
    patch_ends: Vec<i64>,
    patch_seq_len: i64,
}

/// Available tickers for random selection
const AVAILABLE_TICKERS: &[&str] = &["TSLA", "AAPL", "MSFT", "NVDA", "INTC", "AMD"];

impl VecEnv {
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

    fn tensor_from_i64(&self, data: &[i64], size: &[i64]) -> Tensor {
        unsafe {
            Tensor::from_blob(
                data.as_ptr() as *const u8,
                size,
                &[],
                tch::Kind::Int64,
                Device::Cpu,
            )
        }
    }

    fn fill_seq_idx_buf(&mut self) {
        let seq_len = self.patch_ends.len();
        let rows = (NPROCS * TICKERS_COUNT) as usize;
        debug_assert_eq!(seq_len as i64, self.patch_seq_len);
        debug_assert_eq!(self.seq_idx_buf.len(), rows * seq_len);
        for row in 0..rows {
            let out_offset = row * seq_len;
            for (idx, end) in self.patch_ends.iter().enumerate() {
                self.seq_idx_buf[out_offset + idx] = if *end <= 0 { -1 } else { 0 };
            }
        }
    }

    pub fn new(random_start: bool, model_variant: ModelVariant) -> Self {
        let mut envs = Vec::with_capacity(NPROCS as usize);
        envs.push(Env::new_with_recording(random_start, true));
        eprintln!("first env");
        for i in 1..(NPROCS as usize) {
            envs.push(Env::new_with_recording(random_start, false));
            eprintln!("env {}", i);
        }
        for (i, env) in envs.iter_mut().enumerate() {
            env.env_id = i;
        }
        let price_deltas_dim = NPROCS as usize * TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER;
        let static_obs_dim = NPROCS as usize * STATIC_OBSERVATIONS;
        let patch_seq_len = patch_seq_len_for_variant(model_variant);
        let patch_ends = patch_ends_for_variant(model_variant);
        let seq_idx_dim = NPROCS as usize * TICKERS_COUNT as usize * patch_seq_len as usize;
        let done_mask = vec![false; NPROCS as usize];
        let last_static_obs = vec![0.0; NPROCS as usize * STATIC_OBSERVATIONS];
        let last_step_deltas = vec![0.0; NPROCS as usize * TICKERS_COUNT as usize];
        let step_deltas_buf = vec![0.0; NPROCS as usize * TICKERS_COUNT as usize];
        let actions_buf = vec![0.0f64; NPROCS as usize * ACTION_COUNT as usize];
        Self {
            envs,
            done_mask,
            last_static_obs,
            last_step_deltas,
            step_deltas_buf,
            actions_buf,
            reward_buf: vec![0.0; NPROCS as usize],
            reward_per_ticker_buf: vec![0.0; NPROCS as usize * TICKERS_COUNT as usize],
            cash_reward_buf: vec![0.0; NPROCS as usize],
            is_done_buf: vec![0.0; NPROCS as usize],
            price_deltas_buf: vec![0.0; price_deltas_dim],
            static_obs_buf: vec![0.0; static_obs_dim],
            seq_idx_buf: vec![0; seq_idx_dim],
            patch_ends,
            patch_seq_len,
        }
    }

    pub fn reset(&mut self) -> (Tensor, Tensor, Tensor) {
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

        self.price_deltas_buf.copy_from_slice(&all_price_deltas);
        self.static_obs_buf.copy_from_slice(&all_static_obs);
        self.fill_seq_idx_buf();

        let price_deltas = Tensor::from_slice(&all_price_deltas)
            .view([NPROCS, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let static_obs =
            Tensor::from_slice(&all_static_obs).view([NPROCS, STATIC_OBSERVATIONS as i64]);
        let seq_idx = self
            .tensor_from_i64(
                &self.seq_idx_buf,
                &[NPROCS * TICKERS_COUNT, self.patch_seq_len],
            )
            .view([NPROCS, TICKERS_COUNT * self.patch_seq_len]);

        self.done_mask.fill(false);
        self.last_static_obs.clone_from(&all_static_obs);

        (price_deltas, static_obs, seq_idx)
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
        let static_obs =
            Tensor::from_slice(&all_static_obs).view([NPROCS, STATIC_OBSERVATIONS as i64]);

        self.done_mask.fill(false);
        self.last_static_obs.clone_from(&all_static_obs);
        self.last_step_deltas.fill(0.0);

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

        let static_obs =
            Tensor::from_slice(&all_static_obs).view([NPROCS, STATIC_OBSERVATIONS as i64]);

        let mut deltas_flat = Vec::with_capacity(NPROCS as usize * TICKERS_COUNT as usize);
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
        out_price_deltas: &mut Tensor, // [NPROCS, price_deltas_dim] on device
        out_static_obs: &mut Tensor,   // [NPROCS, static_obs_dim] on device
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
            &[NPROCS, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64],
        );
        out_price_deltas.copy_(&pd_cpu);

        let so_cpu =
            self.tensor_from_f32(&self.static_obs_buf, &[NPROCS, STATIC_OBSERVATIONS as i64]);
        out_static_obs.copy_(&so_cpu);

        // Return small tensors - these go to GPU via arithmetic ops later
        let reward = self.tensor_from_f32(&self.reward_buf, &[NPROCS]);
        let is_done = self.tensor_from_f32(&self.is_done_buf, &[NPROCS]);

        (reward, is_done)
    }

    pub fn step_into_full(
        &mut self,
        all_actions: &[Vec<f64>],
        out_price_deltas: &mut Tensor,
        out_static_obs: &mut Tensor,
        out_reward_per_ticker: &mut Tensor,
        out_cash_reward: &mut Tensor,
        out_is_done: &mut Tensor,
        out_seq_idx: &mut Tensor,
    ) {
        debug_assert_eq!(all_actions.len(), self.envs.len());

        let price_deltas_dim = (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64) as usize;
        let static_obs_dim = STATIC_OBSERVATIONS;

        for (i, env) in self.envs.iter_mut().enumerate() {
            let step = env.step_single(&all_actions[i]);
            let reward_start = i * TICKERS_COUNT as usize;
            self.reward_per_ticker_buf[reward_start..reward_start + TICKERS_COUNT as usize]
                .copy_from_slice(&step.reward_per_ticker);
            self.cash_reward_buf[i] = step.cash_reward;
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
            &[NPROCS, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64],
        );
        out_price_deltas.copy_(&pd_cpu);

        let so_cpu =
            self.tensor_from_f32(&self.static_obs_buf, &[NPROCS, STATIC_OBSERVATIONS as i64]);
        out_static_obs.copy_(&so_cpu);

        self.fill_seq_idx_buf();
        let seq_idx_cpu = self
            .tensor_from_i64(
                &self.seq_idx_buf,
                &[NPROCS * TICKERS_COUNT, self.patch_seq_len],
            )
            .view([NPROCS, TICKERS_COUNT * self.patch_seq_len]);
        out_seq_idx.copy_(&seq_idx_cpu);

        let rpt_cpu = self.tensor_from_f32(&self.reward_per_ticker_buf, &[NPROCS, TICKERS_COUNT]);
        out_reward_per_ticker.copy_(&rpt_cpu);
        out_cash_reward.copy_(&self.tensor_from_f32(&self.cash_reward_buf, &[NPROCS]));
        out_is_done.copy_(&self.tensor_from_f32(&self.is_done_buf, &[NPROCS]));
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
            let step = env.step_step_single(&all_actions[i]);
            rewards[i] = step.reward as f32;
            is_dones[i] = step.is_done;
            all_step_deltas.extend(step.step_deltas);
            all_static_obs.extend(step.static_obs);
        }

        let deltas_cpu = Tensor::from_slice(&all_step_deltas).view([NPROCS, TICKERS_COUNT]);
        out_step_deltas.copy_(&deltas_cpu);

        let so_cpu = Tensor::from_slice(&all_static_obs).view([NPROCS, STATIC_OBSERVATIONS as i64]);
        out_static_obs.copy_(&so_cpu);

        let reward = Tensor::from_slice(&rewards);
        let is_done = Tensor::from_slice(&is_dones);

        (reward, is_done)
    }

    pub fn step_incremental(
        &mut self,
        actions_flat: &[f64],
        out_static_obs: &mut Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let action_dim = ACTION_COUNT as usize;
        let envs_len = self.envs.len();
        debug_assert_eq!(actions_flat.len(), envs_len * action_dim);

        let mut rewards = [0f32; NPROCS as usize];
        let mut rewards_per_ticker = vec![0f32; NPROCS as usize * TICKERS_COUNT as usize];
        let mut cash_rewards = [0f32; NPROCS as usize];
        let mut is_dones = [0f32; NPROCS as usize];
        let mut all_step_deltas = Vec::with_capacity(NPROCS as usize * TICKERS_COUNT as usize);
        let mut all_static_obs = Vec::with_capacity(NPROCS as usize * STATIC_OBSERVATIONS);

        for (i, env) in self.envs.iter_mut().enumerate() {
            let action_start = i * action_dim;
            let action_slice = &actions_flat[action_start..action_start + action_dim];

            if self.done_mask[i] {
                rewards[i] = 0.0;
                cash_rewards[i] = 0.0;
                is_dones[i] = 1.0;
                let step_start = i * TICKERS_COUNT as usize;
                let static_start = i * STATIC_OBSERVATIONS;
                all_step_deltas.extend_from_slice(
                    &self.last_step_deltas[step_start..step_start + TICKERS_COUNT as usize],
                );
                all_static_obs.extend_from_slice(
                    &self.last_static_obs[static_start..static_start + STATIC_OBSERVATIONS],
                );
                continue;
            }

            let step = env.step_step_single(action_slice);
            rewards[i] = step.reward as f32;
            let reward_start = i * TICKERS_COUNT as usize;
            rewards_per_ticker[reward_start..reward_start + TICKERS_COUNT as usize]
                .copy_from_slice(&step.reward_per_ticker);
            cash_rewards[i] = step.cash_reward;
            is_dones[i] = step.is_done;
            all_step_deltas.extend_from_slice(&step.step_deltas);
            all_static_obs.extend_from_slice(&step.static_obs);

            let step_start = i * TICKERS_COUNT as usize;
            self.last_step_deltas[step_start..step_start + TICKERS_COUNT as usize]
                .copy_from_slice(&step.step_deltas);
            let static_start = i * STATIC_OBSERVATIONS;
            self.last_static_obs[static_start..static_start + STATIC_OBSERVATIONS]
                .copy_from_slice(&step.static_obs);
            if step.is_done == 1.0 {
                self.done_mask[i] = true;
            }
        }

        let so_cpu = Tensor::from_slice(&all_static_obs).view([NPROCS, STATIC_OBSERVATIONS as i64]);
        out_static_obs.copy_(&so_cpu);

        let reward = Tensor::from_slice(&rewards);
        let reward_per_ticker =
            Tensor::from_slice(&rewards_per_ticker).view([NPROCS, TICKERS_COUNT]);
        let cash_reward = Tensor::from_slice(&cash_rewards);
        let is_done = Tensor::from_slice(&is_dones);
        let step_deltas = Tensor::from_slice(&all_step_deltas).view([NPROCS, TICKERS_COUNT]);

        (reward, reward_per_ticker, cash_reward, is_done, step_deltas)
    }

    pub fn step_incremental_tensor_into(
        &mut self,
        actions: &Tensor,
        out_static_obs: &mut Tensor,
        out_reward: &mut Tensor,
        out_reward_per_ticker: &mut Tensor,
        out_cash_reward: &mut Tensor,
        out_is_done: &mut Tensor,
        out_step_deltas: &mut Tensor,
    ) {
        let actions_cpu = actions.to_device(Device::Cpu);
        let actions_flat = Vec::<f64>::try_from(actions_cpu.flatten(0, -1)).unwrap();
        let (reward, reward_per_ticker, cash_reward, is_done, step_deltas) =
            self.step_incremental(&actions_flat, out_static_obs);
        out_reward.copy_(&reward);
        out_reward_per_ticker.copy_(&reward_per_ticker);
        out_cash_reward.copy_(&cash_reward);
        out_is_done.copy_(&is_done);
        out_step_deltas.copy_(&step_deltas);
    }

    pub fn step_incremental_tensor(
        &mut self,
        actions: &Tensor,
        out_static_obs: &mut Tensor,
    ) -> (Tensor, Tensor, Tensor, Tensor, Tensor) {
        let actions_cpu = actions.to_device(Device::Cpu);
        let actions_flat = Vec::<f64>::try_from(actions_cpu.flatten(0, -1)).unwrap();
        self.step_incremental(&actions_flat, out_static_obs)
    }

    pub fn step_into_ring_flat(
        &mut self,
        all_actions: &[f64],
        out_step_deltas: &mut Tensor,
        out_static_obs: &mut Tensor,
        out_reward_per_ticker: &mut Tensor,
        out_cash_reward: &mut Tensor,
        out_is_done: &mut Tensor,
    ) -> (Vec<usize>, Vec<f32>) {
        let action_dim = ACTION_COUNT as usize;
        debug_assert_eq!(all_actions.len(), self.envs.len() * action_dim);

        let static_obs_dim = STATIC_OBSERVATIONS;
        let step_deltas_dim = TICKERS_COUNT as usize;
        let price_deltas_dim = TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER;

        let mut reset_indices = Vec::new();
        let mut reset_price_deltas = Vec::new();

        for (i, env) in self.envs.iter_mut().enumerate() {
            let action_start = i * action_dim;
            let action_slice = &all_actions[action_start..action_start + action_dim];
            let step = env.step_step_single(action_slice);
            let reward_start = i * TICKERS_COUNT as usize;
            self.reward_per_ticker_buf[reward_start..reward_start + TICKERS_COUNT as usize]
                .copy_from_slice(&step.reward_per_ticker);
            self.cash_reward_buf[i] = step.cash_reward;
            self.is_done_buf[i] = step.is_done;

            let step_offset = i * step_deltas_dim;
            let so_offset = i * static_obs_dim;
            if step.is_done == 1.0 {
                let (price_deltas, static_obs) = env.reset_single();
                let tail_base = PRICE_DELTAS_PER_TICKER - 1;
                for t in 0..TICKERS_COUNT as usize {
                    let idx = t * PRICE_DELTAS_PER_TICKER + tail_base;
                    self.step_deltas_buf[step_offset + t] = price_deltas[idx];
                }
                self.static_obs_buf[so_offset..so_offset + static_obs_dim]
                    .copy_from_slice(&static_obs);

                reset_indices.push(i);
                reset_price_deltas.extend(price_deltas);
            } else {
                self.step_deltas_buf[step_offset..step_offset + step_deltas_dim]
                    .copy_from_slice(&step.step_deltas);
                self.static_obs_buf[so_offset..so_offset + static_obs_dim]
                    .copy_from_slice(&step.static_obs);
            }
        }

        let deltas_cpu = self.tensor_from_f32(&self.step_deltas_buf, &[NPROCS, TICKERS_COUNT]);
        out_step_deltas.copy_(&deltas_cpu);

        let so_cpu =
            self.tensor_from_f32(&self.static_obs_buf, &[NPROCS, STATIC_OBSERVATIONS as i64]);
        out_static_obs.copy_(&so_cpu);

        let rpt_cpu = self.tensor_from_f32(&self.reward_per_ticker_buf, &[NPROCS, TICKERS_COUNT]);
        out_reward_per_ticker.copy_(&rpt_cpu);
        out_cash_reward.copy_(&self.tensor_from_f32(&self.cash_reward_buf, &[NPROCS]));
        out_is_done.copy_(&self.tensor_from_f32(&self.is_done_buf, &[NPROCS]));

        (reset_indices, reset_price_deltas)
    }

    pub fn step_into_ring_tensor(
        &mut self,
        actions: &Tensor,
        out_step_deltas: &mut Tensor,
        out_static_obs: &mut Tensor,
        out_reward_per_ticker: &mut Tensor,
        out_cash_reward: &mut Tensor,
        out_is_done: &mut Tensor,
    ) -> (Vec<usize>, Vec<f32>) {
        let actions_cpu = actions.to_device(Device::Cpu).to_kind(tch::Kind::Double);
        let actions_len = self.actions_buf.len();
        actions_cpu.copy_data(&mut self.actions_buf, actions_len);
        let actions_ptr = self.actions_buf.as_ptr();
        let actions_len = self.actions_buf.len();
        let actions_slice = unsafe { std::slice::from_raw_parts(actions_ptr, actions_len) };
        self.step_into_ring_flat(
            actions_slice,
            out_step_deltas,
            out_static_obs,
            out_reward_per_ticker,
            out_cash_reward,
            out_is_done,
        )
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
