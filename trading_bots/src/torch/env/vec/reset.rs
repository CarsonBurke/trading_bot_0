use super::core::{EnvGroupEpisode, VecEnv};
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
use tch::Tensor;

impl VecEnv {
    pub(super) fn reset_group_full_obs(
        &mut self,
        group_idx: usize,
        used_specs: &mut Vec<EnvGroupEpisode>,
    ) {
        let (group_start, group_end) = self.group_bounds(group_idx);
        let (mut leader_price_deltas, mut leader_static_obs) =
            self.envs[group_start].reset_single_resampled_training_episode();
        let mut spec = self.current_group_episode(group_start);
        let mut attempts = 0;
        while Self::has_used_market_episode(used_specs, &spec) && attempts < 128 {
            let reset = self.envs[group_start].reset_single_resampled_training_episode();
            leader_price_deltas = reset.0;
            leader_static_obs = reset.1;
            spec = self.current_group_episode(group_start);
            attempts += 1;
        }
        assert!(
            !Self::has_used_market_episode(used_specs, &spec),
            "failed to sample distinct env reset group after {} attempts",
            attempts
        );
        used_specs.push(spec.clone());

        self.write_full_obs(group_start, &leader_price_deltas, &leader_static_obs);
        for env_idx in group_start + 1..group_end {
            let (price_deltas, static_obs) =
                self.envs[env_idx].reset_single_to_episode(&spec.market, spec.start_offset);
            self.write_full_obs(env_idx, &price_deltas, &static_obs);
        }
    }

    pub(super) fn reset_group_step_obs(
        &mut self,
        group_idx: usize,
        used_specs: &mut Vec<EnvGroupEpisode>,
    ) {
        let (group_start, group_end) = self.group_bounds(group_idx);
        let (mut leader_step_deltas, mut leader_static_obs) =
            self.envs[group_start].reset_step_single_resampled_training_episode();
        let mut spec = self.current_group_episode(group_start);
        let mut attempts = 0;
        while Self::has_used_market_episode(used_specs, &spec) && attempts < 128 {
            let reset = self.envs[group_start].reset_step_single_resampled_training_episode();
            leader_step_deltas = reset.0;
            leader_static_obs = reset.1;
            spec = self.current_group_episode(group_start);
            attempts += 1;
        }
        assert!(
            !Self::has_used_market_episode(used_specs, &spec),
            "failed to sample distinct env reset group after {} attempts",
            attempts
        );
        used_specs.push(spec.clone());

        self.write_step_obs(group_start, &leader_step_deltas, &leader_static_obs);
        for env_idx in group_start + 1..group_end {
            let (step_deltas, static_obs) =
                self.envs[env_idx].reset_step_single_to_episode(&spec.market, spec.start_offset);
            self.write_step_obs(env_idx, &step_deltas, &static_obs);
        }
    }

    pub(super) fn write_full_obs(
        &mut self,
        env_idx: usize,
        price_deltas: &[f32],
        static_obs: &[f32],
    ) {
        let pd_dim = TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER;
        let pd_offset = env_idx * pd_dim;
        let so_offset = env_idx * STATIC_OBSERVATIONS;
        self.price_deltas_buf[pd_offset..pd_offset + pd_dim].copy_from_slice(price_deltas);
        self.static_obs_buf[so_offset..so_offset + STATIC_OBSERVATIONS].copy_from_slice(static_obs);
        let step_offset = env_idx * TICKERS_COUNT as usize;
        let tail_base = PRICE_DELTAS_PER_TICKER - 1;
        for t in 0..TICKERS_COUNT as usize {
            self.step_deltas_buf[step_offset + t] =
                price_deltas[t * PRICE_DELTAS_PER_TICKER + tail_base];
        }
    }

    pub(super) fn write_step_obs(
        &mut self,
        env_idx: usize,
        step_deltas: &[f32],
        static_obs: &[f32],
    ) {
        let step_offset = env_idx * TICKERS_COUNT as usize;
        let so_offset = env_idx * STATIC_OBSERVATIONS;
        self.step_deltas_buf[step_offset..step_offset + TICKERS_COUNT as usize]
            .copy_from_slice(step_deltas);
        self.static_obs_buf[so_offset..so_offset + STATIC_OBSERVATIONS].copy_from_slice(static_obs);
    }

    pub(super) fn write_reset_ring_obs(
        &mut self,
        env_idx: usize,
        price_deltas: &[f32],
        static_obs: &[f32],
        reset_indices: &mut Vec<usize>,
        reset_price_deltas: &mut Vec<f32>,
    ) {
        let tail_base = PRICE_DELTAS_PER_TICKER - 1;
        let step_offset = env_idx * TICKERS_COUNT as usize;
        let so_offset = env_idx * STATIC_OBSERVATIONS;
        for t in 0..TICKERS_COUNT as usize {
            let idx = t * PRICE_DELTAS_PER_TICKER + tail_base;
            self.step_deltas_buf[step_offset + t] = price_deltas[idx];
        }
        self.static_obs_buf[so_offset..so_offset + STATIC_OBSERVATIONS].copy_from_slice(static_obs);
        reset_indices.push(env_idx);
        reset_price_deltas.extend(price_deltas);
    }

    pub fn reset(&mut self) -> (Tensor, Tensor) {
        assert_eq!(
            self.envs.len(),
            self.nprocs,
            "VecEnv desync: envs.len={} != nprocs={}",
            self.envs.len(),
            self.nprocs
        );
        let mut used_specs = Vec::with_capacity(self.env_group_count());
        for group_idx in 0..self.env_group_count() {
            self.reset_group_full_obs(group_idx, &mut used_specs);
        }

        let price_deltas = Tensor::from_slice(&self.price_deltas_buf).view([
            self.nprocs_i64(),
            TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64,
        ]);
        let static_obs = Tensor::from_slice(&self.static_obs_buf)
            .view([self.nprocs_i64(), STATIC_OBSERVATIONS as i64]);

        self.done_mask.fill(false);
        self.last_static_obs.clone_from(&self.static_obs_buf);

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
        let mut used_specs = Vec::with_capacity(self.env_group_count());
        for group_idx in 0..self.env_group_count() {
            self.reset_group_step_obs(group_idx, &mut used_specs);
        }

        let step_deltas =
            Tensor::from_slice(&self.step_deltas_buf).view([self.nprocs_i64(), TICKERS_COUNT]);
        let static_obs = Tensor::from_slice(&self.static_obs_buf)
            .view([self.nprocs_i64(), STATIC_OBSERVATIONS as i64]);

        self.done_mask.fill(false);
        self.last_static_obs.clone_from(&self.static_obs_buf);
        self.last_step_deltas.fill(0.0);

        (step_deltas, static_obs)
    }

    pub fn reset_incremental(&mut self) -> (Vec<Vec<f32>>, Tensor) {
        assert_eq!(self.envs.len(), self.nprocs);
        let mut used_specs = Vec::with_capacity(self.env_group_count());
        for group_idx in 0..self.env_group_count() {
            self.reset_group_full_obs(group_idx, &mut used_specs);
        }

        let static_obs = Tensor::from_slice(&self.static_obs_buf)
            .view([self.nprocs_i64(), STATIC_OBSERVATIONS as i64]);

        let pd_dim = TICKERS_COUNT as usize * PRICE_DELTAS_PER_TICKER;
        let mut all_deltas_per_env: Vec<Vec<f32>> = Vec::with_capacity(self.nprocs);
        for env_idx in 0..self.nprocs {
            let offset = env_idx * pd_dim;
            all_deltas_per_env.push(self.price_deltas_buf[offset..offset + pd_dim].to_vec());
        }
        self.done_mask.fill(false);
        self.last_static_obs.clone_from(&self.static_obs_buf);
        self.last_step_deltas.clone_from(&self.step_deltas_buf);

        (all_deltas_per_env, static_obs)
    }
}
