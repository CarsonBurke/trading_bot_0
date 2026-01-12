use tch::{Kind, Tensor};

use super::{ModelOutput, StreamState, TradingModel, FINEST_PATCH_INDEX, FINEST_PATCH_SIZE, SSM_DIM};
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, TICKERS_COUNT};

impl StreamState {
    pub fn reset(&mut self) {
        let _ = self.delta_ring.zero_();
        self.ring_pos = 0;
        let _ = self.patch_buf.zero_();
        self.patch_pos = 0;
        for s in &mut self.ssm_states {
            s.reset();
        }
        self.initialized = false;
    }
}

impl TradingModel {
    pub fn init_stream_state(&self) -> StreamState {
        StreamState {
            delta_ring: Tensor::zeros(
                &[TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64],
                (Kind::Float, self.device),
            ),
            ring_pos: 0,
            patch_buf: Tensor::zeros(
                &[TICKERS_COUNT, FINEST_PATCH_SIZE],
                (Kind::Float, self.device),
            ),
            patch_pos: 0,
            ssm_states: (0..(TICKERS_COUNT * self.ssm_layers.len() as i64))
                .map(|_| self.ssm_layers[0].init_state(1, self.device))
                .collect(),
            initialized: false,
        }
    }

    pub fn init_stream_state_batched(&self, batch_size: i64) -> StreamState {
        StreamState {
            delta_ring: Tensor::zeros(
                &[batch_size * TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64],
                (Kind::Float, self.device),
            ),
            ring_pos: 0,
            patch_buf: Tensor::zeros(
                &[batch_size * TICKERS_COUNT, FINEST_PATCH_SIZE],
                (Kind::Float, self.device),
            ),
            patch_pos: 0,
            ssm_states: (0..(batch_size * TICKERS_COUNT * self.ssm_layers.len() as i64) as usize)
                .map(|_| self.ssm_layers[0].init_state(1, self.device))
                .collect(),
            initialized: false,
        }
    }

    pub fn step(
        &self,
        new_deltas: &Tensor,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let new_deltas = self.cast_inputs(&new_deltas.to_device(self.device));
        let static_features = self.cast_inputs(&static_features.to_device(self.device));

        let full_obs = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
        let is_full = (new_deltas.dim() == 1 && new_deltas.size()[0] == full_obs)
            || (new_deltas.dim() == 2 && new_deltas.size()[1] == full_obs);

        if is_full {
            return self.init_from_full(&new_deltas, &static_features, state);
        }

        let new_deltas = if new_deltas.dim() == 1 {
            new_deltas
        } else {
            new_deltas.squeeze_dim(0)
        };

        for t in 0..TICKERS_COUNT {
            let _ = state
                .delta_ring
                .get(t)
                .narrow(0, state.ring_pos, 1)
                .copy_(&new_deltas.get(t).unsqueeze(0));
        }
        state.ring_pos = (state.ring_pos + 1) % PRICE_DELTAS_PER_TICKER as i64;

        let _ = state
            .patch_buf
            .narrow(1, state.patch_pos, 1)
            .copy_(&new_deltas.unsqueeze(1));
        state.patch_pos += 1;

        let static_features = if static_features.dim() == 1 {
            static_features.unsqueeze(0)
        } else {
            static_features
        };
        let (global_static, per_ticker_static) = self.parse_static(&static_features, 1);

        if state.patch_pos >= FINEST_PATCH_SIZE {
            state.patch_pos = 0;
            let x_last = self.process_new_patch(state);
            let _ = state.patch_buf.zero_();
            return self
                .head_with_temporal_pool(&x_last, &global_static, &per_ticker_static, 1, false)
                .0;
        }

        self.head_with_temporal_pool(
            &Tensor::zeros(&[TICKERS_COUNT, 1, SSM_DIM], (Kind::Float, self.device)),
            &global_static,
            &per_ticker_static,
            1,
            false,
        )
        .0
    }

    fn init_from_full(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let price = if price_deltas.dim() == 1 {
            price_deltas.unsqueeze(0)
        } else {
            price_deltas.shallow_clone()
        };
        let static_features = if static_features.dim() == 1 {
            static_features.unsqueeze(0)
        } else {
            static_features.shallow_clone()
        };
        let price = self.cast_inputs(&price);
        let static_features = self.cast_inputs(&static_features);

        let reshaped = price.view([TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64]);
        let _ = state.delta_ring.copy_(&reshaped);
        state.ring_pos = 0;
        state.patch_pos = 0;
        let _ = state.patch_buf.zero_();

        let (global_static, per_ticker_static) = self.parse_static(&static_features, 1);
        let dt_scale = self.patch_dt_scale.shallow_clone();

        let mut outputs = Vec::with_capacity(TICKERS_COUNT as usize);
        for t in 0..TICKERS_COUNT as usize {
            let ticker_data = reshaped.get(t as i64).unsqueeze(0);
            let x_stem = self.patch_embed_single(&ticker_data);
            let mut x = x_stem;
            for (layer, (ssm, norm)) in
                self.ssm_layers.iter().zip(self.ssm_norms.iter()).enumerate()
            {
                let state_idx = layer * TICKERS_COUNT as usize + t;
                let normed = norm.forward(&x);
                let out = ssm.forward_with_state_dt_scale(
                    &normed,
                    &mut state.ssm_states[state_idx],
                    Some(&dt_scale),
                );
                x = x + out;
            }
            outputs.push(x);
        }

        state.initialized = true;
        let x_ssm = Tensor::cat(&outputs, 0);
        self.head_with_temporal_pool(&x_ssm, &global_static, &per_ticker_static, 1, false)
            .0
    }

    fn process_new_patch(&self, state: &mut StreamState) -> Tensor {
        let patches = state
            .patch_buf
            .view([TICKERS_COUNT, 1, FINEST_PATCH_SIZE]);
        let patch_emb = self
            .embed_patch_config(&patches, FINEST_PATCH_INDEX as i64)
            .squeeze_dim(1);
        let dt_scale = FINEST_PATCH_SIZE as f64;

        let mut outputs = Vec::with_capacity(TICKERS_COUNT as usize);
        for t in 0..TICKERS_COUNT as usize {
            let mut x_in = patch_emb.get(t as i64).unsqueeze(0);
            for (layer, (ssm, norm)) in
                self.ssm_layers.iter().zip(self.ssm_norms.iter()).enumerate()
            {
                let state_idx = layer * TICKERS_COUNT as usize + t;
                let normed = norm.forward(&x_in);
                let out = ssm.step_with_dt_scale(&normed, &mut state.ssm_states[state_idx], dt_scale);
                x_in = x_in + out;
            }
            outputs.push(x_in.unsqueeze(1));
        }
        Tensor::cat(&outputs, 0)
    }
}
