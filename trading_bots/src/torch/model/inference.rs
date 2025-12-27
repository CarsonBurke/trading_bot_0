use tch::{Kind, Tensor};

use super::{ModelOutput, StreamState, TradingModel, PATCH_SIZE, SEQ_LEN, SSM_DIM};
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
        let _ = self.ssm_cache.zero_();
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
            patch_buf: Tensor::zeros(&[TICKERS_COUNT, PATCH_SIZE], (Kind::Float, self.device)),
            patch_pos: 0,
            ssm_states: (0..TICKERS_COUNT)
                .map(|_| self.ssm.init_state(1, self.device))
                .collect(),
            ssm_cache: Tensor::zeros(
                &[TICKERS_COUNT, SSM_DIM, SEQ_LEN],
                (Kind::Float, self.device),
            ),
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
                &[batch_size * TICKERS_COUNT, PATCH_SIZE],
                (Kind::Float, self.device),
            ),
            patch_pos: 0,
            ssm_states: (0..(batch_size * TICKERS_COUNT) as usize)
                .map(|_| self.ssm.init_state(1, self.device))
                .collect(),
            ssm_cache: Tensor::zeros(
                &[batch_size * TICKERS_COUNT, SSM_DIM, SEQ_LEN],
                (Kind::Float, self.device),
            ),
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
        let static_ssm = self.per_ticker_static_ssm(&per_ticker_static, 1);

        if state.patch_pos >= PATCH_SIZE {
            state.patch_pos = 0;
            self.process_new_patch(state, &static_ssm);
            let _ = state.patch_buf.zero_();
        }

        self.head_with_temporal_pool(
            &state.ssm_cache,
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
        let static_ssm = self.per_ticker_static_ssm(&per_ticker_static, 1);

        for t in 0..TICKERS_COUNT as usize {
            let ticker_data = reshaped.get(t as i64).unsqueeze(0);
            let x_stem = self.patch_embed_single(&ticker_data, &static_ssm.get(t as i64));
            let x_ssm = self
                .ssm
                .forward_with_state(&x_stem.permute([0, 2, 1]), &mut state.ssm_states[t]);
            let _ = state
                .ssm_cache
                .get(t as i64)
                .copy_(&x_ssm.squeeze_dim(0).permute([1, 0]));
        }

        state.initialized = true;
        self.head_with_temporal_pool(
            &state.ssm_cache,
            &global_static,
            &per_ticker_static,
            1,
            false,
        )
        .0
    }

    fn process_new_patch(&self, state: &mut StreamState, static_ssm: &Tensor) {
        let patch_emb = state
            .patch_buf
            .view([TICKERS_COUNT, 1, PATCH_SIZE])
            .apply(&self.patch_embed);
        let patch_emb = self.patch_ln.forward(&patch_emb).squeeze_dim(1) + static_ssm;

        for t in 0..TICKERS_COUNT as usize {
            let x_in = patch_emb.get(t as i64).unsqueeze(0);
            let out = self.ssm.step(&x_in, &mut state.ssm_states[t]);

            let old_cache = state.ssm_cache.get(t as i64);
            let shifted = old_cache.narrow(1, 1, SEQ_LEN - 1);
            let _ = old_cache.narrow(1, 0, SEQ_LEN - 1).copy_(&shifted);
            let _ = old_cache
                .narrow(1, SEQ_LEN - 1, 1)
                .copy_(&out.squeeze_dim(0).unsqueeze(1));
        }
    }
}
