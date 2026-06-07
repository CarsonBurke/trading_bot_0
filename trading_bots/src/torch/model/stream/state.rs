use tch::{Kind, Tensor};

use super::super::config::{
    ModelVariant, UNIFORM_STREAM_LAYOUT_LEN, UNIFORM_STREAM_PATCH_COUNT, UNIFORM_STREAM_PATCH_SIZE,
};
use super::super::trading_model::{StreamState, TradingModel};
use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, TICKERS_COUNT};

impl StreamState {
    pub fn reset(&mut self) {
        let _ = self.delta_ring.zero_();
        self.ring_pos = 0;
        let _ = self.patch_buf.zero_();
        self.patch_pos = 0;
        let _ = self.uniform_layout.fill_(f64::NAN);
        let _ = self.uniform_patch_tokens.zero_();
        let _ = self.uniform_live_fill.zero_();
        self.uniform_live_fill_host.fill(0);
        self.initialized = false;
    }
}

impl TradingModel {
    pub fn detach_stream_state(&self, state: &mut StreamState) {
        if self.variant != ModelVariant::UniformStream {
            return;
        }
        state.uniform_layout = state.uniform_layout.detach();
        state.uniform_patch_tokens = state.uniform_patch_tokens.detach();
    }

    pub fn uniform_stream_snapshot(&self, state: &StreamState) -> Tensor {
        state
            .uniform_layout
            .view([
                state.uniform_live_fill.size()[0],
                TICKERS_COUNT * UNIFORM_STREAM_LAYOUT_LEN,
            ])
            .to_kind(Kind::Float)
    }

    pub(super) fn ordered_price_from_ring(&self, state: &StreamState, batch_size: i64) -> Tensor {
        let ring_len = PRICE_DELTAS_PER_TICKER as i64;
        let idx = (Tensor::arange(ring_len, (Kind::Int64, self.device)) + state.ring_pos)
            .remainder(ring_len);
        state
            .delta_ring
            .index_select(1, &idx)
            .view([batch_size, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64])
    }

    fn build_stream_state(&self, batch_size: i64) -> StreamState {
        let uniform_rows = batch_size * TICKERS_COUNT;
        let activation_kind = self.activation_kind();
        let (delta_ring, patch_buf) = if self.variant == ModelVariant::UniformStream {
            (
                Tensor::zeros(
                    [0, PRICE_DELTAS_PER_TICKER as i64],
                    (Kind::Float, self.device),
                ),
                Tensor::zeros([0, self.finest_patch_size], (Kind::Float, self.device)),
            )
        } else {
            (
                Tensor::zeros(
                    &[uniform_rows, PRICE_DELTAS_PER_TICKER as i64],
                    (Kind::Float, self.device),
                ),
                Tensor::zeros(
                    &[uniform_rows, self.finest_patch_size],
                    (Kind::Float, self.device),
                ),
            )
        };
        StreamState {
            delta_ring,
            ring_pos: 0,
            patch_buf,
            patch_pos: 0,
            uniform_layout: Tensor::full(
                [
                    uniform_rows,
                    UNIFORM_STREAM_PATCH_COUNT,
                    UNIFORM_STREAM_PATCH_SIZE,
                ],
                f64::NAN,
                (activation_kind, self.device),
            ),
            uniform_patch_tokens: Tensor::zeros(
                [uniform_rows, UNIFORM_STREAM_PATCH_COUNT, self.model_dim],
                (activation_kind, self.device),
            ),
            uniform_live_fill: Tensor::zeros([batch_size], (Kind::Int64, self.device)),
            uniform_live_fill_host: vec![0; batch_size as usize],
            initialized: false,
        }
    }

    pub fn init_stream_state(&self) -> StreamState {
        self.build_stream_state(1)
    }

    pub fn init_stream_state_batched(&self, batch_size: i64) -> StreamState {
        self.build_stream_state(batch_size)
    }

    pub fn init_replay_stream_state_batched(&self, batch_size: i64) -> StreamState {
        self.build_stream_state(batch_size)
    }
}
