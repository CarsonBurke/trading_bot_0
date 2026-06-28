use tch::{Kind, Tensor};

use super::super::config::{
    ModelVariant, UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL, UNIFORM_STREAM_LAYOUT_LEN,
    UNIFORM_STREAM_PATCH_COUNT, UNIFORM_STREAM_PATCH_SIZE,
};
use super::super::trading_model::{ModelOutput, StreamState, TradingModel};
use crate::torch::constants::{ACTION_COUNT, PRICE_DELTAS_PER_TICKER, TICKERS_COUNT};

impl TradingModel {
    pub(super) fn cached_readout_forward(
        &self,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let batch_size = state.uniform_live_fill.size()[0];
        let (global_static, per_ticker_static) = self.parse_static(static_features, batch_size);
        let exo_tokens = self.build_exo_tokens(&global_static, &per_ticker_static, batch_size);
        self.readout_from_cached_patches(&exo_tokens, batch_size, state)
    }

    pub fn forward_stream_state_on_device(
        &self,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let static_features = self.ensure_batched(static_features);
        let static_features = self.cast_inputs(&static_features);
        if self.variant == ModelVariant::UniformStream {
            return self.cached_readout_forward(&static_features, state);
        }
        let batch_size = state.delta_ring.size()[0] / TICKERS_COUNT;
        let price = self.ordered_price_from_ring(state, batch_size);
        self.forward_on_device(&price, &static_features, false)
    }

    pub fn reset_uniform_stream_envs_from_layout(
        &self,
        state: &mut StreamState,
        env_indices: &[usize],
        layouts: &Tensor,
    ) {
        if self.variant != ModelVariant::UniformStream || env_indices.is_empty() {
            return;
        }
        let layouts = self.ensure_batched(layouts);
        let layouts = self.cast_inputs(&layouts);
        let expected = TICKERS_COUNT * UNIFORM_STREAM_LAYOUT_LEN;
        assert_eq!(
            layouts.size()[1],
            expected,
            "reset_uniform_stream_envs_from_layout expects flattened uniform layouts"
        );
        assert_eq!(
            layouts.size()[0] as usize,
            env_indices.len(),
            "layout/reset batch mismatch"
        );
        let layouts = layouts.view([
            layouts.size()[0] * TICKERS_COUNT,
            UNIFORM_STREAM_PATCH_COUNT,
            UNIFORM_STREAM_PATCH_SIZE,
        ]);
        let patch_tokens =
            self.patch_embed(&layouts.view([layouts.size()[0], UNIFORM_STREAM_LAYOUT_LEN]));
        let live_fill = Self::live_fill_from_layout(&layouts);
        let live_fill_host = Vec::<i64>::try_from(live_fill.to_device(tch::Device::Cpu)).unwrap();
        let row_indices = env_indices
            .iter()
            .flat_map(|&env_idx| {
                (0..TICKERS_COUNT)
                    .map(move |ticker_idx| (env_idx as i64) * TICKERS_COUNT + ticker_idx)
            })
            .collect::<Vec<_>>();
        let row_idx = Tensor::from_slice(&row_indices)
            .to_kind(Kind::Int64)
            .to_device(self.device);
        state.uniform_layout = state.uniform_layout.index_copy(
            0,
            &row_idx,
            &layouts.to_kind(state.uniform_layout.kind()),
        );
        state.uniform_patch_tokens = state.uniform_patch_tokens.index_copy(
            0,
            &row_idx,
            &patch_tokens.to_kind(state.uniform_patch_tokens.kind()),
        );
        let env_idx = Tensor::from_slice(
            &env_indices
                .iter()
                .map(|&idx| idx as i64)
                .collect::<Vec<_>>(),
        )
        .to_kind(Kind::Int64)
        .to_device(self.device);
        let reset_live_fill = Tensor::from_slice(
            &env_indices
                .iter()
                .enumerate()
                .map(|(reset_i, _)| live_fill_host[(reset_i as i64 * TICKERS_COUNT) as usize])
                .collect::<Vec<_>>(),
        )
        .to_kind(Kind::Int64)
        .to_device(self.device);
        state.uniform_live_fill = state
            .uniform_live_fill
            .index_copy(0, &env_idx, &reset_live_fill);
        for (reset_i, env_idx) in env_indices.iter().enumerate() {
            state.uniform_live_fill_host[*env_idx] =
                live_fill_host[(reset_i as i64 * TICKERS_COUNT) as usize];
        }
    }

    pub(crate) fn reset_uniform_stream_envs_from_layout_indexed(
        &self,
        state: &mut StreamState,
        env_idx: &Tensor,
        row_idx: &Tensor,
        layouts: &Tensor,
    ) {
        if self.variant != ModelVariant::UniformStream || env_idx.size()[0] == 0 {
            return;
        }
        let layouts = self.ensure_batched(layouts);
        let layouts = self.cast_inputs(&layouts);
        let expected = TICKERS_COUNT * UNIFORM_STREAM_LAYOUT_LEN;
        assert_eq!(
            layouts.size()[1],
            expected,
            "reset_uniform_stream_envs_from_layout_indexed expects flattened uniform layouts"
        );
        assert_eq!(
            layouts.size()[0] as usize,
            env_idx.size()[0] as usize,
            "layout/reset batch mismatch"
        );
        let layouts = layouts.view([
            layouts.size()[0] * TICKERS_COUNT,
            UNIFORM_STREAM_PATCH_COUNT,
            UNIFORM_STREAM_PATCH_SIZE,
        ]);
        let patch_tokens =
            self.patch_embed(&layouts.view([layouts.size()[0], UNIFORM_STREAM_LAYOUT_LEN]));
        state.uniform_layout = state.uniform_layout.index_copy(
            0,
            row_idx,
            &layouts.to_kind(state.uniform_layout.kind()),
        );
        state.uniform_patch_tokens = state.uniform_patch_tokens.index_copy(
            0,
            row_idx,
            &patch_tokens.to_kind(state.uniform_patch_tokens.kind()),
        );
        // PPO replay uses the device tensor directly; the host mirror has no read sites here.
        let _ = state
            .uniform_live_fill
            .index_fill_(0, env_idx, UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL);
    }

    pub fn reset_uniform_stream_envs(
        &self,
        state: &mut StreamState,
        env_indices: &[usize],
        reset_price_deltas: &[f32],
    ) {
        if self.variant != ModelVariant::UniformStream || env_indices.is_empty() {
            return;
        }
        let raw_pd_dim = (TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64) as usize;
        for (reset_i, env_idx) in env_indices.iter().enumerate() {
            let start = reset_i * raw_pd_dim;
            let end = start + raw_pd_dim;
            let raw = Tensor::from_slice(&reset_price_deltas[start..end])
                .view([TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
                .to_device(self.device);
            let layout = self.uniform_stream_layout_from_raw(&raw).view([
                TICKERS_COUNT,
                UNIFORM_STREAM_PATCH_COUNT,
                UNIFORM_STREAM_PATCH_SIZE,
            ]);
            let patch_tokens =
                self.patch_embed(&layout.view([TICKERS_COUNT, UNIFORM_STREAM_LAYOUT_LEN]));
            let row_start = (*env_idx as i64) * TICKERS_COUNT;
            let _ = state
                .uniform_layout
                .narrow(0, row_start, TICKERS_COUNT)
                .copy_(&layout);
            let _ = state
                .uniform_patch_tokens
                .narrow(0, row_start, TICKERS_COUNT)
                .copy_(&patch_tokens);
            let _ = state
                .uniform_live_fill
                .get(*env_idx as i64)
                .fill_(UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL);
            state.uniform_live_fill_host[*env_idx] = UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL;
        }
    }

    pub(super) fn init_uniform_from_full_on_device(&self, price: &Tensor, state: &mut StreamState) {
        let batch_size = price.size()[0];
        let expected_layout = TICKERS_COUNT * UNIFORM_STREAM_LAYOUT_LEN;
        assert_eq!(
            price.size()[1],
            expected_layout,
            "UniformStream init expects anchored layout input"
        );
        let layout = price
            .view([
                batch_size * TICKERS_COUNT,
                UNIFORM_STREAM_PATCH_COUNT,
                UNIFORM_STREAM_PATCH_SIZE,
            ])
            .copy();
        let live_fill = Self::live_fill_from_layout(&layout);
        state.uniform_layout = layout;
        state.uniform_patch_tokens = self.patch_embed(
            &state
                .uniform_layout
                .view([batch_size * TICKERS_COUNT, UNIFORM_STREAM_LAYOUT_LEN]),
        );
        state.uniform_live_fill = live_fill.shallow_clone();
        state.uniform_live_fill_host =
            Vec::<i64>::try_from(live_fill.to_device(tch::Device::Cpu)).unwrap();
        state.initialized = true;
    }

    pub(super) fn step_uniform_stream_state_on_device(
        &self,
        new_deltas: &Tensor,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let new_deltas = self.ensure_batched(new_deltas);
        let batch_size = new_deltas.size()[0];
        let state_batch_size = state.uniform_live_fill.size()[0];
        assert_eq!(
            state_batch_size, batch_size,
            "stream state batch size mismatch"
        );

        self.advance_layout_and_reembed_inplace(state, &new_deltas);
        self.cached_readout_forward(static_features, state)
    }

    pub fn step(
        &self,
        new_deltas: &Tensor,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let new_deltas = self.cast_inputs(&new_deltas.to_device(self.device));
        let static_features = self.cast_inputs(&static_features.to_device(self.device));
        self.step_on_device(&new_deltas, &static_features, state)
    }

    pub fn step_on_device(
        &self,
        new_deltas: &Tensor,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        if new_deltas.device() != self.device || static_features.device() != self.device {
            panic!("step_on_device requires tensors on {:?}", self.device);
        }
        let new_deltas = self.cast_inputs(new_deltas);
        let static_features = self.cast_inputs(static_features);

        let is_full = self.is_full_obs(&new_deltas);
        let static_features = self.ensure_batched(&static_features);

        if self.variant == ModelVariant::UniformStream {
            if is_full {
                return self.init_from_full_on_device(&new_deltas, &static_features, state);
            }
            return self.step_uniform_stream_state_on_device(&new_deltas, &static_features, state);
        }

        if is_full {
            return self.init_from_full_on_device(&new_deltas, &static_features, state);
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

        if state.patch_pos >= self.finest_patch_size {
            state.patch_pos = 0;
            let _ = state.patch_buf.zero_();
            // Full forward pass from ring buffer
            let price_deltas = state
                .delta_ring
                .view([1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
            return self.forward_on_device(&price_deltas, &static_features, false);
        }

        // Not enough deltas for a new patch yet; return a uniform Beta (alpha=beta=1)
        (
            Tensor::zeros(
                &[1, crate::torch::value::hl_gauss::NUM_BINS],
                (Kind::Float, self.device),
            ),
            Tensor::ones(&[1, ACTION_COUNT], (Kind::Float, self.device)),
            Tensor::ones(&[1, ACTION_COUNT], (Kind::Float, self.device)),
        )
    }

    pub(super) fn init_from_full_on_device(
        &self,
        price_deltas: &Tensor,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        if price_deltas.device() != self.device || static_features.device() != self.device {
            panic!(
                "init_from_full_on_device requires tensors on {:?}",
                self.device
            );
        }
        let price = self.ensure_batched(price_deltas);
        let static_features = self.ensure_batched(static_features);
        let price = self.cast_inputs(&price);
        let static_features = self.cast_inputs(&static_features);

        if self.variant == ModelVariant::UniformStream {
            self.init_uniform_from_full_on_device(&price, state);
            return self.cached_readout_forward(&static_features, state);
        }

        let reshaped = price.view([TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64]);
        let _ = state.delta_ring.copy_(&reshaped);
        state.ring_pos = 0;
        state.patch_pos = 0;
        let _ = state.patch_buf.zero_();
        state.initialized = true;

        // Full forward pass
        self.forward_on_device(
            &price.view([1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]),
            &static_features,
            false,
        )
    }
}
