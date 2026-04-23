use tch::{Kind, Tensor};

use super::{ModelOutput, StreamState, TradingModel};
use crate::torch::constants::{ACTION_COUNT, PRICE_DELTAS_PER_TICKER, TICKERS_COUNT};

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
        let _ = self.uniform_layer0_prefix_hidden.zero_();
        let _ = self.uniform_layer0_prefix_k.zero_();
        let _ = self.uniform_layer0_prefix_v.zero_();
        let _ = self.uniform_prefix_x0.zero_();
        self.uniform_prefix_k.clear();
        self.uniform_prefix_v.clear();
        self.uniform_cached_static_features = None;
        self.uniform_cached_exo_tokens = None;
        self.initialized = false;
    }
}

impl TradingModel {
    fn ensure_stream_cache_kind(
        &self,
        state: &mut StreamState,
        x0: &Tensor,
        hidden: &Tensor,
        k: &Tensor,
        v: &Tensor,
    ) {
        if state.uniform_prefix_x0.kind() != x0.kind() {
            state.uniform_prefix_x0 = state.uniform_prefix_x0.to_kind(x0.kind());
        }
        if state.uniform_layer0_prefix_hidden.kind() != hidden.kind() {
            state.uniform_layer0_prefix_hidden =
                state.uniform_layer0_prefix_hidden.to_kind(hidden.kind());
        }
        if state.uniform_layer0_prefix_k.kind() != k.kind() {
            state.uniform_layer0_prefix_k = state.uniform_layer0_prefix_k.to_kind(k.kind());
        }
        if state.uniform_layer0_prefix_v.kind() != v.kind() {
            state.uniform_layer0_prefix_v = state.uniform_layer0_prefix_v.to_kind(v.kind());
        }
    }

    pub fn detach_stream_state(&self, state: &mut StreamState) {
        if self.variant != super::ModelVariant::Uniform256Stream {
            return;
        }
        state.uniform_layout = state.uniform_layout.detach();
        state.uniform_patch_tokens = state.uniform_patch_tokens.detach();
        state.uniform_layer0_prefix_hidden = state.uniform_layer0_prefix_hidden.detach();
        state.uniform_layer0_prefix_k = state.uniform_layer0_prefix_k.detach();
        state.uniform_layer0_prefix_v = state.uniform_layer0_prefix_v.detach();
        state.uniform_prefix_x0 = state.uniform_prefix_x0.detach();
        state.uniform_prefix_k = state.uniform_prefix_k.iter().map(|t| t.detach()).collect();
        state.uniform_prefix_v = state.uniform_prefix_v.iter().map(|t| t.detach()).collect();
        state.uniform_cached_static_features = state
            .uniform_cached_static_features
            .as_ref()
            .map(|t| t.detach());
        state.uniform_cached_exo_tokens =
            state.uniform_cached_exo_tokens.as_ref().map(|t| t.detach());
    }

    fn recompute_live_token(&self, state: &StreamState) -> Tensor {
        let live_patch = state
            .uniform_layout
            .select(1, super::UNIFORM_STREAM_PATCH_COUNT - 1);
        let rows = live_patch.size()[0];
        let batch_size = state.uniform_live_fill.size()[0];
        let fill_counts = state
            .uniform_live_fill
            .unsqueeze(1)
            .expand([batch_size, TICKERS_COUNT], false)
            .reshape([rows]);
        self.patch_embed_stream_batch(&live_patch, &fill_counts)
    }

    fn prefill_uniform_prefix_base_cache(&self, state: &mut StreamState) {
        let x = state
            .uniform_patch_tokens
            .narrow(1, 0, super::UNIFORM_STREAM_PATCH_COUNT - 1)
            .shallow_clone();
        let x0 = self.input_ln.forward(&x);
        let (x_next, k, v) = self.gqa_layers[0].forward_prefix_and_cache(&x0, &x0, &self.rope);
        state.uniform_prefix_x0 = x0;
        state.uniform_layer0_prefix_hidden = x_next;
        state.uniform_layer0_prefix_k = k.index_select(1, &self.gqa_kv_head_index);
        state.uniform_layer0_prefix_v = v.index_select(1, &self.gqa_kv_head_index);
        state.uniform_prefix_k.clear();
        state.uniform_prefix_v.clear();
        state.uniform_cached_static_features = None;
        state.uniform_cached_exo_tokens = None;
    }

    fn prefill_uniform_prefix_base_cache_indexed(&self, state: &mut StreamState, row_idx: &Tensor) {
        let x = state.uniform_patch_tokens.index_select(0, row_idx).narrow(
            1,
            0,
            super::UNIFORM_STREAM_PATCH_COUNT - 1,
        );
        let x0 = self.input_ln.forward(&x);
        let (x_next, k, v) = self.gqa_layers[0].forward_prefix_and_cache(&x0, &x0, &self.rope);
        let k = k.index_select(1, &self.gqa_kv_head_index);
        let v = v.index_select(1, &self.gqa_kv_head_index);
        self.ensure_stream_cache_kind(state, &x0, &x_next, &k, &v);
        state.uniform_prefix_x0 = state.uniform_prefix_x0.index_copy(0, row_idx, &x0);
        state.uniform_layer0_prefix_hidden = state
            .uniform_layer0_prefix_hidden
            .index_copy(0, row_idx, &x_next);
        state.uniform_layer0_prefix_k = state.uniform_layer0_prefix_k.index_copy(0, row_idx, &k);
        state.uniform_layer0_prefix_v = state.uniform_layer0_prefix_v.index_copy(0, row_idx, &v);
        state.uniform_prefix_k.clear();
        state.uniform_prefix_v.clear();
        state.uniform_cached_static_features = None;
        state.uniform_cached_exo_tokens = None;
    }

    fn conditioned_prefix_cache_is_fresh(
        &self,
        static_features: &Tensor,
        state: &StreamState,
    ) -> bool {
        state.uniform_prefix_k.len() == self.gqa_layers.len()
            && state.uniform_prefix_v.len() == self.gqa_layers.len()
            && state.uniform_cached_exo_tokens.is_some()
            && state
                .uniform_cached_static_features
                .as_ref()
                .map(|cached| cached.allclose(static_features, 0.0, 0.0, true))
                .unwrap_or(false)
    }

    fn rebuild_uniform_conditioned_prefix_cache(
        &self,
        static_features: &Tensor,
        exo_tokens: &Tensor,
        state: &mut StreamState,
    ) {
        let x0 = &state.uniform_prefix_x0;
        let mut x = self
            .exogenous_ticker_block
            .forward(&state.uniform_layer0_prefix_hidden, exo_tokens);
        state.uniform_prefix_k = vec![state.uniform_layer0_prefix_k.shallow_clone()];
        state.uniform_prefix_v = vec![state.uniform_layer0_prefix_v.shallow_clone()];
        for layer in self.gqa_layers.iter().skip(1) {
            let (x_next, k, v) = layer.forward_prefix_and_cache(&x, x0, &self.rope);
            state
                .uniform_prefix_k
                .push(k.index_select(1, &self.gqa_kv_head_index));
            state
                .uniform_prefix_v
                .push(v.index_select(1, &self.gqa_kv_head_index));
            x = x_next;
        }
        state.uniform_cached_static_features = Some(static_features.shallow_clone());
        state.uniform_cached_exo_tokens = Some(exo_tokens.shallow_clone());
    }

    fn uniform_stream_replay_forward(
        &self,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let batch_size = state.uniform_live_fill.size()[0];
        let (global_static, per_ticker_static) = self.parse_static(static_features, batch_size);
        let exo_tokens = self.build_exo_tokens(&global_static, &per_ticker_static, batch_size);
        let x0 = &state.uniform_prefix_x0;
        let mut prefix_hidden = self
            .exogenous_ticker_block
            .forward(&state.uniform_layer0_prefix_hidden, &exo_tokens);
        let mut prefix_k = Vec::with_capacity(self.gqa_layers.len());
        let mut prefix_v = Vec::with_capacity(self.gqa_layers.len());
        prefix_k.push(state.uniform_layer0_prefix_k.shallow_clone());
        prefix_v.push(state.uniform_layer0_prefix_v.shallow_clone());
        for layer in self.gqa_layers.iter().skip(1) {
            let (x_next, k, v) = layer.forward_prefix_and_cache(&prefix_hidden, x0, &self.rope);
            prefix_k.push(k.index_select(1, &self.gqa_kv_head_index));
            prefix_v.push(v.index_select(1, &self.gqa_kv_head_index));
            prefix_hidden = x_next;
        }

        let live_token =
            state
                .uniform_patch_tokens
                .narrow(1, super::UNIFORM_STREAM_PATCH_COUNT - 1, 1);
        let x0_suffix = self.input_ln.forward(&live_token);
        let mut x_suffix = x0_suffix.shallow_clone();
        let prefix_len = super::UNIFORM_STREAM_PATCH_COUNT - 1;
        for (layer_idx, layer) in self.gqa_layers.iter().enumerate() {
            x_suffix = layer.forward_suffix_with_cache(
                &x_suffix,
                &x0_suffix,
                &prefix_k[layer_idx],
                &prefix_v[layer_idx],
                &self.rope,
                &self.gqa_kv_head_index,
                prefix_len,
            );
            if layer_idx == 0 {
                x_suffix = self.exogenous_ticker_block.forward(&x_suffix, &exo_tokens);
            }
            x_suffix = self.maybe_apply_endogenous_ticker(&x_suffix, layer_idx);
        }
        let x_suffix = self.final_ln.forward(&x_suffix);
        self.head_from_final_hidden(&x_suffix, batch_size)
    }

    fn uniform_stream_cached_forward(
        &self,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let batch_size = state.uniform_live_fill.size()[0];
        let exo_tokens = if self.conditioned_prefix_cache_is_fresh(static_features, state) {
            state
                .uniform_cached_exo_tokens
                .as_ref()
                .unwrap()
                .shallow_clone()
        } else {
            let (global_static, per_ticker_static) = self.parse_static(static_features, batch_size);
            let exo_tokens = self.build_exo_tokens(&global_static, &per_ticker_static, batch_size);
            self.rebuild_uniform_conditioned_prefix_cache(static_features, &exo_tokens, state);
            exo_tokens
        };
        let live_token =
            state
                .uniform_patch_tokens
                .narrow(1, super::UNIFORM_STREAM_PATCH_COUNT - 1, 1);
        let x0_suffix = self.input_ln.forward(&live_token);
        let mut x_suffix = x0_suffix.shallow_clone();
        let prefix_len = super::UNIFORM_STREAM_PATCH_COUNT - 1;
        for (layer_idx, layer) in self.gqa_layers.iter().enumerate() {
            x_suffix = layer.forward_suffix_with_cache(
                &x_suffix,
                &x0_suffix,
                &state.uniform_prefix_k[layer_idx],
                &state.uniform_prefix_v[layer_idx],
                &self.rope,
                &self.gqa_kv_head_index,
                prefix_len,
            );
            if layer_idx == 0 {
                x_suffix = self.exogenous_ticker_block.forward(&x_suffix, &exo_tokens);
            }
            x_suffix = self.maybe_apply_endogenous_ticker(&x_suffix, layer_idx);
        }
        let x_suffix = self.final_ln.forward(&x_suffix);
        self.head_from_final_hidden(&x_suffix, batch_size)
    }

    pub fn forward_stream_state_on_device(
        &self,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let static_features = if static_features.dim() == 1 {
            static_features.unsqueeze(0)
        } else {
            static_features.shallow_clone()
        };
        let static_features = self.cast_inputs(&static_features);
        if self.variant == super::ModelVariant::Uniform256Stream {
            return self.uniform_stream_cached_forward(&static_features, state);
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
        if self.variant != super::ModelVariant::Uniform256Stream || env_indices.is_empty() {
            return;
        }
        let layouts = if layouts.dim() == 1 {
            layouts.unsqueeze(0)
        } else {
            layouts.shallow_clone()
        };
        let layouts = self.cast_inputs(&layouts);
        let expected = TICKERS_COUNT * super::UNIFORM_STREAM_LAYOUT_LEN;
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
            super::UNIFORM_STREAM_PATCH_COUNT,
            super::UNIFORM_STREAM_PATCH_SIZE,
        ]);
        let patch_tokens =
            self.patch_embed(&layouts.view([layouts.size()[0], super::UNIFORM_STREAM_LAYOUT_LEN]));
        let live_fill = layouts
            .select(1, super::UNIFORM_STREAM_PATCH_COUNT - 1)
            .isnan()
            .logical_not()
            .sum_dim_intlist([1].as_slice(), false, Kind::Int64);
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
        self.prefill_uniform_prefix_base_cache_indexed(state, &row_idx);
    }

    pub(crate) fn reset_uniform_stream_envs_from_layout_indexed(
        &self,
        state: &mut StreamState,
        env_idx: &Tensor,
        row_idx: &Tensor,
        layouts: &Tensor,
    ) {
        if self.variant != super::ModelVariant::Uniform256Stream || env_idx.size()[0] == 0 {
            return;
        }
        let layouts = if layouts.dim() == 1 {
            layouts.unsqueeze(0)
        } else {
            layouts.shallow_clone()
        };
        let layouts = self.cast_inputs(&layouts);
        let expected = TICKERS_COUNT * super::UNIFORM_STREAM_LAYOUT_LEN;
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
            super::UNIFORM_STREAM_PATCH_COUNT,
            super::UNIFORM_STREAM_PATCH_SIZE,
        ]);
        let patch_tokens =
            self.patch_embed(&layouts.view([layouts.size()[0], super::UNIFORM_STREAM_LAYOUT_LEN]));
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
        let _ = state.uniform_live_fill.index_fill_(
            0,
            env_idx,
            super::UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL,
        );
        self.prefill_uniform_prefix_base_cache_indexed(state, row_idx);
    }

    pub fn uniform_stream_snapshot(&self, state: &StreamState) -> Tensor {
        state
            .uniform_layout
            .view([
                state.uniform_live_fill.size()[0],
                TICKERS_COUNT * super::UNIFORM_STREAM_LAYOUT_LEN,
            ])
            .to_kind(Kind::Float)
    }

    pub fn reset_uniform_stream_envs(
        &self,
        state: &mut StreamState,
        env_indices: &[usize],
        reset_price_deltas: &[f32],
    ) {
        if self.variant != super::ModelVariant::Uniform256Stream || env_indices.is_empty() {
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
                super::UNIFORM_STREAM_PATCH_COUNT,
                super::UNIFORM_STREAM_PATCH_SIZE,
            ]);
            let patch_tokens =
                self.patch_embed(&layout.view([TICKERS_COUNT, super::UNIFORM_STREAM_LAYOUT_LEN]));
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
                .fill_(super::UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL);
            state.uniform_live_fill_host[*env_idx] = super::UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL;
        }
        self.prefill_uniform_prefix_base_cache(state);
    }

    fn ordered_price_from_ring(&self, state: &StreamState, batch_size: i64) -> Tensor {
        let ring_len = PRICE_DELTAS_PER_TICKER as i64;
        let idx = (Tensor::arange(ring_len, (Kind::Int64, self.device)) + state.ring_pos)
            .remainder(ring_len);
        state
            .delta_ring
            .index_select(1, &idx)
            .view([batch_size, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64])
    }

    fn build_stream_state(&self, batch_size: i64, _cache_conditioned_prefix: bool) -> StreamState {
        let uniform_rows = batch_size * TICKERS_COUNT;
        let (delta_ring, patch_buf) = if self.variant == super::ModelVariant::Uniform256Stream {
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
                    super::UNIFORM_STREAM_PATCH_COUNT,
                    super::UNIFORM_STREAM_PATCH_SIZE,
                ],
                f64::NAN,
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_patch_tokens: Tensor::zeros(
                [
                    uniform_rows,
                    super::UNIFORM_STREAM_PATCH_COUNT,
                    self.model_dim,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_live_fill: Tensor::zeros([batch_size], (Kind::Int64, self.device)),
            uniform_live_fill_host: vec![0; batch_size as usize],
            uniform_layer0_prefix_hidden: Tensor::zeros(
                [
                    uniform_rows,
                    super::UNIFORM_STREAM_PATCH_COUNT - 1,
                    self.model_dim,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_layer0_prefix_k: Tensor::zeros(
                [
                    uniform_rows,
                    super::GQA_NUM_KV_HEADS,
                    super::UNIFORM_STREAM_PATCH_COUNT - 1,
                    self.model_dim / super::GQA_NUM_Q_HEADS,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_layer0_prefix_v: Tensor::zeros(
                [
                    uniform_rows,
                    super::GQA_NUM_KV_HEADS,
                    super::UNIFORM_STREAM_PATCH_COUNT - 1,
                    self.model_dim / super::GQA_NUM_Q_HEADS,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_prefix_x0: Tensor::zeros(
                [
                    uniform_rows,
                    super::UNIFORM_STREAM_PATCH_COUNT - 1,
                    self.model_dim,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_prefix_k: Vec::new(),
            uniform_prefix_v: Vec::new(),
            uniform_cached_static_features: None,
            uniform_cached_exo_tokens: None,
            initialized: false,
        }
    }

    fn init_uniform_from_full_on_device(&self, price: &Tensor, state: &mut StreamState) {
        let batch_size = price.size()[0];
        let expected_layout = TICKERS_COUNT * super::UNIFORM_STREAM_LAYOUT_LEN;
        assert_eq!(
            price.size()[1],
            expected_layout,
            "Uniform256Stream init expects anchored layout input"
        );
        let layout = price
            .view([
                batch_size * TICKERS_COUNT,
                super::UNIFORM_STREAM_PATCH_COUNT,
                super::UNIFORM_STREAM_PATCH_SIZE,
            ])
            .copy();
        let live_fill = layout
            .select(1, super::UNIFORM_STREAM_PATCH_COUNT - 1)
            .isnan()
            .logical_not()
            .sum_dim_intlist([1].as_slice(), false, Kind::Int64);
        state.uniform_layout = layout;
        state.uniform_patch_tokens = self.patch_embed(
            &state
                .uniform_layout
                .view([batch_size * TICKERS_COUNT, super::UNIFORM_STREAM_LAYOUT_LEN]),
        );
        state.uniform_live_fill = live_fill.shallow_clone();
        state.uniform_live_fill_host =
            Vec::<i64>::try_from(live_fill.to_device(tch::Device::Cpu)).unwrap();
        state.initialized = true;
        self.prefill_uniform_prefix_base_cache(state);
    }

    fn step_uniform_stream_state_on_device(
        &self,
        new_deltas: &Tensor,
        static_features: &Tensor,
        state: &mut StreamState,
        replay_mode: bool,
    ) -> ModelOutput {
        let new_deltas = if new_deltas.dim() == 1 {
            new_deltas.unsqueeze(0)
        } else {
            new_deltas.shallow_clone()
        };
        let batch_size = new_deltas.size()[0];
        let state_batch_size = state.uniform_live_fill.size()[0];
        assert_eq!(
            state_batch_size, batch_size,
            "stream state batch size mismatch"
        );

        let rows = batch_size * TICKERS_COUNT;
        let row_deltas = new_deltas.reshape([rows, 1]);
        let history_len = PRICE_DELTAS_PER_TICKER as i64;
        let flat_layout = state
            .uniform_layout
            .view([rows, super::UNIFORM_STREAM_LAYOUT_LEN]);
        let mut shifted_valid = Tensor::zeros(
            [rows, history_len - 1],
            (flat_layout.kind(), flat_layout.device()),
        );
        let _ = shifted_valid.copy_(&flat_layout.narrow(1, 1, history_len - 1));
        let _ = flat_layout
            .narrow(1, 0, history_len - 1)
            .copy_(&shifted_valid);
        let _ = flat_layout.narrow(1, history_len - 1, 1).copy_(&row_deltas);
        state.uniform_patch_tokens = self.patch_embed(&flat_layout);
        state
            .uniform_live_fill_host
            .fill(super::UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL);
        let _ = state
            .uniform_live_fill
            .fill_(super::UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL);
        state.initialized = true;
        self.prefill_uniform_prefix_base_cache(state);
        if replay_mode {
            self.uniform_stream_replay_forward(static_features, state)
        } else {
            self.uniform_stream_cached_forward(static_features, state)
        }
    }

    pub fn init_stream_state(&self) -> StreamState {
        self.build_stream_state(1, true)
    }

    pub fn init_stream_state_batched(&self, batch_size: i64) -> StreamState {
        self.build_stream_state(batch_size, true)
    }

    pub fn init_replay_stream_state_batched(&self, batch_size: i64) -> StreamState {
        self.build_stream_state(batch_size, false)
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

        let raw_full_obs = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
        let layout_full_obs = self.price_input_dim();
        let is_full = match new_deltas.dim() {
            1 => {
                let width = new_deltas.size()[0];
                width == raw_full_obs || width == layout_full_obs
            }
            2 => {
                let width = new_deltas.size()[1];
                width == raw_full_obs || width == layout_full_obs
            }
            _ => false,
        };
        let static_features = if static_features.dim() == 1 {
            static_features.unsqueeze(0)
        } else {
            static_features.shallow_clone()
        };

        if self.variant == super::ModelVariant::Uniform256Stream {
            if is_full {
                return self.init_from_full_on_device(&new_deltas, &static_features, state);
            }
            return self.step_uniform_stream_state_on_device(
                &new_deltas,
                &static_features,
                state,
                false,
            );
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

        // Not enough deltas for a new patch yet; return zeros
        (
            Tensor::zeros(
                &[1, crate::torch::hl_gauss::NUM_BINS],
                (Kind::Float, self.device),
            ),
            Tensor::zeros(&[1, ACTION_COUNT], (Kind::Float, self.device)),
            Tensor::ones(&[1, ACTION_COUNT], (Kind::Float, self.device)),
        )
    }

    pub fn forward_stream_state_on_device_for_replay(
        &self,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let static_features = if static_features.dim() == 1 {
            static_features.unsqueeze(0)
        } else {
            static_features.shallow_clone()
        };
        let static_features = self.cast_inputs(&static_features);
        if self.variant == super::ModelVariant::Uniform256Stream {
            return self.uniform_stream_replay_forward(&static_features, state);
        }
        self.forward_stream_state_on_device(&static_features, state)
    }

    /// Pure-tensor helper: given a layout of shape `[rows, UNIFORM_STREAM_LAYOUT_LEN]`
    /// and new deltas of shape `[rows, 1]` (one delta per ticker row), return a
    /// NEW layout tensor with the oldest delta dropped, the remaining history
    /// shifted left by one, and the new delta appended at position
    /// `PRICE_DELTAS_PER_TICKER - 1`. Trailing padding past `PRICE_DELTAS_PER_TICKER`
    /// is preserved. Does not mutate `layout`.
    pub(crate) fn shift_layout_append_delta(&self, layout: &Tensor, row_deltas: &Tensor) -> Tensor {
        let history_len = PRICE_DELTAS_PER_TICKER as i64;
        let rows = layout.size()[0];
        let layout_len = layout.size()[1];
        let kept = layout.narrow(1, 1, history_len - 1);
        let appended = row_deltas.view([rows, 1]).to_kind(layout.kind());
        if layout_len > history_len {
            let pad = layout.narrow(1, history_len, layout_len - history_len);
            Tensor::cat(&[&kept, &appended, &pad], 1)
        } else {
            Tensor::cat(&[&kept, &appended], 1)
        }
    }

    /// Advance the uniform stream state by one delta step without running the
    /// post-layer-0 forward or the head. Used by replay callers that will
    /// reset some envs immediately afterwards, so the replay forward's output
    /// would be discarded. Non-reset envs still have their layout shifted,
    /// patch tokens recomputed, and layer-0 prefix cache refreshed; reset envs
    /// are subsequently overridden via `reset_uniform_stream_envs_from_layout_indexed`.
    pub fn advance_replay_stream_state(&self, new_deltas: &Tensor, state: &mut StreamState) {
        assert_eq!(
            self.variant,
            super::ModelVariant::Uniform256Stream,
            "advance_replay_stream_state requires uniform stream variant"
        );
        if new_deltas.device() != self.device {
            panic!(
                "advance_replay_stream_state requires tensors on {:?}",
                self.device
            );
        }
        let new_deltas = self.cast_inputs(new_deltas);
        let new_deltas = if new_deltas.dim() == 1 {
            new_deltas.unsqueeze(0)
        } else {
            new_deltas.shallow_clone()
        };
        let batch_size = new_deltas.size()[0];
        let state_batch_size = state.uniform_live_fill.size()[0];
        assert_eq!(
            state_batch_size, batch_size,
            "stream state batch size mismatch"
        );
        let rows = batch_size * TICKERS_COUNT;
        let row_deltas = new_deltas.reshape([rows, 1]);
        let flat_layout = state
            .uniform_layout
            .view([rows, super::UNIFORM_STREAM_LAYOUT_LEN]);
        // Apply shift+append in-place on the state's layout storage. This is the
        // streaming-rollout hot path; we avoid an extra copy by mutating directly.
        let history_len = PRICE_DELTAS_PER_TICKER as i64;
        let mut shifted_valid = Tensor::zeros(
            [rows, history_len - 1],
            (flat_layout.kind(), flat_layout.device()),
        );
        let _ = shifted_valid.copy_(&flat_layout.narrow(1, 1, history_len - 1));
        let _ = flat_layout
            .narrow(1, 0, history_len - 1)
            .copy_(&shifted_valid);
        let _ = flat_layout.narrow(1, history_len - 1, 1).copy_(&row_deltas);
        state.uniform_patch_tokens = self.patch_embed(&flat_layout);
        state
            .uniform_live_fill_host
            .fill(super::UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL);
        let _ = state
            .uniform_live_fill
            .fill_(super::UNIFORM_STREAM_BOOTSTRAP_LIVE_FILL);
        state.initialized = true;
        self.prefill_uniform_prefix_base_cache(state);
    }

    /// Stateless batched replay forward over B pre-built windows.
    ///
    /// `layouts` has shape `[B * TICKERS_COUNT, UNIFORM_STREAM_LAYOUT_LEN]` (per-ticker flat
    /// layout rows, concatenated across B windows), and `static_features` has shape
    /// `[B, STATIC_OBS]`. The returned ModelOutput has batch `B`.
    ///
    /// Semantically equivalent to calling `uniform_stream_replay_forward` once per
    /// window with the appropriate state prefilled; intended for batching PPO
    /// minibatch sub-chunks where the state threading between time steps can be
    /// flattened into the batch dimension.
    pub(crate) fn windowed_replay_forward(
        &self,
        layouts: &Tensor,
        static_features: &Tensor,
        batch_size: i64,
    ) -> ModelOutput {
        assert_eq!(
            self.variant,
            super::ModelVariant::Uniform256Stream,
            "windowed_replay_forward requires uniform stream variant"
        );
        let layouts = self.cast_inputs(layouts);
        let patch_tokens = self.patch_embed(&layouts);
        let static_features = self.cast_inputs(static_features);
        let static_features = if static_features.dim() == 1 {
            static_features.unsqueeze(0)
        } else {
            static_features.shallow_clone()
        };
        let (global_static, per_ticker_static) = self.parse_static(&static_features, batch_size);
        let exo_tokens = self.build_exo_tokens(&global_static, &per_ticker_static, batch_size);

        // Prefix layer-0 prefill: first PATCH_COUNT - 1 tokens per row.
        let prefix_patch = patch_tokens.narrow(1, 0, super::UNIFORM_STREAM_PATCH_COUNT - 1);
        let x0 = self.input_ln.forward(&prefix_patch);
        let (layer0_hidden, layer0_k_raw, layer0_v_raw) =
            self.gqa_layers[0].forward_prefix_and_cache(&x0, &x0, &self.rope);
        let layer0_k = layer0_k_raw.index_select(1, &self.gqa_kv_head_index);
        let layer0_v = layer0_v_raw.index_select(1, &self.gqa_kv_head_index);

        let mut prefix_hidden = self
            .exogenous_ticker_block
            .forward(&layer0_hidden, &exo_tokens);
        let mut prefix_k = Vec::with_capacity(self.gqa_layers.len());
        let mut prefix_v = Vec::with_capacity(self.gqa_layers.len());
        prefix_k.push(layer0_k);
        prefix_v.push(layer0_v);
        for layer in self.gqa_layers.iter().skip(1) {
            let (x_next, k, v) = layer.forward_prefix_and_cache(&prefix_hidden, &x0, &self.rope);
            prefix_k.push(k.index_select(1, &self.gqa_kv_head_index));
            prefix_v.push(v.index_select(1, &self.gqa_kv_head_index));
            prefix_hidden = x_next;
        }

        let live_token = patch_tokens.narrow(1, super::UNIFORM_STREAM_PATCH_COUNT - 1, 1);
        let x0_suffix = self.input_ln.forward(&live_token);
        let mut x_suffix = x0_suffix.shallow_clone();
        let prefix_len = super::UNIFORM_STREAM_PATCH_COUNT - 1;
        for (layer_idx, layer) in self.gqa_layers.iter().enumerate() {
            x_suffix = layer.forward_suffix_with_cache(
                &x_suffix,
                &x0_suffix,
                &prefix_k[layer_idx],
                &prefix_v[layer_idx],
                &self.rope,
                &self.gqa_kv_head_index,
                prefix_len,
            );
            if layer_idx == 0 {
                x_suffix = self.exogenous_ticker_block.forward(&x_suffix, &exo_tokens);
            }
            x_suffix = self.maybe_apply_endogenous_ticker(&x_suffix, layer_idx);
        }
        let x_suffix = self.final_ln.forward(&x_suffix);
        self.head_from_final_hidden(&x_suffix, batch_size)
    }

    pub fn step_on_device_for_replay(
        &self,
        new_deltas: &Tensor,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        if new_deltas.device() != self.device || static_features.device() != self.device {
            panic!(
                "step_on_device_for_replay requires tensors on {:?}",
                self.device
            );
        }
        let new_deltas = self.cast_inputs(new_deltas);
        let static_features = self.cast_inputs(static_features);

        let raw_full_obs = TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64;
        let layout_full_obs = self.price_input_dim();
        let is_full = match new_deltas.dim() {
            1 => {
                let width = new_deltas.size()[0];
                width == raw_full_obs || width == layout_full_obs
            }
            2 => {
                let width = new_deltas.size()[1];
                width == raw_full_obs || width == layout_full_obs
            }
            _ => false,
        };
        let static_features = if static_features.dim() == 1 {
            static_features.unsqueeze(0)
        } else {
            static_features.shallow_clone()
        };

        if self.variant == super::ModelVariant::Uniform256Stream {
            if is_full {
                let price = if new_deltas.dim() == 1 {
                    new_deltas.unsqueeze(0)
                } else {
                    new_deltas.shallow_clone()
                };
                self.init_uniform_from_full_on_device(&price, state);
                return self.uniform_stream_replay_forward(&static_features, state);
            }
            return self.step_uniform_stream_state_on_device(
                &new_deltas,
                &static_features,
                state,
                true,
            );
        }

        self.step_on_device(&new_deltas, &static_features, state)
    }

    fn init_from_full_on_device(
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

        if self.variant == super::ModelVariant::Uniform256Stream {
            self.init_uniform_from_full_on_device(&price, state);
            return self.uniform_stream_cached_forward(&static_features, state);
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
