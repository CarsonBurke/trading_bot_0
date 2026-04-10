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
        self.uniform_prefix_k.clear();
        self.uniform_prefix_v.clear();
        self.uniform_cached_static_features = None;
        self.initialized = false;
    }
}

impl TradingModel {
    pub fn refresh_stream_state_storage_for_autograd(&self, state: &mut StreamState) {
        if self.variant != super::ModelVariant::Uniform256Stream {
            return;
        }
        state.uniform_layout = &state.uniform_layout + 0.0;
        state.uniform_patch_tokens = &state.uniform_patch_tokens + 0.0;
        state.uniform_live_fill = &state.uniform_live_fill + 0;
        state.uniform_layer0_prefix_hidden = &state.uniform_layer0_prefix_hidden + 0.0;
        state.uniform_layer0_prefix_k = &state.uniform_layer0_prefix_k + 0.0;
        state.uniform_layer0_prefix_v = &state.uniform_layer0_prefix_v + 0.0;
        state.uniform_prefix_k = state.uniform_prefix_k.iter().map(|t| t + 0.0).collect();
        state.uniform_prefix_v = state.uniform_prefix_v.iter().map(|t| t + 0.0).collect();
        state.uniform_cached_static_features = state
            .uniform_cached_static_features
            .as_ref()
            .map(|t| t + 0.0);
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
        let (x_next, k, v) = self.gqa_layers[0].forward_prefix_and_cache(&x, &self.rope);
        state.uniform_layer0_prefix_hidden = x_next;
        state.uniform_layer0_prefix_k = k.index_select(1, &self.gqa_kv_head_index);
        state.uniform_layer0_prefix_v = v.index_select(1, &self.gqa_kv_head_index);
        state.uniform_prefix_k.clear();
        state.uniform_prefix_v.clear();
        state.uniform_cached_static_features = None;
    }

    fn conditioned_prefix_cache_is_fresh(
        &self,
        static_features: &Tensor,
        state: &StreamState,
    ) -> bool {
        state.uniform_prefix_k.len() == self.gqa_layers.len()
            && state.uniform_prefix_v.len() == self.gqa_layers.len()
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
        let mut x = self
            .cross_attn
            .forward(&state.uniform_layer0_prefix_hidden, exo_tokens);
        state.uniform_prefix_k = vec![state.uniform_layer0_prefix_k.shallow_clone()];
        state.uniform_prefix_v = vec![state.uniform_layer0_prefix_v.shallow_clone()];
        for layer in self.gqa_layers.iter().skip(1) {
            let (x_next, k, v) = layer.forward_prefix_and_cache(&x, &self.rope);
            state
                .uniform_prefix_k
                .push(k.index_select(1, &self.gqa_kv_head_index));
            state
                .uniform_prefix_v
                .push(v.index_select(1, &self.gqa_kv_head_index));
            x = x_next;
        }
        state.uniform_cached_static_features = Some(static_features.shallow_clone());
    }

    fn uniform_stream_cached_forward(
        &self,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let batch_size = state.uniform_live_fill.size()[0];
        let (global_static, per_ticker_static) = self.parse_static(static_features, batch_size);
        let exo_tokens = self.build_exo_tokens(&global_static, &per_ticker_static, batch_size);
        if !self.conditioned_prefix_cache_is_fresh(static_features, state) {
            self.rebuild_uniform_conditioned_prefix_cache(static_features, &exo_tokens, state);
        }
        let batch_tokens = batch_size * TICKERS_COUNT;
        let live_token =
            state
                .uniform_patch_tokens
                .narrow(1, super::UNIFORM_STREAM_PATCH_COUNT - 1, 1);
        let kind = live_token.kind();
        let actor_cls = self
            .actor_cls_token
            .to_kind(kind)
            .expand([batch_tokens, 1, self.model_dim], false);
        let critic_cls = self
            .critic_cls_token
            .to_kind(kind)
            .expand([batch_tokens, 1, self.model_dim], false);
        let sde_cls = self
            .sde_cls_token
            .to_kind(kind)
            .expand([batch_tokens, 1, self.model_dim], false);
        let mut x_suffix = Tensor::cat(&[&live_token, &actor_cls, &critic_cls, &sde_cls], 1);
        let prefix_len = super::UNIFORM_STREAM_PATCH_COUNT - 1;
        for (layer_idx, layer) in self.gqa_layers.iter().enumerate() {
            x_suffix = layer.forward_suffix_with_cache(
                &x_suffix,
                &state.uniform_prefix_k[layer_idx],
                &state.uniform_prefix_v[layer_idx],
                &self.rope,
                &self.gqa_kv_head_index,
                prefix_len,
            );
            if layer_idx == 0 {
                x_suffix = self.cross_attn.forward(&x_suffix, &exo_tokens);
            }
            x_suffix = self.maybe_apply_inter_ticker(&x_suffix, layer_idx);
        }
        self.head_from_uniform_suffix(&x_suffix, batch_size)
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
        self.prefill_uniform_prefix_base_cache(state);
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

    pub fn init_stream_state(&self) -> StreamState {
        StreamState {
            delta_ring: Tensor::zeros(
                &[TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64],
                (Kind::Float, self.device),
            ),
            ring_pos: 0,
            patch_buf: Tensor::zeros(
                &[TICKERS_COUNT, self.finest_patch_size],
                (Kind::Float, self.device),
            ),
            patch_pos: 0,
            uniform_layout: Tensor::full(
                [
                    TICKERS_COUNT,
                    super::UNIFORM_STREAM_PATCH_COUNT,
                    super::UNIFORM_STREAM_PATCH_SIZE,
                ],
                f64::NAN,
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_patch_tokens: Tensor::zeros(
                [
                    TICKERS_COUNT,
                    super::UNIFORM_STREAM_PATCH_COUNT,
                    self.model_dim,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_live_fill: Tensor::zeros([1], (Kind::Int64, self.device)),
            uniform_live_fill_host: vec![0; 1],
            uniform_layer0_prefix_hidden: Tensor::zeros(
                [
                    TICKERS_COUNT,
                    super::UNIFORM_STREAM_PATCH_COUNT - 1,
                    self.model_dim,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_layer0_prefix_k: Tensor::zeros(
                [
                    TICKERS_COUNT,
                    super::GQA_NUM_KV_HEADS,
                    super::UNIFORM_STREAM_PATCH_COUNT - 1,
                    self.model_dim / super::GQA_NUM_Q_HEADS,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_layer0_prefix_v: Tensor::zeros(
                [
                    TICKERS_COUNT,
                    super::GQA_NUM_KV_HEADS,
                    super::UNIFORM_STREAM_PATCH_COUNT - 1,
                    self.model_dim / super::GQA_NUM_Q_HEADS,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_prefix_k: Vec::new(),
            uniform_prefix_v: Vec::new(),
            uniform_cached_static_features: None,
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
                &[batch_size * TICKERS_COUNT, self.finest_patch_size],
                (Kind::Float, self.device),
            ),
            patch_pos: 0,
            uniform_layout: Tensor::full(
                [
                    batch_size * TICKERS_COUNT,
                    super::UNIFORM_STREAM_PATCH_COUNT,
                    super::UNIFORM_STREAM_PATCH_SIZE,
                ],
                f64::NAN,
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_patch_tokens: Tensor::zeros(
                [
                    batch_size * TICKERS_COUNT,
                    super::UNIFORM_STREAM_PATCH_COUNT,
                    self.model_dim,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_live_fill: Tensor::zeros([batch_size], (Kind::Int64, self.device)),
            uniform_live_fill_host: vec![0; batch_size as usize],
            uniform_layer0_prefix_hidden: Tensor::zeros(
                [
                    batch_size * TICKERS_COUNT,
                    super::UNIFORM_STREAM_PATCH_COUNT - 1,
                    self.model_dim,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_layer0_prefix_k: Tensor::zeros(
                [
                    batch_size * TICKERS_COUNT,
                    super::GQA_NUM_KV_HEADS,
                    super::UNIFORM_STREAM_PATCH_COUNT - 1,
                    self.model_dim / super::GQA_NUM_Q_HEADS,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_layer0_prefix_v: Tensor::zeros(
                [
                    batch_size * TICKERS_COUNT,
                    super::GQA_NUM_KV_HEADS,
                    super::UNIFORM_STREAM_PATCH_COUNT - 1,
                    self.model_dim / super::GQA_NUM_Q_HEADS,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_prefix_k: Vec::new(),
            uniform_prefix_v: Vec::new(),
            uniform_cached_static_features: None,
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

            let new_deltas = if new_deltas.dim() == 1 {
                new_deltas.unsqueeze(0)
            } else {
                new_deltas.shallow_clone()
            };
            let batch_size = new_deltas.size()[0];
            let state_batch_size = state.delta_ring.size()[0] / TICKERS_COUNT;
            assert_eq!(
                state_batch_size, batch_size,
                "stream state batch size mismatch"
            );

            let rows = batch_size * TICKERS_COUNT;
            let target_kind = self.patch_embed_weight.kind();
            let has_rollover = state
                .uniform_live_fill_host
                .iter()
                .any(|&fill| fill >= super::UNIFORM_STREAM_PATCH_SIZE);

            // Handle rollover: shift layout and patch tokens for envs whose live patch is full
            let cache_dirty = has_rollover;
            if has_rollover {
                let rollover_envs = state
                    .uniform_live_fill_host
                    .iter()
                    .enumerate()
                    .filter_map(|(env_idx, &fill)| {
                        (fill >= super::UNIFORM_STREAM_PATCH_SIZE).then_some(env_idx as i64)
                    })
                    .collect::<Vec<_>>();
                let rollover_rows = rollover_envs
                    .iter()
                    .flat_map(|&env_idx| {
                        (0..TICKERS_COUNT)
                            .map(move |ticker_idx| env_idx * TICKERS_COUNT + ticker_idx)
                    })
                    .collect::<Vec<_>>();
                let rollover_row_idx = Tensor::from_slice(&rollover_rows)
                    .to_kind(Kind::Int64)
                    .to_device(self.device);
                let shifted_layout = Tensor::cat(
                    &[
                        &state
                            .uniform_layout
                            .index_select(0, &rollover_row_idx)
                            .narrow(1, 1, super::UNIFORM_STREAM_PATCH_COUNT - 1),
                        &Tensor::full(
                            [
                                rollover_rows.len() as i64,
                                1,
                                super::UNIFORM_STREAM_PATCH_SIZE,
                            ],
                            f64::NAN,
                            (target_kind, self.device),
                        ),
                    ],
                    1,
                );
                let shifted_patch_tokens = Tensor::cat(
                    &[
                        &state
                            .uniform_patch_tokens
                            .index_select(0, &rollover_row_idx)
                            .narrow(1, 1, super::UNIFORM_STREAM_PATCH_COUNT - 1),
                        &Tensor::zeros(
                            [rollover_rows.len() as i64, 1, self.model_dim],
                            (target_kind, self.device),
                        ),
                    ],
                    1,
                );
                state.uniform_layout = state.uniform_layout.index_copy(
                    0,
                    &rollover_row_idx,
                    &shifted_layout.to_kind(state.uniform_layout.kind()),
                );
                state.uniform_patch_tokens = state.uniform_patch_tokens.index_copy(
                    0,
                    &rollover_row_idx,
                    &shifted_patch_tokens.to_kind(state.uniform_patch_tokens.kind()),
                );
                for &env_idx in &rollover_envs {
                    state.uniform_live_fill_host[env_idx as usize] = 0;
                }
            }

            // Write new delta into layout at current fill position
            let current_fill_host = state.uniform_live_fill_host.clone();
            let current_fill = Tensor::from_slice(&current_fill_host)
                .to_kind(Kind::Int64)
                .to_device(self.device);
            let row_deltas = new_deltas.reshape([rows, 1]);
            let fill_rows = current_fill
                .unsqueeze(1)
                .expand([batch_size, TICKERS_COUNT], false)
                .reshape([rows, 1]);
            let updated_last = state
                .uniform_layout
                .select(1, super::UNIFORM_STREAM_PATCH_COUNT - 1)
                .scatter(1, &fill_rows, &row_deltas);
            let prefix_layout =
                state
                    .uniform_layout
                    .narrow(1, 0, super::UNIFORM_STREAM_PATCH_COUNT - 1);
            state.uniform_layout = Tensor::cat(&[&prefix_layout, &updated_last.unsqueeze(1)], 1);

            // Increment fill counts
            let next_fill_host = current_fill_host
                .iter()
                .map(|&fill| fill + 1)
                .collect::<Vec<_>>();
            state.uniform_live_fill_host = next_fill_host.clone();
            state.uniform_live_fill = Tensor::from_slice(&next_fill_host)
                .to_kind(Kind::Int64)
                .to_device(self.device);

            // Recompute live token via linear projection
            let live_tokens = self.recompute_live_token(state);
            let prefix_tokens =
                state
                    .uniform_patch_tokens
                    .narrow(1, 0, super::UNIFORM_STREAM_PATCH_COUNT - 1);
            state.uniform_patch_tokens =
                Tensor::cat(&[&prefix_tokens, &live_tokens.unsqueeze(1)], 1);
            state.initialized = true;
            if cache_dirty {
                self.prefill_uniform_prefix_base_cache(state);
            }
            return self.uniform_stream_cached_forward(&static_features, state);
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
                &[1, crate::torch::two_hot::NUM_BINS],
                (Kind::Float, self.device),
            ),
            Tensor::zeros(&[1, ACTION_COUNT], (Kind::Float, self.device)),
            Tensor::ones(&[1, ACTION_COUNT], (Kind::Float, self.device)),
        )
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
            let batch_size = price.size()[0];
            let expected_layout = TICKERS_COUNT * super::UNIFORM_STREAM_LAYOUT_LEN;
            assert_eq!(
                price.size()[1],
                expected_layout,
                "Uniform256Stream init expects anchored layout input"
            );
            let layout = price.view([
                batch_size * TICKERS_COUNT,
                super::UNIFORM_STREAM_PATCH_COUNT,
                super::UNIFORM_STREAM_PATCH_SIZE,
            ]);
            let live_fill = layout
                .select(1, super::UNIFORM_STREAM_PATCH_COUNT - 1)
                .isnan()
                .logical_not()
                .sum_dim_intlist([1].as_slice(), false, Kind::Int64);
            state.uniform_layout = layout;
            let patch_tokens = self.patch_embed(
                &state
                    .uniform_layout
                    .view([batch_size * TICKERS_COUNT, super::UNIFORM_STREAM_LAYOUT_LEN]),
            );
            state.uniform_patch_tokens = patch_tokens;
            state.uniform_live_fill = live_fill.shallow_clone();
            state.uniform_live_fill_host =
                Vec::<i64>::try_from(live_fill.to_device(tch::Device::Cpu)).unwrap();
            state.initialized = true;
            self.prefill_uniform_prefix_base_cache(state);

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
