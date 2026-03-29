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
        let _ = self.uniform_live_conv_state.zero_();
        let _ = self.uniform_live_sum.zero_();
        let _ = self.uniform_live_sum_sq.zero_();
        let _ = self.uniform_live_first.zero_();
        let _ = self.uniform_live_last.zero_();
        self.uniform_prefix_k.clear();
        self.uniform_prefix_v.clear();
        self.initialized = false;
    }
}

impl TradingModel {
    fn patch_embed_stream_single_patch(&self, patch_vals: &Tensor) -> Tensor {
        self.patch_embed_stream_batch(patch_vals)
    }

    fn rebuild_uniform_live_state(&self, state: &mut StreamState) {
        let live_patch = state
            .uniform_layout
            .select(1, super::UNIFORM_STREAM_PATCH_COUNT - 1);
        let rows = live_patch.size()[0];
        let target_kind = self.patch_embed_weight.kind();
        let clean_vals = live_patch.nan_to_num(0.0, None, None).to_kind(target_kind);
        let clean_vals_f = clean_vals.to_kind(Kind::Float);
        let live_fill_rows_host = state
            .uniform_live_fill_host
            .iter()
            .flat_map(|&fill| std::iter::repeat_n(fill, TICKERS_COUNT as usize))
            .collect::<Vec<_>>();
        let live_fill_rows = Tensor::from_slice(&live_fill_rows_host).to_device(self.device);
        let counts_f = live_fill_rows.to_kind(Kind::Float).unsqueeze(-1);
        let _ = state.uniform_live_sum.copy_(&clean_vals_f.sum_dim_intlist(
            [1].as_slice(),
            true,
            Kind::Float,
        ));
        let _ =
            state
                .uniform_live_sum_sq
                .copy_(&clean_vals_f.pow_tensor_scalar(2.0).sum_dim_intlist(
                    [1].as_slice(),
                    true,
                    Kind::Float,
                ));
        let _ = state
            .uniform_live_first
            .copy_(&clean_vals_f.narrow(1, 0, 1));
        let last_idx = live_fill_rows.clamp_min(1).g_add_scalar(-1).unsqueeze(-1);
        let _ = state
            .uniform_live_last
            .copy_(&clean_vals_f.gather(1, &last_idx, false));
        let lifted = clean_vals
            .unsqueeze(-1)
            .reshape([rows * super::UNIFORM_STREAM_PATCH_SIZE, 1])
            .apply(&self.patch_stream_lift)
            .reshape([
                rows,
                super::UNIFORM_STREAM_PATCH_SIZE,
                super::STREAM_PATCH_INNER_DIM,
            ]);
        let _ = state.uniform_live_conv_state.zero_();
        for row in 0..rows {
            let fill = live_fill_rows_host[row as usize];
            let take = fill.min(super::STREAM_PATCH_CONV_KERNEL);
            if take <= 0 {
                continue;
            }
            let src_start = fill - take;
            let dst_start = super::STREAM_PATCH_CONV_KERNEL - take;
            let src = lifted.get(row).narrow(0, src_start, take).transpose(0, 1);
            let _ = state
                .uniform_live_conv_state
                .get(row)
                .narrow(1, dst_start, take)
                .copy_(&src);
        }
        let counts_safe = counts_f.clamp_min(1.0);
        let mean = &state.uniform_live_sum / &counts_safe;
        let var = (&state.uniform_live_sum_sq / &counts_safe - mean.pow_tensor_scalar(2.0))
            .clamp_min(0.0);
        let std = (var + 1e-5).sqrt();
        let slope = &state.uniform_live_last - &state.uniform_live_first;
        let fill_fraction = counts_f / super::UNIFORM_STREAM_PATCH_SIZE as f64;
        let conv_w = self.patch_stream_conv_w.squeeze_dim(1).unsqueeze(0);
        let final_hidden = (&state.uniform_live_conv_state * &conv_w)
            .sum_dim_intlist([2].as_slice(), false, target_kind)
            .g_add(&self.patch_stream_conv_b)
            .silu()
            .apply(&self.patch_stream_mix);
        let scalar = Tensor::cat(&[&mean, &std, &slope, &fill_fraction], 1)
            .to_kind(target_kind)
            .apply(&self.patch_stream_scalar_proj);
        let live_tokens = final_hidden + scalar;
        let _ = state
            .uniform_patch_tokens
            .select(1, super::UNIFORM_STREAM_PATCH_COUNT - 1)
            .copy_(&live_tokens);
    }

    fn prefill_uniform_prefix_cache(&self, state: &mut StreamState) {
        let mut x = state
            .uniform_patch_tokens
            .narrow(1, 0, super::UNIFORM_STREAM_PATCH_COUNT - 1)
            .shallow_clone();
        state.uniform_prefix_k.clear();
        state.uniform_prefix_v.clear();
        for layer in &self.gqa_layers {
            let (x_next, k, v) = layer.forward_prefix_and_cache(&x, &self.rope);
            state
                .uniform_prefix_k
                .push(k.index_select(1, &self.gqa_kv_head_index));
            state
                .uniform_prefix_v
                .push(v.index_select(1, &self.gqa_kv_head_index));
            x = x_next;
        }
    }

    fn uniform_stream_cached_forward(
        &self,
        static_features: &Tensor,
        state: &mut StreamState,
    ) -> ModelOutput {
        let batch_size = state.uniform_live_fill.size()[0];
        if state.uniform_prefix_k.len() != self.gqa_layers.len() {
            self.prefill_uniform_prefix_cache(state);
        }
        let batch_tokens = batch_size * TICKERS_COUNT;
        let live_token =
            state
                .uniform_patch_tokens
                .narrow(1, super::UNIFORM_STREAM_PATCH_COUNT - 1, 1);
        let kind = live_token.kind();
        let (global_static, per_ticker_static) = self.parse_static(static_features, batch_size);
        let tail_cond = self.tail_condition(&global_static, &per_ticker_static, batch_size, kind);
        let actor_cls = self
            .actor_cls_token
            .to_kind(kind)
            .expand([batch_tokens, 1, self.model_dim], false)
            + &tail_cond;
        let critic_cls = self
            .critic_cls_token
            .to_kind(kind)
            .expand([batch_tokens, 1, self.model_dim], false)
            + &tail_cond;
        let sde_cls = self
            .sde_cls_token
            .to_kind(kind)
            .expand([batch_tokens, 1, self.model_dim], false)
            + &tail_cond;
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
        }
        self.head_from_uniform_suffix(&x_suffix, batch_size)
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
        self.rebuild_uniform_live_state(state);
        self.prefill_uniform_prefix_cache(state);
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
            uniform_live_conv_state: Tensor::zeros(
                [
                    TICKERS_COUNT,
                    super::STREAM_PATCH_INNER_DIM,
                    super::STREAM_PATCH_CONV_KERNEL,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_live_sum: Tensor::zeros([TICKERS_COUNT, 1], (Kind::Float, self.device)),
            uniform_live_sum_sq: Tensor::zeros([TICKERS_COUNT, 1], (Kind::Float, self.device)),
            uniform_live_first: Tensor::zeros([TICKERS_COUNT, 1], (Kind::Float, self.device)),
            uniform_live_last: Tensor::zeros([TICKERS_COUNT, 1], (Kind::Float, self.device)),
            uniform_prefix_k: Vec::new(),
            uniform_prefix_v: Vec::new(),
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
            uniform_live_conv_state: Tensor::zeros(
                [
                    batch_size * TICKERS_COUNT,
                    super::STREAM_PATCH_INNER_DIM,
                    super::STREAM_PATCH_CONV_KERNEL,
                ],
                (self.patch_embed_weight.kind(), self.device),
            ),
            uniform_live_sum: Tensor::zeros(
                [batch_size * TICKERS_COUNT, 1],
                (Kind::Float, self.device),
            ),
            uniform_live_sum_sq: Tensor::zeros(
                [batch_size * TICKERS_COUNT, 1],
                (Kind::Float, self.device),
            ),
            uniform_live_first: Tensor::zeros(
                [batch_size * TICKERS_COUNT, 1],
                (Kind::Float, self.device),
            ),
            uniform_live_last: Tensor::zeros(
                [batch_size * TICKERS_COUNT, 1],
                (Kind::Float, self.device),
            ),
            uniform_prefix_k: Vec::new(),
            uniform_prefix_v: Vec::new(),
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

            let conv_shift_idx = &self.uniform_conv_shift_idx;
            let rows = batch_size * TICKERS_COUNT;
            let target_kind = self.patch_embed_weight.kind();
            let live_fill = state.uniform_live_fill.shallow_clone();
            let has_rollover = state
                .uniform_live_fill_host
                .iter()
                .any(|&fill| fill >= super::UNIFORM_STREAM_PATCH_SIZE);
            if !has_rollover {
                let row_deltas = new_deltas.reshape([rows, 1]);
                let row_deltas_f = row_deltas.to_kind(Kind::Float);
                let fill_rows = live_fill
                    .unsqueeze(1)
                    .expand([batch_size, TICKERS_COUNT], false)
                    .reshape([rows, 1]);
                let updated_last = state
                    .uniform_layout
                    .select(1, super::UNIFORM_STREAM_PATCH_COUNT - 1)
                    .scatter(1, &fill_rows, &row_deltas);
                let _ = state
                    .uniform_layout
                    .select(1, super::UNIFORM_STREAM_PATCH_COUNT - 1)
                    .copy_(&updated_last);
                let first_mask = fill_rows.eq(0).to_kind(Kind::Float);
                let first_keep = Tensor::ones_like(&first_mask) - &first_mask;
                let _ = state.uniform_live_first.copy_(
                    &(&state.uniform_live_first * &first_keep + &row_deltas_f * &first_mask),
                );
                let _ = state.uniform_live_last.copy_(&row_deltas_f);
                let _ = state
                    .uniform_live_sum
                    .copy_(&(&state.uniform_live_sum + &row_deltas_f));
                let _ = state
                    .uniform_live_sum_sq
                    .copy_(&(&state.uniform_live_sum_sq + row_deltas_f.pow_tensor_scalar(2.0)));
                let lifted = row_deltas
                    .reshape([rows, 1])
                    .apply(&self.patch_stream_lift)
                    .reshape([rows, super::STREAM_PATCH_INNER_DIM, 1]);
                let kept_state = state
                    .uniform_live_conv_state
                    .index_select(2, &conv_shift_idx);
                let new_conv_state = Tensor::cat(&[&kept_state, &lifted], 2);
                let _ = state.uniform_live_conv_state.copy_(&new_conv_state);
                let next_fill = &live_fill + 1;
                let _ = state.uniform_live_fill.copy_(&next_fill);
                for fill in &mut state.uniform_live_fill_host {
                    *fill += 1;
                }
                let counts_f = next_fill
                    .unsqueeze(1)
                    .expand([batch_size, TICKERS_COUNT], false)
                    .reshape([rows, 1])
                    .to_kind(Kind::Float);
                let mean = &state.uniform_live_sum / &counts_f;
                let var = (&state.uniform_live_sum_sq / &counts_f - mean.pow_tensor_scalar(2.0))
                    .clamp_min(0.0);
                let std = (var + 1e-5).sqrt();
                let slope = &state.uniform_live_last - &state.uniform_live_first;
                let fill_fraction = counts_f / super::UNIFORM_STREAM_PATCH_SIZE as f64;
                let conv_w = self.patch_stream_conv_w.squeeze_dim(1).unsqueeze(0);
                let final_hidden = (&state.uniform_live_conv_state * &conv_w)
                    .sum_dim_intlist([2].as_slice(), false, target_kind)
                    .g_add(&self.patch_stream_conv_b)
                    .silu()
                    .apply(&self.patch_stream_mix);
                let scalar = Tensor::cat(&[&mean, &std, &slope, &fill_fraction], 1)
                    .to_kind(target_kind)
                    .apply(&self.patch_stream_scalar_proj);
                let live_tokens = final_hidden + scalar;
                let _ = state
                    .uniform_patch_tokens
                    .select(1, super::UNIFORM_STREAM_PATCH_COUNT - 1)
                    .copy_(&live_tokens);
                state.initialized = true;
                return self.uniform_stream_cached_forward(&static_features, state);
            }

            let mut cache_dirty = false;
            let shift_idx = &self.uniform_patch_shift_idx;
            for env_idx in 0..batch_size {
                let live_fill = state.uniform_live_fill_host[env_idx as usize];
                if live_fill >= super::UNIFORM_STREAM_PATCH_SIZE {
                    for ticker_idx in 0..TICKERS_COUNT {
                        let row = env_idx * TICKERS_COUNT + ticker_idx;
                        let kept = state.uniform_layout.get(row).index_select(0, &shift_idx);
                        let _ = state
                            .uniform_layout
                            .get(row)
                            .narrow(0, 0, super::UNIFORM_STREAM_PATCH_COUNT - 1)
                            .copy_(&kept);
                        let _ = state
                            .uniform_layout
                            .get(row)
                            .get(super::UNIFORM_STREAM_PATCH_COUNT - 1)
                            .fill_(f64::NAN);
                        let kept_tokens = state
                            .uniform_patch_tokens
                            .get(row)
                            .index_select(0, &shift_idx);
                        let _ = state
                            .uniform_patch_tokens
                            .get(row)
                            .narrow(0, 0, super::UNIFORM_STREAM_PATCH_COUNT - 1)
                            .copy_(&kept_tokens);
                        let _ = state
                            .uniform_patch_tokens
                            .get(row)
                            .get(super::UNIFORM_STREAM_PATCH_COUNT - 1)
                            .zero_();
                    }
                    state.uniform_live_fill_host[env_idx as usize] = 0;
                    cache_dirty = true;
                }
                let cur_fill = state.uniform_live_fill_host[env_idx as usize];
                for ticker_idx in 0..TICKERS_COUNT {
                    let row = env_idx * TICKERS_COUNT + ticker_idx;
                    let delta = new_deltas.get(env_idx).narrow(0, ticker_idx, 1);
                    let delta_f = delta.to_kind(Kind::Float);
                    let _ = state
                        .uniform_layout
                        .get(row)
                        .get(super::UNIFORM_STREAM_PATCH_COUNT - 1)
                        .narrow(0, cur_fill, 1)
                        .copy_(&delta);
                    if cur_fill == 0 {
                        let _ = state.uniform_live_first.get(row).copy_(&delta_f);
                    }
                    let _ = state.uniform_live_last.get(row).copy_(&delta_f);
                    let _ = state.uniform_live_sum.get(row).g_add_(&delta_f);
                    let _ = state
                        .uniform_live_sum_sq
                        .get(row)
                        .g_add_(&delta_f.pow_tensor_scalar(2.0));
                    let lifted = delta
                        .reshape([1, 1])
                        .apply(&self.patch_stream_lift)
                        .reshape([super::STREAM_PATCH_INNER_DIM, 1]);
                    let kept_state = state
                        .uniform_live_conv_state
                        .get(row)
                        .index_select(1, &conv_shift_idx);
                    let _ = state
                        .uniform_live_conv_state
                        .get(row)
                        .narrow(1, 0, super::STREAM_PATCH_CONV_KERNEL - 1)
                        .copy_(&kept_state);
                    let _ = state
                        .uniform_live_conv_state
                        .get(row)
                        .narrow(1, super::STREAM_PATCH_CONV_KERNEL - 1, 1)
                        .copy_(&lifted);
                }
                state.uniform_live_fill_host[env_idx as usize] = cur_fill + 1;
            }
            let next_fill =
                Tensor::from_slice(&state.uniform_live_fill_host).to_device(self.device);
            let _ = state.uniform_live_fill.copy_(&next_fill);
            if cache_dirty {
                self.rebuild_uniform_live_state(state);
            } else {
                let rows = batch_size * TICKERS_COUNT;
                let target_kind = self.patch_embed_weight.kind();
                let counts_f = state
                    .uniform_live_fill
                    .unsqueeze(1)
                    .expand([batch_size, TICKERS_COUNT], false)
                    .reshape([rows, 1])
                    .to_kind(Kind::Float);
                let counts_safe = counts_f.clamp_min(1.0);
                let mean = &state.uniform_live_sum / &counts_safe;
                let var = (&state.uniform_live_sum_sq / &counts_safe - mean.pow_tensor_scalar(2.0))
                    .clamp_min(0.0);
                let std = (var + 1e-5).sqrt();
                let slope = &state.uniform_live_last - &state.uniform_live_first;
                let fill_fraction = counts_f / super::UNIFORM_STREAM_PATCH_SIZE as f64;
                let conv_w = self.patch_stream_conv_w.squeeze_dim(1).unsqueeze(0);
                let final_hidden = (&state.uniform_live_conv_state * &conv_w)
                    .sum_dim_intlist([2].as_slice(), false, target_kind)
                    .g_add(&self.patch_stream_conv_b)
                    .silu()
                    .apply(&self.patch_stream_mix);
                let scalar = Tensor::cat(&[&mean, &std, &slope, &fill_fraction], 1)
                    .to_kind(target_kind)
                    .apply(&self.patch_stream_scalar_proj);
                let live_tokens = final_hidden + scalar;
                let _ = state
                    .uniform_patch_tokens
                    .select(1, super::UNIFORM_STREAM_PATCH_COUNT - 1)
                    .copy_(&live_tokens);
            }
            state.initialized = true;
            if cache_dirty {
                self.prefill_uniform_prefix_cache(state);
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
            let _ = state.uniform_layout.copy_(&layout);
            let patch_tokens = self.patch_embed(
                &layout.view([batch_size * TICKERS_COUNT, super::UNIFORM_STREAM_LAYOUT_LEN]),
            );
            let _ = state.uniform_patch_tokens.copy_(&patch_tokens);
            let _ = state.uniform_live_fill.copy_(&live_fill);
            state.uniform_live_fill_host =
                Vec::<i64>::try_from(live_fill.to_device(tch::Device::Cpu)).unwrap();
            state.initialized = true;
            self.rebuild_uniform_live_state(state);
            self.prefill_uniform_prefix_cache(state);

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
