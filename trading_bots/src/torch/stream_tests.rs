#[cfg(test)]
mod tests {
    use tch::{nn, Device, Kind, Tensor};

    use crate::torch::constants::{PRICE_DELTAS_PER_TICKER, STATIC_OBSERVATIONS, TICKERS_COUNT};
    use crate::torch::model::{ModelVariant, TradingModel, TradingModelConfig};

    fn assert_close(lhs: &Tensor, rhs: &Tensor, name: &str) {
        let max_diff = (lhs - rhs).abs().max().double_value(&[]);
        assert!(
            lhs.allclose(rhs, 1e-4, 3e-4, false),
            "{} max diff: {}",
            name,
            max_diff
        );
    }

    fn assert_not_close(lhs: &Tensor, rhs: &Tensor, name: &str) {
        let max_diff = (lhs - rhs).abs().max().double_value(&[]);
        assert!(max_diff > 1e-6, "{} max diff: {}", name, max_diff);
    }

    #[test]
    fn fresh_uniform_stream_model_outputs_depend_on_price_history() {
        tch::manual_seed(20260425);

        let vs = nn::VarStore::new(Device::Cpu);
        let model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant: ModelVariant::UniformStream,
            },
        );

        let raw_0 = Tensor::randn(
            [1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64],
            (Kind::Float, Device::Cpu),
        );
        let raw_1 = &raw_0
            + 0.25
                * Tensor::randn(
                    [1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64],
                    (Kind::Float, Device::Cpu),
                );
        let static_features =
            Tensor::randn([1, STATIC_OBSERVATIONS as i64], (Kind::Float, Device::Cpu));

        let out_0 = model.forward_on_device(
            &model.uniform_stream_layout_from_raw_input(&raw_0),
            &static_features,
            false,
        );
        let out_1 = model.forward_on_device(
            &model.uniform_stream_layout_from_raw_input(&raw_1),
            &static_features,
            false,
        );

        assert_not_close(
            &out_0.0,
            &out_1.0,
            "value logits should depend on price history",
        );
        assert_not_close(
            &out_0.1,
            &out_1.1,
            "action mean should depend on price history",
        );
    }

    #[test]
    fn uniform_stream_step_matches_full_forward_after_static_change() {
        tch::manual_seed(42);

        let vs = nn::VarStore::new(Device::Cpu);
        let model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant: ModelVariant::UniformStream,
            },
        );

        let raw = Tensor::randn(
            [1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64],
            (Kind::Float, Device::Cpu),
        );
        let static_0 = Tensor::randn([1, STATIC_OBSERVATIONS as i64], (Kind::Float, Device::Cpu));
        let layout_0 = model.uniform_stream_layout_from_raw_input(&raw);

        let mut stream_state = model.init_stream_state();
        let streamed_0 = model.step_on_device(&layout_0, &static_0, &mut stream_state);
        let full_0 = model.forward_on_device(&layout_0, &static_0, false);
        assert_close(&streamed_0.0, &full_0.0, "init values");
        assert_close(&streamed_0.1, &full_0.1, "init mean");
        assert_close(&streamed_0.2, &full_0.2, "init std");

        let next_delta = Tensor::randn([1, TICKERS_COUNT], (Kind::Float, Device::Cpu));
        let static_1 = Tensor::randn([1, STATIC_OBSERVATIONS as i64], (Kind::Float, Device::Cpu));

        let raw_next = Tensor::cat(
            &[
                &raw.view([TICKERS_COUNT, PRICE_DELTAS_PER_TICKER as i64])
                    .narrow(1, 1, PRICE_DELTAS_PER_TICKER as i64 - 1),
                &next_delta.transpose(0, 1),
            ],
            1,
        )
        .view([1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64]);
        let layout_1 = model.uniform_stream_layout_from_raw_input(&raw_next);

        let streamed_1 = model.step_on_device(&next_delta, &static_1, &mut stream_state);
        let full_1 = model.forward_on_device(&layout_1, &static_1, false);
        assert_close(&streamed_1.0, &full_1.0, "step values");
        assert_close(&streamed_1.1, &full_1.1, "step mean");
        assert_close(&streamed_1.2, &full_1.2, "step std");
    }

    /// Equivalence gate for the batched sub-chunk forward refactor: run a small
    /// "sub-chunk" both sequentially (using `step_on_device_for_replay` like the
    /// PPO training loop) and via the stateless `windowed_replay_forward` over
    /// pre-built windowed layouts. Outputs must match to near-float32 precision.
    #[test]
    fn windowed_replay_matches_sequential_subchunk() {
        tch::manual_seed(4242);

        let batch = 2i64;
        let sub_chunk_len = 4i64;

        let vs = nn::VarStore::new(Device::Cpu);
        let model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant: ModelVariant::UniformStream,
            },
        );

        let raw = Tensor::randn(
            [batch, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64],
            (Kind::Float, Device::Cpu),
        );
        let boundary_layout = model.uniform_stream_layout_from_raw_input(&raw);
        let static_seq = Tensor::randn(
            [sub_chunk_len, batch, STATIC_OBSERVATIONS as i64],
            (Kind::Float, Device::Cpu),
        );
        let delta_seq = Tensor::randn(
            [sub_chunk_len, batch, TICKERS_COUNT],
            (Kind::Float, Device::Cpu),
        );

        // Sequential path: step_on_device_for_replay one step at a time.
        let mut seq_outputs = Vec::with_capacity(sub_chunk_len as usize);
        let mut state = model.init_replay_stream_state_batched(batch);
        // Step 0 uses the boundary layout directly (is_full init path).
        let s0 =
            model.step_on_device_for_replay(&boundary_layout, &static_seq.select(0, 0), &mut state);
        seq_outputs.push(s0);
        for t in 1..sub_chunk_len {
            let st = model.step_on_device_for_replay(
                &delta_seq.select(0, t - 1),
                &static_seq.select(0, t),
                &mut state,
            );
            seq_outputs.push(st);
        }

        // Batched path: build windowed layouts by shift-append, then run
        // windowed_replay_forward with B*T batch dim.
        let layout_rows = batch * TICKERS_COUNT;
        let flat_layout_len = boundary_layout.size()[1] / TICKERS_COUNT;
        let mut current_flat = boundary_layout
            .view([layout_rows, flat_layout_len])
            .shallow_clone();
        let mut windowed_layouts: Vec<Tensor> = Vec::with_capacity(sub_chunk_len as usize);
        // Window 0 = boundary unchanged (t=0 special case matching is_full init).
        windowed_layouts.push(current_flat.shallow_clone());
        for t in 1..sub_chunk_len {
            let deltas_row = delta_seq.select(0, t - 1).reshape([layout_rows, 1]);
            current_flat = model.shift_layout_append_delta(&current_flat, &deltas_row);
            windowed_layouts.push(current_flat.shallow_clone());
        }
        // Stack into [T, layout_rows, flat_layout_len] then permute to [batch, T, TICKERS, flat_layout_len].
        // Effective batch dim ordering: (b, t) -> b * T + t; per-ticker rows go (b, t, ticker).
        let windowed = Tensor::stack(&windowed_layouts, 0)
            .view([sub_chunk_len, batch, TICKERS_COUNT, flat_layout_len])
            .permute([1, 0, 2, 3])
            .contiguous()
            .view([batch * sub_chunk_len * TICKERS_COUNT, flat_layout_len]);
        let static_windowed = static_seq
            .permute([1, 0, 2])
            .contiguous()
            .view([batch * sub_chunk_len, STATIC_OBSERVATIONS as i64]);
        let batched =
            model.windowed_replay_forward(&windowed, &static_windowed, batch * sub_chunk_len);

        // Reshape batched outputs to [T, batch, ...] and compare per step.
        let bt = batch * sub_chunk_len;
        let value_logits_bt = batched.0.view([batch, sub_chunk_len, -1]);
        let action_mean_bt = batched.1.view([batch, sub_chunk_len, -1]);
        let action_std_bt = batched.2.view([batch, sub_chunk_len, -1]);
        for t in 0..sub_chunk_len {
            let seq = &seq_outputs[t as usize];
            let batched_v = value_logits_bt.select(1, t).contiguous();
            let batched_m = action_mean_bt.select(1, t).contiguous();
            let batched_s = action_std_bt.select(1, t).contiguous();
            assert_close(&seq.0, &batched_v, &format!("value_logits t={}", t));
            assert_close(&seq.1, &batched_m, &format!("action_mean t={}", t));
            assert_close(&seq.2, &batched_s, &format!("action_std t={}", t));
        }
        let _ = bt;
    }

    /// Same equivalence check as `windowed_replay_matches_sequential_subchunk`
    /// but with a simulated reset at step t=2: the reset-env's layout is
    /// overwritten with a fresh layout from the "bank" at that step. Ensures
    /// the batched builder's post-shift index_copy reset matches the sequential
    /// advance→reset→forward ordering.
    #[test]
    fn windowed_replay_matches_sequential_subchunk_with_reset() {
        tch::manual_seed(777);

        let batch = 2i64;
        let sub_chunk_len = 4i64;
        let reset_at = 2i64; // reset slot fired at step reset_at-1, applied to window reset_at

        let vs = nn::VarStore::new(Device::Cpu);
        let model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant: ModelVariant::UniformStream,
            },
        );

        let raw = Tensor::randn(
            [batch, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64],
            (Kind::Float, Device::Cpu),
        );
        let boundary_layout = model.uniform_stream_layout_from_raw_input(&raw);
        let static_seq = Tensor::randn(
            [sub_chunk_len, batch, STATIC_OBSERVATIONS as i64],
            (Kind::Float, Device::Cpu),
        );
        let delta_seq = Tensor::randn(
            [sub_chunk_len, batch, TICKERS_COUNT],
            (Kind::Float, Device::Cpu),
        );

        // Reset bank: one reset layout for env 0, applied at step reset_at-1 (so it
        // materializes in window reset_at).
        let reset_raw = Tensor::randn(
            [1, TICKERS_COUNT * PRICE_DELTAS_PER_TICKER as i64],
            (Kind::Float, Device::Cpu),
        );
        let reset_layout_bank = model.uniform_stream_layout_from_raw_input(&reset_raw);
        let reset_env_idx = 0i64;

        // Sequential path with reset.
        let mut seq_outputs = Vec::with_capacity(sub_chunk_len as usize);
        let mut state = model.init_replay_stream_state_batched(batch);
        let s0 =
            model.step_on_device_for_replay(&boundary_layout, &static_seq.select(0, 0), &mut state);
        seq_outputs.push(s0);
        for t in 1..sub_chunk_len {
            let prev_deltas = delta_seq.select(0, t - 1);
            let cur_static = static_seq.select(0, t);
            if t == reset_at {
                // Mirror sequential: advance, reset env 0, forward.
                model.advance_replay_stream_state(&prev_deltas, &mut state);
                let env_idx_t = Tensor::from_slice(&[reset_env_idx]).to_kind(Kind::Int64);
                let row_idx = Tensor::from_slice(
                    &(0..TICKERS_COUNT)
                        .map(|i| reset_env_idx * TICKERS_COUNT + i)
                        .collect::<Vec<_>>(),
                )
                .to_kind(Kind::Int64);
                model.reset_uniform_stream_envs_from_layout_indexed(
                    &mut state,
                    &env_idx_t,
                    &row_idx,
                    &reset_layout_bank,
                );
                let st = model.forward_stream_state_on_device_for_replay(&cur_static, &mut state);
                seq_outputs.push(st);
            } else {
                let st = model.step_on_device_for_replay(&prev_deltas, &cur_static, &mut state);
                seq_outputs.push(st);
            }
        }

        // Batched path with reset.
        let layout_rows = batch * TICKERS_COUNT;
        let flat_layout_len = boundary_layout.size()[1] / TICKERS_COUNT;
        let mut current = boundary_layout
            .view([layout_rows, flat_layout_len])
            .shallow_clone();
        let mut windowed = Vec::with_capacity(sub_chunk_len as usize);
        windowed.push(current.shallow_clone());
        for t in 1..sub_chunk_len {
            let prev_deltas = delta_seq.select(0, t - 1);
            let row_deltas = prev_deltas.reshape([layout_rows, 1]);
            current = model.shift_layout_append_delta(&current, &row_deltas);
            if t == reset_at {
                let row_idx = Tensor::from_slice(
                    &(0..TICKERS_COUNT)
                        .map(|i| reset_env_idx * TICKERS_COUNT + i)
                        .collect::<Vec<_>>(),
                )
                .to_kind(Kind::Int64);
                current = current.index_copy(
                    0,
                    &row_idx,
                    &reset_layout_bank.view([TICKERS_COUNT, flat_layout_len]),
                );
            }
            windowed.push(current.shallow_clone());
        }

        let windowed_stack = Tensor::stack(&windowed, 0)
            .view([sub_chunk_len, batch, TICKERS_COUNT, flat_layout_len])
            .permute([1, 0, 2, 3])
            .contiguous()
            .view([batch * sub_chunk_len * TICKERS_COUNT, flat_layout_len]);
        let static_windowed = static_seq
            .permute([1, 0, 2])
            .contiguous()
            .view([batch * sub_chunk_len, STATIC_OBSERVATIONS as i64]);
        let batched =
            model.windowed_replay_forward(&windowed_stack, &static_windowed, batch * sub_chunk_len);

        let value_logits_bt = batched.0.view([batch, sub_chunk_len, -1]);
        let action_mean_bt = batched.1.view([batch, sub_chunk_len, -1]);
        let action_std_bt = batched.2.view([batch, sub_chunk_len, -1]);
        for t in 0..sub_chunk_len {
            let seq = &seq_outputs[t as usize];
            let bv = value_logits_bt.select(1, t).contiguous();
            let bm = action_mean_bt.select(1, t).contiguous();
            let bs = action_std_bt.select(1, t).contiguous();
            assert_close(&seq.0, &bv, &format!("reset value_logits t={}", t));
            assert_close(&seq.1, &bm, &format!("reset action_mean t={}", t));
            assert_close(&seq.2, &bs, &format!("reset action_std t={}", t));
        }
    }
}
