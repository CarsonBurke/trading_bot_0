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

    #[test]
    fn uniform_stream_step_matches_full_forward_after_static_change() {
        tch::manual_seed(42);

        let vs = nn::VarStore::new(Device::Cpu);
        let model = TradingModel::new_with_config(
            &vs.root(),
            TradingModelConfig {
                variant: ModelVariant::Uniform256Stream,
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
}
