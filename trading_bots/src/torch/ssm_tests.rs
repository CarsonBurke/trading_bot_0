#[cfg(test)]
mod tests {
    use crate::torch::ssm::*;
    use tch::{nn, Kind, Tensor};

    #[test]
    fn test_mamba2_shapes() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let config = Mamba2Config {
            d_model: 64,
            headdim: 32, // 64*2/32 = 4 heads
            d_state: 16,
            chunk_size: 32,
            ..Default::default()
        };
        let mamba = mamba_block_cfg(&vs.root(), config);

        let x = Tensor::randn(&[2, 100, 64], (Kind::Float, tch::Device::Cpu));
        let y = mamba(&x, false);

        assert_eq!(y.size(), vec![2, 100, 64]);
    }

    #[test]
    fn test_mamba2_stack() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let config = Mamba2Config {
            d_model: 32,
            headdim: 16,
            d_state: 16,
            chunk_size: 16,
            ..Default::default()
        };
        let stack = mamba_stack_cfg(&vs.root(), config, 3);

        let x = Tensor::randn(&[1, 50, 32], (Kind::Float, tch::Device::Cpu));
        let y = stack(&x, false);

        assert_eq!(y.size(), vec![1, 50, 32]);
    }

    #[test]
    fn test_mamba2_default_d_state() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let config = Mamba2Config {
            d_model: 64,
            d_state: 128,
            headdim: 64, // 64*2/64 = 2 heads
            chunk_size: 256,
            ..Default::default()
        };
        let mamba = mamba_block_cfg(&vs.root(), config);

        let x = Tensor::randn(&[1, 256, 64], (Kind::Float, tch::Device::Cpu));
        let y = mamba(&x, false);

        assert_eq!(y.size(), vec![1, 256, 64]);
    }

    #[test]
    fn test_mamba2_d_has_hdim() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let config = Mamba2Config {
            d_model: 64,
            headdim: 32,
            d_state: 16,
            d_has_hdim: true,
            ..Default::default()
        };
        let mamba = mamba_block_cfg(&vs.root(), config);

        let x = Tensor::randn(&[2, 50, 64], (Kind::Float, tch::Device::Cpu));
        let y = mamba(&x, false);

        assert_eq!(y.size(), vec![2, 50, 64]);
    }

    #[test]
    fn test_mamba2_multi_groups() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let config = Mamba2Config {
            d_model: 64,
            headdim: 32, // 4 heads
            d_state: 16,
            ngroups: 2, // 2 groups, each controls 2 heads
            ..Default::default()
        };
        let mamba = mamba_block_cfg(&vs.root(), config);

        let x = Tensor::randn(&[2, 100, 64], (Kind::Float, tch::Device::Cpu));
        let y = mamba(&x, false);

        assert_eq!(y.size(), vec![2, 100, 64]);
    }

    #[test]
    fn test_mamba2_step_shape() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let config = Mamba2Config {
            d_model: 64,
            headdim: 32,
            d_state: 16,
            ..Default::default()
        };
        let mamba = Mamba2::new(&vs.root(), config);
        let mut state = mamba.init_state(2, tch::Device::Cpu);

        // Single step input
        let x = Tensor::randn(&[2, 64], (Kind::Float, tch::Device::Cpu));
        let y = mamba.step(&x, &mut state);

        assert_eq!(y.size(), vec![2, 64]);
    }

    #[test]
    fn test_mamba2_step_sequence() {
        // Verify step-by-step produces same output as forward for initial steps
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let d_conv = 4i64;
        let config = Mamba2Config {
            d_model: 32,
            headdim: 16,
            d_state: 8,
            d_conv,
            chunk_size: 64,
            ..Default::default()
        };
        let mamba = Mamba2::new(&vs.root(), config);

        let seq_len = 10;
        let x_seq = Tensor::randn(&[1, seq_len, 32], (Kind::Float, tch::Device::Cpu));

        // Run forward on full sequence
        let y_forward = mamba.forward(&x_seq, false);

        // Run step-by-step
        let mut state = mamba.init_state(1, tch::Device::Cpu);
        let mut y_steps = Vec::new();
        for t in 0..seq_len {
            let x_t = x_seq.narrow(1, t, 1).squeeze_dim(1);
            let y_t = mamba.step(&x_t, &mut state);
            y_steps.push(y_t);
        }
        let y_stepped = Tensor::stack(&y_steps, 1);

        // Check shapes match
        assert_eq!(y_forward.size(), y_stepped.size());

        // After d_conv steps, outputs should be very close (conv buffer filled)
        // Compare from step d_conv onwards
        let y_fwd_tail = y_forward.narrow(1, d_conv, seq_len - d_conv);
        let y_step_tail = y_stepped.narrow(1, d_conv, seq_len - d_conv);
        let diff = (&y_fwd_tail - &y_step_tail).abs().max();
        let max_diff: f64 = diff.double_value(&[]);
        assert!(max_diff < 1e-4, "Max diff {} too large", max_diff);
    }

    #[test]
    fn test_mamba2_state_reset() {
        let vs = nn::VarStore::new(tch::Device::Cpu);
        let config = Mamba2Config {
            d_model: 32,
            headdim: 16,
            d_state: 8,
            ..Default::default()
        };
        let mamba = Mamba2::new(&vs.root(), config);
        let mut state = mamba.init_state(1, tch::Device::Cpu);

        // Run a few steps
        let x = Tensor::randn(&[1, 32], (Kind::Float, tch::Device::Cpu));
        let _ = mamba.step(&x, &mut state);
        let _ = mamba.step(&x, &mut state);

        // State should be non-zero
        let ssm_sum: f64 = state.ssm_state.abs().sum(Kind::Float).double_value(&[]);
        assert!(ssm_sum > 0.0);

        // Reset and verify zeros
        state.reset();
        let ssm_sum_after: f64 = state.ssm_state.abs().sum(Kind::Float).double_value(&[]);
        assert_eq!(ssm_sum_after, 0.0);
    }
}
