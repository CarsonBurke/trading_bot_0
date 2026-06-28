use tch::nn::Init;
use tch::{nn, Tensor};

use crate::torch::model::init::{linear_residual_out, linear_truncated, relu_sq_linear};
use crate::torch::model::rmsnorm::RMSNorm;

pub(in crate::torch::model) struct ScaledFfn {
    ln: RMSNorm,
    fc1: nn::Linear,
    fc2: nn::Linear,
    scale: Tensor,
}

impl ScaledFfn {
    /// Canonical names: `ffn_fc1`, `ffn_fc2`, `mlp_scale`.
    pub(in crate::torch::model) fn new(p: &nn::Path, model_dim: i64, ff_dim: i64) -> Self {
        Self::new_named(p, model_dim, ff_dim, "ffn_fc1", "ffn_fc2", "mlp_scale")
    }

    pub(in crate::torch::model) fn new_named(
        p: &nn::Path,
        model_dim: i64,
        ff_dim: i64,
        fc1_name: &str,
        fc2_name: &str,
        scale_name: &str,
    ) -> Self {
        let ln = RMSNorm::new(model_dim, 1e-6);
        let fc1 = linear_truncated(p, fc1_name, model_dim, ff_dim);
        let fc2 = linear_residual_out(p, fc2_name, ff_dim, model_dim);
        let scale = p.var(scale_name, &[model_dim], Init::Const(1.0));
        Self {
            ln,
            fc1,
            fc2,
            scale,
        }
    }

    pub(in crate::torch::model) fn forward(&self, x: &Tensor) -> Tensor {
        let h = relu_sq_linear(&self.ln.forward_linear(x, &self.fc1), &self.fc2);
        let h = &h * self.scale.to_kind(h.kind()).view([1, 1, -1]);
        x + h
    }
}
