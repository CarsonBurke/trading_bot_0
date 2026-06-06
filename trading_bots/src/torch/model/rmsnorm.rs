use tch::nn::{self, Linear};
use tch::Tensor;

use super::init::linear_with_same_dtype;

pub(in crate::torch::model) struct RMSNorm {
    dim: i64,
    eps: f64,
}

impl RMSNorm {
    pub(in crate::torch::model) fn new(_p: &nn::Path, dim: i64, eps: f64) -> Self {
        Self { dim, eps }
    }

    pub(in crate::torch::model) fn forward(&self, x: &Tensor) -> Tensor {
        x.internal_fused_rms_norm([self.dim], None::<&Tensor>, Some(self.eps))
            .0
    }

    pub(in crate::torch::model) fn forward_linear(&self, x: &Tensor, linear: &Linear) -> Tensor {
        let x = self.forward(x);
        linear_with_same_dtype(&x, linear)
    }
}
