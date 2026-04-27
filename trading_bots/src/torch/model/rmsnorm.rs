use tch::nn::{self, Linear};
use tch::Tensor;

pub(super) struct RMSNorm {
    dim: i64,
    eps: f64,
}

impl RMSNorm {
    pub(super) fn new(_p: &nn::Path, dim: i64, eps: f64) -> Self {
        Self { dim, eps }
    }

    pub(super) fn forward(&self, x: &Tensor) -> Tensor {
        x.internal_fused_rms_norm([self.dim], None::<&Tensor>, Some(self.eps))
            .0
    }

    pub(super) fn forward_linear(&self, x: &Tensor, linear: &Linear) -> Tensor {
        let x = self.forward(x);
        super::linear_with_same_dtype(&x, linear)
    }
}
