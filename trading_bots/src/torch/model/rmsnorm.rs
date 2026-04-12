use tch::nn::{self, Init, Linear};
use tch::Tensor;

pub(super) struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    pub(super) fn new(p: &nn::Path, dim: i64, eps: f64) -> Self {
        let weight = p.var("weight", &[dim], Init::Const(1.0));
        Self { weight, eps }
    }

    pub(super) fn forward(&self, x: &Tensor) -> Tensor {
        let normalized_shape = [self.weight.size()[0]];
        let weight = if self.weight.kind() == x.kind() {
            self.weight.shallow_clone()
        } else {
            self.weight.to_kind(x.kind())
        };
        x.internal_fused_rms_norm(normalized_shape, Some(&weight), Some(self.eps))
            .0
    }

    pub(super) fn forward_linear(&self, x: &Tensor, linear: &Linear) -> Tensor {
        let x = self.forward(x);
        super::linear_with_same_dtype(&x, linear)
    }

    pub(super) fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub(super) fn eps(&self) -> f64 {
        self.eps
    }
}
