use tch::nn::{self, Init};
use tch::{Kind, Tensor};

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
        let x_f32 = x.to_kind(Kind::Float);
        let rms = (x_f32.pow_tensor_scalar(2).mean_dim(-1, true, Kind::Float) + self.eps).sqrt();
        (x_f32 / rms * &self.weight).to_kind(x.kind())
    }
}
