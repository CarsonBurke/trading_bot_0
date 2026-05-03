use tch::{Kind, Tensor};

pub(in crate::torch::model) const ROPE_DIMS: i64 = 16;

fn rotate_half(x: &Tensor) -> Tensor {
    let last_dim = *x.size().last().unwrap();
    let half = last_dim / 2;
    let x1 = x.narrow(-1, 0, half);
    let x2 = x.narrow(-1, half, half);
    Tensor::cat(&[&(-&x2), &x1], -1)
}

pub(in crate::torch::model) struct RotaryEmbedding {
    cos_cached: Tensor, // [max_seq_len, rope_dims]
    sin_cached: Tensor, // [max_seq_len, rope_dims]
    rope_dims: i64,
}

impl RotaryEmbedding {
    pub(in crate::torch::model) fn new(
        max_seq_len: i64,
        head_dim: i64,
        rope_dims: i64,
        device: tch::Device,
    ) -> Self {
        let rd = rope_dims.min(head_dim);
        let half_rd = rd / 2;
        let exponents = Tensor::arange(half_rd, (Kind::Float, device)) * (2.0 / rd as f64);
        let inv_freq = (exponents * -(10000.0_f64.ln())).exp();
        let positions = Tensor::arange(max_seq_len, (Kind::Float, device));
        let angles = positions.unsqueeze(1) * inv_freq.unsqueeze(0);
        let cos_half = angles.cos();
        let sin_half = angles.sin();
        Self {
            cos_cached: Tensor::cat(&[&cos_half, &cos_half], -1).set_requires_grad(false),
            sin_cached: Tensor::cat(&[&sin_half, &sin_half], -1).set_requires_grad(false),
            rope_dims: rd,
        }
    }

    pub(in crate::torch::model) fn apply_from(&self, x: &Tensor, offset: i64) -> Tensor {
        // x: [batch, heads, seq_len, head_dim]
        let seq_len = x.size()[2];
        let head_dim = *x.size().last().unwrap();
        let cos = self.cos_cached.narrow(0, offset, seq_len).to_kind(x.kind());
        let sin = self.sin_cached.narrow(0, offset, seq_len).to_kind(x.kind());
        self.apply_with_cached(x, &cos, &sin, head_dim)
    }

    pub(in crate::torch::model) fn apply_positions(
        &self,
        x: &Tensor,
        positions: &Tensor,
    ) -> Tensor {
        let head_dim = *x.size().last().unwrap();
        let positions = positions.to_kind(Kind::Int64).to_device(x.device());
        let cos = self
            .cos_cached
            .index_select(0, &positions)
            .to_kind(x.kind());
        let sin = self
            .sin_cached
            .index_select(0, &positions)
            .to_kind(x.kind());
        self.apply_with_cached(x, &cos, &sin, head_dim)
    }

    fn apply_with_cached(&self, x: &Tensor, cos: &Tensor, sin: &Tensor, head_dim: i64) -> Tensor {
        if self.rope_dims < head_dim {
            let x_rope = x.narrow(-1, 0, self.rope_dims);
            let x_pass = x.narrow(-1, self.rope_dims, head_dim - self.rope_dims);
            let rotated = &x_rope * cos + rotate_half(&x_rope) * sin;
            Tensor::cat(&[&rotated, &x_pass], -1)
        } else {
            x * cos + rotate_half(x) * sin
        }
    }
}
