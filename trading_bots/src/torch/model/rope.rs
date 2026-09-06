use tch::{Kind, Tensor};

pub(in crate::torch::model) const ROPE_DIMS: i64 = 16;

/// `x·cos + rotate_half(x)·sin` written as the two half-width products it is.
///
/// Bit-identical to the `cat([-x2, x1])` form it replaces (`(-x2)·sin == -(x2·sin)` and
/// `a + (-b) == a - b` are exact in IEEE), and strictly cheaper on both passes: one `split`
/// node instead of two `narrow`s, so backward builds ONE zero-padded full-width gradient
/// instead of two plus the add that reduced them, and the negation and the full-width
/// concatenation of the rotated copy disappear from forward.
fn rotate(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
    let parts = x.split(*x.size().last().unwrap() / 2, -1);
    let (x1, x2) = (&parts[0], &parts[1]);
    Tensor::cat(&[x1 * cos - x2 * sin, x2 * cos + x1 * sin], -1)
}

pub(crate) struct RotaryEmbedding {
    cos_cached: Tensor, // [max_seq_len, rope_dims / 2]
    sin_cached: Tensor, // [max_seq_len, rope_dims / 2]
    rope_dims: i64,
}

impl RotaryEmbedding {
    pub(crate) fn new(
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
            cos_cached: cos_half.set_requires_grad(false),
            sin_cached: sin_half.set_requires_grad(false),
            rope_dims: rd,
        }
    }

    pub(crate) fn apply_positions(&self, x: &Tensor, positions: &Tensor) -> Tensor {
        let (cos, sin) = self.cached_rotation(positions, x.kind());
        self.apply_cached(x, &cos, &sin)
    }

    /// Half-width `cos`/`sin` rows for `positions`, broadcastable over `[.., seq, rope_dims/2]`.
    /// Callers whose positions never change hoist this out of the step: it is four small kernels
    /// per attention tensor per layer otherwise.
    pub(crate) fn cached_rotation(&self, positions: &Tensor, kind: Kind) -> (Tensor, Tensor) {
        let positions = positions
            .to_kind(Kind::Int64)
            .to_device(self.cos_cached.device());
        (
            self.cos_cached.index_select(0, &positions).to_kind(kind),
            self.sin_cached.index_select(0, &positions).to_kind(kind),
        )
    }

    pub(crate) fn apply_cached(&self, x: &Tensor, cos: &Tensor, sin: &Tensor) -> Tensor {
        let head_dim = *x.size().last().unwrap();
        if self.rope_dims < head_dim {
            let parts = x.split_with_sizes([self.rope_dims, head_dim - self.rope_dims], -1);
            Tensor::cat(&[rotate(&parts[0], cos, sin), parts[1].shallow_clone()], -1)
        } else {
            rotate(x, cos, sin)
        }
    }
}
