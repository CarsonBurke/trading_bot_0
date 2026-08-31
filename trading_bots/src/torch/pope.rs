use std::f64::consts::PI;
use std::sync::Mutex;

use tch::nn::{self, Init};
use tch::{Device, Kind, Tensor};

pub const POPE_DIM: i64 = 64;
pub const POPE_QK_DIM: i64 = 2 * POPE_DIM;
pub const POPE_ATTENTION_SCALE: f64 = 0.125;
pub const POPE_FREQUENCY_BASE: f64 = 10_000.0;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PopeThetaInit {
    Zero,
    TwoPi,
}

pub struct PolarQk {
    pub query: Tensor,
    pub key: Tensor,
}

/// Layer-invariant PoPE phases for one forward pass.
///
/// Q and K read the same absolute position table, and only the magnitudes and the
/// query's own phase bias vary by layer, so one trunk pass shares a single frequency
/// ladder, position outer product and key `cos`/`sin` pair.
pub struct PopePhases {
    /// `[1, S, 1, POPE_DIM]` position/frequency outer product, fp32.
    angle: Tensor,
    /// `angle.cos()` and `angle.sin()`. The key side takes no phase bias, so these
    /// are its finished phases.
    cos: Tensor,
    sin: Tensor,
}

impl PopePhases {
    pub fn new(positions: &Tensor, device: Device, frequency_base: f64) -> Self {
        assert_eq!(positions.dim(), 1, "PoPE positions must be rank 1");
        assert!(
            frequency_base.is_finite() && frequency_base > 1.0,
            "invalid PoPE frequency base"
        );
        let angle = positions
            .to_device(device)
            .to_kind(Kind::Float)
            .view([1, -1, 1, 1])
            * inv_frequency_table(device, frequency_base).view([1, 1, 1, POPE_DIM]);
        Self {
            cos: angle.cos(),
            sin: angle.sin(),
            angle,
        }
    }

    /// Returns a view over a contiguous subset of this precomputed phase table.
    ///
    /// This keeps variable-length callers from rebuilding the frequency ladder and
    /// evaluating `cos`/`sin` when a longer cached table is already available.
    pub fn narrow(&self, start: i64, length: i64) -> Self {
        let table_length = self.angle.size()[1];
        assert!(start >= 0, "PoPE phase start must be non-negative");
        assert!(
            length > 0 && length <= table_length,
            "PoPE phase length must fit the cached table"
        );
        assert!(
            start <= table_length - length,
            "PoPE phase range must fit the cached table"
        );
        Self {
            angle: self.angle.narrow(1, start, length),
            cos: self.cos.narrow(1, start, length),
            sin: self.sin.narrow(1, start, length),
        }
    }
}

/// `frequency_base ** (-i / POPE_DIM)`, a constant of the architecture. Built once
/// per (device, base) instead of once per layer; 64 floats that every caller reads.
fn inv_frequency_table(device: Device, frequency_base: f64) -> Tensor {
    static CACHE: Mutex<Vec<(Device, u64, Tensor)>> = Mutex::new(Vec::new());
    let base = frequency_base.to_bits();
    let mut cache = CACHE.lock().expect("PoPE frequency table cache poisoned");
    if let Some((_, _, table)) = cache.iter().find(|(d, b, _)| *d == device && *b == base) {
        return table.shallow_clone();
    }
    let table = (Tensor::arange(POPE_DIM, (Kind::Float, device))
        * (-frequency_base.ln() / POPE_DIM as f64))
        .exp();
    cache.push((device, base, table.shallow_clone()));
    table
}

pub fn init_pope_theta_bias(
    p: &nn::Path,
    name: &str,
    query_heads: i64,
    dim: i64,
    train_len: i64,
    init: PopeThetaInit,
) -> Tensor {
    assert!(query_heads > 0, "PoPE requires at least one query head");
    assert_eq!(dim, POPE_DIM, "full PoPE64 requires dim=64");
    assert!(train_len > 0, "PoPE training length must be positive");
    match init {
        PopeThetaInit::Zero => p.var(name, &[query_heads, dim], Init::Const(0.0)),
        PopeThetaInit::TwoPi => {
            let device = p.device();
            let inv_frequency = inv_frequency_table(device, POPE_FREQUENCY_BASE);
            let min_frequency = 1.0 / train_len as f64;
            let lower = -2.0 * PI * &inv_frequency / inv_frequency.clamp_min(min_frequency);
            let initial = Tensor::rand([query_heads, dim], (Kind::Float, device))
                * -lower.view([1, dim])
                + lower.view([1, dim]);
            p.var_copy(name, &initial)
        }
    }
}

/// Expands one layer's Q/K into PoPE's 128-wide polar form, already in `kind`.
///
/// The position phases are built and biased in fp32 and stay there through `cos`/`sin`, where
/// the width is load-bearing: a phase reaches the sequence length in radians, and bf16 resolves
/// ~896 radians only to about 4. Each half is cast after its multiply and before the
/// concatenation, which is elementwise-identical to casting the concatenation and keeps the
/// 128-wide fp32 transient off the device.
///
/// The trailing `to_kind(kind)` on each product is LOAD-BEARING, not tidying. `aten::softplus`
/// sits on autocast's fp32 cast-policy list, so under `autocast(bf16)` it returns fp32 whatever
/// dtype it is handed; each product then inherits fp32 by promotion, and without the cast
/// `Tensor::cat` yields an fp32 polar pair that the strict FA4 bridge rejects at runtime with
/// `FA4 query must be fp16 or bf16, got Float`. The CPU tests cannot catch that - off CUDA
/// `attention_kind` is `Float` and an fp32 query is legal - so only a pretrain launch does.
///
/// Narrowing the magnitude to shrink what `softplus_backward` retains is equally a dead end,
/// and was measured rather than reasoned: autocast materializes the fp32 copy INSIDE the
/// `softplus` call, so passing `kind` instead of `Float` leaves the saved tensor the same
/// `[B, S, H, POPE_DIM]` fp32 - 36.7 MB at batch 20 - for a bit-identical polar output and
/// 0.0 MB saved. The width here is not the caller's to take.
pub fn pope_expand_qk(
    query: &Tensor,
    key: &Tensor,
    phases: &PopePhases,
    phase_bias: &Tensor,
    kind: Kind,
) -> PolarQk {
    validate_inputs(query, key, phases, phase_bias);

    // Moving each repeated GQA key phase by delta is score-equivalent to
    // moving its corresponding query phase by -delta. This retains the compact
    // KV-head layout while allowing an independent bias for every query head.
    let query_angle = &phases.angle
        - phase_bias
            .to_device(query.device())
            .to_kind(Kind::Float)
            .clamp(-2.0 * PI, 0.0)
            .view([1, 1, query.size()[2], POPE_DIM]);
    let query_magnitude = query.to_kind(Kind::Float).softplus();
    let key_magnitude = key.to_kind(Kind::Float).softplus();

    PolarQk {
        query: Tensor::cat(
            &[
                (&query_magnitude * query_angle.cos()).to_kind(kind),
                (&query_magnitude * query_angle.sin()).to_kind(kind),
            ],
            -1,
        ),
        key: Tensor::cat(
            &[
                (&key_magnitude * &phases.cos).to_kind(kind),
                (&key_magnitude * &phases.sin).to_kind(kind),
            ],
            -1,
        ),
    }
}

/// Exact unequal-width reference for CPU tests and numerical validation.
/// Production CUDA prefill and decode must use the strict FA4 bridge.
pub fn pope_attention_reference(qk: &PolarQk, value_bshd: &Tensor, causal: bool) -> Tensor {
    assert_eq!(qk.query.dim(), 4, "PoPE query must be BSHD");
    assert_eq!(qk.key.dim(), 4, "PoPE key must be BSHD");
    assert_eq!(value_bshd.dim(), 4, "PoPE value must be BSHD");
    let query = qk.query.to_kind(Kind::Float).transpose(1, 2);
    let mut key = qk.key.to_kind(Kind::Float).transpose(1, 2);
    let mut value = value_bshd.to_kind(Kind::Float).transpose(1, 2);
    let head_ratio = query.size()[1] / key.size()[1];
    if head_ratio > 1 {
        key = key.repeat_interleave_self_int(head_ratio, 1, None);
        value = value.repeat_interleave_self_int(head_ratio, 1, None);
    }
    let mut scores = query.matmul(&key.transpose(-2, -1)) * POPE_ATTENTION_SCALE;
    if causal {
        let query_length = query.size()[2];
        let key_length = key.size()[2];
        assert_eq!(
            query_length, key_length,
            "causal PoPE reference requires equal Q/K lengths"
        );
        let mask = Tensor::ones([query_length, key_length], (Kind::Bool, query.device())).triu(1);
        scores = scores.masked_fill(&mask, f64::NEG_INFINITY);
    }
    scores
        .softmax(-1, Kind::Float)
        .matmul(&value)
        .transpose(1, 2)
}

fn validate_inputs(query: &Tensor, key: &Tensor, phases: &PopePhases, phase_bias: &Tensor) {
    assert_eq!(query.dim(), 4, "PoPE query must be [B,S,H,64]");
    assert_eq!(key.dim(), 4, "PoPE key must be [B,S,Hkv,64]");
    assert_eq!(query.size()[0], key.size()[0], "PoPE Q/K batch mismatch");
    assert_eq!(query.size()[3], POPE_DIM, "PoPE query width must be 64");
    assert_eq!(key.size()[3], POPE_DIM, "PoPE key width must be 64");
    assert_eq!(
        query.size()[1],
        key.size()[1],
        "PoPE Q/K share one position table"
    );
    assert_eq!(
        phases.angle.size()[1],
        query.size()[1],
        "PoPE position count mismatch"
    );
    assert_eq!(
        phase_bias.size(),
        [query.size()[2], POPE_DIM],
        "PoPE phase bias must be [Hq,64]"
    );
    assert!(
        query.size()[2] % key.size()[2] == 0,
        "PoPE query heads must be divisible by KV heads"
    );
}
