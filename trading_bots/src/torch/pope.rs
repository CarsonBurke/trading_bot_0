use std::f64::consts::PI;

use tch::nn::{self, Init};
use tch::{Kind, Tensor};

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
            let inv_frequency = (Tensor::arange(dim, (Kind::Float, device))
                * (-POPE_FREQUENCY_BASE.ln() / dim as f64))
                .exp();
            let min_frequency = 1.0 / train_len as f64;
            let lower = -2.0 * PI * &inv_frequency / inv_frequency.clamp_min(min_frequency);
            let initial = Tensor::rand([query_heads, dim], (Kind::Float, device))
                * -lower.view([1, dim])
                + lower.view([1, dim]);
            p.var_copy(name, &initial)
        }
    }
}

pub fn pope_expand_qk_fp32(
    query: &Tensor,
    key: &Tensor,
    query_positions: &Tensor,
    key_positions: &Tensor,
    phase_bias: &Tensor,
    frequency_base: f64,
) -> PolarQk {
    validate_inputs(
        query,
        key,
        query_positions,
        key_positions,
        phase_bias,
        frequency_base,
    );

    let device = query.device();
    let inv_frequency = (Tensor::arange(POPE_DIM, (Kind::Float, device))
        * (-frequency_base.ln() / POPE_DIM as f64))
        .exp();
    let angles = |positions: &Tensor| {
        positions
            .to_device(device)
            .to_kind(Kind::Float)
            .view([1, -1, 1, 1])
            * inv_frequency.view([1, 1, 1, POPE_DIM])
    };

    // Moving each repeated GQA key phase by delta is score-equivalent to
    // moving its corresponding query phase by -delta. This retains the compact
    // KV-head layout while allowing an independent bias for every query head.
    let query_angle = angles(query_positions)
        - phase_bias
            .to_device(device)
            .to_kind(Kind::Float)
            .clamp(-2.0 * PI, 0.0)
            .view([1, 1, query.size()[2], POPE_DIM]);
    let key_angle = angles(key_positions);
    let query_magnitude = query.to_kind(Kind::Float).softplus();
    let key_magnitude = key.to_kind(Kind::Float).softplus();

    PolarQk {
        query: Tensor::cat(
            &[
                &(&query_magnitude * query_angle.cos()),
                &(&query_magnitude * query_angle.sin()),
            ],
            -1,
        ),
        key: Tensor::cat(
            &[
                &(&key_magnitude * key_angle.cos()),
                &(&key_magnitude * key_angle.sin()),
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

fn validate_inputs(
    query: &Tensor,
    key: &Tensor,
    query_positions: &Tensor,
    key_positions: &Tensor,
    phase_bias: &Tensor,
    frequency_base: f64,
) {
    assert_eq!(query.dim(), 4, "PoPE query must be [B,S,H,64]");
    assert_eq!(key.dim(), 4, "PoPE key must be [B,S,Hkv,64]");
    assert_eq!(query.size()[0], key.size()[0], "PoPE Q/K batch mismatch");
    assert_eq!(query.size()[3], POPE_DIM, "PoPE query width must be 64");
    assert_eq!(key.size()[3], POPE_DIM, "PoPE key width must be 64");
    assert_eq!(
        query_positions.dim(),
        1,
        "PoPE query positions must be rank 1"
    );
    assert_eq!(key_positions.dim(), 1, "PoPE key positions must be rank 1");
    assert_eq!(
        query_positions.size()[0],
        query.size()[1],
        "PoPE query position count mismatch"
    );
    assert_eq!(
        key_positions.size()[0],
        key.size()[1],
        "PoPE key position count mismatch"
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
    assert!(
        frequency_base.is_finite() && frequency_base > 1.0,
        "invalid PoPE frequency base"
    );
}
