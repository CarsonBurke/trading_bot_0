//! Shared GPT-style decoder transformer + synthetic LM task used by the
//! optimizer benchmarks (`optim_loss.rs` sweep and `optim_grid.rs` full grid).
//!
//! This replaces the prior 4-layer square MLP, which was NOT architecturally
//! representative of the real model: it had no embedding/head to exclude from
//! Muon, only square weights, and therefore never exercised the real
//! NorMuon-vs-AdamW routing split. The absolute optimal LR from an MLP does not
//! reliably transfer to a transformer.
//!
//! Design:
//!   - Pre-norm GPT decoder: token embed + learned position embed, N blocks of
//!     (RMSNorm -> causal MHA via SDPA/FlashAttention -> residual) and
//!     (RMSNorm -> MLP 4x -> residual), final RMSNorm + LM head.
//!   - Attention uses `Tensor::scaled_dot_product_attention(is_causal=true)`,
//!     which dispatches to the FlashAttention kernel on CUDA. q/k/v are
//!     contiguous bf16 in [b, heads, seq, head_dim]. No hand-rolled softmax.
//!   - All trainable params are built as NAMED vars so the optimizer can route
//!     them: token/pos embeddings + LM head -> AdamW (forced, even though 2D);
//!     RMSNorm gains (1D) -> AdamW automatically; attention Q/K/V/O and MLP
//!     up/down 2D weights -> NorMuon.
//!   - Task: deterministic, learnable in-context associative recall (induction
//!     head). Each sequence is a stream of (key, value) pairs from a fixed
//!     vocabulary; the model must, on seeing a key it has seen before, predict
//!     the value that followed it earlier in the SAME sequence. This is the
//!     canonical task that transformers solve and MLPs cannot, so the optimizer
//!     comparison is on a representative loss surface. Fixed seed => identical
//!     init + data for every optimizer config.

use tch::nn::Init;
use tch::{nn, Device, Kind, Tensor};

pub const D_MODEL: i64 = 256;
pub const N_LAYERS: usize = 4;
pub const N_HEADS: i64 = 4;
pub const HEAD_DIM: i64 = D_MODEL / N_HEADS; // 64
pub const SEQ_LEN: i64 = 128;
pub const VOCAB: i64 = 256;
pub const MLP_RATIO: i64 = 4;
pub const MLP_HIDDEN: i64 = D_MODEL * MLP_RATIO; // 1024

/// Compute dtype for the model + SDPA. bf16 is required for the FlashAttention
/// SDPA backend on CUDA (and matches real training).
pub const COMPUTE_KIND: Kind = Kind::BFloat16;

/// Substrings that force a 2D param onto AdamW (token/pos embeddings + LM head).
/// Mirrors the original Muon routing: embeddings and the final head go to Adam.
pub fn force_adamw_substrings() -> Vec<String> {
    vec![
        "tok_embed".to_string(),
        "pos_embed".to_string(),
        "lm_head".to_string(),
    ]
}

/// Learnable RMSNorm (gain is 1D -> routed to AdamW).
struct RMSNorm {
    gain: Tensor,
    eps: f64,
}

impl RMSNorm {
    fn new(p: &nn::Path, dim: i64, name: &str) -> Self {
        Self {
            gain: p.var(name, &[dim], Init::Const(1.0)),
            eps: 1e-6,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        // x: [..., dim]. Compute in fp32 for stability, scale by learnable gain.
        let xf = x.to_kind(Kind::Float);
        let var = (&xf * &xf).mean_dim([-1i64].as_slice(), true, Kind::Float);
        let normed = &xf * (var + self.eps).rsqrt();
        let g = self.gain.to_kind(Kind::Float);
        (normed * g).to_kind(x.kind())
    }
}

/// A named 2D weight, stored as [out, in] and applied as x @ wᵀ. Built via
/// `p.var` so the optimizer routes it by name. Truncated-normal init scaled by
/// 1/sqrt(fan_in), the standard transformer init.
struct LinearW {
    w: Tensor, // [out_features, in_features]
}

impl LinearW {
    fn new(p: &nn::Path, name: &str, in_features: i64, out_features: i64) -> Self {
        let std = 1.0 / (in_features as f64).sqrt();
        Self {
            w: p.var(
                name,
                &[out_features, in_features],
                Init::Randn {
                    mean: 0.0,
                    stdev: std,
                },
            ),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.w.to_kind(x.kind()).tr())
    }
}

struct Block {
    norm1: RMSNorm,
    q_proj: LinearW,
    k_proj: LinearW,
    v_proj: LinearW,
    o_proj: LinearW,
    norm2: RMSNorm,
    fc1: LinearW,
    fc2: LinearW,
}

impl Block {
    fn new(p: &nn::Path, i: usize) -> Self {
        let bp = p / format!("block{}", i);
        Self {
            norm1: RMSNorm::new(&bp, D_MODEL, "norm1_gain"),
            q_proj: LinearW::new(&bp, "attn_q", D_MODEL, D_MODEL),
            k_proj: LinearW::new(&bp, "attn_k", D_MODEL, D_MODEL),
            v_proj: LinearW::new(&bp, "attn_v", D_MODEL, D_MODEL),
            o_proj: LinearW::new(&bp, "attn_o", D_MODEL, D_MODEL),
            norm2: RMSNorm::new(&bp, D_MODEL, "norm2_gain"),
            fc1: LinearW::new(&bp, "mlp_fc1", D_MODEL, MLP_HIDDEN),
            fc2: LinearW::new(&bp, "mlp_fc2", MLP_HIDDEN, D_MODEL),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let (b, t, _d) = x.size3().unwrap();
        // --- Causal self-attention (pre-norm). ---
        let h = self.norm1.forward(x);
        let split = |proj: &LinearW| -> Tensor {
            proj.forward(&h)
                .reshape([b, t, N_HEADS, HEAD_DIM])
                .permute([0, 2, 1, 3])
                .contiguous()
        };
        let q = split(&self.q_proj);
        let k = split(&self.k_proj);
        let v = split(&self.v_proj);
        // SDPA -> FlashAttention kernel on CUDA. is_causal=true, default scale.
        let attn = Tensor::scaled_dot_product_attention(
            &q,
            &k,
            &v,
            None::<&Tensor>,
            0.0,
            true,
            None,
            false,
        )
        .permute([0, 2, 1, 3])
        .contiguous()
        .reshape([b, t, D_MODEL]);
        let x = x + self.o_proj.forward(&attn);

        // --- MLP (pre-norm). ---
        let h = self.norm2.forward(&x);
        let h = self.fc1.forward(&h).gelu("none");
        let h = self.fc2.forward(&h);
        &x + h
    }
}

pub struct GptModel {
    tok_embed: Tensor, // [VOCAB, D_MODEL]
    pos_embed: Tensor, // [SEQ_LEN, D_MODEL]
    blocks: Vec<Block>,
    final_norm: RMSNorm,
    lm_head: LinearW, // [VOCAB, D_MODEL]
}

impl GptModel {
    pub fn new(p: &nn::Path) -> Self {
        let tok_embed = p.var(
            "tok_embed",
            &[VOCAB, D_MODEL],
            Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        let pos_embed = p.var(
            "pos_embed",
            &[SEQ_LEN, D_MODEL],
            Init::Randn {
                mean: 0.0,
                stdev: 0.02,
            },
        );
        let blocks = (0..N_LAYERS).map(|i| Block::new(p, i)).collect();
        let final_norm = RMSNorm::new(p, D_MODEL, "final_norm_gain");
        let lm_head = LinearW::new(p, "lm_head", D_MODEL, VOCAB);
        Self {
            tok_embed,
            pos_embed,
            blocks,
            final_norm,
            lm_head,
        }
    }

    /// tokens: [b, t] int64. Returns logits [b, t, VOCAB].
    pub fn forward(&self, tokens: &Tensor) -> Tensor {
        let t = tokens.size()[1];
        let tok = self
            .tok_embed
            .to_kind(COMPUTE_KIND)
            .index_select(0, &tokens.reshape([-1]))
            .reshape([tokens.size()[0], t, D_MODEL]);
        let pos = self
            .pos_embed
            .narrow(0, 0, t)
            .to_kind(COMPUTE_KIND)
            .unsqueeze(0);
        let mut x = tok + pos;
        for block in &self.blocks {
            x = block.forward(&x);
        }
        let x = self.final_norm.forward(&x);
        self.lm_head.forward(&x)
    }
}

/// Deterministic, learnable in-context associative-recall (induction) task.
///
/// Each sequence is a flat stream of tokens: positions alternate key, value,
/// key, value, ... Keys are drawn from a key sub-vocabulary, values from a value
/// sub-vocabulary. Within a sequence, every key has a FIXED associated value
/// (sampled once per sequence). The model sees a key at an even position and
/// must predict its value at the next (odd) position. Only repeated keys are
/// scored; first occurrences are masked because their value is not causally
/// available yet. Because associations are consistent within a sequence and a
/// key recurs, this is solvable only by attending back to the earlier
/// (key,value) occurrence — the induction-head mechanism. There is genuine,
/// decreasing CE signal; pure-random tokens have none.
///
/// Returns (inputs [n, SEQ_LEN], targets [n, SEQ_LEN]) where targets are the
/// next-token labels (shifted inputs), with non-recall positions masked to -100
/// so cross-entropy only scores repeated-key recall positions.
pub struct LmDataset {
    pub inputs: Tensor,  // [n, SEQ_LEN] int64
    pub targets: Tensor, // [n, SEQ_LEN] int64 (-100 = ignore)
}

const N_KEYS: i64 = 32;
const KEY_BASE: i64 = 0;
const VALUE_BASE: i64 = N_KEYS; // values occupy [N_KEYS, N_KEYS + N_VALUES)
const N_VALUES: i64 = 64;

pub fn make_dataset(device: Device, n_seqs: i64) -> LmDataset {
    let _guard = tch::no_grad_guard();
    let pairs = SEQ_LEN / 2; // (key,value) slots per sequence

    // Per-sequence key stream: which key appears at each pair slot.
    let key_stream = Tensor::randint(N_KEYS, [n_seqs, pairs], (Kind::Int64, device)) + KEY_BASE;
    // Per-sequence fixed key->value map: [n_seqs, N_KEYS].
    let key_to_value =
        Tensor::randint(N_VALUES, [n_seqs, N_KEYS], (Kind::Int64, device)) + VALUE_BASE;
    // Gather the value for each key in the stream: [n_seqs, pairs].
    let key_idx = &key_stream - KEY_BASE;
    let value_stream = key_to_value.gather(1, &key_idx, false);

    // Interleave into [n_seqs, SEQ_LEN]: even=key, odd=value.
    let inputs = Tensor::zeros([n_seqs, SEQ_LEN], (Kind::Int64, device));
    let even_idx = Tensor::arange_start_step(0, SEQ_LEN, 2, (Kind::Int64, device));
    let odd_idx = Tensor::arange_start_step(1, SEQ_LEN, 2, (Kind::Int64, device));
    let inputs = inputs.index_copy(1, &even_idx, &key_stream);
    let inputs = inputs.index_copy(1, &odd_idx, &value_stream);

    // Next-token targets = inputs shifted left by 1; last position has no label.
    // We only score repeated keys at even input positions, where the matching
    // key,value pair already exists in the causal context.
    let targets = Tensor::full([n_seqs, SEQ_LEN], -100i64, (Kind::Int64, device));
    let key_one_hot = key_idx.one_hot(N_KEYS).to_kind(Kind::Int64);
    let prior_key_counts = key_one_hot.cumsum(1, Kind::Int64) - &key_one_hot;
    let repeated_key = prior_key_counts
        .gather(2, &key_idx.unsqueeze(-1), false)
        .squeeze_dim(-1)
        .gt(0);

    // target at position p = input at position p+1, for repeated even p.
    let score_pos = Tensor::arange_start_step(0, SEQ_LEN - 1, 2, (Kind::Int64, device));
    let next_pos = &score_pos + 1;
    let value_labels = inputs.index_select(1, &next_pos); // [n_seqs, pairs]
    let ignored_labels = Tensor::full([n_seqs, pairs], -100i64, (Kind::Int64, device));
    let value_labels = value_labels.where_self(&repeated_key, &ignored_labels);
    let targets = targets.index_copy(1, &score_pos, &value_labels);

    LmDataset { inputs, targets }
}

/// Named trainable variables from a VarStore, sorted by name for a stable,
/// deterministic order (so every optimizer config sees the same param order).
pub fn named_trainable(vs: &nn::VarStore) -> Vec<(String, Tensor)> {
    let mut named: Vec<(String, Tensor)> = vs
        .variables()
        .into_iter()
        .filter(|(_, t)| t.requires_grad())
        .collect();
    named.sort_by(|a, b| a.0.cmp(&b.0));
    named
}

/// Cross-entropy over scored (non -100) positions only.
pub fn lm_loss(logits: &Tensor, targets: &Tensor) -> Tensor {
    let v = logits.size()[2];
    logits
        .to_kind(Kind::Float)
        .reshape([-1, v])
        .cross_entropy_loss::<&Tensor>(
            &targets.reshape([-1]),
            None,
            tch::Reduction::Mean,
            -100,
            0.0,
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_scores_only_repeated_keys() {
        tch::manual_seed(123);
        let data = make_dataset(Device::Cpu, 8);
        let inputs: Vec<i64> = data.inputs.reshape([-1]).iter::<i64>().unwrap().collect();
        let targets: Vec<i64> = data.targets.reshape([-1]).iter::<i64>().unwrap().collect();
        let pairs = (SEQ_LEN / 2) as usize;
        let seq_len = SEQ_LEN as usize;

        let mut scored = 0;
        for seq in 0..8usize {
            let mut seen = vec![false; N_KEYS as usize];
            for pair in 0..pairs {
                let pos = pair * 2;
                let flat = seq * seq_len + pos;
                let key = inputs[flat] as usize;
                let value = inputs[flat + 1];
                if seen[key] {
                    assert_eq!(targets[flat], value, "repeated key should be scored");
                    scored += 1;
                } else {
                    assert_eq!(targets[flat], -100, "first key occurrence must be ignored");
                    seen[key] = true;
                }
                assert_eq!(targets[flat + 1], -100, "value positions are never scored");
            }
        }
        assert!(scored > 0, "test seed should include repeated keys");
    }
}
