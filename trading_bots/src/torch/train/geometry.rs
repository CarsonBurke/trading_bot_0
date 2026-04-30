use std::env;

use super::config::{
    parse_positive_i64_env, DEFAULT_NPROCS, DEFAULT_PPO_CHUNK_LEN, DEFAULT_PPO_MINIBATCH_RATIO,
    DEFAULT_SEQ_LEN,
};

#[derive(Clone, Copy, Debug)]
pub(crate) struct RolloutGeometry {
    pub(crate) nprocs: i64,
    pub(crate) seq_len: i64,
    pub(crate) ppo_chunk_len: i64,
    pub(crate) total_samples: i64,
}

pub(crate) fn align_up(value: i64, multiple: i64) -> i64 {
    debug_assert!(multiple > 0);
    ((value + multiple - 1) / multiple) * multiple
}

pub(crate) fn largest_divisor_at_most(value: i64, max_divisor: i64) -> i64 {
    for divisor in (1..=max_divisor.min(value)).rev() {
        if value % divisor == 0 {
            return divisor;
        }
    }
    1
}

pub(crate) fn default_chunk_len_for_seq_len(seq_len: i64) -> i64 {
    if seq_len % DEFAULT_PPO_CHUNK_LEN == 0 {
        DEFAULT_PPO_CHUNK_LEN
    } else {
        largest_divisor_at_most(seq_len, DEFAULT_PPO_CHUNK_LEN)
    }
}

pub(crate) fn rollout_geometry() -> RolloutGeometry {
    let nprocs = parse_positive_i64_env("PPO_NPROCS").unwrap_or(DEFAULT_NPROCS);

    let requested_seq_len = if let Some(seq_len) = parse_positive_i64_env("PPO_SEQ_LEN") {
        seq_len
    } else if let Some(target_total_samples) = parse_positive_i64_env("PPO_TOTAL_SAMPLES") {
        (target_total_samples + nprocs - 1) / nprocs
    } else {
        DEFAULT_SEQ_LEN
    };
    let ppo_chunk_len = parse_positive_i64_env("PPO_CHUNK_LEN")
        .unwrap_or_else(|| default_chunk_len_for_seq_len(requested_seq_len));
    let seq_len = align_up(requested_seq_len.max(ppo_chunk_len), ppo_chunk_len);

    RolloutGeometry {
        nprocs,
        seq_len,
        ppo_chunk_len,
        total_samples: nprocs * seq_len,
    }
}

pub(crate) fn minibatch_samples_from_total(total_samples: i64, nprocs: i64) -> i64 {
    let ratio = env::var("PPO_MINIBATCH_RATIO")
        .ok()
        .and_then(|v| v.parse::<f64>().ok())
        .filter(|&v| v > 0.0 && v <= 1.0)
        .unwrap_or(DEFAULT_PPO_MINIBATCH_RATIO);
    let target = ((total_samples as f64) * ratio).round() as i64;
    let aligned = ((target + nprocs - 1) / nprocs).max(1) * nprocs;
    aligned.min(total_samples)
}
