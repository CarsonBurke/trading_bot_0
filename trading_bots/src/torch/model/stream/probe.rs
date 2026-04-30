use std::sync::atomic::{AtomicBool, Ordering};
use tch::{Kind, Tensor};

use super::super::trading_model::TradingModel;

static REPLAY_NUMERIC_PROBE: AtomicBool = AtomicBool::new(false);

pub(super) fn probe_replay_tensor(tag: &str, t: &Tensor) {
    if !replay_numeric_probe_enabled() {
        return;
    }
    if t.isfinite().all().int64_value(&[]) != 0 {
        return;
    }
    let nan_count = t
        .isnan()
        .to_kind(Kind::Int64)
        .sum(Kind::Int64)
        .int64_value(&[]);
    let inf_count = t
        .isinf()
        .to_kind(Kind::Int64)
        .sum(Kind::Int64)
        .int64_value(&[]);
    let mean = t.mean(Kind::Float).double_value(&[]);
    let min = t.min().double_value(&[]);
    let max = t.max().double_value(&[]);
    let abs_max = t.abs().max().double_value(&[]);
    println!(
        "NUMERIC REPLAY PROBE: {} shape={:?} kind={:?} mean={:.6} min={:.6} max={:.6} abs_max={:.6} nan={} inf={}",
        tag,
        t.size(),
        t.kind(),
        mean,
        min,
        max,
        abs_max,
        nan_count,
        inf_count
    );
}

fn replay_numeric_probe_enabled() -> bool {
    REPLAY_NUMERIC_PROBE.load(Ordering::Relaxed)
}

impl TradingModel {
    pub(crate) fn set_replay_numeric_probe(enabled: bool) {
        REPLAY_NUMERIC_PROBE.store(enabled, Ordering::Relaxed);
    }
}
