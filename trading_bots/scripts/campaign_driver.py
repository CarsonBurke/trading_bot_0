#!/usr/bin/env python3
"""Cumulative ablation campaign driver (audit Part 3, W0.6).

Queues `pretrain` arms through mlq, reads their pinned per-window artifact, shells to
`pretrain-compare` against the arm's incumbent, applies the section 3.3 autocull rules and
appends every number it saw to an append-only JSONL ledger.

It reimplements no statistics. Every interval, standard error and paired difference is read
out of `pretrain-compare` stdout or out of `*.windows.json`; the driver only compares numbers
to thresholds and records what it compared.

Stdlib only. No third-party imports, by design: this thing has to run unattended beside a
training job and must not be able to fail on an environment change.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

DRIVER_VERSION = 3

# Training uses an already-built binary behind a CUDA environment launcher. Both paths and
# digests are mandatory in the spec; rebuilding through `cargo run` is not a campaign identity.
GEOMETRY_CHANGE_COMPARE_FLAG = "--allow-geometry-change"
#: `RunDir::create_fresh(runs_path, name)` (shared/src/run_dir.rs:69-119).
DEFAULT_RUNS_DIR = "training/runs"
#: Diagnostic context every stage-0 arm is selected at (pretrain.rs:102,840-845).
DEFAULT_DIAGNOSTIC_CONTEXT = 896
#: Deployment bar resolution; the supports sidecar is named after it
#: (world_model.rs:1078-1080).
DEFAULT_RESOLUTION_SECS = 300
#: `POLICY_NAMES[POLICY_MODEL]` (trade_bench.rs:312,320-321). `TradePolicySummary.policy`
#: carries the name, so rows are matched by name rather than by index.
POLICY_MODEL_NAME = "model"
#: `SELECTION_CAP` (pretrain.rs:394-400); the cap the primary metric is quoted at.
SELECTION_CAP = 0.25
#: Schema of the per-window vector this driver reads. W0.1 bumped it 2 -> 3 to carry
#: `selection_edge_bps`, `model_growth_bps`, `traded_blocks`, `supports_sha256` and
#: `calibration`, and `WindowScores::load` now REJECTS v2 outright. Matching that here means
#: a stale incumbent artifact is named as such by the driver instead of surfacing later as an
#: unexplained comparator failure.
ACCEPTED_WINDOW_SCORES_VERSIONS = (3,)

#: `ensure_finite_step_metrics` (pretrain.rs:10276-10305). Tripwire 1 reads the captured
#: launcher output for these, because the trainer aborts before optimizer mutation and the
#: only surviving record of WHY is the message.
NONFINITE_PATTERNS = (
    "loss is not finite at step",
    "gradient norm is not finite at step",
    "packed loss/gradient diagnostics are not finite at step",
    "row learned-LR diagnostics are partially non-finite at step",
    "SMD-IDBD diagnostics are partially non-finite at step",
)

ARM_CLASSES = ("geometry_preserving", "geometry_changing")
TIER_ORDER = ("screen", "confirm", "deploy")


# ---------------------------------------------------------------------------
# COMPARATOR CONTRACT — confirmed with agent `EdgePersist`, who wrote the W0.1/W0.2/W0.3
# edits. Verified against the format strings in `impl Display for PairedComparison`
# (pretrain_stats.rs), dispatched at main.rs:2056-2066. Re-verify with
# `campaign_driver_selftest.py`, whose fixtures are those strings rendered.
# ---------------------------------------------------------------------------
#
#     paired comparison over 4096 identical pinned windows, scoring smoothed
#       baseline  control_base_s24301_screen   3.1416 nats/bar
#       candidate B1_base_s24301_screen        3.1350 nats/bar
#       paired delta [nats/bar, candidate - baseline, NEGATIVE is better] <dispersion>
#       conditional delta (u,v scored only where s != 0) <dispersion>
#       delta r  <dispersion>                       <- `{name:<2}`, so TWO spaces after `r`
#       delta s  <dispersion>                          ... in BAR_DOF_NAMES [r,s,u,v,w] order
#       selection edge delta [bps/bar, candidate - baseline, POSITIVE is better] \
#           <dispersion>; candidate advantage +0.0234, MDE 0.0255, verdict: SIGNIFICANT at 95%
#       model growth delta [bps/bar, candidate - baseline, POSITIVE is better] <dispersion>; ...
#       per-window correlation 0.9971, candidate worse on 1503 of 4096 windows
#       detectable at 80% power: 0.0059 nats; verdict: SIGNIFICANT at 95%
#
# `<dispersion>` is always `Dispersion`'s own Display (pretrain_stats.rs:118-126):
# `"{:.4} +/- {:.4} (95% CI {:.4}..{:.4}, {} blocks / {} windows)"`.
#
# The two economic rows ALWAYS print. When the vector is absent they take a second form,
# which the parser records as an explicit absence WITH ITS REASON rather than as silence:
#
#       selection edge delta [bps/bar]: not recorded (pre-v3 artifact, or a pass with no
#       traded windows)
#
# Units are bps PER BAR, already scaled by 1e4 at the source; both economic rows are
# `candidate - baseline` with POSITIVE better, so no orientation flip happens here.
# `candidate advantage` is the same number pre-flipped into "more is better" and is parsed
# only as a cross-check on the sign convention.
#
# NOT on stdout: the Mincer-Zarnowitz slope, the raw per-window vectors, and
# `supports_sha256`. All four live on `WindowScores` in the `.windows.json`:
#   selection_edge_bps  model_growth_bps  traded_blocks   -- arrays over the TRADED PREFIX
#   supports_sha256                                       -- W0.2's geometry fingerprint
#   calibration {mean_beta, mean_beta_se, variance_beta}  -- W0.3, or null with no bench
#
# By default `pretrain-compare` hard-refuses a supports mismatch. Geometry-changing arms use
# its fixed `--allow-geometry-change` contract, which withholds every support-decoded quantity
# and emits `model_growth_difference` alone. Differing traded-prefix lengths always refuse.
# Any refusal means "not comparable", NEVER "no effect", and its stderr stays in the ledger.
#
# Sign conventions the decision engine assumes:
#   * `edge_difference`         bps/bar at SELECTION_CAP, HIGHER better
#   * `model_growth_difference` bps/bar at SELECTION_CAP, HIGHER better
#   * `nll_difference`, `dof_r` nats/bar, LOWER better
#   * `calibration.mean_beta`   a LEVEL per run, not a difference; better = closer to 1.0


@dataclasses.dataclass(frozen=True)
class ComparatorContract:
    """Where the driver looks for each metric. Overridable from the spec, so a wording
    change on the Rust side is an operator edit rather than a driver patch."""

    #: Ordered (regex, canonical-key) label rules, applied after the exact table misses.
    #: Ordered because "model growth delta" must not be captured by the bare-edge rule.
    label_patterns: tuple[tuple[str, str], ...] = (
        (r"model[ _-]*growth", "model_growth_difference"),
        (r"\bedge\b", "edge_difference"),
        (r"^conditional delta\b", "conditional_difference"),
        (r"^paired delta\b", "nll_difference"),
        (r"^delta ([rsuvw])\b", "dof_"),
    )
    #: Per-window vectors over the TRADED PREFIX, and the geometry fingerprint.
    selection_edge_keys: tuple[str, ...] = ("selection_edge_bps",)
    model_growth_keys: tuple[str, ...] = ("model_growth_bps",)
    traded_blocks_keys: tuple[str, ...] = ("traded_blocks",)
    supports_sha256_keys: tuple[str, ...] = ("supports_sha256",)
    #: W0.3's Mincer-Zarnowitz mean slope level and its standard error.
    mz_slope_keys: tuple[str, ...] = ("calibration.mean_beta",)
    mz_slope_se_keys: tuple[str, ...] = ("calibration.mean_beta_se",)
    #: Markers that identify a geometry refusal in the comparator's stderr. Belt and braces:
    #: tripwire 6 also compares the two fingerprints directly, which depends on no wording.
    supports_refusal_markers: tuple[str, ...] = (
        "DIFFERENT BIN GEOMETRIES",
        "does not record the bin geometry",
        "supports_sha256",
    )
    #: Marker of the traded-prefix refusal, which is NOT a geometry problem and must not be
    #: reported as one.
    prefix_refusal_markers: tuple[str, ...] = ("traded different prefixes",)

    @staticmethod
    def from_spec(raw: Any) -> "ComparatorContract":
        base = ComparatorContract()
        if raw is None:
            return base
        if not isinstance(raw, dict):
            raise SpecError("`comparator_contract` must be an object")
        fields = {f.name for f in dataclasses.fields(base)}
        unknown = sorted(set(raw) - fields)
        if unknown:
            raise SpecError(f"unknown comparator_contract keys: {', '.join(unknown)}")
        patched: dict[str, Any] = {}
        for key, value in raw.items():
            if key == "label_patterns":
                patched[key] = tuple(
                    (str(pattern), str(canonical)) for pattern, canonical in value
                )
            else:
                patched[key] = tuple(str(item) for item in value)
        return dataclasses.replace(base, **patched)


#: Labels matched verbatim before the pattern rules run. Both the current bracketed nats
#: label and the pre-W0.1 parenthesized one, so stdout captured in an older ledger still
#: reads back.
EXACT_LABELS = {
    "paired delta [nats/bar, candidate - baseline, negative is better]": "nll_difference",
    "paired delta (candidate - baseline)": "nll_difference",
    "conditional delta (u,v scored only where s != 0)": "conditional_difference",
    "selection edge delta [bps/bar, candidate - baseline, positive is better]": "edge_difference",
    "model growth delta [bps/bar, candidate - baseline, positive is better]": "model_growth_difference",
    "delta r": "dof_r",
    "delta s": "dof_s",
    "delta u": "dof_u",
    "delta v": "dof_v",
    "delta w": "dof_w",
}


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class DriverError(Exception):
    """Anything the operator has to fix before the campaign can continue."""


class SpecError(DriverError):
    pass


class RefusalError(DriverError):
    """A decision the driver refuses to take, rather than taking it on a default."""


# ---------------------------------------------------------------------------
# Metric parsing
# ---------------------------------------------------------------------------

_NUM = r"[-+]?(?:\d+(?:\.\d*)?(?:[eE][-+]?\d+)?|NaN|nan|inf|Infinity)"

#: `Dispersion`'s Display impl, verbatim (pretrain_stats.rs:118-126):
#: `"{:.4} +/- {:.4} (95% CI {:.4}..{:.4}, {} blocks / {} windows)"`.
#:
#: `tail` absorbs whatever the row appends after the interval. The two economic rows append
#: `"; candidate advantage {:+.4}, MDE {:.4}, verdict: {}"`; the nats and per-DOF rows append
#: nothing. Matching the tail loosely rather than exactly is deliberate: an added field must
#: not make an existing metric unreadable.
DISPERSION_RE = re.compile(
    r"^\s{2,}(?P<label>.*?)\s+"
    rf"(?P<mean>{_NUM})\s+\+/-\s+(?P<se>{_NUM})\s+"
    rf"\(95% CI\s+(?P<ci_low>{_NUM})\.\.(?P<ci_high>{_NUM}),\s+"
    r"(?P<blocks>\d+)\s+blocks\s*/\s*(?P<samples>\d+)\s+windows\)(?P<tail>.*)$"
)
#: The economic rows' absent form: `"  {label} [{units}]: not recorded ({reason})"`. Parsed
#: so the ledger records WHY a metric was missing instead of showing an empty slot.
ABSENT_RE = re.compile(
    r"^\s{2,}(?P<label>.*?)\s*:\s*not recorded\s*\((?P<reason>.*)\)\s*$"
)
WITHHELD_RE = re.compile(
    r"^\s{2,}(?P<label>.*?)\s*:\s*NOT COMPARABLE\s*\((?P<reason>.*)\)\s*$"
)
GEOMETRY_BANNER_RE = re.compile(
    r"^\s{2,}BIN GEOMETRY CHANGED and --allow-geometry-change was given:"
)
#: `"; candidate advantage {:+.4}, MDE {:.4}, verdict: {}"`, pre-flipped into
#: more-is-better orientation. Parsed only to cross-check the sign convention.
ADVANTAGE_RE = re.compile(
    rf"candidate advantage\s+(?P<advantage>{_NUM}),\s+MDE\s+(?P<mde>{_NUM}),"
    r"\s+verdict:\s+(?P<verdict>.+?)\s*$"
)
HEADER_RE = re.compile(
    r"^paired comparison over (?P<windows>\d+) identical pinned windows, "
    r"scoring (?P<scoring>.+?)\s*$"
)
CORRELATION_RE = re.compile(
    rf"^\s{{2,}}per-window correlation (?P<correlation>{_NUM}), "
    r"candidate worse on (?P<worse>\d+) of (?P<total>\d+) windows\s*$"
)
LEVEL_RE = re.compile(
    rf"^\s{{2,}}(?P<side>baseline|candidate)\s+(?P<run>\S+)\s+(?P<mean>{_NUM}) nats/bar\s*$"
)
VERDICT_RE = re.compile(
    rf"^\s{{2,}}detectable at 80% power: (?P<mde>{_NUM}) nats; verdict: (?P<verdict>.+?)\s*$"
)


def _to_float(text: str) -> float:
    """Rust's `{:.4}` prints `NaN` and `inf`; Python's float() wants `nan`/`inf`."""
    return float(text.replace("NaN", "nan").replace("Infinity", "inf"))


@dataclasses.dataclass
class Dispersion:
    mean: float
    se: float
    ci_low: float
    ci_high: float
    blocks: int
    samples: int

    def as_json(self) -> dict[str, Any]:
        return {
            "mean": _json_float(self.mean),
            "se": _json_float(self.se),
            "ci_low": _json_float(self.ci_low),
            "ci_high": _json_float(self.ci_high),
            "blocks": self.blocks,
            "samples": self.samples,
        }

    def resolved_at(self, multiple: float) -> bool:
        """True when the sign of `mean` is resolved at `multiple` standard errors."""
        return (
            math.isfinite(self.mean)
            and math.isfinite(self.se)
            and self.se > 0.0
            and abs(self.mean) > multiple * self.se
        )


@dataclasses.dataclass
class Comparison:
    """One `pretrain-compare` invocation, parsed."""

    argv: list[str]
    exit_code: int
    stdout: str
    stderr: str
    windows: int | None = None
    scoring: str | None = None
    baseline_run: str | None = None
    candidate_run: str | None = None
    baseline_mean: float | None = None
    candidate_mean: float | None = None
    correlation: float | None = None
    worse_windows: int | None = None
    total_windows: int | None = None
    mde: float | None = None
    verdict: str | None = None
    dispersions: dict[str, Dispersion] = dataclasses.field(default_factory=dict)
    #: Metrics the comparator explicitly reported as absent, keyed canonically, with the
    #: reason it printed. An absence with a reason is evidence; a silent gap is not.
    absent: dict[str, str] = dataclasses.field(default_factory=dict)
    #: `candidate advantage` per metric, already in more-is-better orientation.
    advantage: dict[str, float] = dataclasses.field(default_factory=dict)
    unmapped: list[str] = dataclasses.field(default_factory=list)
    #: Set when the comparator refused the pair outright: `geometry`, `traded_prefix` or
    #: `unclassified`. A refusal means NOT COMPARABLE, never "no effect".
    refusal: str | None = None
    #: True only when the comparator printed its explicit geometry-change banner. In that
    #: mode every support-decoded metric is withheld and model growth is the sole dispersion.
    model_growth_only: bool = False

    def as_json(self) -> dict[str, Any]:
        return {
            "argv": self.argv,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "windows": self.windows,
            "scoring": self.scoring,
            "baseline_run": self.baseline_run,
            "candidate_run": self.candidate_run,
            "baseline_mean": _json_float(self.baseline_mean),
            "candidate_mean": _json_float(self.candidate_mean),
            "correlation": _json_float(self.correlation),
            "worse_windows": self.worse_windows,
            "total_windows": self.total_windows,
            "minimum_detectable_effect": _json_float(self.mde),
            "verdict": self.verdict,
            "dispersions": {
                key: value.as_json() for key, value in sorted(self.dispersions.items())
            },
            "absent": dict(sorted(self.absent.items())),
            "candidate_advantage": {
                key: _json_float(value) for key, value in sorted(self.advantage.items())
            },
            "unmapped_dispersion_lines": self.unmapped,
            "refusal": self.refusal,
            "model_growth_only": self.model_growth_only,
        }


def canonical_label(label: str, contract: ComparatorContract) -> str | None:
    normalized = " ".join(label.strip().lower().split())
    if normalized in EXACT_LABELS:
        return EXACT_LABELS[normalized]
    for pattern, canonical in contract.label_patterns:
        match = re.search(pattern, normalized)
        if not match:
            continue
        # A trailing `_` on the canonical name means "append capture group 1", which is how
        # the five per-DOF rows stay readable if their `{name:<2}` padding ever changes.
        if canonical.endswith("_") and match.groups():
            return canonical + match.group(1)
        return canonical
    return None


def classify_refusal(text: str, contract: ComparatorContract) -> str:
    """Which of the comparator's hard refusals fired. Geometry and traded-prefix refusals
    are different findings and must not be reported as each other."""
    if any(marker in text for marker in contract.prefix_refusal_markers):
        return "traded_prefix"
    if any(marker in text for marker in contract.supports_refusal_markers):
        return "geometry"
    return "unclassified"


def parse_comparison(
    argv: list[str],
    exit_code: int,
    stdout: str,
    stderr: str,
    contract: ComparatorContract,
) -> Comparison:
    """Parse `pretrain-compare` stdout.

    Deliberately tolerant of NEW lines and intolerant of MISSING ones: unrecognized
    dispersion lines are recorded, never dropped, and the decision engine refuses when a
    metric it needs is absent.
    """
    out = Comparison(argv=argv, exit_code=exit_code, stdout=stdout, stderr=stderr)
    for line in stdout.splitlines():
        if GEOMETRY_BANNER_RE.match(line):
            out.model_growth_only = True
            continue
        if header := HEADER_RE.match(line):
            out.windows = int(header["windows"])
            out.scoring = header["scoring"]
            continue
        if level := LEVEL_RE.match(line):
            if level["side"] == "baseline":
                out.baseline_run = level["run"]
                out.baseline_mean = _to_float(level["mean"])
            else:
                out.candidate_run = level["run"]
                out.candidate_mean = _to_float(level["mean"])
            continue
        if correlation := CORRELATION_RE.match(line):
            out.correlation = _to_float(correlation["correlation"])
            out.worse_windows = int(correlation["worse"])
            out.total_windows = int(correlation["total"])
            continue
        if verdict := VERDICT_RE.match(line):
            out.mde = _to_float(verdict["mde"])
            out.verdict = verdict["verdict"]
            continue
        if match := DISPERSION_RE.match(line):
            key = canonical_label(match["label"], contract)
            if key is None:
                out.unmapped.append(line.rstrip())
                continue
            out.dispersions[key] = Dispersion(
                mean=_to_float(match["mean"]),
                se=_to_float(match["se"]),
                ci_low=_to_float(match["ci_low"]),
                ci_high=_to_float(match["ci_high"]),
                blocks=int(match["blocks"]),
                samples=int(match["samples"]),
            )
            if tail := ADVANTAGE_RE.search(match["tail"]):
                out.advantage[key] = _to_float(tail["advantage"])
            continue
        if absent := ABSENT_RE.match(line):
            key = canonical_label(absent["label"], contract)
            out.absent[key or absent["label"].strip()] = absent["reason"]
            continue
        if withheld := WITHHELD_RE.match(line):
            key = canonical_label(withheld["label"], contract)
            out.absent[key or withheld["label"].strip()] = withheld["reason"]
            continue
    if exit_code != 0:
        out.refusal = classify_refusal(f"{stderr}\n{stdout}", contract)
    return out


def validate_geometry_comparison(comparison: Comparison) -> None:
    if comparison.exit_code != 0:
        return
    if not comparison.model_growth_only:
        raise DriverError(
            "geometry-changing comparison succeeded without the comparator's explicit "
            "model-growth-only banner"
        )
    measured = set(comparison.dispersions)
    if measured != {"model_growth_difference"}:
        raise DriverError(
            "geometry-changing comparison must emit model-growth-only evidence; measured "
            f"{sorted(measured) or 'nothing'}"
        )
    required_withheld = {
        "nll_difference",
        "conditional_difference",
        "edge_difference",
        "per-DOF deltas [nats/bar]",
        "per-window correlation [unitless]",
    }
    missing = sorted(required_withheld - set(comparison.absent))
    if missing:
        raise DriverError(
            "geometry-changing comparison did not explicitly withhold forbidden metrics: "
            + ", ".join(missing)
        )


# ---------------------------------------------------------------------------
# Artifact reading
# ---------------------------------------------------------------------------


def _dig(obj: Any, dotted: str) -> Any:
    cursor = obj
    for part in dotted.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor


def _first_present(obj: Any, keys: Iterable[str]) -> tuple[str | None, Any]:
    for key in keys:
        value = _dig(obj, key)
        if value is not None:
            return key, value
    return None, None


def _json_float(value: float | None) -> Any:
    """JSON has no NaN/inf. Round-trip them as names, matching `JsonF64`
    (pretrain_stats.rs:439-451) so the ledger reads back with `json.loads` defaults."""
    if value is None:
        return None
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return value
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return value


def _read_jsonf64(value: Any) -> float | None:
    """Inverse of `JsonF64`'s Serialize impl."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return {"nan": math.nan, "inf": math.inf, "-inf": -math.inf}.get(value.lower())
    return None


@dataclasses.dataclass
class ArtifactFacts:
    """Everything the driver reads out of one `*.windows.json`."""

    path: Path
    sha256: str
    format_version: int
    run: str
    global_step: int
    split: str
    context: int
    windows: int
    scoring: str | None
    corpus_fingerprint: str
    split_bounds: tuple[int, int] | None
    eval_window_seed: int | None
    realized_batch: int | None
    realized_steps: int | None
    supports_sha256: str | None
    supports_sha256_source: str | None
    #: Lengths of the per-window vectors W0.1 persists. These are over the TRADED PREFIX of
    #: the pinned window set, not the whole set, which is why they differ from `windows`.
    traded_edge_windows: int | None
    traded_growth_windows: int | None
    traded_blocks: int | None
    mz_slope: float | None
    mz_slope_se: float | None
    mz_slope_source: str | None
    free_kelly_saturated: float | None
    model_ruin_bars: int | None
    selection_cap_ruin_bars: int | None
    trade_present: bool

    def as_json(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["path"] = str(self.path)
        payload["split_bounds"] = list(self.split_bounds) if self.split_bounds else None
        payload["mz_slope"] = _json_float(self.mz_slope)
        payload["mz_slope_se"] = _json_float(self.mz_slope_se)
        payload["free_kelly_saturated"] = _json_float(self.free_kelly_saturated)
        return payload


def read_artifact(path: Path, contract: ComparatorContract) -> ArtifactFacts:
    raw_bytes = path.read_bytes()
    document = json.loads(raw_bytes)
    version = int(document.get("format_version", 0))
    if version not in ACCEPTED_WINDOW_SCORES_VERSIONS:
        raise DriverError(
            f"{path}: window-scores format version {version} is not one of "
            f"{ACCEPTED_WINDOW_SCORES_VERSIONS}; this driver would be reading a schema it "
            "does not understand"
        )
    trade = document.get("trade")
    free_kelly = None
    model_ruin = None
    selection_ruin = None
    if isinstance(trade, dict):
        free_kelly = _read_jsonf64(trade.get("free_kelly_saturated"))
        for row in trade.get("policies") or []:
            if isinstance(row, dict) and row.get("policy") == POLICY_MODEL_NAME:
                model_ruin = row.get("ruin_bars")
                break
        for point in trade.get("cap_curve") or []:
            if not isinstance(point, dict):
                continue
            cap = _read_jsonf64(point.get("cap"))
            if cap is not None and abs(cap - SELECTION_CAP) < 1e-12:
                selection_ruin = point.get("ruin_bars")
                break
    supports_key, supports_value = _first_present(document, contract.supports_sha256_keys)
    mz_key, mz_value = _first_present(document, contract.mz_slope_keys)
    _, mz_se_value = _first_present(document, contract.mz_slope_se_keys)
    bounds = document.get("split_bounds")
    _, edge_vector = _first_present(document, contract.selection_edge_keys)
    _, growth_vector = _first_present(document, contract.model_growth_keys)
    _, blocks_vector = _first_present(document, contract.traded_blocks_keys)
    return ArtifactFacts(
        path=path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        format_version=version,
        run=str(document.get("run", "")),
        global_step=int(document.get("global_step", -1)),
        split=str(document.get("split", "")),
        context=int(document.get("context", -1)),
        windows=len(document.get("windows") or []),
        scoring=document.get("scoring"),
        corpus_fingerprint=str(document.get("corpus_fingerprint", "")),
        split_bounds=(int(bounds[0]), int(bounds[1]))
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2
        else None,
        eval_window_seed=document.get("eval_window_seed"),
        realized_batch=document.get("realized_batch"),
        realized_steps=document.get("realized_steps"),
        supports_sha256=str(supports_value) if supports_value is not None else None,
        supports_sha256_source=supports_key,
        traded_edge_windows=len(edge_vector) if isinstance(edge_vector, list) else None,
        traded_growth_windows=len(growth_vector) if isinstance(growth_vector, list) else None,
        traded_blocks=len(blocks_vector) if isinstance(blocks_vector, list) else None,
        mz_slope=_read_jsonf64(mz_value),
        mz_slope_se=_read_jsonf64(mz_se_value),
        mz_slope_source=mz_key,
        free_kelly_saturated=free_kelly,
        model_ruin_bars=model_ruin,
        selection_cap_ruin_bars=selection_ruin,
        trade_present=isinstance(trade, dict),
    )


# ---------------------------------------------------------------------------
# Spec
# ---------------------------------------------------------------------------
#
# JSON, not TOML. Three reasons, in order of weight:
#
#  1. The ledger has to contain the arm definition VERBATIM, and the stdlib has a JSON
#     writer but no TOML writer (`tomllib` is read-only). With TOML the driver could only
#     record a lossy re-rendering of what it was handed, which is precisely the thing
#     "justifiable from the ledger alone" forbids.
#  2. Everything the driver already reads and writes is JSON: `*.windows.json`, `meta.json`,
#     the JSONL ledger. One parser, one set of number conventions.
#  3. `json.dumps(sort_keys=True)` gives a stable digest of the spec, so the ledger can pin
#     WHICH spec produced a decision by hash.
#
# JSON's real cost is comments. Mitigated: every object accepts a free-text `"note"`, and
# notes are carried into the ledger rather than discarded.


@dataclasses.dataclass(frozen=True)
class Tier:
    name: str
    steps: int
    validation_windows: int
    seeds: tuple[int, ...]
    artifact: str
    context: int
    extra_flags: tuple[str, ...]
    note: str | None
    #: Appended to every run name at this tier. Non-empty by default: 3.4's
    #: `<wave><n>_<stack-hash>_s<seed>` does not distinguish tiers, and a cumulative campaign
    #: runs the SAME arm and seed at screen and then at confirm, which would collide in
    #: `RunDir::create_fresh` (run_dir.rs:86-98) — the exact failure 3.4's naming rule exists
    #: to prevent. Set `"run_tag": ""` on a tier to get the audit's literal form.
    run_tag: str

    def as_json(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["seeds"] = list(self.seeds)
        payload["extra_flags"] = list(self.extra_flags)
        return payload


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    wave: str
    arm_class: str
    carried_stack: tuple[str, ...]
    flags: tuple[str, ...]
    tiers: tuple[str, ...]
    incumbent: str | None
    is_control: bool
    exempt_from_cull: bool
    supports: str | None
    supports_sha256: str | None
    exempt_reason: str | None
    raw: dict[str, Any]
    note: str | None = None

    @property
    def stack_tag(self) -> str:
        """`<wave><n>_<stack>_s<seed>` naming from 3.4: `C1_a1a2b1_s24301`."""
        return "".join(name.lower() for name in self.carried_stack) or "base"

    def run_name(self, seed: int, tier: Tier) -> str:
        suffix = f"_{tier.run_tag}" if tier.run_tag else ""
        return f"{self.name}_{self.stack_tag}_s{seed}{suffix}"


@dataclasses.dataclass(frozen=True)
class Spec:
    path: Path
    sha256: str
    document: dict[str, Any]
    launcher: str
    launcher_sha256: str
    executable: str
    executable_sha256: str
    mlq_max_parallel_runs: int
    supports: str
    supports_sha256: str
    market_supports: str | None
    market_supports_sha256: str | None
    mlq_time_limit: str
    runs_dir: str
    ledger_path: str
    diagnostic_context: int
    resolution_secs: int
    common_flags: tuple[str, ...]
    tiers: dict[str, Tier]
    arms: dict[str, Arm]
    arm_order: tuple[str, ...]
    control_arm: str
    s_seed: dict[str, Any]
    s_seed_provenance: str | None
    contract: ComparatorContract
    note: str | None

    def artifact_path(self, run_name: str, tier: Tier) -> Path:
        return (
            Path(self.runs_dir) / run_name / "weights" / f"{tier.artifact}.windows.json"
        )

    def resolve_s_seed(self, tier: str, metric: str) -> tuple[float, str]:
        """Return `(s_seed, provenance)` or refuse. NEVER defaulted: see 3.3/W0.5."""
        for scope in (tier, "all"):
            bucket = self.s_seed.get(scope)
            if isinstance(bucket, dict) and metric in bucket:
                value = _read_jsonf64(bucket[metric])
                if value is None or not math.isfinite(value) or value < 0.0:
                    raise RefusalError(
                        f"s_seed[{scope}][{metric}] = {bucket[metric]!r} is not a "
                        "finite non-negative number"
                    )
                if not self.s_seed_provenance:
                    raise RefusalError(
                        "s_seed was supplied without `s_seed_provenance`; a cull threshold "
                        "denominated in an unattributed number is not justifiable from the "
                        "ledger later. Record which W0.5 runs measured it."
                    )
                return value, f"{self.s_seed_provenance} (via s_seed.{scope}.{metric})"
        raise RefusalError(
            f"REFUSING to decide {metric} at tier {tier}: no `s_seed` supplied. Every "
            "threshold in audit 3.3 is denominated in the W0.5 between-seed sd, and the "
            "driver will not substitute a default for a measurement. Run the control on 3 "
            "seeds at this tier, take the between-seed sd of the primary metric, and put it "
            f'in the spec as {{"s_seed": {{"{tier}": {{"{metric}": <value>}}}}, '
            '"s_seed_provenance": "<runs that measured it>"}, or pass --s-seed.'
        )


def _flag_list(raw: Any, what: str) -> tuple[str, ...]:
    if raw is None:
        return ()
    if not isinstance(raw, list) or any(not isinstance(item, (str, int, float)) for item in raw):
        raise SpecError(f"{what} must be a list of scalars, already split into argv tokens")
    return tuple(str(item) for item in raw)

def _required_string(obj: dict[str, Any], key: str, scope: str = "defaults") -> str:
    value = obj.get(key)
    if not isinstance(value, str) or not value.strip():
        raise SpecError(f"`{scope}.{key}` must be an explicit non-empty string")
    return value


def _required_sha256(
    obj: dict[str, Any], key: str, scope: str = "defaults"
) -> str:
    value = _required_string(obj, key, scope).lower()
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise SpecError(f"`{scope}.{key}` must be a full 64-character SHA-256 digest")
    return value


def load_spec(path: Path, overrides: dict[str, Any] | None = None) -> Spec:
    raw_bytes = path.read_bytes()
    try:
        document = json.loads(raw_bytes)
    except json.JSONDecodeError as error:
        raise SpecError(f"{path} is not valid JSON: {error}") from error
    if not isinstance(document, dict):
        raise SpecError(f"{path} must hold a JSON object")
    for key, value in (overrides or {}).items():
        document[key] = value

    defaults = document.get("defaults") or {}
    if not isinstance(defaults, dict):
        raise SpecError("`defaults` must be an object")
    launcher = _required_string(defaults, "launcher")
    launcher_sha256 = _required_sha256(defaults, "launcher_sha256")
    executable = _required_string(defaults, "executable")
    executable_sha256 = _required_sha256(defaults, "executable_sha256")
    mlq_time_limit = _required_string(defaults, "mlq_time_limit")
    supports = _required_string(defaults, "supports")
    supports_sha256 = _required_sha256(defaults, "supports_sha256")
    market_supports = defaults.get("market_supports")
    market_supports_sha256 = defaults.get("market_supports_sha256")
    if (market_supports is None) != (market_supports_sha256 is None):
        raise SpecError(
            "`defaults.market_supports` and `defaults.market_supports_sha256` must be "
            "declared together"
        )
    if market_supports is not None:
        market_supports = _required_string(defaults, "market_supports")
        market_supports_sha256 = _required_sha256(defaults, "market_supports_sha256")
    try:
        mlq_max_parallel_runs = int(defaults["mlq_max_parallel_runs"])
    except (KeyError, TypeError, ValueError) as error:
        raise SpecError("`defaults.mlq_max_parallel_runs` must be an explicit positive integer") from error
    if mlq_max_parallel_runs < 1:
        raise SpecError("`defaults.mlq_max_parallel_runs` must be at least 1")
    if "geometry_change_compare_flags" in defaults:
        raise SpecError(
            "`defaults.geometry_change_compare_flags` is obsolete: geometry-changing arms "
            "always use the fixed --allow-geometry-change / model-growth-only contract"
        )
    contract = ComparatorContract.from_spec(document.get("comparator_contract"))
    diagnostic_context = int(defaults.get("diagnostic_context", DEFAULT_DIAGNOSTIC_CONTEXT))

    tier_docs = document.get("tiers")
    if not isinstance(tier_docs, dict) or not tier_docs:
        raise SpecError("`tiers` must be a non-empty object")
    tiers: dict[str, Tier] = {}
    for name, body in tier_docs.items():
        if not isinstance(body, dict):
            raise SpecError(f"tier {name} must be an object")
        if "steps" not in body:
            raise SpecError(f"tier {name} must declare `steps`")
        seeds = body.get("seeds")
        if not isinstance(seeds, list) or not seeds:
            raise SpecError(f"tier {name} must declare a non-empty `seeds` list")
        if any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds):
            raise SpecError(f"tier {name} seeds must be explicit JSON integers")
        if len(set(seeds)) != len(seeds):
            raise SpecError(f"tier {name} seeds must be unique")
        if any(seed < 0 or seed > 2**64 - 1 for seed in seeds):
            raise SpecError(f"tier {name} seeds must fit the trainer's unsigned 64-bit seed")
        context = int(body.get("context", diagnostic_context))
        default_artifact = (
            f"pretrain_best_diag{context}" if name == "screen" else "pretrain_best"
        )
        tiers[name] = Tier(
            name=name,
            steps=int(body["steps"]),
            validation_windows=int(body.get("validation_windows", 4096)),
            seeds=tuple(int(seed) for seed in seeds),
            artifact=str(body.get("artifact", default_artifact)),
            context=context,
            extra_flags=_flag_list(body.get("extra_flags"), f"tier {name} extra_flags"),
            note=body.get("note"),
            run_tag=str(body.get("run_tag", name)),
        )

    arm_docs = document.get("arms")
    if not isinstance(arm_docs, list) or not arm_docs:
        raise SpecError("`arms` must be a non-empty list")
    arms: dict[str, Arm] = {}
    order: list[str] = []
    controls: list[str] = []
    for body in arm_docs:
        if not isinstance(body, dict):
            raise SpecError("every arm must be an object")
        name = body.get("name")
        if not isinstance(name, str) or not name:
            raise SpecError("every arm needs a non-empty string `name`")
        if name in arms:
            raise SpecError(f"duplicate arm name {name}")
        if name == "latest" or name.startswith(".") or "/" in name:
            raise SpecError(
                f"arm name {name!r} would not survive `validate_run_name` "
                "(shared/src/run_dir.rs:343-354)"
            )
        arm_class = body.get("class")
        is_control = bool(body.get("control", False))
        if arm_class not in ARM_CLASSES:
            raise SpecError(
                f"arm {name}: `class` must be one of {ARM_CLASSES}, got {arm_class!r}. "
                "This is the field that decides which metric may judge the arm (3.3); it is "
                "not defaulted."
            )
        arm_tiers = body.get("tiers") or [tier for tier in TIER_ORDER if tier in tiers]
        if not isinstance(arm_tiers, list) or not arm_tiers:
            raise SpecError(f"arm {name}: `tiers` must be a non-empty list")
        for tier in arm_tiers:
            if tier not in tiers:
                raise SpecError(f"arm {name}: unknown tier {tier!r}")
        exempt = bool(body.get("exempt_from_cull", False))
        reason = body.get("exempt_reason")
        if exempt and not reason:
            raise SpecError(
                f"arm {name}: `exempt_from_cull` requires `exempt_reason` — a correctness "
                "arm adopted regardless of sign (3.2 Wave F4) must say so in the ledger"
            )
        if ("supports" in body) != ("supports_sha256" in body):
            raise SpecError(
                f"arm {name}: `supports` and `supports_sha256` must be declared together"
            )
        arms[name] = Arm(
            name=name,
            wave=str(body.get("wave", name[:1])),
            arm_class=arm_class,
            carried_stack=tuple(str(item) for item in body.get("carried_stack") or []),
            flags=_flag_list(body.get("flags"), f"arm {name} flags"),
            tiers=tuple(str(tier) for tier in arm_tiers),
            incumbent=body.get("incumbent"),
            is_control=is_control,
            exempt_from_cull=exempt,
            supports=(
                _required_string(body, "supports", f"arm {name}")
                if "supports" in body
                else None
            ),
            supports_sha256=(
                _required_sha256(body, "supports_sha256", f"arm {name}")
                if "supports" in body
                else None
            ),
            exempt_reason=reason,
            note=body.get("note"),
            raw=body,
        )
        order.append(name)
        if is_control:
            controls.append(name)

    if len(controls) != 1:
        raise SpecError(
            "exactly one arm must be marked `\"control\": true`; it is the incumbent of "
            f"every arm with an empty carried stack. Found: {controls or 'none'}"
        )
    control = controls[0]
    for arm in arms.values():
        for parent in arm.carried_stack:
            if parent not in arms:
                raise SpecError(f"arm {arm.name}: carried_stack names unknown arm {parent!r}")
        if arm.incumbent is not None and arm.incumbent not in arms:
            raise SpecError(f"arm {arm.name}: incumbent {arm.incumbent!r} is not an arm")
        if arm.is_control and (arm.carried_stack or arm.incumbent):
            raise SpecError("the control arm may not carry a stack or name an incumbent")

    s_seed = document.get("s_seed") or {}
    if not isinstance(s_seed, dict):
        raise SpecError("`s_seed` must be an object keyed by tier (or `all`)")
    # A flat `{"edge_difference": x}` is sugar for `{"all": {...}}`.
    if s_seed and not any(key in s_seed for key in (*tiers, "all")):
        s_seed = {"all": s_seed}

    return Spec(
        path=path,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        document=document,
        launcher=launcher,
        launcher_sha256=launcher_sha256,
        executable=executable,
        executable_sha256=executable_sha256,
        mlq_max_parallel_runs=mlq_max_parallel_runs,
        supports=supports,
        supports_sha256=supports_sha256,
        market_supports=market_supports,
        market_supports_sha256=market_supports_sha256,
        mlq_time_limit=mlq_time_limit,
        runs_dir=str(defaults.get("runs_dir", DEFAULT_RUNS_DIR)),
        ledger_path=str(defaults.get("ledger", "training/campaign_ledger.jsonl")),
        diagnostic_context=diagnostic_context,
        resolution_secs=int(defaults.get("resolution_secs", DEFAULT_RESOLUTION_SECS)),
        common_flags=_flag_list(defaults.get("common_flags"), "defaults.common_flags"),
        tiers=tiers,
        arms=arms,
        arm_order=tuple(order),
        control_arm=control,
        s_seed=s_seed,
        s_seed_provenance=document.get("s_seed_provenance"),
        contract=contract,
        note=document.get("note"),
    )


def incumbent_of(spec: Spec, arm: Arm) -> str | None:
    """The arm this one is measured against. `None` only for the control."""
    if arm.is_control:
        return None
    if arm.incumbent:
        return arm.incumbent
    if arm.carried_stack:
        return arm.carried_stack[-1]
    return spec.control_arm


# ---------------------------------------------------------------------------
# Decision engine
# ---------------------------------------------------------------------------

PRIMARY_METRIC = {
    "geometry_preserving": "edge_difference",
    "geometry_changing": "model_growth_difference",
}
GUARD_METRIC = {
    "geometry_preserving": "dof_r",
    "geometry_changing": "mz_slope",
}
#: 3.3: nats and CRPS are meaningless across a geometry change, and so is the DIFFERENCED
#: edge, whose null leg is support-derived. Naming an arm geometry_changing forbids them.
FORBIDDEN_FOR_GEOMETRY_CHANGE = (
    "nll_difference",
    "conditional_difference",
    "dof_r",
    "dof_s",
    "dof_u",
    "dof_v",
    "dof_w",
    "edge_difference",
    "crps_dof",
)
SCREEN_CULL_MULTIPLE = -1.0
CONFIRM_CARRY_MULTIPLE = 2.0
GUARD_RESOLUTION_SE = 2.0
CORRELATION_FLOOR = 0.9
SATURATION_RISE_LIMIT = 0.05

CULL = "CULL"
SURVIVE = "SURVIVE_SCREEN"
CARRY = "CARRY"
DISCARD = "DISCARD"
ADOPT_EXEMPT = "ADOPT_EXEMPT"


@dataclasses.dataclass
class Tripwire:
    name: str
    status: str  # TRIPPED | CLEAR | UNEVALUATED | NOT_APPLICABLE
    binding: bool
    evidence: str

    def as_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class SeedResult:
    seed: int
    run_name: str
    launch_exit_code: int | None
    log_path: str | None
    artifact: ArtifactFacts | None
    incumbent_artifact: ArtifactFacts | None
    comparison: Comparison | None

    def as_json(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "run_name": self.run_name,
            "launch_exit_code": self.launch_exit_code,
            "log_path": self.log_path,
            "artifact": self.artifact.as_json() if self.artifact else None,
            "incumbent_artifact": self.incumbent_artifact.as_json()
            if self.incumbent_artifact
            else None,
            "comparison": self.comparison.as_json() if self.comparison else None,
        }


def evaluate_tripwires(arm: Arm, seeds: Sequence[SeedResult]) -> list[Tripwire]:
    """The six hard tripwires of 3.3, in order, over every seed of the arm.

    A binding tripwire that cannot be evaluated is reported UNEVALUATED rather than passed.
    Screen tier only culls, so an unevaluated tripwire does not kill an arm there; the carry
    gate treats one as blocking, because carrying on unmeasured safety is the failure mode the
    list exists to prevent. Contractually meaningless checks are instead NOT_APPLICABLE and
    non-binding (notably cross-geometry correlation).
    """
    wires: list[Tripwire] = []

    # 1. Non-finite loss or gradient norm.
    hits: list[str] = []
    crashes: list[str] = []
    for seed in seeds:
        if seed.launch_exit_code not in (0, None):
            crashes.append(f"{seed.run_name} exited {seed.launch_exit_code}")
        text = ""
        if seed.log_path and Path(seed.log_path).exists():
            text = Path(seed.log_path).read_text(errors="replace")
        for pattern in NONFINITE_PATTERNS:
            for line in text.splitlines():
                if pattern in line:
                    hits.append(f"{seed.run_name}: {line.strip()}")
    if hits or crashes:
        wires.append(
            Tripwire(
                "nonfinite_loss_or_grad_norm",
                "TRIPPED",
                True,
                "; ".join(hits + crashes)[:2000],
            )
        )
    else:
        wires.append(
            Tripwire(
                "nonfinite_loss_or_grad_norm",
                "CLEAR",
                True,
                "no ensure_finite_step_metrics message in any seed's captured output; "
                "every launch exited 0",
            )
        )

    # 2. free_kelly_saturated rises by more than 5 pp.
    deltas: list[str] = []
    tripped = False
    unmeasured = False
    for seed in seeds:
        cand = seed.artifact.free_kelly_saturated if seed.artifact else None
        base = (
            seed.incumbent_artifact.free_kelly_saturated
            if seed.incumbent_artifact
            else None
        )
        if cand is None or base is None:
            unmeasured = True
            deltas.append(
                f"{seed.run_name}: unavailable "
                f"(candidate {cand!r}, incumbent {base!r}; `trade` is absent on a vector "
                "written from a pass the bench did not ride, pretrain_stats.rs:728-736)"
            )
            continue
        rise = cand - base
        deltas.append(f"{seed.run_name}: {base:.4f} -> {cand:.4f} ({rise:+.4f})")
        if rise > SATURATION_RISE_LIMIT:
            tripped = True
    wires.append(
        Tripwire(
            "free_kelly_saturated_rise_over_5pp",
            "TRIPPED" if tripped else ("UNEVALUATED" if unmeasured else "CLEAR"),
            True,
            "; ".join(deltas) or "no seeds",
        )
    )

    # 3. ruin_bars > 0 where the incumbent had 0.
    notes: list[str] = []
    tripped = False
    unmeasured = False
    for seed in seeds:
        for label, getter in (
            ("model policy", lambda a: a.model_ruin_bars),
            (f"cap {SELECTION_CAP:.2f}x", lambda a: a.selection_cap_ruin_bars),
        ):
            cand = getter(seed.artifact) if seed.artifact else None
            base = getter(seed.incumbent_artifact) if seed.incumbent_artifact else None
            if cand is None or base is None:
                unmeasured = True
                notes.append(f"{seed.run_name} {label}: unavailable")
                continue
            notes.append(f"{seed.run_name} {label}: incumbent {base}, candidate {cand}")
            if base == 0 and cand > 0:
                tripped = True
    wires.append(
        Tripwire(
            "ruin_bars_appeared",
            "TRIPPED" if tripped else ("UNEVALUATED" if unmeasured else "CLEAR"),
            True,
            "; ".join(notes) or "no seeds",
        )
    )

    # 4. Paired correlation below 0.9. The geometry-changing comparator deliberately
    # withholds correlation: it is decoded through bins and no correction can make two
    # discretizations comparable. Record that contractual absence as non-binding N/A rather
    # than turning every otherwise-qualified geometry arm into an unevaluated discard.
    if arm.arm_class == "geometry_changing":
        wires.append(
            Tripwire(
                "paired_correlation_below_0.9",
                "NOT_APPLICABLE",
                False,
                "not applicable: the arm is declared geometry_changing and "
                "--allow-geometry-change deliberately withholds cross-geometry correlation",
            )
        )
    else:
        notes = []
        tripped = False
        unmeasured = False
        for seed in seeds:
            value = seed.comparison.correlation if seed.comparison else None
            if value is None:
                unmeasured = True
                notes.append(f"{seed.run_name}: no parsed correlation")
                continue
            notes.append(f"{seed.run_name}: {value:.4f}")
            if not math.isfinite(value) or value < CORRELATION_FLOOR:
                tripped = True
        wires.append(
            Tripwire(
                "paired_correlation_below_0.9",
                "TRIPPED" if tripped else ("UNEVALUATED" if unmeasured else "CLEAR"),
                True,
                "; ".join(notes) or "no seeds",
            )
        )

    # 5. Resolved dof_difference[r] regression at 2.0 SE. Geometry-preserving arms only:
    #    r-NLL is measured in the geometry, so it says nothing once the geometry moves.
    if arm.arm_class != "geometry_preserving":
        wires.append(
            Tripwire(
                "r_nll_regression_at_2se",
                "CLEAR",
                False,
                "not applicable: the arm is declared geometry_changing, and every nll_* is "
                "FORBIDDEN there (3.3)",
            )
        )
    else:
        notes = []
        tripped = False
        unmeasured = False
        for seed in seeds:
            dispersion = (seed.comparison.dispersions.get("dof_r") if seed.comparison else None)
            if dispersion is None:
                unmeasured = True
                notes.append(f"{seed.run_name}: no `delta r` line parsed")
                continue
            regressed = dispersion.mean > 0.0 and dispersion.resolved_at(GUARD_RESOLUTION_SE)
            notes.append(
                f"{seed.run_name}: {dispersion.mean:+.5f} +/- {dispersion.se:.5f} nats "
                f"({'|mean| > 2 SE' if dispersion.resolved_at(GUARD_RESOLUTION_SE) else 'unresolved'}, "
                f"{'REGRESSION' if regressed else 'ok'})"
            )
            if regressed:
                tripped = True
        wires.append(
            Tripwire(
                "r_nll_regression_at_2se",
                "TRIPPED" if tripped else ("UNEVALUATED" if unmeasured else "CLEAR"),
                True,
                "; ".join(notes) or "no seeds",
            )
        )

    # 6. supports_sha256 mismatch on an arm not declared geometry-changing. Checked off the
    #    two artifacts directly, so it does not depend on any error-message wording; the
    #    comparator's own refusal is picked up as corroborating evidence.
    notes = []
    tripped = False
    unmeasured = False
    for seed in seeds:
        cand = seed.artifact.supports_sha256 if seed.artifact else None
        base = seed.incumbent_artifact.supports_sha256 if seed.incumbent_artifact else None
        refusal = ""
        if seed.comparison and seed.comparison.refusal == "geometry":
            refusal = " (pretrain-compare REFUSED the pair on bin geometry)"
            tripped = True
        if cand is None or base is None:
            unmeasured = True
            notes.append(
                f"{seed.run_name}: unavailable{refusal} "
                "(W0.2's `supports_sha256` is not on this artifact)"
            )
            continue
        notes.append(f"{seed.run_name}: incumbent {base[:12]}, candidate {cand[:12]}{refusal}")
        if cand != base:
            tripped = True
    if arm.arm_class == "geometry_changing":
        wires.append(
            Tripwire(
                "undeclared_geometry_change",
                "CLEAR",
                False,
                "arm is declared geometry_changing, so a supports mismatch is expected. "
                "Observed: " + ("; ".join(notes) or "no seeds"),
            )
        )
    else:
        wires.append(
            Tripwire(
                "undeclared_geometry_change",
                "TRIPPED" if tripped else ("UNEVALUATED" if unmeasured else "CLEAR"),
                True,
                "; ".join(notes) or "no seeds",
            )
        )
    return wires


@dataclasses.dataclass
class Decision:
    arm: str
    tier: str
    arm_class: str
    incumbent: str | None
    verdict: str
    primary_metric: str
    primary_per_seed: list[float]
    primary_mean: float | None
    se_win: float | None
    s_seed: float | None
    s_seed_provenance: str | None
    threshold_expression: str
    threshold_value: float | None
    guard_metric: str
    guard_verdict: str
    guard_evidence: str
    tripwires: list[Tripwire]
    forbidden_metrics_withheld: list[str]
    justification: str

    def as_json(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["primary_per_seed"] = [_json_float(v) for v in self.primary_per_seed]
        payload["primary_mean"] = _json_float(self.primary_mean)
        payload["se_win"] = _json_float(self.se_win)
        payload["s_seed"] = _json_float(self.s_seed)
        payload["threshold_value"] = _json_float(self.threshold_value)
        payload["tripwires"] = [wire.as_json() for wire in self.tripwires]
        return payload


def evaluate_guard(
    arm: Arm, seeds: Sequence[SeedResult]
) -> tuple[str, str]:
    """Return `(verdict, evidence)` where verdict is VETO | PASS | UNEVALUATED.

    Geometry-preserving: `dof_difference[r]`, and a resolved regression vetoes.

    Geometry-changing: the Mincer-Zarnowitz mean slope, which must not move AWAY from 1.0.
    `beta` is a LEVEL, not a difference, and `pretrain-compare` does not print a paired MZ
    difference — W0.3 persists `calibration.{mean_beta, mean_beta_se}` on each artifact
    instead. So this is an UNPAIRED comparison of two published levels, and it is scored
    against 2.0 x the LARGER of the two published standard errors: the conservative choice,
    which makes a veto harder rather than easier. The evidence string says so on every row,
    because a reader must not mistake this band for the paired one the primary metric uses.
    Deriving a paired MZ interval here would mean implementing a statistic, which this driver
    does not do.
    """
    if arm.arm_class == "geometry_preserving":
        notes: list[str] = []
        veto = False
        unmeasured = False
        for seed in seeds:
            dispersion = seed.comparison.dispersions.get("dof_r") if seed.comparison else None
            if dispersion is None:
                unmeasured = True
                notes.append(f"{seed.run_name}: no `delta r` line")
                continue
            regressed = dispersion.mean > 0.0 and dispersion.resolved_at(GUARD_RESOLUTION_SE)
            notes.append(
                f"{seed.run_name}: delta r {dispersion.mean:+.5f} +/- {dispersion.se:.5f} nats"
                f"{' RESOLVED REGRESSION' if regressed else ''}"
            )
            veto = veto or regressed
        return (
            "VETO" if veto else ("UNEVALUATED" if unmeasured else "PASS"),
            "; ".join(notes) or "no seeds",
        )

    notes = []
    veto = False
    unmeasured = False
    for seed in seeds:
        cand = seed.artifact if seed.artifact else None
        base = seed.incumbent_artifact if seed.incumbent_artifact else None
        beta_cand = cand.mz_slope if cand else None
        beta_base = base.mz_slope if base else None
        if beta_cand is None or beta_base is None:
            unmeasured = True
            notes.append(
                f"{seed.run_name}: MZ slope unavailable on "
                + ", ".join(
                    side
                    for side, value in (("candidate", beta_cand), ("incumbent", beta_base))
                    if value is None
                )
                + " (W0.3 writes it to `calibration.mean_beta`; it is null on a pass that ran "
                "no bench)"
            )
            continue
        drift = abs(beta_cand - 1.0) - abs(beta_base - 1.0)
        ses = [
            value
            for value in (cand.mz_slope_se if cand else None, base.mz_slope_se if base else None)
            if value is not None and math.isfinite(value) and value > 0.0
        ]
        band = GUARD_RESOLUTION_SE * max(ses) if ses else None
        resolved = band is not None and drift > band
        notes.append(
            f"{seed.run_name}: beta {beta_base:.4f} -> {beta_cand:.4f}, |beta-1| "
            f"{abs(beta_base - 1.0):.4f} -> {abs(beta_cand - 1.0):.4f}, drift {drift:+.4f}"
            + (
                f" vs UNPAIRED band 2.0 x max(published mean_beta_se) = {band:.4f}"
                if band is not None
                else " with NO published mean_beta_se, so the drift cannot be resolved"
            )
            + (" RESOLVED DRIFT AWAY FROM 1.0" if resolved else "")
        )
        if band is None:
            unmeasured = True
        veto = veto or resolved
    return (
        "VETO" if veto else ("UNEVALUATED" if unmeasured else "PASS"),
        "; ".join(notes) or "no seeds",
    )


def decide(
    spec: Spec, arm: Arm, tier: Tier, seeds: Sequence[SeedResult]
) -> Decision:
    """Apply 3.3 to one settled arm at one tier. Raises `RefusalError` without `s_seed`."""
    metric = PRIMARY_METRIC[arm.arm_class]
    guard_metric = GUARD_METRIC[arm.arm_class]
    withheld = list(FORBIDDEN_FOR_GEOMETRY_CHANGE) if arm.arm_class == "geometry_changing" else []
    tripwires = evaluate_tripwires(arm, seeds)
    guard_verdict, guard_evidence = evaluate_guard(arm, seeds)

    binding_trips = [wire for wire in tripwires if wire.binding and wire.status == "TRIPPED"]
    unevaluated = [wire for wire in tripwires if wire.binding and wire.status == "UNEVALUATED"]

    per_seed: list[float] = []
    ses: list[float] = []
    missing: list[str] = []
    for seed in seeds:
        dispersion = seed.comparison.dispersions.get(metric) if seed.comparison else None
        if dispersion is None:
            missing.append(seed.run_name)
            continue
        per_seed.append(dispersion.mean)
        ses.append(dispersion.se)
    mean = sum(per_seed) / len(per_seed) if per_seed else None
    # `se_win` is the paired block-bootstrap SE of ONE comparison, undivided: 3.3 divides
    # only `s_seed` by the seed count because every seed is scored on the SAME pinned
    # windows, so window noise is common across replicates and does not shrink with them.
    se_win = sum(ses) / len(ses) if ses else None

    # Tripwire 1 is fatal at every tier including for a cull-exempt arm: a crashed run has
    # no result to adopt.
    fatal = [wire for wire in binding_trips if wire.name == "nonfinite_loss_or_grad_norm"]

    if arm.exempt_from_cull and not fatal:
        return Decision(
            arm=arm.name,
            tier=tier.name,
            arm_class=arm.arm_class,
            incumbent=incumbent_of(spec, arm),
            verdict=ADOPT_EXEMPT,
            primary_metric=metric,
            primary_per_seed=per_seed,
            primary_mean=mean,
            se_win=se_win,
            s_seed=None,
            s_seed_provenance=None,
            threshold_expression="exempt from the cull rules by declaration",
            threshold_value=None,
            guard_metric=guard_metric,
            guard_verdict=guard_verdict,
            guard_evidence=guard_evidence,
            tripwires=tripwires,
            forbidden_metrics_withheld=withheld,
            justification=(
                f"{arm.name} is declared exempt from the cull rules: {arm.exempt_reason}. "
                f"Primary {metric} recorded but not decisive: mean "
                f"{_fmt(mean)} over {len(per_seed)} seed(s). Tripwires evaluated and "
                "reported; only a non-finite loss would have blocked adoption."
            ),
        )

    if fatal:
        return _tripwire_cull(spec, arm, tier, metric, guard_metric, per_seed, mean, se_win,
                              guard_verdict, guard_evidence, tripwires, withheld, fatal)

    if binding_trips:
        return _tripwire_cull(spec, arm, tier, metric, guard_metric, per_seed, mean, se_win,
                              guard_verdict, guard_evidence, tripwires, withheld, binding_trips)

    if missing or mean is None:
        raise RefusalError(_explain_missing_primary(arm, tier, metric, seeds, missing))

    s_seed, provenance = spec.resolve_s_seed(tier.name, metric)

    if tier.name == "screen":
        threshold = SCREEN_CULL_MULTIPLE * s_seed
        expression = f"cull if mean({metric}) < {SCREEN_CULL_MULTIPLE:+.1f} * s_seed"
        culled = mean < threshold
        verdict = CULL if culled else SURVIVE
        justification = (
            f"Screen tier, {len(per_seed)} seed(s). mean({metric}) = {mean:+.6f} bps/bar "
            f"from per-seed {[round(v, 6) for v in per_seed]}. Threshold "
            f"{SCREEN_CULL_MULTIPLE:+.1f} * s_seed = {threshold:+.6f} with s_seed = "
            f"{s_seed:.6f} [{provenance}]. "
            + (
                "Below the threshold: a clear loser, culled."
                if culled
                else "At or above the threshold: survives to confirm. NOTHING is carried at "
                "screen tier (3.3)."
            )
            + f" Guard {guard_metric}: {guard_verdict} — {guard_evidence}."
            + " Guard does not gate at screen tier; recorded for the confirm decision."
        )
    elif tier.name in ("confirm", "deploy"):
        band = math.sqrt(s_seed * s_seed / max(len(per_seed), 1) + (se_win or 0.0) ** 2)
        threshold = CONFIRM_CARRY_MULTIPLE * band
        expression = (
            f"carry if mean({metric}) > {CONFIRM_CARRY_MULTIPLE:+.1f} * "
            f"sqrt(s_seed^2/{len(per_seed)} + se_win^2) and no guard veto"
        )
        cleared = mean > threshold
        vetoed = guard_verdict != "PASS"
        verdict = CARRY if (cleared and not vetoed) else DISCARD
        justification = (
            f"{tier.name.capitalize()} tier, {len(per_seed)} seed(s). mean({metric}) = "
            f"{mean:+.6f} bps/bar from per-seed {[round(v, 6) for v in per_seed]}. "
            f"s_seed = {s_seed:.6f} [{provenance}], se_win = {_fmt(se_win)} (mean paired "
            f"block-bootstrap SE, undivided: every seed scores the SAME pinned windows). "
            f"Threshold {CONFIRM_CARRY_MULTIPLE:+.1f} * sqrt({s_seed:.6f}^2/"
            f"{len(per_seed)} + {_fmt(se_win)}^2) = {threshold:+.6f}. "
            + ("Cleared the band. " if cleared else "Did NOT clear the band. ")
            + f"Guard {guard_metric}: {guard_verdict} — {guard_evidence}. "
            + (
                f"Verdict {CARRY}: the arm enters the carried stack."
                if verdict == CARRY
                else f"Verdict {DISCARD}: keep the incumbent."
            )
        )
        if unevaluated and verdict == CARRY:
            verdict = DISCARD
            justification += (
                " OVERRIDDEN to DISCARD: binding tripwires could not be evaluated ("
                + ", ".join(wire.name for wire in unevaluated)
                + "). A carry on unmeasured safety is exactly what the tripwire list exists "
                "to prevent; re-run with the instrumentation present."
            )
    else:
        raise SpecError(f"tier {tier.name!r} has no rule in audit 3.3")

    return Decision(
        arm=arm.name,
        tier=tier.name,
        arm_class=arm.arm_class,
        incumbent=incumbent_of(spec, arm),
        verdict=verdict,
        primary_metric=metric,
        primary_per_seed=per_seed,
        primary_mean=mean,
        se_win=se_win,
        s_seed=s_seed,
        s_seed_provenance=provenance,
        threshold_expression=expression,
        threshold_value=threshold,
        guard_metric=guard_metric,
        guard_verdict=guard_verdict,
        guard_evidence=guard_evidence,
        tripwires=tripwires,
        forbidden_metrics_withheld=withheld,
        justification=justification,
    )


#: Human text for each of the comparator's hard refusals. A refusal is "not comparable",
#: which is a different finding from "no effect" and must never collapse into it.
REFUSAL_EXPLANATION = {
    "geometry": (
        "`pretrain-compare` REFUSED the pair on bin geometry. For a geometry_preserving arm "
        "this is tripwire 6 and the arm is not what it claims to be. Geometry-changing arms "
        "are always invoked with `--allow-geometry-change`; a geometry refusal there means an "
        "artifact did not record a usable supports identity or the pinned executable did not "
        "honour the model-growth-only contract. It is not evidence of no effect."
    ),
    "traded_prefix": (
        "`pretrain-compare` REFUSED the pair because the two runs traded different prefixes "
        "of the pinned window set. The per-window economic vectors are over the TRADED PREFIX, "
        "so a differing prefix length is not a pairing. This is unrelated to the treatment and "
        "must not be read as a result."
    ),
    "unclassified": (
        "`pretrain-compare` exited non-zero for a reason this driver does not recognize. Read "
        "the `stderr` recorded in the ledger beside this refusal."
    ),
}


def _explain_missing_primary(
    arm: Arm, tier: Tier, metric: str, seeds: Sequence[SeedResult], missing: Sequence[str]
) -> str:
    refusals = {
        seed.comparison.refusal: seed.comparison.stderr.strip()[-600:]
        for seed in seeds
        if seed.comparison and seed.comparison.refusal
    }
    stated = {
        f"{seed.run_name}: {reason}"
        for seed in seeds
        if seed.comparison
        for key, reason in seed.comparison.absent.items()
        if key == metric
    }
    unmapped = sorted(
        {line for seed in seeds if seed.comparison for line in seed.comparison.unmapped}
    )
    parts = [
        f"REFUSING to decide {arm.name} at tier {tier.name}: the primary metric `{metric}` "
        f"is not available for {', '.join(missing) or 'any seed'}. There is nothing economic "
        "to judge the arm on, and the aggregate nats `difference` is explicitly forbidden as "
        "a substitute (audit 3.3)."
    ]
    for kind, stderr in refusals.items():
        parts.append(REFUSAL_EXPLANATION[kind] + (f" stderr: {stderr}" if stderr else ""))
    if stated:
        parts.append("The comparator stated the absence: " + "; ".join(sorted(stated)) + ".")
    if unmapped:
        parts.append(
            "Unrecognized dispersion lines, which may be this metric under a changed label — "
            "correct `comparator_contract.label_patterns` in the spec if so: "
            + json.dumps(unmapped)
        )
    return " ".join(parts)


def _tripwire_cull(
    spec: Spec,
    arm: Arm,
    tier: Tier,
    metric: str,
    guard_metric: str,
    per_seed: list[float],
    mean: float | None,
    se_win: float | None,
    guard_verdict: str,
    guard_evidence: str,
    tripwires: list[Tripwire],
    withheld: list[str],
    trips: list[Tripwire],
) -> Decision:
    return Decision(
        arm=arm.name,
        tier=tier.name,
        arm_class=arm.arm_class,
        incumbent=incumbent_of(spec, arm),
        verdict=CULL,
        primary_metric=metric,
        primary_per_seed=per_seed,
        primary_mean=mean,
        se_win=se_win,
        s_seed=None,
        s_seed_provenance=None,
        threshold_expression="hard tripwire: immediate cull at any tier, regardless of the "
        "primary metric (3.3)",
        threshold_value=None,
        guard_metric=guard_metric,
        guard_verdict=guard_verdict,
        guard_evidence=guard_evidence,
        tripwires=tripwires,
        forbidden_metrics_withheld=withheld,
        justification="Culled by hard tripwire, no threshold consulted: "
        + "; ".join(f"{wire.name} -> {wire.evidence}" for wire in trips)
        + f". Primary {metric} mean {_fmt(mean)} recorded for the record only.",
    )


def _fmt(value: float | None) -> str:
    return "unavailable" if value is None else f"{value:+.6f}"


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


class Ledger:
    """Append-only JSONL. One `write` per record, flushed and fsynced.

    Append-only because the campaign's whole defence against carrying noise is that a cull
    is re-derivable months later from what the driver actually read. A mutable state file
    would let a re-run overwrite the evidence for a decision already taken.
    """

    def __init__(self, path: Path, campaign_id: str, dry_run: bool) -> None:
        self.path = path
        self.campaign_id = campaign_id
        self.dry_run = dry_run
        self._lock = threading.Lock()
        if not dry_run:
            path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: str, **payload: Any) -> dict[str, Any]:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace(
                "+00:00", "Z"
            ),
            "record": record,
            "campaign_id": self.campaign_id,
            "driver_version": DRIVER_VERSION,
            **payload,
        }
        line = json.dumps(entry, sort_keys=True, allow_nan=False)
        if self.dry_run:
            return entry
        with self._lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return entry

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records = []
        for number, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise DriverError(
                    f"{self.path}:{number} is not valid JSON ({error}); the ledger is "
                    "append-only and must not be hand-edited"
                ) from error
        return records


@dataclasses.dataclass
class ResumeState:
    """What a previous invocation of this driver already established, replayed off the ledger."""

    settled: dict[tuple[str, str], str]
    launched: dict[str, dict[str, Any]]
    completed: dict[str, dict[str, Any]]
    retained: set[str]

    @staticmethod
    def replay(records: Sequence[dict[str, Any]]) -> "ResumeState":
        settled: dict[tuple[str, str], str] = {}
        launched: dict[str, dict[str, Any]] = {}
        completed: dict[str, dict[str, Any]] = {}
        retained: set[str] = set()
        for entry in records:
            kind = entry.get("record")
            if kind == "arm_launch":
                launched[entry["run_name"]] = entry
            elif kind == "arm_complete" and entry.get("exit_code") == 0:
                completed[entry["run_name"]] = entry
            elif kind == "decision":
                decision = entry.get("decision") or {}
                settled[(decision.get("arm"), decision.get("tier"))] = decision.get("verdict")
            elif kind == "retention":
                retained.add(entry.get("arm"))
        return ResumeState(settled, launched, completed, retained)

    def carried(self) -> set[str]:
        return {arm for (arm, _tier), verdict in self.settled.items() if verdict in (CARRY, ADOPT_EXEMPT)}


# ---------------------------------------------------------------------------
# Process execution
# ---------------------------------------------------------------------------


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved_file(
    cwd: Path,
    configured: str,
    expected_sha256: str,
    label: str,
    *,
    require_executable: bool = True,
) -> dict[str, str]:
    path = Path(configured)
    resolved = (cwd / path).resolve() if not path.is_absolute() else path.resolve()
    if not resolved.is_file():
        raise SpecError(f"{label} {configured!r} does not resolve to a regular file")
    actual = _sha256_file(resolved)
    if actual != expected_sha256:
        raise SpecError(
            f"{label} digest mismatch for {resolved}: spec pins {expected_sha256}, found {actual}"
        )
    if require_executable and not os.access(resolved, os.X_OK):
        raise SpecError(f"{label} {resolved} is not executable")
    return {"configured_path": configured, "resolved_path": str(resolved), "sha256": actual}


def _option_values(flags: Sequence[str], option: str) -> list[str | None]:
    values: list[str | None] = []
    index = 0
    while index < len(flags):
        token = flags[index]
        if token == option:
            values.append(flags[index + 1] if index + 1 < len(flags) else None)
            index += 2
            continue
        if token.startswith(option + "="):
            values.append(token.split("=", 1)[1])
        index += 1
    return values


def _one_value(flags: Sequence[str], option: str, context: str) -> str:
    values = _option_values(flags, option)
    if len(values) != 1 or values[0] in (None, ""):
        raise SpecError(f"{context} must contain exactly one explicit `{option} VALUE` pin")
    return str(values[0])


def validate_launch_contract(spec: Spec, cwd: Path) -> dict[str, Any]:
    """Resolve every campaign input before any mlq submission.

    Pin validation is performed on every arm/tier's final argv, rather than merely checking
    `common_flags`, so a later arm-local override cannot silently defeat the campaign contract.
    """
    if re.fullmatch(r"\d+(?:ms|s|m|h|d)", spec.mlq_time_limit) is None:
        raise SpecError("`defaults.mlq_time_limit` must be explicit, e.g. `12h` or `90m`")
    launcher = _resolved_file(cwd, spec.launcher, spec.launcher_sha256, "launcher")
    executable = _resolved_file(cwd, spec.executable, spec.executable_sha256, "executable")
    if (spec.market_supports is None) != (spec.market_supports_sha256 is None):
        raise SpecError(
            "market supports path and digest must both be present in the resolved spec"
        )
    market_supports = (
        _resolved_file(
            cwd,
            spec.market_supports,
            spec.market_supports_sha256,
            "market supports",
            require_executable=False,
        )
        if spec.market_supports is not None and spec.market_supports_sha256 is not None
        else None
    )
    forbidden_driver_options = (
        "--run",
        "--steps",
        "--seed",
        "--validation-windows",
        "--supports",
        "--derive-split-bounds",
        GEOMETRY_CHANGE_COMPARE_FLAG,
    )
    pinned: set[tuple[str, int]] = set()
    supports_identities: dict[str, dict[str, str]] = {}
    for tier in spec.tiers.values():
        if len(set(tier.seeds)) != len(tier.seeds):
            raise SpecError(f"tier {tier.name} seeds must be unique")
        for arm in spec.arms.values():
            if tier.name not in arm.tiers:
                continue
            context = f"resolved flags for arm {arm.name} at tier {tier.name}"
            flags = (*spec.common_flags, *tier.extra_flags, *arm.flags)
            for option in forbidden_driver_options:
                if _option_values(flags, option):
                    raise SpecError(f"{context} may not override driver-owned `{option}`")
            split = _one_value(flags, "--split-bounds", context)
            try:
                left_text, right_text = split.split(",", 1)
                split_pair = (int(left_text), int(right_text))
            except (ValueError, TypeError) as error:
                raise SpecError(f"{context} has malformed --split-bounds {split!r}") from error
            if split_pair[0] >= split_pair[1]:
                raise SpecError(f"{context} has non-ascending --split-bounds {split!r}")
            batch_text = _one_value(flags, "--batch-size", context)
            try:
                batch = int(batch_text)
            except ValueError as error:
                raise SpecError(f"{context} has non-integer --batch-size {batch_text!r}") from error
            if batch < 1:
                raise SpecError(f"{context} has non-positive --batch-size {batch}")
            for required in ("--freeze-supports", "--freeze-market-supports", "--exact-batch"):
                if sum(token == required for token in flags) != 1:
                    raise SpecError(f"{context} must contain exactly one `{required}`")
            if arm.arm_class == "geometry_changing" and not arm.supports:
                raise SpecError(
                    f"{context} must declare an arm-level supports path for its changed geometry"
                )
            if arm.arm_class == "geometry_preserving" and arm.supports not in (None, spec.supports):
                raise SpecError(
                    f"{context} may not replace the geometry-preserving default supports"
                )
            configured_supports = arm.supports or spec.supports
            expected_supports_sha256 = (
                arm.supports_sha256 if arm.supports is not None else spec.supports_sha256
            )
            if expected_supports_sha256 is None:
                raise SpecError(
                    f"{context} declares supports {configured_supports!r} without "
                    "`supports_sha256`"
                )
            identity = _resolved_file(
                cwd,
                configured_supports,
                expected_supports_sha256,
                f"{context} supports",
                require_executable=False,
            )
            previous = supports_identities.setdefault(configured_supports, identity)
            if previous != identity:
                raise SpecError(f"supports identity changed while resolving {configured_supports}")
            pinned.add((split, batch))
    default_identity = supports_identities.get(spec.supports)
    if default_identity is None:
        raise SpecError("no controlled geometry-preserving arm resolved defaults.supports")
    for arm in spec.arms.values():
        if (
            arm.arm_class == "geometry_changing"
            and arm.supports in supports_identities
            and supports_identities[arm.supports]["sha256"] == default_identity["sha256"]
        ):
            raise SpecError(
                f"arm {arm.name} declares a geometry change but pins the control supports digest"
            )
    if len(pinned) != 1 or not supports_identities:
        raise SpecError("all controlled arms must resolve to one split/exact-batch contract")
    split, batch = next(iter(pinned))
    return {
        "launcher": launcher,
        "executable": executable,
        "supports": dict(sorted(supports_identities.items())),
        "market_supports": market_supports,
        "split_bounds": [int(value) for value in split.split(",", 1)],
        "batch_size": batch,
        "exact_batch": True,
        "freeze_supports": True,
        "freeze_market_supports": True,
        "tier_seeds": {name: list(tier.seeds) for name, tier in spec.tiers.items()},
        "mlq": {
            "max_parallel_runs": spec.mlq_max_parallel_runs,
            "time_limit": spec.mlq_time_limit,
        },
    }


def _resolved_launch_argv(argv: Sequence[str], identity: dict[str, Any]) -> list[str]:
    return [
        identity["launcher"]["resolved_path"],
        identity["executable"]["resolved_path"],
        *argv[2:],
    ]


EXECUTION_PIN_GUARD = """\
set -euo pipefail
while [[ $# -ge 3 && "$1" != "--" ]]; do
    expected="$1"
    path="$2"
    label="$3"
    if ! digest_line="$(sha256sum -- "$path")"; then
        printf 'campaign SHA-256 input missing or unreadable for %s (%s)\n' \
            "$label" "$path" >&2
        exit 125
    fi
    actual="${digest_line%% *}"
    if [[ "$actual" != "$expected" ]]; then
        printf 'campaign SHA-256 mismatch for %s (%s): expected %s, found %s\n' \
            "$label" "$path" "$expected" "$actual" >&2
        exit 125
    fi
    shift 3
done
if [[ $# -eq 0 || "$1" != "--" ]]; then
    printf 'campaign SHA-256 guard: missing command delimiter\n' >&2
    exit 125
fi
shift
exec "$@"
"""


def _execution_verified_argv(
    resolved_argv: Sequence[str],
    launch_contract: dict[str, Any],
    configured_supports: str,
) -> list[str]:
    """Wrap an mlq payload so mutable campaign inputs are re-authenticated on the worker.

    Pre-submit validation catches bad specs. This guard closes the queue-time race: mlq may
    start much later, so the worker hashes every selected path immediately before `exec`.
    """
    pins = [
        ("launcher", launch_contract["launcher"]),
        ("executable", launch_contract["executable"]),
        ("supports", launch_contract["supports"][configured_supports]),
    ]
    if launch_contract.get("market_supports") is not None:
        pins.append(("market_supports", launch_contract["market_supports"]))
    guarded = ["/bin/bash", "-lc", EXECUTION_PIN_GUARD, "campaign-sha256-guard"]
    for label, identity in pins:
        guarded.extend((identity["sha256"], identity["resolved_path"], label))
    return [*guarded, "--", *resolved_argv]


@dataclasses.dataclass
class MlqRun:
    job_id: int
    submit_argv: list[str]
    exit_code: int
    stdout: str
    stderr: str


def _mlq_job_id(document: Any) -> int:
    if isinstance(document, dict):
        for key in ("id", "jobId", "job_id"):
            value = document.get(key)
            if isinstance(value, int):
                return value
        for value in document.values():
            try:
                return _mlq_job_id(value)
            except DriverError:
                pass
    raise DriverError(f"mlq submit returned JSON without a numeric job id: {document!r}")


def run_through_mlq(
    argv: Sequence[str],
    run_name: str,
    cwd: Path,
    max_parallel_runs: int,
    time_limit: str,
    idempotency_key: str,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    on_submitted: Callable[[int, list[str]], None] | None = None,
) -> MlqRun:
    submit_argv = [
        "mlq",
        "submit",
        "--json",
        "--idempotency-key",
        idempotency_key,
        "--name",
        run_name,
        "--cwd",
        str(cwd),
        "--max-parallel-runs",
        str(max_parallel_runs),
        "--time-limit",
        time_limit,
        "--",
        *argv,
    ]
    submitted = command_runner(submit_argv, capture_output=True, text=True, check=False)
    if submitted.returncode != 0:
        raise DriverError(f"mlq submit refused {run_name}: {submitted.stderr.strip()}")
    try:
        job_id = _mlq_job_id(json.loads(submitted.stdout))
    except json.JSONDecodeError as error:
        raise DriverError(f"mlq submit returned invalid JSON: {submitted.stdout!r}") from error
    if on_submitted is not None:
        on_submitted(job_id, submit_argv)
    waited = command_runner(
        ["mlq", "wait", "--json", str(job_id)], capture_output=True, text=True, check=False
    )
    stdout = command_runner(
        ["mlq", "logs", str(job_id)], capture_output=True, text=True, check=False
    )
    stderr = command_runner(
        ["mlq", "logs", "--stderr", str(job_id)], capture_output=True, text=True, check=False
    )
    if stdout.returncode != 0 or stderr.returncode != 0:
        raise DriverError(f"mlq logs failed for job {job_id}")
    return MlqRun(job_id, submit_argv, waited.returncode, stdout.stdout, stderr.stdout)


def write_mlq_capture(path: Path, run: MlqRun) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = run.stdout
    if run.stderr:
        text += "\n--- mlq captured stderr ---\n" + run.stderr
    path.write_text(text, encoding="utf-8")
    return text


@dataclasses.dataclass
class Launch:
    argv: list[str]
    config: dict[str, Any]
    run_name: str
    log_path: Path
    staging_log: Path


def build_launch(spec: Spec, arm: Arm, tier: Tier, seed: int) -> Launch:
    run_name = arm.run_name(seed, tier)
    argv = [
        spec.launcher,
        spec.executable,
        "pretrain",
        "--run",
        run_name,
        "--steps",
        str(tier.steps),
        "--seed",
        str(seed),
        "--validation-windows",
        str(tier.validation_windows),
        "--supports",
        arm.supports or spec.supports,
        *spec.common_flags,
        *tier.extra_flags,
        *arm.flags,
    ]
    return Launch(
        argv=argv,
        config={
            "arm": arm.name,
            "arm_class": arm.arm_class,
            "tier": tier.as_json(),
            "seed": seed,
            "supports": arm.supports or spec.supports,
            "supports_sha256": arm.supports_sha256 or spec.supports_sha256,
            "common_flags": list(spec.common_flags),
            "tier_flags": list(tier.extra_flags),
            "arm_flags": list(arm.flags),
        },
        run_name=run_name,
        log_path=Path(spec.runs_dir) / run_name / "campaign_stdout.log",
        # `RunDir::create_fresh` BAILS on an existing directory (run_dir.rs:86-98), so the
        # capture cannot start inside the run dir. It is moved in after the process exits.
        staging_log=Path("training/campaign_logs") / f"{run_name}.stdout.log",
    )


def build_compare(spec: Spec, arm: Arm, baseline: Path, candidate: Path) -> list[str]:
    extra = (GEOMETRY_CHANGE_COMPARE_FLAG,) if arm.arm_class == "geometry_changing" else ()
    return [
        spec.launcher,
        spec.executable,
        "pretrain-compare",
        str(baseline),
        str(candidate),
        *extra,
    ]


def run_compare(argv: Sequence[str], cwd: Path) -> tuple[int, str, str]:
    """`pretrain-compare` writes the report to stdout and panics to stderr; keep them apart
    so a parse never sees a panic message as a metric line."""
    completed = subprocess.run(
        list(argv), cwd=str(cwd), capture_output=True, text=True, check=False
    )
    return completed.returncode, completed.stdout, completed.stderr


# ---------------------------------------------------------------------------
# Retention (3.4)
# ---------------------------------------------------------------------------


def retention_plan(spec: Spec, arm: Arm, tier: Tier, seed: int) -> dict[str, Any]:
    """What retention would delete under `weights/`, and what it protects.

    3.4 retains `meta.json`, `pretrain_best_diag896.{ot,windows.json,metadata.json}` and
    `gens/`, and deletes the optimizer bundles. The supports sidecar of the RETAINED
    checkpoint is protected too, because the trainer's own pruning contract is that a kept
    checkpoint keeps every sidecar that makes it loadable (pretrain.rs:12162-12169) — a
    retained `.ot` with no supports is unloadable and therefore worthless.
    """
    weights = Path(spec.runs_dir) / arm.run_name(seed, tier) / "weights"
    keep = {
        f"{tier.artifact}.ot",
        f"{tier.artifact}.windows.json",
        f"{tier.artifact}.metadata.json",
        f"{tier.artifact}.supports.{spec.resolution_secs}.json",
    }
    doomed: list[str] = []
    kept: list[str] = []
    bytes_freed = 0
    if weights.is_dir():
        for entry in sorted(weights.iterdir()):
            if not entry.is_file():
                continue
            if entry.name in keep:
                kept.append(entry.name)
                continue
            # Only the two families a stage-0 arm's 1.5 GB actually lives in are eligible:
            # optimizer bundles, and non-retained checkpoints with their sidecars.
            deletable = entry.name.endswith(".optimizer.ot") or entry.name.startswith(
                ("pretrain_step_", "pretrain_last", "pretrain_epoch_", "pretrain_promotion_candidate")
            )
            if deletable:
                doomed.append(entry.name)
                bytes_freed += entry.stat().st_size
            else:
                kept.append(entry.name)
    return {
        "weights_dir": str(weights),
        "keep_patterns": sorted(keep),
        "delete": doomed,
        "keep": kept,
        "bytes_freed": bytes_freed,
    }


def apply_retention(plan: dict[str, Any]) -> dict[str, Any]:
    weights = Path(plan["weights_dir"])
    removed = []
    for name in plan["delete"]:
        target = weights / name
        if target.exists():
            target.unlink()
            removed.append(name)
    return {**plan, "removed": removed}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def dependency_levels(spec: Spec, arm_names: Sequence[str]) -> list[list[str]]:
    """Group arms into waves that may run concurrently.

    An arm may launch only once its incumbent has settled, because the incumbent's artifact
    is the baseline of its comparison. Everything within a level is independent.
    """
    selected = list(arm_names)
    remaining = set(selected)
    levels: list[list[str]] = []
    while remaining:
        level = [
            name
            for name in selected
            if name in remaining
            and (
                (parent := incumbent_of(spec, spec.arms[name])) is None
                or parent not in remaining
            )
        ]
        if not level:
            raise SpecError(
                "incumbent cycle among arms: " + ", ".join(sorted(remaining))
            )
        levels.append(level)
        remaining -= set(level)
    return levels


class Driver:
    def __init__(
        self,
        spec: Spec,
        args: argparse.Namespace,
        command_runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
    ) -> None:
        self.spec = spec
        self.args = args
        self.cwd = Path(args.repo_root).resolve()
        self.launch_contract = validate_launch_contract(spec, self.cwd)
        self.command_runner = command_runner
        self.campaign_id = args.campaign_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:8]}"
        self.ledger = Ledger(self.cwd / spec.ledger_path, self.campaign_id, args.dry_run)
        self.state = ResumeState.replay(self.ledger.read())
        # This bounds how many submissions this driver waits on concurrently. mlq remains the
        # sole GPU admission authority and enforces each job's declared global parallel limit.
        self._slots = threading.Semaphore(max(args.concurrency, 1))
        self._console = threading.Lock()

    def emit(self, text: str) -> None:
        with self._console:
            print(text, flush=True)

    # -- launching ------------------------------------------------------

    def _partial_run_action(self, launch: Launch) -> str:
        root = self.cwd / self.spec.runs_dir / launch.run_name
        if not root.exists():
            return "launch"
        if launch.run_name in self.state.completed:
            return "skip"
        return self.args.on_partial

    def launch_seed(self, arm: Arm, tier: Tier, seed: int) -> SeedResult:
        launch = build_launch(self.spec, arm, tier, seed)
        action = self._partial_run_action(launch)
        if action == "refuse":
            raise DriverError(
                f"{self.spec.runs_dir}/{launch.run_name} exists but the ledger has no "
                "successful `arm_complete` for it. `RunDir::create_fresh` would bail. Choose: "
                "--on-partial adopt (trust the artifacts already there), --on-partial rerun "
                "(move the directory aside and relaunch), or delete it yourself."
            )
        if action == "rerun":
            root = self.cwd / self.spec.runs_dir / launch.run_name
            aside = root.with_name(f"{launch.run_name}.abandoned.{int(time.time())}")
            root.rename(aside)
            self.ledger.append(
                "run_abandoned", run_name=launch.run_name, moved_to=str(aside)
            )
            action = "launch"

        if action in ("skip", "adopt"):
            self.emit(f"  [{launch.run_name}] {action}: reusing artifacts already on disk")
            return SeedResult(
                seed=seed,
                run_name=launch.run_name,
                launch_exit_code=0,
                log_path=str(self.cwd / launch.log_path)
                if (self.cwd / launch.log_path).exists()
                else None,
                artifact=None,
                incumbent_artifact=None,
                comparison=None,
            )

        current_contract = validate_launch_contract(self.spec, self.cwd)
        if current_contract != self.launch_contract:
            raise DriverError("pinned launch inputs changed after campaign validation")
        resolved_argv = _resolved_launch_argv(launch.argv, current_contract)
        execution_argv = _execution_verified_argv(
            resolved_argv, current_contract, launch.config["supports"]
        )
        idempotency_key = f"campaign-{self.campaign_id}-{launch.run_name}"

        def record_submission(job_id: int, submit_argv: list[str]) -> None:
            self.ledger.append(
                "arm_launch",
                arm=arm.name,
                tier=tier.name,
                seed=seed,
                run_name=launch.run_name,
                configured_argv=launch.argv,
                resolved_argv=resolved_argv,
                supports=current_contract["supports"][launch.config["supports"]],
                command=shlex.join(resolved_argv),
                execution_argv=execution_argv,
                execution_command=shlex.join(execution_argv),
                resolved_config=launch.config,
                launch_contract=current_contract,
                launcher=current_contract["launcher"],
                executable=current_contract["executable"],
                cwd=str(self.cwd),
                driver_git_commit=git_describe(self.cwd),
                arm_spec=arm.raw,
                tier_spec=tier.as_json(),
                spec_sha256=self.spec.sha256,
                staging_log=str(launch.staging_log),
                mlq_job_id=job_id,
                mlq_submit_argv=submit_argv,
                mlq_idempotency_key=idempotency_key,
                mlq_max_parallel_runs=self.spec.mlq_max_parallel_runs,
                mlq_time_limit=self.spec.mlq_time_limit,
            )

        self.emit(f"  [{launch.run_name}] mlq <- {shlex.join(resolved_argv)}")
        started = time.monotonic()
        with self._slots:
            mlq_run = run_through_mlq(
                execution_argv,
                launch.run_name,
                self.cwd,
                self.spec.mlq_max_parallel_runs,
                self.spec.mlq_time_limit,
                idempotency_key,
                command_runner=self.command_runner,
                on_submitted=record_submission,
            )
            write_mlq_capture(self.cwd / launch.staging_log, mlq_run)
        code = mlq_run.exit_code
        elapsed = time.monotonic() - started

        final_log = self.cwd / launch.log_path
        if final_log.parent.is_dir():
            shutil.move(str(self.cwd / launch.staging_log), str(final_log))
            log_path = final_log
        else:
            log_path = self.cwd / launch.staging_log

        self.ledger.append(
            "arm_complete",
            arm=arm.name,
            tier=tier.name,
            seed=seed,
            run_name=launch.run_name,
            exit_code=code,
            mlq_job_id=mlq_run.job_id,
            wall_seconds=round(elapsed, 3),
            log_path=str(log_path),
            run_meta=read_run_meta(self.cwd / self.spec.runs_dir / launch.run_name),
        )
        return SeedResult(
            seed=seed,
            run_name=launch.run_name,
            launch_exit_code=code,
            log_path=str(log_path),
            artifact=None,
            incumbent_artifact=None,
            comparison=None,
        )

    # -- comparison -----------------------------------------------------

    @staticmethod
    def _require_trade_evidence(tier: Tier, artifact: ArtifactFacts, role: str) -> None:
        if tier.name in ("confirm", "deploy") and not artifact.trade_present:
            raise DriverError(
                f"{role} artifact {artifact.path} has no `trade` summary. {tier.name} must "
                "consume promotion evidence such as pretrain_best.windows.json, not a "
                "diagnostic context sidecar written from a pass the trade bench did not ride."
            )


    def compare_seed(
        self, arm: Arm, tier: Tier, result: SeedResult, incumbent_run: str | None
    ) -> None:
        candidate = self.cwd / self.spec.artifact_path(result.run_name, tier)
        if not candidate.exists():
            raise DriverError(
                f"{candidate} is missing. A `--steps {tier.steps}` arm writes it at every "
                "improving validation via `keep_context_best` (pretrain.rs:6200-6201, "
                "7199-7253); its absence means the run never reached a validation."
            )
        result.artifact = read_artifact(candidate, self.spec.contract)
        self._require_trade_evidence(tier, result.artifact, "candidate")
        if incumbent_run is None:
            self.ledger.append(
                "artifact_read",
                arm=arm.name,
                tier=tier.name,
                run_name=result.run_name,
                artifact=result.artifact.as_json(),
                note="control arm: no incumbent to compare against",
            )
            return
        baseline = self.cwd / self.spec.artifact_path(incumbent_run, tier)
        if not baseline.exists():
            raise DriverError(
                f"incumbent artifact {baseline} is missing; run {incumbent_run} first "
                "(the driver orders arms by incumbent dependency, so this means the "
                "incumbent's run directory was removed)"
            )
        current_contract = validate_launch_contract(self.spec, self.cwd)
        if current_contract != self.launch_contract:
            raise DriverError("pinned launch inputs changed before comparison")
        result.incumbent_artifact = read_artifact(baseline, self.spec.contract)
        self._require_trade_evidence(tier, result.incumbent_artifact, "incumbent")
        configured_argv = build_compare(self.spec, arm, baseline, candidate)
        argv = _resolved_launch_argv(configured_argv, self.launch_contract)
        code, stdout, stderr = run_compare(argv, self.cwd)
        result.comparison = parse_comparison(argv, code, stdout, stderr, self.spec.contract)
        if arm.arm_class == "geometry_changing":
            validate_geometry_comparison(result.comparison)
        self.ledger.append(
            "comparison",
            arm=arm.name,
            tier=tier.name,
            seed=result.seed,
            candidate_run=result.run_name,
            baseline_run=incumbent_run,
            configured_argv=configured_argv,
            resolved_argv=argv,
            command=shlex.join(argv),
            launch_contract=self.launch_contract,
            candidate_artifact=result.artifact.as_json(),
            baseline_artifact=result.incumbent_artifact.as_json(),
            parsed=result.comparison.as_json(),
        )

    # -- one arm --------------------------------------------------------

    def run_arm(self, arm: Arm, tier: Tier) -> Decision | None:
        if (arm.name, tier.name) in self.state.settled:
            verdict = self.state.settled[(arm.name, tier.name)]
            self.emit(f"[{arm.name} @ {tier.name}] already settled: {verdict}; skipping")
            return None
        incumbent_name = incumbent_of(self.spec, arm)
        self.emit(
            f"[{arm.name} @ {tier.name}] class={arm.arm_class} stack={arm.stack_tag} "
            f"incumbent={incumbent_name or '(none: control)'}"
        )
        results: list[SeedResult] = []
        seeds = tier.seeds
        run_in_parallel(
            [
                lambda seed=seed: results.append(self.launch_seed(arm, tier, seed))
                for seed in seeds
            ]
        )
        results.sort(key=lambda item: item.seed)
        for result in results:
            incumbent_run = (
                self.spec.arms[incumbent_name].run_name(result.seed, tier)
                if incumbent_name
                else None
            )
            self.compare_seed(arm, tier, result, incumbent_run)

        if arm.is_control:
            # The control never settles: it takes no decision and is the permanent baseline
            # of every comparison, so nothing of its is deleted. Said out loud because a
            # control costs ~1.5 GB per seed and an operator will otherwise wonder why.
            self.ledger.append(
                "retention",
                arm=arm.name,
                tier=tier.name,
                skipped=True,
                reason="control arm: never settles, and its artifacts are the baseline of "
                "every downstream comparison",
            )
            self.emit("  control arm: no decision to take, and nothing retained away")
            return None

        try:
            decision = decide(self.spec, arm, tier, results)
        except RefusalError as error:
            self.ledger.append(
                "decision_refused",
                arm=arm.name,
                tier=tier.name,
                reason=str(error),
                seeds=[result.as_json() for result in results],
            )
            raise
        self.ledger.append(
            "decision",
            arm=arm.name,
            tier=tier.name,
            decision=decision.as_json(),
            seeds=[result.as_json() for result in results],
        )
        self.state.settled[(arm.name, tier.name)] = decision.verdict
        self.emit(f"  verdict {decision.verdict}: {decision.justification}")

        if self.args.retention:
            self.settle_retention(arm, tier, decision)
        return decision

    def settle_retention(self, arm: Arm, tier: Tier, decision: Decision) -> None:
        if decision.verdict in (CARRY, ADOPT_EXEMPT):
            self.ledger.append(
                "retention",
                arm=arm.name,
                tier=tier.name,
                skipped=True,
                reason=f"arm was {decision.verdict}; a carried arm's artifacts are never deleted",
            )
            self.emit("  retention: skipped, the arm was carried")
            return
        for seed in tier.seeds:
            plan = retention_plan(self.spec, arm, tier, seed)
            applied = apply_retention(plan)
            self.ledger.append(
                "retention",
                arm=arm.name,
                tier=tier.name,
                seed=seed,
                verdict=decision.verdict,
                **applied,
            )
            self.emit(
                f"  retention: freed {applied['bytes_freed']:,} bytes from "
                f"{plan['weights_dir']} ({len(applied['removed'])} files)"
            )

    # -- campaign -------------------------------------------------------

    def run(self, arm_names: Sequence[str], tier_names: Sequence[str]) -> int:
        self.ledger.append(
            "campaign_start",
            spec_path=str(self.spec.path),
            spec_sha256=self.spec.sha256,
            spec=self.spec.document,
            note=self.spec.note,
            argv=sys.argv,
            cwd=str(self.cwd),
            git=git_describe(self.cwd),
            launch_contract=self.launch_contract,
            arms=list(arm_names),
            tiers=list(tier_names),
            concurrency=self.args.concurrency,
            retention_enabled=self.args.retention,
        )
        failures = 0
        for tier_name in tier_names:
            tier = self.spec.tiers[tier_name]
            eligible = [name for name in arm_names if tier_name in self.spec.arms[name].tiers]
            for level in dependency_levels(self.spec, eligible):
                errors: list[tuple[str, DriverError]] = []

                def attempt(name: str) -> None:
                    try:
                        self.run_arm(self.spec.arms[name], tier)
                    except DriverError as error:
                        errors.append((name, error))

                run_in_parallel([lambda name=name: attempt(name) for name in level])
                for name, error in errors:
                    failures += 1
                    self.emit(f"[{name} @ {tier_name}] ERROR: {error}")
                if errors and not self.args.keep_going:
                    return 1
        return 1 if failures else 0


def run_in_parallel(thunks: Sequence[Callable[[], None]]) -> None:
    """Run every thunk, then re-raise the first exception. Unbounded by design: the launch
    budget lives on `Driver._slots`, so a queued thunk costs one blocked thread."""
    if len(thunks) <= 1:
        for thunk in thunks:
            thunk()
        return
    errors: list[BaseException] = []

    def wrapper(thunk: Callable[[], None]) -> None:
        try:
            thunk()
        except BaseException as error:  # noqa: BLE001 - re-raised after the join
            errors.append(error)

    threads = [threading.Thread(target=wrapper, args=(thunk,)) for thunk in thunks]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]


def read_run_meta(root: Path) -> dict[str, Any] | None:
    """`meta.json` records the commit the run ACTUALLY executed (run_dir.rs:102-109), which
    is the provenance a decision needs — not the driver's HEAD."""
    path = root / "meta.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {"unparseable": str(path)}


def git_describe(cwd: Path) -> dict[str, Any]:
    def capture(*args: str) -> str | None:
        try:
            done = subprocess.run(
                ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=False
            )
        except OSError:
            return None
        return done.stdout.strip() if done.returncode == 0 else None

    status = capture("status", "--porcelain")
    return {
        "commit": capture("rev-parse", "HEAD"),
        "short": capture("rev-parse", "--short", "HEAD"),
        "branch": capture("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


    cwd = Path(args.repo_root).resolve()
    launch_contract = validate_launch_contract(spec, cwd)
    out = print
    out("DRY RUN — nothing is launched, written or deleted.")
    out(f"spec           {spec.path} (sha256 {spec.sha256[:16]})")
    out(
        f"launcher       {launch_contract['launcher']['resolved_path']} "
        f"(sha256 {launch_contract['launcher']['sha256']})"
    )
    out(
        f"executable     {launch_contract['executable']['resolved_path']} "
        f"(sha256 {launch_contract['executable']['sha256']})"
    )
    out(
        f"mlq            max-parallel-runs={spec.mlq_max_parallel_runs} "
        f"time-limit={spec.mlq_time_limit}"
    )
    out(f"control arm    {spec.control_arm}")
    out(f"concurrency    {args.concurrency}")
    out(f"retention      {'on' if args.retention else 'off (--no-retention)'}")
    if spec.note:
        out(f"note           {spec.note}")
    out("")

    for tier_name in tier_names:
        tier = spec.tiers[tier_name]
        eligible = [name for name in arm_names if tier_name in spec.arms[name].tiers]
        out("=" * 78)
        out(
            f"TIER {tier_name}: --steps {tier.steps} --validation-windows "
            f"{tier.validation_windows} seeds {list(tier.seeds)} context {tier.context} "
            f"artifact {tier.artifact}.windows.json"
        )
        if tier.note:
            out(f"  note: {tier.note}")
        out("=" * 78)
        for level_number, level in enumerate(dependency_levels(spec, eligible)):
            out(f"-- dependency level {level_number} ({len(level)} arm(s), may run concurrently)")
            for name in level:
                arm = spec.arms[name]
                incumbent = incumbent_of(spec, arm)
                out("")
                out(
                    f"  ARM {arm.name}  wave={arm.wave}  class={arm.arm_class}  "
                    f"stack=[{', '.join(arm.carried_stack) or 'none'}]  "
                    f"incumbent={incumbent or '(none: control)'}"
                )
                if arm.note:
                    out(f"    note: {arm.note}")
                if arm.exempt_from_cull:
                    out(f"    EXEMPT FROM CULL: {arm.exempt_reason}")
                for seed in tier.seeds:
                    launch = build_launch(spec, arm, tier, seed)
                    out(f"    mlq queue: {shlex.join(launch.argv)}")
                    out(
                        f"      max-parallel-runs={spec.mlq_max_parallel_runs} "
                        f"time-limit={spec.mlq_time_limit}; stdout -> {launch.staging_log}, "
                        f"moved to {launch.log_path}"
                    )
                    if incumbent is None:
                        continue
                    baseline = spec.artifact_path(
                        spec.arms[incumbent].run_name(seed, tier), tier
                    )
                    candidate = spec.artifact_path(launch.run_name, tier)
                    out(
                        f"    compare: "
                        f"{shlex.join(build_compare(spec, arm, baseline, candidate))}"
                    )
                if incumbent is None:
                    out("    decision: none — the control defines the incumbent")
                    continue
                _describe_decision(out, spec, arm, tier)
                if args.retention:
                    plan = retention_plan(spec, arm, tier, tier.seeds[0])
                    out(
                        f"    retention (only if NOT carried): under {plan['weights_dir']}, "
                        f"protect {', '.join(plan['keep_patterns'])}; delete "
                        "*.optimizer.ot and every pretrain_{step_*,last,epoch_*,"
                        "promotion_candidate}* sidecar"
                    )
                    if plan["delete"]:
                        out(
                            f"      on disk right now this would remove {len(plan['delete'])} "
                            f"file(s), {plan['bytes_freed']:,} bytes"
                        )
        out("")

    out("=" * 78)
    out("TRIPWIRES applied to every arm at every tier, before any threshold is consulted:")
    out("  1 non-finite loss or global_grad_norm  <- captured launcher output, exit code")
    out("  2 free_kelly_saturated rises > 5 pp    <- trade.free_kelly_saturated, both arms")
    out("  3 ruin_bars > 0 where incumbent had 0  <- trade.policies[model], cap 0.25x point")
    out("  4 paired correlation < 0.9             <- pretrain-compare correlation line")
    out("  5 resolved delta r regression at 2 SE  <- geometry_preserving arms ONLY")
    out("  6 supports_sha256 mismatch             <- unless the arm is geometry_changing")
    out("A tripwire whose inputs are absent reports UNEVALUATED. Screen tier only culls, so")
    out("an UNEVALUATED wire does not kill an arm there; a CARRY is downgraded to DISCARD.")
    ledger = cwd / spec.ledger_path
    out("")
    out(f"ledger would be appended at {ledger} ({'exists' if ledger.exists() else 'would be created'})")
    return 0


def _describe_decision(out: Callable[..., None], spec: Spec, arm: Arm, tier: Tier) -> None:
    metric = PRIMARY_METRIC[arm.arm_class]
    guard = GUARD_METRIC[arm.arm_class]
    out(f"    decision: primary = paired `{metric}` (bps/bar at {SELECTION_CAP:.2f}x cap, higher better)")
    if arm.arm_class == "geometry_preserving":
        out(f"              guard   = paired `{guard}`, a resolved regression at 2 SE vetoes")
        out("              FORBIDDEN: the aggregate `difference` — dominated by s/u/v, which")
        out("                         cannot affect P&L (3.3, T1.2)")
    else:
        out(
            f"              guard   = `{guard}` toward 1.0. Read as a LEVEL from each "
            "artifact's"
        )
        out("                         `calibration.mean_beta`; the band is 2.0 x the larger")
        out("                         published `mean_beta_se`, i.e. UNPAIRED, because the")
        out("                         comparator prints no paired MZ difference")
        out(
            "              FORBIDDEN: "
            + ", ".join(FORBIDDEN_FOR_GEOMETRY_CHANGE)
            + " — nats and CRPS move with the geometry, and the differenced edge has a"
        )
        out(
            "              compare contract: --allow-geometry-change; the comparator must "
            "explicitly withhold every support-decoded metric and emit model growth alone"
        )
    if arm.exempt_from_cull:
        out("              rule: exempt — adopted regardless of sign; tripwire 1 still fatal")
        return
    try:
        s_seed, provenance = spec.resolve_s_seed(tier.name, metric)
    except RefusalError as error:
        out("              rule: REFUSED —")
        for line in textwrap.wrap(" ".join(str(error).split()), width=88):
            out(f"                {line}")
        return
    if tier.name == "screen":
        out(
            f"              rule: CULL if mean({metric}) < {SCREEN_CULL_MULTIPLE:+.1f} * "
            f"s_seed = {SCREEN_CULL_MULTIPLE * s_seed:+.6f}"
        )
        out(f"              s_seed = {s_seed:.6f} [{provenance}]")
        out("              nothing is carried at screen tier; survivors advance to confirm")
    else:
        n = len(tier.seeds)
        out(
            f"              rule: CARRY if mean({metric}) > {CONFIRM_CARRY_MULTIPLE:+.1f} * "
            f"sqrt(s_seed^2/{n} + se_win^2), and no guard veto"
        )
        floor = CONFIRM_CARRY_MULTIPLE * math.sqrt(s_seed * s_seed / n)
        out(
            f"              s_seed = {s_seed:.6f} [{provenance}]; se_win is measured per "
            f"comparison, so the threshold is >= {floor:+.6f}"
        )
        out("              otherwise DISCARD and keep the incumbent")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="campaign_driver.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
        epilog=(
            "Examples\n"
            "  # See exactly what the campaign would do, touching nothing:\n"
            "  python3 trading_bots/scripts/campaign_driver.py --spec "
            "trading_bots/scripts/campaign_arms.example.json --tier screen --dry-run\n"
            "\n"
            "  # Run the screen tier for two arms, one at a time:\n"
            "  python3 trading_bots/scripts/campaign_driver.py --spec arms.json "
            "--tier screen --arm B1 --arm B2\n"
            "\n"
            "  # Resume: settled arms are skipped, half-finished runs must be dispositioned.\n"
            "  python3 trading_bots/scripts/campaign_driver.py --spec arms.json "
            "--tier screen --on-partial adopt\n"
        ),
    )
    parser.add_argument(
        "--spec",
        type=Path,
        help="arm spec (JSON). Required for everything except --print-ledger-schema",
    )
    parser.add_argument(
        "--tier",
        action="append",
        dest="tiers",
        default=None,
        help="tier to run; repeatable; default every tier in the spec, in screen/confirm/deploy order",
    )
    parser.add_argument(
        "--arm",
        action="append",
        dest="arms",
        default=None,
        help="arm to run; repeatable; default every arm in the spec, in spec order",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="arms/seeds in flight at once. DEFAULT 1. 2-way may fit at stage 0 / ctx 896 on "
        "the single 5090 but the recorded OOM was at ctx 1472 — test it once, do not assume "
        "it (audit 3.4)",
    )
    parser.add_argument(
        "--s-seed",
        action="append",
        default=None,
        metavar="TIER:METRIC=VALUE",
        help="W0.5 between-seed sd, overriding the spec. Repeatable. No default exists: "
        "without it the driver REFUSES to take a cull decision",
    )
    parser.add_argument(
        "--s-seed-provenance",
        default=None,
        help="which W0.5 runs measured --s-seed; required whenever --s-seed is given",
    )
    parser.add_argument("--dry-run", action="store_true", help="print and exit; touch nothing")
    parser.add_argument(
        "--no-retention",
        dest="retention",
        action="store_false",
        default=True,
        help="do not delete anything after an arm settles (3.4 retention is on by default; a "
        "CARRIED arm is never touched either way)",
    )
    parser.add_argument(
        "--on-partial",
        choices=("refuse", "adopt", "rerun"),
        default="refuse",
        help="what to do when a run directory exists with no successful completion in the "
        "ledger. DEFAULT refuse",
    )
    parser.add_argument("--keep-going", action="store_true", help="continue after an arm errors")
    parser.add_argument("--campaign-id", default=None, help="reuse a campaign id when resuming")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="repository root every relative path is resolved against",
    )
    parser.add_argument(
        "--print-ledger-schema", action="store_true", help="describe the ledger records and exit"
    )
    return parser.parse_args(argv)


LEDGER_SCHEMA = """\
Ledger: append-only JSONL, one JSON object per line. Every record carries
  ts               RFC3339 UTC, millisecond precision
  record           the record kind, below
  campaign_id      groups the records of one invocation chain; reuse with --campaign-id
  driver_version   this file's DRIVER_VERSION

campaign_start   spec_path, spec_sha256, spec (verbatim), note, argv, cwd, git{commit,short,
                 branch,dirty}, launch_contract{launcher,executable,supports,market_supports,
                 split_bounds,batch_size,exact_batch,freeze_supports,freeze_market_supports,
                 tier_seeds,mlq}, arms, tiers, concurrency, retention_enabled
arm_launch       arm, tier, seed, run_name, configured_argv, resolved_argv, command,
                 execution_argv, execution_command, resolved_config, launch_contract,
                 supports{configured_path,resolved_path,sha256},
                 launcher{configured_path,resolved_path,sha256},
                 executable{configured_path,resolved_path,sha256}, cwd, driver_git_commit,
                 arm_spec (verbatim), tier_spec, spec_sha256, staging_log, mlq_job_id,
                 mlq_submit_argv, mlq_idempotency_key, mlq_max_parallel_runs, mlq_time_limit
arm_complete     arm, tier, seed, run_name, exit_code, mlq_job_id, wall_seconds, log_path,
                 run_meta (the run's OWN meta.json, i.e. the commit it actually ran)
run_abandoned    run_name, moved_to                       (--on-partial rerun only)
artifact_read    arm, tier, run_name, artifact             (control arm; nothing to compare)
comparison       arm, tier, seed, candidate_run, baseline_run, configured_argv,
                 resolved_argv, command, launch_contract, candidate_artifact,
                 baseline_artifact, parsed{argv, exit_code, stdout (VERBATIM), stderr, windows,
                        baseline_run, candidate_run, baseline_mean, candidate_mean,
                        correlation, worse_windows, total_windows,
                        minimum_detectable_effect, verdict,
                        dispersions{<metric>: {mean, se, ci_low, ci_high, blocks, samples}},
                        absent{<metric>: reason the comparator printed},
                        candidate_advantage{<metric>: value, more-is-better orientation},
                        unmapped_dispersion_lines, model_growth_only,
                        refusal: null | geometry | traded_prefix | unclassified}
                 Each *_artifact carries: path, sha256, format_version, run, global_step,
                 split, context, windows, scoring, corpus_fingerprint, split_bounds,
                 eval_window_seed, realized_batch, realized_steps, supports_sha256,
                 traded_{edge,growth}_windows, traded_blocks, mz_slope, mz_slope_se,
                 free_kelly_saturated, model_ruin_bars, selection_cap_ruin_bars,
                 trade_present.
decision         arm, tier, seeds[<every SeedResult>], decision{
                   arm, tier, arm_class, incumbent, verdict, primary_metric,
                   primary_per_seed, primary_mean, se_win, s_seed, s_seed_provenance,
                   threshold_expression, threshold_value, guard_metric, guard_verdict,
                   guard_evidence, tripwires[{name, status, binding, evidence}],
                   forbidden_metrics_withheld, justification}
decision_refused arm, tier, reason, seeds
                 Raised, never defaulted around, when: s_seed is absent or has no
                 provenance; the primary metric is absent; or pretrain-compare refused the
                 pair. A refusal is "not comparable", NEVER "no effect".
retention        arm, tier, seed, verdict, weights_dir, keep_patterns, keep, delete,
                 removed, bytes_freed  — or {skipped: true, reason} for a carried arm

Non-finite floats round-trip as the strings "nan" / "inf" / "-inf", matching JsonF64
(pretrain_stats.rs:439-451), so every line loads with plain json.loads.

Verdicts: CULL, SURVIVE_SCREEN, CARRY, DISCARD, ADOPT_EXEMPT.

Resume: the driver replays the ledger before doing anything. An (arm, tier) with a
`decision` record is settled and skipped. A run_name with a zero-exit `arm_complete` is
reused. A run directory with neither makes the driver stop and ask, because
RunDir::create_fresh bails on an existing directory: --on-partial adopt trusts what is on
disk, --on-partial rerun moves it to <name>.abandoned.<epoch> and relaunches.
"""


def _parse_s_seed_overrides(entries: Sequence[str] | None) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for entry in entries or []:
        if "=" not in entry:
            raise SpecError(f"--s-seed {entry!r} must look like TIER:METRIC=VALUE or METRIC=VALUE")
        key, _, value = entry.partition("=")
        tier, _, metric = key.rpartition(":")
        out.setdefault(tier or "all", {})[metric] = float(value)
    return out

def validate_decision_inputs(
    spec: Spec, arm_names: Sequence[str], tier_names: Sequence[str]
) -> None:
    """Refuse an undecidable campaign before its control or any other arm is queued."""
    for tier_name in tier_names:
        tier = spec.tiers[tier_name]
        for arm_name in arm_names:
            arm = spec.arms[arm_name]
            if tier_name not in arm.tiers or arm.is_control or arm.exempt_from_cull:
                continue
            spec.resolve_s_seed(tier_name, PRIMARY_METRIC[arm.arm_class])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.print_ledger_schema:
        print(LEDGER_SCHEMA)
        return 0
    try:
        overrides: dict[str, Any] = {}
        supplied = _parse_s_seed_overrides(args.s_seed)
        if supplied:
            overrides["s_seed"] = supplied
            if args.s_seed_provenance:
                overrides["s_seed_provenance"] = args.s_seed_provenance
        if args.spec is None:
            raise SpecError("--spec is required")
        spec = load_spec(args.spec, overrides)
        arm_names = args.arms or list(spec.arm_order)
        for name in arm_names:
            if name not in spec.arms:
                raise SpecError(f"unknown arm {name!r}; spec has {', '.join(spec.arm_order)}")
        tier_names = args.tiers or [tier for tier in TIER_ORDER if tier in spec.tiers] or list(
            spec.tiers
        )
        for name in tier_names:
            if name not in spec.tiers:
                raise SpecError(f"unknown tier {name!r}; spec has {', '.join(spec.tiers)}")
        if args.concurrency < 1:
            raise SpecError("--concurrency must be at least 1")
        # An arm's own carried stack must be settled before it launches, so the control has
        # to be present whenever anything depends on it.
        if spec.control_arm not in arm_names and any(
            incumbent_of(spec, spec.arms[name]) == spec.control_arm for name in arm_names
        ):
            print(
                f"note: {spec.control_arm} is not in the selection; its artifacts must "
                "already be on disk for the comparisons below to resolve",
                file=sys.stderr,
            )
        if args.dry_run:
            return dry_run(spec, args, arm_names, tier_names)
        validate_launch_contract(spec, Path(args.repo_root).resolve())
        validate_decision_inputs(spec, arm_names, tier_names)
        return Driver(spec, args).run(arm_names, tier_names)
    except DriverError as error:
        print(f"campaign_driver: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
