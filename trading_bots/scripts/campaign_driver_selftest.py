#!/usr/bin/env python3
"""Offline checks for `campaign_driver.py`. No GPU, no cargo, no training run.

Exercises the parts that decide whether an arm lives: the `pretrain-compare` parser, all six
hard tripwires, both threshold gates, the `s_seed` refusal, the retention plan and ledger
resume. Run it after touching the driver, and again once `EdgePersist`'s W0.1/W0.3 output
lands, with `REAL_COMPARE_OUTPUT` replaced by a genuine capture.

    python3 trading_bots/scripts/campaign_driver_selftest.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import campaign_driver as cd  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, what: str) -> None:
    if condition:
        print(f"  ok   {what}")
    else:
        FAILURES.append(what)
        print(f"  FAIL {what}")


# ---------------------------------------------------------------------------
# Synthetic `pretrain-compare` stdout.
#
# Rendered from the format strings in `impl Display for PairedComparison`, as confirmed by
# agent `EdgePersist` who wrote the W0.1/W0.2/W0.3 edits. `NATS_BLOCK` is the geometry-
# dependent half; `ECONOMIC_LINES` is the pair of rows W0.1 adds, which always print and take
# the `: not recorded (...)` form when the vector is absent.
# ---------------------------------------------------------------------------

NATS_BLOCK = """\
paired comparison over 2048 identical pinned windows, scoring smoothed
  baseline  control_base_s24301_screen   3.1416 nats/bar
  candidate B1_base_s24301_screen        3.1350 nats/bar
  paired delta [nats/bar, candidate - baseline, NEGATIVE is better] -0.0066 +/- 0.0021 (95% CI -0.0108..-0.0025, 412 blocks / 2048 windows)
  conditional delta (u,v scored only where s != 0) -0.0041 +/- 0.0018 (95% CI -0.0077..-0.0006, 412 blocks / 2048 windows)
  delta r  -0.0012 +/- 0.0004 (95% CI -0.0020..-0.0004, 412 blocks / 2048 windows)
  delta s  -0.0030 +/- 0.0011 (95% CI -0.0052..-0.0009, 412 blocks / 2048 windows)
  delta u  -0.0015 +/- 0.0009 (95% CI -0.0033..0.0003, 412 blocks / 2048 windows)
  delta v  -0.0009 +/- 0.0007 (95% CI -0.0023..0.0005, 412 blocks / 2048 windows)
  delta w  0.0000 +/- 0.0000 (95% CI 0.0000..0.0000, 412 blocks / 2048 windows)
"""

ECONOMIC_LINES = """\
  selection edge delta [bps/bar, candidate - baseline, POSITIVE is better] {edge} +/- {edge_se} (95% CI 0.0055..0.0413, 40 blocks / 1901 windows); candidate advantage {edge_adv}, MDE 0.0255, verdict: SIGNIFICANT at 95%
  model growth delta [bps/bar, candidate - baseline, POSITIVE is better] {growth} +/- 0.0180 (95% CI -0.0100..0.0600, 40 blocks / 1901 windows); candidate advantage {growth_adv}, MDE 0.0504, verdict: not distinguishable from zero
"""

CLOSING = """\
  per-window correlation 0.9971, candidate worse on 803 of 2048 windows
  detectable at 80% power: 0.0059 nats; verdict: SIGNIFICANT at 95%
"""


def compare_output(edge: float = 0.0700, edge_se: float = 0.0200, growth: float = 0.0250) -> str:
    return (
        NATS_BLOCK
        + ECONOMIC_LINES.format(
            edge=f"{edge:.4f}",
            edge_se=f"{edge_se:.4f}",
            edge_adv=f"{edge:+.4f}",
            growth=f"{growth:.4f}",
            growth_adv=f"{growth:+.4f}",
        )
        + CLOSING
    )


ABSENT_OUTPUT = (
    NATS_BLOCK
    + "  selection edge delta [bps/bar]: not recorded (pre-v3 artifact, or a pass with no "
    "traded windows)\n"
    + "  model growth delta [bps/bar]: not recorded (pre-v3 artifact, or a pass with no "
    "traded windows)\n"
    + CLOSING
)

GEOMETRY_REFUSAL_STDERR = (
    "Error: REFUSING to pair: the two runs were fitted on DIFFERENT BIN GEOMETRIES "
    "(supports_sha256 aaaaaaaa vs bbbbbbbb)"
)
PREFIX_REFUSAL_STDERR = (
    "Error: REFUSING to pair: the two runs traded different prefixes (1901 vs 1874)"
)

GEOMETRY_OUTPUT = """\
paired comparison over 2048 identical pinned windows, scoring smoothed
  BIN GEOMETRY CHANGED and --allow-geometry-change was given: baseline supports aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, candidate supports bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb. Every quantity decoded through the bins is WITHHELD rather than corrected, because no correction makes two discretizations' log densities comparable and a corrected number would look usable. `model growth delta` is the ONLY valid cull metric below.
  paired delta [nats/bar]: NOT COMPARABLE (forbidden across bin geometries)
  conditional delta [nats/bar]: NOT COMPARABLE (forbidden across bin geometries)
  per-DOF deltas [nats/bar]: NOT COMPARABLE (forbidden across bin geometries)
  per-window correlation [unitless]: NOT COMPARABLE (forbidden across bin geometries)
  selection edge delta [bps/bar]: NOT COMPARABLE (forbidden across bin geometries)
  model growth delta [bps/bar, candidate - baseline, POSITIVE is better] 0.0500 +/- 0.0180 (95% CI 0.0147..0.0853, 40 blocks / 1901 windows); candidate advantage +0.0500, MDE 0.0504, verdict: SIGNIFICANT at 95%
"""

CONTRACT = cd.ComparatorContract()


def parse(stdout: str) -> cd.Comparison:
    return cd.parse_comparison(["compare"], 0, stdout, "", CONTRACT)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


def windows_json(
    run: str,
    *,
    saturated: float = 0.848,
    ruin: int = 0,
    supports: str = "a" * 64,
    mz: float | None = 0.6653,
    mz_se: float = 0.0286,
    version: int = 3,
    with_trade: bool = True,
) -> dict:
    document = {
        "format_version": version,
        "run": run,
        "global_step": 2999,
        "split": "val",
        "context": 896,
        "eval_window_seed": 0xE7A15E7D0001,
        "corpus_fingerprint": "f" * 64,
        "split_bounds": [1_759_838_400_000, 1_762_430_400_000],
        "marginal_nll_bar": 3.9,
        "scoring": "smoothed",
        "realized_batch": 23,
        "realized_steps": 3000,
        "windows": [
            {
                "symbol": "SPY",
                "bar_index": 1024,
                "ts_ms": 1_760_000_000_000,
                "nll_dof": [0.5, 0.9, 0.7, 0.8, 0.0],
                "conditional_nll": {"numerator": [0.0] * 5, "denominator": [1.0] * 5},
            }
        ],
        "supports_sha256": supports,
        # W0.3: `calibration` is null on a pass that ran no bench.
        "calibration": None
        if mz is None
        else {"mean_beta": mz, "mean_beta_se": mz_se, "variance_beta": 1.31},
        # W0.1: all three are over the TRADED PREFIX, not the full window list.
        "selection_edge_bps": [0.31, 0.42],
        "model_growth_bps": [0.55, 0.61],
        "traded_blocks": [24301, 24301],
    }
    if with_trade:
        document["trade"] = {
            "policies": [
                {"policy": "model", "ruin_bars": ruin, "net_growth": 3.1e-05},
                {"policy": "marginal null", "ruin_bars": 0, "net_growth": 1.0e-05},
            ],
            "free_kelly_saturated": saturated,
            "cap_curve": [{"cap": 0.25, "edge": 3.8e-05, "ruin_bars": ruin}],
        }
    return document


def write_artifacts(root: Path, **overrides) -> tuple[cd.ArtifactFacts, cd.ArtifactFacts]:
    base_path = root / "base.windows.json"
    cand_path = root / "cand.windows.json"
    base_path.write_text(json.dumps(windows_json("base")))
    cand_path.write_text(json.dumps(windows_json("cand", **overrides)))
    return (
        cd.read_artifact(base_path, CONTRACT),
        cd.read_artifact(cand_path, CONTRACT),
    )


def seed_result(
    root: Path,
    *,
    seed: int = 24301,
    stdout: str | None = None,
    log_text: str | None = None,
    exit_code: int = 0,
    compare_exit: int = 0,
    compare_stderr: str = "",
    **artifact_overrides,
) -> cd.SeedResult:
    base, cand = write_artifacts(root, **artifact_overrides)
    log_path = None
    if log_text is not None:
        log_path = root / f"s{seed}.log"
        log_path.write_text(log_text)
    text = compare_output() if stdout is None else stdout
    comparison = cd.parse_comparison(["compare"], compare_exit, text, compare_stderr, CONTRACT)
    return cd.SeedResult(
        seed=seed,
        run_name=f"arm_base_s{seed}_screen",
        launch_exit_code=exit_code,
        log_path=str(log_path) if log_path else None,
        artifact=cand,
        incumbent_artifact=base,
        comparison=comparison,
    )


# ---------------------------------------------------------------------------
# Spec fixture
# ---------------------------------------------------------------------------

SPEC_DOC = {
    "spec_version": 3,
    "tiers": {
        "screen": {"steps": 3000, "validation_windows": 2048, "seeds": [24301]},
        "confirm": {"steps": 6500, "validation_windows": 4096, "seeds": [24301, 24302, 24303]},
    },
    "arms": [
        {"name": "control", "control": True, "class": "geometry_preserving", "flags": []},
        {"name": "GP", "wave": "B", "class": "geometry_preserving", "flags": ["--x"]},
        {"name": "GC", "wave": "E", "class": "geometry_changing", "flags": ["--y"]},
        {
            "name": "EX",
            "wave": "F",
            "class": "geometry_preserving",
            "exempt_from_cull": True,
            "exempt_reason": "correctness arm",
            "flags": [],
        },
    ],
}


def fixture_defaults(root: Path) -> dict:
    launcher = root / "launcher"
    executable = root / "trainer"
    supports = root / "supports.json"
    launcher.write_text("#!/bin/sh\nexec \"$@\"\n")
    executable.write_text('#!/bin/sh\ntouch "$(dirname "$0")/trainer-executed"\nexit 0\n')
    launcher.chmod(0o755)
    executable.chmod(0o755)
    supports.write_text('{"fixture":true}\n')
    (root / "supports-volstd-v8.json").write_text('{"format_version":8}\n')
    return {
        "launcher": str(launcher),
        "launcher_sha256": cd._sha256_file(launcher),
        "executable": str(executable),
        "executable_sha256": cd._sha256_file(executable),
        "mlq_max_parallel_runs": 1,
        "mlq_time_limit": "1h",
        "supports": str(supports),
        "supports_sha256": cd._sha256_file(supports),
        "runs_dir": "runs",
        "ledger": "ledger.jsonl",
        "common_flags": [
            "--split-bounds", "1759839000000,1773427500000",
            "--freeze-supports",
            "--freeze-market-supports",
            "--exact-batch",
            "--batch-size", "24",
        ],
    }


def load_spec(root: Path, **extra) -> cd.Spec:
    document = json.loads(json.dumps(SPEC_DOC))
    document["defaults"] = fixture_defaults(root)
    for arm in document["arms"]:
        if arm.get("class") == "geometry_changing" and "supports" not in arm:
            arm["supports"] = str(root / "supports-volstd-v8.json")
            arm["supports_sha256"] = cd._sha256_file(root / "supports-volstd-v8.json")
    document.update(extra)
    path = root / "spec.json"
    path.write_text(json.dumps(document))
    return cd.load_spec(path)


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------


def test_parser() -> None:
    print("parser")
    parsed = parse(compare_output())
    check(parsed.windows == 2048 and parsed.scoring == "smoothed", "header line")
    check(parsed.baseline_mean == 3.1416 and parsed.candidate_mean == 3.1350, "level lines")
    check(parsed.correlation == 0.9971 and parsed.worse_windows == 803, "correlation line")
    check(parsed.mde == 0.0059 and parsed.verdict == "SIGNIFICANT at 95%", "verdict line")
    check(parsed.dispersions["nll_difference"].mean == -0.0066, "aggregate difference")
    check(parsed.dispersions["conditional_difference"].ci_high == -0.0006, "conditional delta")
    check(
        [parsed.dispersions[f"dof_{d}"].mean for d in "rsuvw"]
        == [-0.0012, -0.0030, -0.0015, -0.0009, 0.0000],
        "all five per-DOF deltas, in [r,s,u,v,w] order",
    )
    check(parsed.dispersions["dof_r"].blocks == 412, "block count survives")
    check(parsed.dispersions["edge_difference"].mean == 0.0700, "W0.1 edge_difference")
    check(
        parsed.dispersions["model_growth_difference"].mean == 0.0250,
        "W0.1 model_growth_difference, not captured by the bare-edge rule",
    )
    check(parsed.advantage["edge_difference"] == 0.0700, "`candidate advantage` cross-check")
    check(
        parsed.advantage["edge_difference"] == parsed.dispersions["edge_difference"].mean,
        "for a POSITIVE-is-better row the advantage equals the mean, so no flip is needed",
    )
    check(parsed.unmapped == [] and parsed.refusal is None, "nothing unmapped, no refusal")

    # The absent form must be recorded WITH its reason, not silently dropped.
    absent = parse(ABSENT_OUTPUT)
    check("edge_difference" in absent.absent, "the absent form maps to the same canonical key")
    check(
        "pre-v3 artifact" in absent.absent["model_growth_difference"],
        "and the reason it printed is kept",
    )
    check("edge_difference" not in absent.dispersions, "an absent metric has no dispersion")

    # An unrecognized label must be RECORDED, never dropped.
    strange = NATS_BLOCK + (
        "  brand new statistic nobody told the driver about "
        "0.1234 +/- 0.0010 (95% CI 0.1200..0.1260, 12 blocks / 34 windows)\n"
    )
    check(len(parse(strange).unmapped) == 1, "unknown dispersion line is recorded as unmapped")

    # The pre-W0.1 nats label still reads back, so stdout captured in an older ledger parses.
    legacy = NATS_BLOCK.replace(
        "paired delta [nats/bar, candidate - baseline, NEGATIVE is better]",
        "paired delta (candidate - baseline)",
    )
    check(
        parse(legacy).dispersions["nll_difference"].mean == -0.0066,
        "the pre-W0.1 nats label still parses",
    )

    # Refusals are classified, and geometry is not confused with traded prefix.
    geometry = cd.parse_comparison(["c"], 101, "", GEOMETRY_REFUSAL_STDERR, CONTRACT)
    prefix = cd.parse_comparison(["c"], 101, "", PREFIX_REFUSAL_STDERR, CONTRACT)
    check(geometry.refusal == "geometry", "a geometry refusal is classified as such")
    check(prefix.refusal == "traded_prefix", "a traded-prefix refusal is NOT called geometry")
    check(
        cd.parse_comparison(["c"], 101, "", "Error: disk on fire", CONTRACT).refusal
        == "unclassified",
        "an unrecognized refusal is not silently mapped onto a known one",
    )

    # Non-finite prints must round-trip: `{:.4}` on NaN is `NaN`, on inf is `inf`.
    nonfinite = NATS_BLOCK.replace(
        "  delta w  0.0000 +/- 0.0000 (95% CI 0.0000..0.0000, 412 blocks / 2048 windows)",
        "  delta w  NaN +/- inf (95% CI -inf..inf, 0 blocks / 0 windows)",
    )
    parsed_nonfinite = parse(nonfinite)
    check(
        cd.math.isnan(parsed_nonfinite.dispersions["dof_w"].mean)
        and cd.math.isinf(parsed_nonfinite.dispersions["dof_w"].se),
        "Rust's NaN/inf prints parse",
    )
    check(
        json.dumps(parsed_nonfinite.as_json(), allow_nan=False).count('"nan"') >= 1,
        "non-finite values serialize into the ledger without allow_nan",
    )

    geometry_only = parse(GEOMETRY_OUTPUT)
    cd.validate_geometry_comparison(geometry_only)
    check(geometry_only.model_growth_only, "geometry banner selects model-growth-only evidence")
    check(
        set(geometry_only.dispersions) == {"model_growth_difference"},
        "no geometry-dependent dispersion survives",
    )
    check(
        len(geometry_only.absent) == 5
        and all(reason == "forbidden across bin geometries" for reason in geometry_only.absent.values()),
        "every forbidden comparator row is explicitly recorded as withheld",
    )


def test_artifact_reading(root: Path) -> None:
    print("artifact reading")
    base, cand = write_artifacts(root, saturated=0.91, ruin=7, mz=0.94)
    check(base.free_kelly_saturated == 0.848 and cand.free_kelly_saturated == 0.91, "saturation")
    check(cand.model_ruin_bars == 7 and cand.selection_cap_ruin_bars == 7, "ruin bars, both rows")
    check(
        cand.mz_slope == 0.94 and cand.mz_slope_source == "calibration.mean_beta",
        "W0.3 MZ slope level, read from `calibration.mean_beta`",
    )
    check(cand.mz_slope_se == 0.0286, "and its published standard error")
    check(
        write_artifacts(root, mz=None)[1].mz_slope is None,
        "a null `calibration` reads back as absent, not as zero",
    )
    check(
        cand.traded_edge_windows == 2
        and cand.traded_growth_windows == 2
        and cand.traded_blocks == 2,
        "the traded-prefix vector lengths, which differ from the pinned window count",
    )
    check(cand.windows == 1, "and the pinned window count is read separately")
    check(len(cand.supports_sha256 or "") == 64, "supports fingerprint")
    check(cand.sha256 != base.sha256, "artifacts are digested for the ledger")
    spec = load_spec(root)
    no_trade = write_artifacts(root, with_trade=False)[1]
    try:
        cd.Driver._require_trade_evidence(spec.tiers["confirm"], no_trade, "candidate")
        check(False, "confirm must reject an artifact without trade evidence")
    except cd.DriverError:
        check(True, "confirm refuses a trade-free diagnostic artifact")
    try:
        cd.Driver._require_trade_evidence(spec.tiers["screen"], no_trade, "candidate")
        check(True, "screen may consume a diagnostic context artifact")
    except cd.DriverError:
        check(False, "screen may consume a diagnostic context artifact")

    confirm = spec.tiers["confirm"]
    control = spec.arms["control"]
    run_name = control.run_name(confirm.seeds[0], confirm)
    promoted = root / spec.artifact_path(run_name, confirm)
    promoted.parent.mkdir(parents=True, exist_ok=True)
    promoted.write_text(json.dumps(windows_json(run_name, with_trade=True)))
    driver_spec = cd.dataclasses.replace(spec, ledger_path="artifact-resolution-ledger.jsonl")
    args = cd.parse_args(
        ["--spec", str(spec.path), "--repo-root", str(root), "--dry-run"]
    )
    driver = cd.Driver(driver_spec, args)
    result = cd.SeedResult(confirm.seeds[0], run_name, 0, None, None, None, None)
    driver.compare_seed(control, confirm, result, None)
    check(
        result.artifact is not None
        and result.artifact.path == promoted
        and result.artifact.trade_present
        and promoted.name == "pretrain_best.windows.json",
        "confirm resolves and consumes the promoted trade-bearing artifact end to end",
    )

    stale = root / "stale.windows.json"
    stale.write_text(json.dumps(windows_json("stale", version=2)))
    try:
        cd.read_artifact(stale, CONTRACT)
        check(False, "a stale format_version must fail loudly")
    except cd.DriverError:
        check(True, "a stale format_version fails loudly")


def test_tripwires(root: Path) -> None:
    print("tripwires")
    spec = load_spec(root)
    gp, gc = spec.arms["GP"], spec.arms["GC"]

    def statuses(arm, **kwargs) -> dict[str, str]:
        kwargs.setdefault("log_text", "step 10: loss 3.14\n")
        result = seed_result(root, **kwargs)
        return {wire.name: wire.status for wire in cd.evaluate_tripwires(arm, [result])}

    clean = statuses(gp)
    check(all(status == "CLEAR" for status in clean.values()), f"clean arm: all CLEAR ({clean})")
    check(len(clean) == 6, "exactly six tripwires")

    # 1 non-finite loss.
    tripped = statuses(gp, log_text="loss is not finite at step 812: NaN\n")
    check(tripped["nonfinite_loss_or_grad_norm"] == "TRIPPED", "1: non-finite loss message")
    tripped = statuses(gp, log_text="ok\n", exit_code=101)
    check(tripped["nonfinite_loss_or_grad_norm"] == "TRIPPED", "1: non-zero exit")

    # 2 saturation rise. 0.848 -> 0.899 is 5.1 pp; 0.848 -> 0.897 is 4.9 pp.
    check(
        statuses(gp, saturated=0.899)["free_kelly_saturated_rise_over_5pp"] == "TRIPPED",
        "2: +5.1 pp trips",
    )
    check(
        statuses(gp, saturated=0.897)["free_kelly_saturated_rise_over_5pp"] == "CLEAR",
        "2: +4.9 pp does not trip",
    )
    check(
        statuses(gp, with_trade=False)["free_kelly_saturated_rise_over_5pp"] == "UNEVALUATED",
        "2: absent `trade` is UNEVALUATED, not CLEAR",
    )

    # 3 ruin bars.
    check(statuses(gp, ruin=1)["ruin_bars_appeared"] == "TRIPPED", "3: ruin appeared")
    check(statuses(gp, ruin=0)["ruin_bars_appeared"] == "CLEAR", "3: no ruin")

    # 4 correlation.
    low = compare_output().replace("correlation 0.9971", "correlation 0.8500")
    check(
        statuses(gp, stdout=low)[
            "paired_correlation_below_0.9"
        ]
        == "TRIPPED",
        "4: correlation 0.85 trips",
    )
    geometry_correlation = {
        wire.name: wire
        for wire in cd.evaluate_tripwires(
            gc, [seed_result(root, stdout=GEOMETRY_OUTPUT, supports="b" * 64, log_text="ok\n")]
        )
    }["paired_correlation_below_0.9"]
    check(
        geometry_correlation.status == "NOT_APPLICABLE"
        and not geometry_correlation.binding
        and "deliberately withholds" in geometry_correlation.evidence,
        "4: withheld cross-geometry correlation is explicitly non-binding and not applicable",
    )

    # 5 r-NLL regression at 2 SE, geometry-preserving only. +0.0012 +/- 0.0004 is 3 SE worse.
    regressed = compare_output().replace(
        "  delta r  -0.0012 +/- 0.0004", "  delta r  0.0012 +/- 0.0004"
    )
    check(
        statuses(gp, stdout=regressed)["r_nll_regression_at_2se"] == "TRIPPED",
        "5: resolved r regression trips a geometry-preserving arm",
    )
    unresolved = compare_output().replace(
        "  delta r  -0.0012 +/- 0.0004", "  delta r  0.0012 +/- 0.0009"
    )
    check(
        statuses(gp, stdout=unresolved)["r_nll_regression_at_2se"] == "CLEAR",
        "5: an UNRESOLVED r regression does not trip",
    )
    wires = {
        wire.name: wire
        for wire in cd.evaluate_tripwires(
            gc, [seed_result(root, stdout=regressed, log_text="ok\n")]
        )
    }
    check(
        wires["r_nll_regression_at_2se"].status == "CLEAR"
        and not wires["r_nll_regression_at_2se"].binding,
        "5: not applicable to a geometry_changing arm — every nll_* is forbidden there",
    )

    # 6 supports geometry.
    check(
        statuses(gp, supports="b" * 64)["undeclared_geometry_change"] == "TRIPPED",
        "6: supports mismatch trips an undeclared arm",
    )
    wires = {
        wire.name: wire
        for wire in cd.evaluate_tripwires(
            gc, [seed_result(root, supports="b" * 64, log_text="ok\n")]
        )
    }
    check(
        wires["undeclared_geometry_change"].status == "CLEAR"
        and not wires["undeclared_geometry_change"].binding,
        "6: a DECLARED geometry change is expected, not a tripwire",
    )
    refused = seed_result(
        root,
        compare_exit=101,
        stdout="",
        compare_stderr=GEOMETRY_REFUSAL_STDERR,
        log_text="ok\n",
    )
    wires = {w.name: w for w in cd.evaluate_tripwires(gp, [refused])}
    check(
        wires["undeclared_geometry_change"].status == "TRIPPED"
        and "REFUSED the pair on bin geometry" in wires["undeclared_geometry_change"].evidence,
        "6: a comparator geometry refusal is picked up too",
    )
    prefix_refused = seed_result(
        root,
        compare_exit=101,
        stdout="",
        compare_stderr=PREFIX_REFUSAL_STDERR,
        log_text="ok\n",
    )
    wires = {w.name: w for w in cd.evaluate_tripwires(gp, [prefix_refused])}
    check(
        wires["undeclared_geometry_change"].status == "CLEAR",
        "6: a traded-prefix refusal is NOT reported as a geometry change",
    )


def test_s_seed_refusal(root: Path) -> None:
    print("s_seed refusal")
    spec = load_spec(root)
    result = seed_result(root, log_text="ok\n")
    try:
        cd.decide(spec, spec.arms["GP"], spec.tiers["screen"], [result])
        check(False, "decide() must refuse without s_seed")
    except cd.RefusalError as error:
        check("no `s_seed` supplied" in str(error), "decide() refuses without s_seed")

    with_seed = load_spec(
        root,
        s_seed={"screen": {"edge_difference": 0.02}},
        s_seed_provenance="W0.5 control seeds 24301/2/3",
    )
    decision = cd.decide(with_seed, with_seed.arms["GP"], with_seed.tiers["screen"], [result])
    check(decision.verdict == cd.SURVIVE, "with s_seed the same inputs decide")

    no_provenance = load_spec(root, s_seed={"screen": {"edge_difference": 0.02}})
    try:
        cd.decide(no_provenance, no_provenance.arms["GP"], no_provenance.tiers["screen"], [result])
        check(False, "s_seed without provenance must refuse")
    except cd.RefusalError as error:
        check("s_seed_provenance" in str(error), "s_seed without provenance refuses")

    # A stated absence of the PRIMARY metric must refuse, quoting the comparator's reason,
    # rather than falling back to the forbidden aggregate nats difference.
    bare = cd.SeedResult(
        seed=24301,
        run_name="arm",
        launch_exit_code=0,
        log_path=None,
        artifact=result.artifact,
        incumbent_artifact=result.incumbent_artifact,
        comparison=parse(ABSENT_OUTPUT),
    )
    try:
        cd.decide(with_seed, with_seed.arms["GP"], with_seed.tiers["screen"], [bare])
        check(False, "a missing primary metric must refuse")
    except cd.RefusalError as error:
        check(
            "`edge_difference` is not available" in str(error)
            and "explicitly forbidden" in str(error)
            and "pre-v3 artifact" in str(error),
            "a missing primary refuses, names the forbidden fallback, quotes the reason",
        )

    # Even with the fixed escape hatch, a comparator failure is a refusal, never no effect.
    refused = cd.SeedResult(
        seed=24301,
        run_name="E1_base_s24301_screen",
        launch_exit_code=0,
        log_path=None,
        artifact=result.artifact,
        incumbent_artifact=result.incumbent_artifact,
        comparison=cd.parse_comparison(["c"], 101, "", GEOMETRY_REFUSAL_STDERR, CONTRACT),
    )
    gc_spec = load_spec(
        root,
        s_seed={"screen": {"model_growth_difference": 0.03}},
        s_seed_provenance="W0.5",
    )
    try:
        cd.decide(gc_spec, gc_spec.arms["GC"], gc_spec.tiers["screen"], [refused])
        check(False, "a geometry refusal must not be read as `no effect`")
    except cd.RefusalError as error:
        check(
            "--allow-geometry-change" in str(error)
            and "DIFFERENT BIN GEOMETRIES" in str(error),
            "a geometry refusal names the fixed escape contract plus the stderr",
        )

    # A traded-prefix refusal is a DIFFERENT finding and must be reported as one.
    prefix = cd.SeedResult(
        seed=24301,
        run_name="arm",
        launch_exit_code=0,
        log_path=None,
        artifact=result.artifact,
        incumbent_artifact=result.incumbent_artifact,
        comparison=cd.parse_comparison(["c"], 101, "", PREFIX_REFUSAL_STDERR, CONTRACT),
    )
    try:
        cd.decide(with_seed, with_seed.arms["GP"], with_seed.tiers["screen"], [prefix])
        check(False, "a traded-prefix refusal must refuse")
    except cd.RefusalError as error:
        check(
            "traded different prefixes" in str(error)
            and "unrelated to the treatment" in str(error),
            "a traded-prefix refusal is reported as its own finding",
        )


def test_screen_gate(root: Path) -> None:
    print("screen gate: cull if mean < -1.0 * s_seed")
    spec = load_spec(
        root,
        s_seed={"screen": {"edge_difference": 0.02, "model_growth_difference": 0.03}},
        s_seed_provenance="W0.5 control seeds 24301/2/3",
    )
    arm, tier = spec.arms["GP"], spec.tiers["screen"]

    for edge, expected in ((0.0700, cd.SURVIVE), (-0.0199, cd.SURVIVE), (-0.0201, cd.CULL)):
        result = seed_result(root, stdout=compare_output(edge=edge), log_text="ok\n")
        decision = cd.decide(spec, arm, tier, [result])
        check(decision.verdict == expected, f"edge {edge:+.4f} -> {expected}")
        check(decision.primary_metric == "edge_difference", "primary is the paired edge")
        check(
            f"{-0.02:+.6f}" in decision.justification and "s_seed = 0.020000" in decision.justification,
            "threshold and s_seed are both in the justification text",
        )

    # A tripwire beats the primary metric, however good it is.
    result = seed_result(root, stdout=compare_output(edge=5.0), ruin=3, log_text="ok\n")
    decision = cd.decide(spec, arm, tier, [result])
    check(decision.verdict == cd.CULL, "a huge edge does not survive a tripwire")
    check("Culled by hard tripwire" in decision.justification, "tripwire cull says so")
    check(decision.threshold_value is None, "no threshold was consulted")

    # Geometry-changing arms are judged on model growth and withhold every nats metric.
    gc_result = seed_result(root, stdout=compare_output(growth=-0.0400), supports="b" * 64, log_text="ok\n")
    decision = cd.decide(spec, spec.arms["GC"], tier, [gc_result])
    check(decision.primary_metric == "model_growth_difference", "geometry_changing primary")
    check(decision.guard_metric == "mz_slope", "geometry_changing guard")
    check("edge_difference" in decision.forbidden_metrics_withheld, "differenced edge withheld")
    check("nll_difference" in decision.forbidden_metrics_withheld, "NLL comparison withheld")
    check(decision.verdict == cd.CULL, "-0.04 < -1.0 * 0.03 culls")


def test_confirm_gate(root: Path) -> None:
    print("confirm gate: carry if mean > +2.0 * sqrt(s_seed^2/3 + se_win^2)")
    spec = load_spec(
        root,
        s_seed={
            "confirm": {
                "edge_difference": 0.0140,
                "model_growth_difference": 0.0050,
            }
        },
        s_seed_provenance="W0.5 control seeds 24301/2/3",
    )
    arm, tier = spec.arms["GP"], spec.tiers["confirm"]
    # s_seed 0.014, se_win 0.0100 -> 2 * sqrt(0.014^2/3 + 0.01^2) = 0.0257164
    threshold = 2.0 * cd.math.sqrt(0.0140**2 / 3 + 0.0100**2)
    check(abs(threshold - 0.0257164) < 1e-6, f"threshold arithmetic: {threshold:.7f}")

    def seeds(edge: float, **kwargs):
        return [
            seed_result(
                root,
                seed=seed,
                stdout=compare_output(edge=edge, edge_se=0.0100),
                log_text="ok\n",
                **kwargs,
            )
            for seed in tier.seeds
        ]

    decision = cd.decide(spec, arm, tier, seeds(0.0400))
    check(decision.verdict == cd.CARRY, "+0.0400 > 0.0257 carries")
    check(len(decision.primary_per_seed) == 3, "three seed replicates averaged")
    check(abs((decision.se_win or 0) - 0.0100) < 1e-9, "se_win is the paired SE, UNDIVIDED")
    check(abs((decision.threshold_value or 0) - threshold) < 1e-9, "threshold recorded numerically")

    check(cd.decide(spec, arm, tier, seeds(0.0200)).verdict == cd.DISCARD, "+0.0200 discards")
    check(
        cd.decide(spec, arm, tier, seeds(0.0258)).verdict == cd.CARRY,
        "+0.0258 is just over the band and carries",
    )
    check(
        cd.decide(spec, arm, tier, seeds(0.0256)).verdict == cd.DISCARD,
        "+0.0256 is just under the band and does not",
    )

    # Guard veto beats a cleared band.
    vetoed = compare_output(edge=0.0400, edge_se=0.0100).replace(
        "  delta r  -0.0012 +/- 0.0004", "  delta r  0.0012 +/- 0.0004"
    )
    results = [
        seed_result(root, seed=seed, stdout=vetoed, log_text="ok\n") for seed in tier.seeds
    ]
    decision = cd.decide(spec, arm, tier, results)
    check(decision.verdict == cd.CULL, "a resolved r regression is tripwire 5, so it CULLS")
    check(decision.guard_verdict == "VETO", "and the guard also vetoes")

    # An unevaluated binding tripwire downgrades a carry.
    downgraded = cd.decide(
        spec,
        arm,
        tier,
        [
            seed_result(
                root, seed=seed, stdout=compare_output(edge=0.0400, edge_se=0.0100),
                log_text="ok\n", with_trade=False,
            )
            for seed in tier.seeds
        ],
    )
    check(downgraded.verdict == cd.DISCARD, "an UNEVALUATED tripwire downgrades CARRY to DISCARD")

    # Cross-geometry comparison intentionally withholds correlation and every decoded metric.
    # Its model-growth primary and MZ level guard still form a complete confirm decision.
    geometry = spec.arms["GC"]
    geometry_results = [
        seed_result(
            root,
            seed=seed,
            stdout=GEOMETRY_OUTPUT,
            log_text="ok\n",
            supports="b" * 64,
        )
        for seed in tier.seeds
    ]
    geometry_decision = cd.decide(spec, geometry, tier, geometry_results)
    geometry_wire = next(
        wire
        for wire in geometry_decision.tripwires
        if wire.name == "paired_correlation_below_0.9"
    )
    check(
        geometry_decision.verdict == cd.CARRY
        and geometry_wire.status == "NOT_APPLICABLE"
        and not geometry_wire.binding,
        "a qualified geometry-changing confirm carries with intentionally withheld correlation",
    )

    # The same absence remains binding for a same-geometry confirm.
    no_correlation = compare_output(edge=0.0400, edge_se=0.0100).replace(
        "  per-window correlation 0.9971, candidate worse on 803 of 2048 windows\n", ""
    )
    same_geometry_missing = cd.decide(
        spec,
        arm,
        tier,
        [
            seed_result(root, seed=seed, stdout=no_correlation, log_text="ok\n")
            for seed in tier.seeds
        ],
    )
    same_geometry_wire = next(
        wire
        for wire in same_geometry_missing.tripwires
        if wire.name == "paired_correlation_below_0.9"
    )
    check(
        same_geometry_missing.verdict == cd.DISCARD
        and same_geometry_wire.status == "UNEVALUATED"
        and same_geometry_wire.binding,
        "a same-geometry confirm still binds correlation and refuses to carry without it",
    )
    check("OVERRIDDEN to DISCARD" in downgraded.justification, "and the ledger says why")

    # Exempt arm: adopted regardless of sign, but a crash still blocks it.
    exempt = spec.arms["EX"]
    adopted = cd.decide(spec, exempt, tier, seeds(-0.5000))
    check(adopted.verdict == cd.ADOPT_EXEMPT, "an exempt arm is adopted at a negative edge")
    check("correctness arm" in adopted.justification, "and its reason is recorded")
    crashed = cd.decide(
        spec,
        exempt,
        tier,
        [
            seed_result(
                root, seed=seed, stdout=compare_output(edge=-0.5),
                log_text="gradient norm is not finite at step 5: inf\n",
            )
            for seed in tier.seeds
        ],
    )
    check(crashed.verdict == cd.CULL, "a non-finite loss still culls an exempt arm")


def test_mz_guard(root: Path) -> None:
    print("MZ slope guard (geometry_changing arms)")
    spec = load_spec(root)
    gc = spec.arms["GC"]

    def guard(**overrides) -> tuple[str, str]:
        return cd.evaluate_guard(gc, [seed_result(root, log_text="ok\n", **overrides)])

    # Incumbent beta 0.6653 (|beta-1| = 0.3347), published SE 0.0286 -> band 0.0572.
    verdict, evidence = guard(mz=0.9000)
    check(verdict == "PASS", "moving toward 1.0 passes")
    check("drift -0.2347" in evidence, f"and the drift is recorded: {evidence}")
    check("UNPAIRED band" in evidence, "the evidence states the band is UNPAIRED")

    verdict, evidence = guard(mz=0.5000)
    check(verdict == "VETO", "a resolved drift AWAY from 1.0 vetoes")
    check("RESOLVED DRIFT AWAY FROM 1.0" in evidence, "and says so")

    verdict, _ = guard(mz=0.6300)
    check(verdict == "PASS", "an unresolved drift away from 1.0 does not veto")

    verdict, evidence = guard(mz=1.4000)
    check(verdict == "VETO", "overshooting past 1.0 is also a drift away from it")

    verdict, evidence = guard(mz=None)
    check(verdict == "UNEVALUATED", "a null `calibration` is UNEVALUATED, not PASS")
    check("calibration.mean_beta" in evidence, "and names where the driver looked")


def test_naming_and_ordering(root: Path) -> None:
    print("naming and ordering")
    spec = load_spec(
        root,
        arms=[
            {"name": "control", "control": True, "class": "geometry_preserving", "flags": []},
            {"name": "A1", "class": "geometry_preserving", "flags": []},
            {"name": "A2", "class": "geometry_preserving", "carried_stack": ["A1"], "flags": []},
            {
                "name": "C1",
                "class": "geometry_preserving",
                "carried_stack": ["A1", "A2", "B1"],
                "incumbent": "A2",
                "flags": [],
            },
            {"name": "B1", "class": "geometry_preserving", "flags": []},
        ],
    )
    screen, confirm = spec.tiers["screen"], spec.tiers["confirm"]
    check(
        spec.arms["C1"].run_name(24301, screen) == "C1_a1a2b1_s24301_screen",
        "3.4 naming: <arm>_<stack>_s<seed>, plus the tier so two tiers cannot collide",
    )
    check(
        spec.arms["C1"].run_name(24301, screen) != spec.arms["C1"].run_name(24301, confirm),
        "the same arm and seed at two tiers get distinct run dirs",
    )
    check(spec.arms["A1"].run_name(24301, screen).startswith("A1_base_"), "empty stack is `base`")
    levels = cd.dependency_levels(spec, list(spec.arm_order))
    order = {name: index for index, level in enumerate(levels) for name in level}
    check(order["control"] == 0, "the control goes first")
    check(order["A1"] < order["A2"] < order["C1"], "an arm follows its incumbent")
    check(order["B1"] == order["A1"], "independent arms share a level and may run concurrently")
    check(cd.incumbent_of(spec, spec.arms["A2"]) == "A1", "incumbent defaults to the stack head")
    check(cd.incumbent_of(spec, spec.arms["C1"]) == "A2", "an explicit incumbent wins")
    check(cd.incumbent_of(spec, spec.arms["B1"]) == "control", "an empty stack faces the control")


def test_spec_validation(root: Path) -> None:
    print("spec validation")

    def rejects(what: str, **extra) -> None:
        try:
            load_spec(root, **extra)
            check(False, f"must reject: {what}")
        except cd.SpecError:
            check(True, f"rejects: {what}")

    rejects(
        "an arm with no `class` — the field that decides which metric may judge it",
        arms=[
            {"name": "control", "control": True, "class": "geometry_preserving", "flags": []},
            {"name": "X", "flags": []},
        ],
    )
    rejects(
        "two control arms",
        arms=[
            {"name": "a", "control": True, "class": "geometry_preserving", "flags": []},
            {"name": "b", "control": True, "class": "geometry_preserving", "flags": []},
        ],
    )
    rejects("no control arm", arms=[{"name": "a", "class": "geometry_preserving", "flags": []}])
    rejects(
        "exempt_from_cull with no reason",
        arms=[
            {"name": "control", "control": True, "class": "geometry_preserving", "flags": []},
            {"name": "X", "class": "geometry_preserving", "exempt_from_cull": True, "flags": []},
        ],
    )
    rejects(
        "a carried_stack naming an unknown arm",
        arms=[
            {"name": "control", "control": True, "class": "geometry_preserving", "flags": []},
            {"name": "X", "class": "geometry_preserving", "carried_stack": ["nope"], "flags": []},
        ],
    )
    rejects(
        "a run name `validate_run_name` would reject",
        arms=[
            {"name": "control", "control": True, "class": "geometry_preserving", "flags": []},
            {"name": "../escape", "class": "geometry_preserving", "flags": []},
        ],
    )
    rejects(
        "flags given as one string instead of argv tokens",
        arms=[
            {"name": "control", "control": True, "class": "geometry_preserving", "flags": []},
            {"name": "X", "class": "geometry_preserving", "flags": [["--a", "1"]]},
        ],
    )
    rejects("missing immutable launcher/executable and mlq declarations", defaults={})
    missing_supports_digest = fixture_defaults(root)
    del missing_supports_digest["supports_sha256"]
    rejects(
        "a mutable default supports path without its required digest",
        defaults=missing_supports_digest,
    )
    rejects(
        "an arm supports path without its required digest",
        arms=[
            {"name": "control", "control": True, "class": "geometry_preserving", "flags": []},
            {
                "name": "GC",
                "class": "geometry_changing",
                "supports": str(root / "supports-volstd-v8.json"),
                "flags": [],
            },
        ],
    )
    missing_market_digest = fixture_defaults(root)
    missing_market_digest["market_supports"] = str(root / "bar_market_supports.300.json")
    rejects(
        "a configured market supports path without its digest",
        defaults=missing_market_digest,
    )
    bad_geometry_defaults = fixture_defaults(root)
    bad_geometry_defaults["geometry_change_compare_flags"] = ["--allow-geometry-change"]
    rejects(
        "operator-supplied geometry flags instead of the fixed model-growth-only contract",
        defaults=bad_geometry_defaults,
    )

    spec = load_spec(root)
    check(spec.arms["EX"].exempt_reason == "correctness arm", "notes and reasons survive")
    check(
        spec.sha256 == cd.hashlib.sha256((root / "spec.json").read_bytes()).hexdigest(),
        "the spec is digested so the ledger can pin which one decided",
    )
    check(
        spec.tiers["screen"].artifact == "pretrain_best_diag896"
        and spec.tiers["confirm"].artifact == "pretrain_best"
        and spec.artifact_path("GP_base_s24301_confirm", spec.tiers["confirm"])
        == Path("runs/GP_base_s24301_confirm/weights/pretrain_best.windows.json"),
        "screen keeps diagnostic context while confirm resolves the promoted trade artifact",
    )
    geometry_argv = cd.build_compare(
        spec, spec.arms["GC"], Path("baseline.windows.json"), Path("candidate.windows.json")
    )
    check(
        geometry_argv[-1] == "--allow-geometry-change"
        and geometry_argv.count("--allow-geometry-change") == 1,
        "geometry-changing comparison always uses exactly the existing escape hatch",
    )
    preserving_argv = cd.build_compare(
        spec, spec.arms["GP"], Path("baseline.windows.json"), Path("candidate.windows.json")
    )
    check(
        "--allow-geometry-change" not in preserving_argv,
        "geometry-preserving comparison never weakens the supports check",
    )

    example = Path(cd.__file__).parent / "campaign_arms.example.json"
    shipped = cd.load_spec(example)
    check(len(shipped.arms) == 6, "the shipped example spec loads")
    check(shipped.s_seed == {}, "the shipped example ships NO s_seed, so it refuses by default")
    check(
        shipped.supports_sha256
        == "58c87dfc2f9dbba4e45981d8b07d4ec7d5befbe79ed898a63e1dda8f78344d45"
        and shipped.arms["E1"].supports
        == "long_data/bars/bar_supports.volstd.300.seed24301.v8.json"
        and shipped.arms["E1"].supports_sha256
        == "af3070feae1aa17ed1794d483521c35291f0878c1666bca8613a7230aa73ed2a"
        and "--vol-standardize-targets" in shipped.arms["E1"].flags,
        "the example authenticates both raw and standardized geometry paths",
    )
    check(
        shipped.tiers["confirm"].artifact == "pretrain_best",
        "the example explicitly prevents confirm from using a trade-free diagnostic artifact",
    )


def test_launch_refusals_and_mlq(root: Path) -> None:
    print("launch pins, immutable identity, and mlq-only execution")
    spec = load_spec(root)
    contract = cd.validate_launch_contract(spec, root)
    check(contract["batch_size"] == 24 and contract["exact_batch"], "exact batch is resolved")
    check(
        contract["launcher"]["sha256"] == spec.launcher_sha256
        and contract["executable"]["sha256"] == spec.executable_sha256,
        "launcher and executable digests resolve before submission",
    )
    check(
        contract["supports"][spec.supports]["sha256"] == cd._sha256_file(root / "supports.json"),
        "the pinned supports input is content-addressed in the resolved contract",
    )

    def rejects_contract(label: str, candidate: cd.Spec) -> None:
        try:
            cd.validate_launch_contract(candidate, root)
            check(False, f"must refuse before launch: {label}")
        except cd.SpecError:
            check(True, f"refuses before launch: {label}")

    for required in ("--freeze-supports", "--freeze-market-supports", "--exact-batch"):
        flags = tuple(token for token in spec.common_flags if token != required)
        rejects_contract(f"missing {required}", cd.dataclasses.replace(spec, common_flags=flags))
    for option in ("--split-bounds", "--batch-size"):
        flags = list(spec.common_flags)
        index = flags.index(option)
        del flags[index:index + 2]
        rejects_contract(f"missing {option}", cd.dataclasses.replace(spec, common_flags=tuple(flags)))
    rejects_contract("missing explicit supports path", cd.dataclasses.replace(spec, supports=""))
    rejects_contract(
        "launcher digest mismatch",
        cd.dataclasses.replace(spec, launcher_sha256="0" * 64),
    )
    rejects_contract(
        "executable digest mismatch",
        cd.dataclasses.replace(spec, executable_sha256="0" * 64),
    )
    rejects_contract(
        "default supports digest mismatch",
        cd.dataclasses.replace(spec, supports_sha256="0" * 64),
    )
    bad_seed_arm = cd.dataclasses.replace(spec.arms["GP"], flags=("--seed", "7"))
    rejects_contract(
        "arm-local seed override",
        cd.dataclasses.replace(spec, arms={**spec.arms, "GP": bad_seed_arm}),
    )
    bad_supports_arm = cd.dataclasses.replace(spec.arms["GP"], flags=("--supports", "other.json"))
    rejects_contract(
        "arm flags bypassing structured supports identity",
        cd.dataclasses.replace(spec, arms={**spec.arms, "GP": bad_supports_arm}),
    )
    missing_geometry_supports = cd.dataclasses.replace(spec.arms["GC"], supports=None)
    rejects_contract(
        "geometry-changing arm without class-specific supports",
        cd.dataclasses.replace(spec, arms={**spec.arms, "GC": missing_geometry_supports}),
    )
    same_geometry_supports = cd.dataclasses.replace(spec.arms["GC"], supports=spec.supports)
    rejects_contract(
        "geometry-changing arm reusing the control supports",
        cd.dataclasses.replace(spec, arms={**spec.arms, "GC": same_geometry_supports}),
    )
    duplicate_seed_tier = cd.dataclasses.replace(spec.tiers["screen"], seeds=(24301, 24301))
    rejects_contract(
        "duplicate controlled seeds",
        cd.dataclasses.replace(spec, tiers={**spec.tiers, "screen": duplicate_seed_tier}),
    )
    try:
        cd.validate_decision_inputs(spec, ["GP"], ["screen"])
        check(False, "missing s_seed must refuse before any mlq submission")
    except cd.RefusalError:
        check(True, "missing s_seed refuses before any mlq submission")

    geometry_supports = root / "supports-volstd-v8.json"
    geometry_supports.write_text('{"format_version":8}\n')
    geometry_arm = cd.dataclasses.replace(spec.arms["GC"], supports=str(geometry_supports))
    class_specific = cd.dataclasses.replace(
        spec, arms={**spec.arms, "GC": geometry_arm}
    )
    class_contract = cd.validate_launch_contract(class_specific, root)
    geometry_launch = cd.build_launch(
        class_specific, geometry_arm, class_specific.tiers["screen"], 24301
    )
    check(
        geometry_launch.argv.count("--supports") == 1
        and str(geometry_supports) in geometry_launch.argv
        and len(class_contract["supports"]) == 2,
        "geometry arms replace the raw default with one explicit class-specific supports identity",
    )
    market_supports = root / "bar_market_supports.300.json"
    market_supports.write_text('{"fixture":"market"}\n')
    market_spec = cd.dataclasses.replace(
        spec,
        market_supports=str(market_supports),
        market_supports_sha256=cd._sha256_file(market_supports),
    )
    market_contract = cd.validate_launch_contract(market_spec, root)
    check(
        market_contract["market_supports"]["sha256"]
        == market_spec.market_supports_sha256,
        "a configured market supports path is also authenticated",
    )
    market_launch = cd.build_launch(
        market_spec, market_spec.arms["control"], market_spec.tiers["screen"], 24301
    )
    market_execution = cd._execution_verified_argv(
        cd._resolved_launch_argv(market_launch.argv, market_contract),
        market_contract,
        market_spec.supports,
    )
    check(
        market_spec.market_supports_sha256 in market_execution
        and str(market_supports.resolve()) in market_execution,
        "the worker-side command re-authenticates configured market supports before exec",
    )

    calls: list[list[str]] = []

    def fake_runner(argv, **_kwargs):
        command = list(argv)
        calls.append(command)
        if command[1] == "submit":
            return cd.subprocess.CompletedProcess(command, 0, '{"job":{"id":73}}\n', "")
        if command[1] == "wait":
            return cd.subprocess.CompletedProcess(command, 0, '{"state":"succeeded"}\n', "")
        if "--stderr" in command:
            return cd.subprocess.CompletedProcess(command, 0, "trainer stderr\n", "")
        return cd.subprocess.CompletedProcess(command, 0, "trainer stdout\n", "")

    launch = cd.build_launch(spec, spec.arms["control"], spec.tiers["screen"], 24301)
    resolved = cd._resolved_launch_argv(launch.argv, contract)
    execution_argv = cd._execution_verified_argv(resolved, contract, spec.supports)
    queued = cd.run_through_mlq(
        execution_argv,
        launch.run_name,
        root,
        spec.mlq_max_parallel_runs,
        spec.mlq_time_limit,
        "selftest-idempotency",
        command_runner=fake_runner,
    )
    submit = calls[0]
    separator = submit.index("--")
    check(
        queued.job_id == 73
        and submit[separator + 1:] == execution_argv
        and execution_argv[:2] == ["/bin/bash", "-lc"]
        and spec.launcher_sha256 in execution_argv
        and spec.executable_sha256 in execution_argv
        and spec.supports_sha256 in execution_argv
        and 'sha256sum -- "$path"' in execution_argv[2]
        and 'exec "$@"' in execution_argv[2],
        "mlq receives exact execution-time SHA checks plus the resolved training argv",
    )
    check(
        "--max-parallel-runs" in submit and "--time-limit" in submit,
        "mlq submission carries both required resource declarations",
    )
    check(
        queued.stdout == "trainer stdout\n" and queued.stderr == "trainer stderr\n",
        "both mlq logs are retrieved for tripwire evidence",
    )

    # Change a support after pre-submit resolution. The worker-side guard must refuse with
    # the reserved pin-failure exit code and never exec the training binary.
    marker = root / "trainer-executed"
    (root / "supports.json").write_text('{"fixture":"tampered"}\n')
    guarded = cd.subprocess.run(
        execution_argv, cwd=root, capture_output=True, text=True, check=False
    )
    check(
        guarded.returncode == 125
        and "campaign SHA-256 mismatch for supports" in guarded.stderr
        and not marker.exists(),
        "a queue-time supports mismatch refuses before the training executable runs",
    )
    (root / "supports.json").write_text('{"fixture":true}\n')

    driver_spec = cd.dataclasses.replace(spec, ledger_path="mlq-ledger.jsonl")
    args = cd.parse_args(["--spec", str(spec.path), "--repo-root", str(root)])
    driver = cd.Driver(driver_spec, args, command_runner=fake_runner)
    driver.launch_seed(driver_spec.arms["control"], driver_spec.tiers["screen"], 24301)
    launch_record = next(
        record for record in driver.ledger.read() if record["record"] == "arm_launch"
    )
    check(
        launch_record["mlq_job_id"] == 73
        and launch_record["mlq_max_parallel_runs"] == 1
        and launch_record["mlq_time_limit"] == "1h",
        "ledger pins the mlq lease declaration and job id",
    )
    check(
        launch_record["launcher"]["sha256"] == spec.launcher_sha256
        and launch_record["executable"]["sha256"] == spec.executable_sha256,
        "ledger pins both launch-chain digests",
    )
    check(
        launch_record["resolved_argv"] == resolved
        and launch_record["execution_argv"] == execution_argv
        and launch_record["resolved_config"]["seed"] == 24301
        and launch_record["supports"]["sha256"]
        == contract["supports"][spec.supports]["sha256"],
        "ledger retains both resolved training argv and execution-time authenticated argv",
    )


def test_retention(root: Path) -> None:
    print("retention")
    spec = load_spec(root)
    tier = spec.tiers["screen"]
    arm = spec.arms["GP"]
    weights = root / "runs" / arm.run_name(24301, tier) / "weights"
    weights.mkdir(parents=True)
    names = [
        "pretrain_best_diag896.ot",
        "pretrain_best_diag896.windows.json",
        "pretrain_best_diag896.metadata.json",
        "pretrain_best_diag896.supports.300.json",
        "pretrain_best_diag896.optimizer.ot",
        "pretrain_last.ot",
        "pretrain_last.optimizer.ot",
        "pretrain_step_2999.ot",
        "pretrain_step_2999.optimizer.ot",
        "pretrain_step_2999.supports.300.json",
        "pretrain_epoch_0_ctx896.ot",
        "something_unexpected.bin",
    ]
    for name in names:
        (weights / name).write_bytes(b"x" * 1024)

    cwd = Path.cwd()
    try:
        import os

        os.chdir(root)
        plan = cd.retention_plan(spec, arm, tier, 24301)
        check("pretrain_best_diag896.ot" not in plan["delete"], "the retained checkpoint survives")
        check(
            "pretrain_best_diag896.windows.json" not in plan["delete"],
            "the per-window vector survives — it IS the comparison input",
        )
        check(
            "pretrain_best_diag896.supports.300.json" not in plan["delete"],
            "the retained checkpoint keeps the sidecar that makes it loadable",
        )
        check(
            "pretrain_best_diag896.optimizer.ot" in plan["delete"],
            "even the retained checkpoint's optimizer bundle goes",
        )
        check(
            {"pretrain_last.ot", "pretrain_step_2999.ot", "pretrain_epoch_0_ctx896.ot"}
            <= set(plan["delete"]),
            "non-retained checkpoints go",
        )
        check(
            "something_unexpected.bin" not in plan["delete"],
            "an unrecognized file is never deleted",
        )
        applied = cd.apply_retention(plan)
        check(len(applied["removed"]) == len(plan["delete"]), "retention removes what it planned")
        check(
            (weights / "pretrain_best_diag896.ot").exists()
            and not (weights / "pretrain_step_2999.optimizer.ot").exists(),
            "on disk: kept kept, doomed gone",
        )
    finally:
        os.chdir(cwd)


def test_ledger_and_resume(root: Path) -> None:
    print("ledger and resume")
    path = root / "ledger.jsonl"
    ledger = cd.Ledger(path, "campaign-1", dry_run=False)
    ledger.append("campaign_start", spec_sha256="deadbeef")
    ledger.append("arm_launch", run_name="GP_base_s24301_screen", arm="GP")
    ledger.append("arm_complete", run_name="GP_base_s24301_screen", arm="GP", exit_code=0)
    ledger.append("arm_launch", run_name="GC_base_s24301_screen", arm="GC")
    ledger.append(
        "decision",
        arm="GP",
        tier="screen",
        decision={"arm": "GP", "tier": "screen", "verdict": cd.CARRY},
    )
    ledger.append("retention", arm="GP", skipped=True)

    lines = path.read_text().strip().splitlines()
    check(len(lines) == 6, "one JSON object per line, append-only")
    check(all(json.loads(line)["campaign_id"] == "campaign-1" for line in lines), "campaign id")
    check(all("ts" in json.loads(line) for line in lines), "every record is timestamped")

    state = cd.ResumeState.replay(ledger.read())
    check(state.settled[("GP", "screen")] == cd.CARRY, "a decision marks (arm, tier) settled")
    check("GP_base_s24301_screen" in state.completed, "a zero-exit completion is reusable")
    check("GC_base_s24301_screen" not in state.completed, "a launch with no completion is not")
    check(state.carried() == {"GP"}, "carried arms are recoverable from the ledger")

    dry = cd.Ledger(root / "never.jsonl", "campaign-2", dry_run=True)
    dry.append("campaign_start")
    check(not (root / "never.jsonl").exists(), "a dry run writes no ledger")

    (root / "broken.jsonl").write_text("{not json}\n")
    try:
        cd.Ledger(root / "broken.jsonl", "c", dry_run=False).read()
        check(False, "a hand-edited ledger must fail loudly")
    except cd.DriverError:
        check(True, "a hand-edited ledger fails loudly")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        test_parser()
        test_artifact_reading(root)
        test_tripwires(root)
        test_s_seed_refusal(root)
        test_screen_gate(root)
        test_confirm_gate(root)
        test_mz_guard(root)
        test_naming_and_ordering(root)
        test_spec_validation(root)
        test_launch_refusals_and_mlq(root)
        test_retention(root)
        test_ledger_and_resume(root)
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
