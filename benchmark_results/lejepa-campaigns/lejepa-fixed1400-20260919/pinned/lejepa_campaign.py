#!/usr/bin/env python3
"""Plan, submit, and follow one exclusive fixed-step CausalPatch/LeJEPA campaign.

Planning snapshots explicit ELF executables and a supplied corpus contract; it never
loads bars or launches training. All six arms run sequentially in ONE mlq job.
Training metrics remain exclusively in the trainer's .report.bin files, read by the
pinned report_cli. JSON files contain configuration, identities, and lifecycle only.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODES = ("off", "latent-one", "latent-multi", "anchored", "anchored-no-sigreg", "anchored-reconstruct")
DEFAULT_NAMES = ("forecast", "latent-one", "latent-multi", "anchored", "no-sigreg", "reconstruct")
COMMON = {
    "seq-len": 6000, "common-context": 6000, "pred-len": 192, "patch-len": 16,
    "layers": 8, "d-model": 512, "heads": 8, "ffn": 2048, "dropout": 0,
    "min-history": 256, "features": "all", "x0-lambdas": "disabled",
    "future-calendar": "false",
    "horizon-loss": "uniform", "horizon-mean": "free", "amplitude-prior": 0,
    "target-basis": "cumulative", "basis-weight": "uniform",
    "scale-coupling": "decoupled", "horizon-decimation": "lattice",
    "optimizer": "polar-express", "scalar-lr-mult": 5, "mlp-down-lr": "aspect-only",
    "market-min-cross-section": 2000,
}


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def write_new(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json_bytes(value))


def read_json(path):
    return json.loads(Path(path).read_text())


def name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("expected a simple campaign or arm name")
    return value


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def snapshot(source, destination, executable=False):
    source = Path(source).resolve(strict=True)
    if executable:
        with source.open("rb") as stream:
            if stream.read(4) != b"\x7fELF" or not os.access(source, os.X_OK):
                raise ValueError(f"explicit executable must be an ELF binary, not a mutable launcher: {source}")
    before = sha256(source)
    with source.open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
    if before != sha256(destination) or before != sha256(source):
        raise ValueError(f"source changed while being pinned: {source}")
    destination.chmod(0o555 if executable else 0o444)
    return {"source": str(source), "path": str(destination), "sha256": before}


def arm_specs(arguments):
    specs = arguments or [f"{label}={mode}" for label, mode in zip(DEFAULT_NAMES, MODES)]
    result = []
    for spec in specs:
        label, separator, mode = spec.partition("=")
        if not separator or mode not in MODES:
            raise ValueError(f"arm must be NAME=MODE, got {spec}")
        result.append({"name": name(label), "jepa_mode": mode})
    if tuple(arm["jepa_mode"] for arm in result) != MODES:
        raise ValueError("supply all six arms in order: off, latent-one, latent-multi, anchored, anchored-no-sigreg, anchored-reconstruct")
    if len({arm["name"] for arm in result}) != len(result):
        raise ValueError("arm names must be unique")
    return result


def plan(args):
    if args.max_steps <= 8:
        raise ValueError("the fixed-step budget must exceed graph-capture warmup")
    common = COMMON.copy()
    if args.common_config:
        overrides = read_json(args.common_config)
        if not isinstance(overrides, dict) or set(overrides) - (set(COMMON) | {"learning-rate"}):
            raise ValueError("common config must map supported shared CLI knobs to scalar values; no arm-specific data, budgets, or seeds")
        common.update(overrides)
    if any(not isinstance(v, (str, int, float)) or isinstance(v, bool) for v in common.values()):
        raise ValueError("common configuration values must be scalar CLI strings or numbers")
    if any(isinstance(v, float) and not math.isfinite(v) for v in common.values()):
        raise ValueError("common configuration contains a nonfinite number")
    required = {"pred-len": 192, "scale-coupling": "decoupled", "horizon-decimation": "lattice", "target-basis": "cumulative", "basis-weight": "uniform", "future-calendar": "false"}
    if any(common[key] != value for key, value in required.items()):
        raise ValueError("this campaign requires the shared 192-bar decoupled+lattice baseline without calibration or observed future calendar")
    expected = read_json(args.data_contract)
    if not expected.get("tickers") or len({t["ticker"] for t in expected["tickers"]}) != len(expected["tickers"]):
        raise ValueError("expected data contract must identify the complete eligible corpus")
    for ticker in expected["tickers"]:
        if not re.fullmatch(r"[0-9a-f]{64}", ticker["fingerprint"]):
            raise ValueError("expected data contract lacks authenticated ticker fingerprints")
    for key, option in (("context", "seq-len"), ("common_context", "common-context"), ("pred_len", "pred-len")):
        if expected[key] != common[option]:
            raise ValueError(f"expected data contract {key} disagrees with the shared model configuration")
    if expected.get("in_period_sections", 0) or expected.get("cross_section_placement", ""):
        raise ValueError("campaign expects the complete unholed, unmodified corpus placement")
    arms = arm_specs(args.arm)
    root = (args.output_root.resolve() / args.campaign)
    if root.exists():
        raise ValueError(f"campaign already exists, never overwrite or relaunch it: {root}")
    data_dir = args.data_dir.resolve(strict=True)
    if not data_dir.is_dir():
        raise ValueError("data-dir must name the actual corpus directory")
    if args.campaign_timeout_seconds < len(arms) * (args.arm_timeout_seconds + 30) + 120:
        raise ValueError("campaign watchdog must allow every independent arm watchdog plus 30s termination grace and 120s collection")
    environment = {}
    for assignment in args.env:
        key, separator, value = assignment.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key == "LEJEPA_QUEUED_PLAN":
            raise ValueError("environment must use NAME=VALUE; LEJEPA_QUEUED_PLAN is reserved")
        environment[key] = value
    root.mkdir(parents=True)
    pinned = root / "pinned"
    pinned.mkdir()
    assets = {
        "executable": snapshot(args.executable, pinned / "trading_bot_0", True),
        "report_cli": snapshot(args.report_cli, pinned / "report_cli", True),
        "driver": snapshot(Path(__file__), pinned / "lejepa_campaign.py"),
        "data_contract": snapshot(args.data_contract, pinned / "expected-data-contract.json"),
    }
    for arm in arms:
        arm["run_name"] = f"{args.campaign}-{arm['name']}"
        arm["run_root"] = str(ROOT / "training/runs" / arm["run_name"])
        if Path(arm["run_root"]).exists():
            raise ValueError(f"run already exists: {arm['run_root']}")
        command = [assets["executable"]["path"], "train-timexer-segment", "--research-panel",
                   "--run", arm["run_name"], "--data-dir", str(data_dir),
                   "--jepa-mode", arm["jepa_mode"], "--max-steps", str(args.max_steps),
                   "--schedule-budget", str(args.max_steps), "--seed", str(args.seed),
                   "--batch-size", str(args.batch_size), "--eval-every", str(args.eval_every),
                   "--eval-origins", str(args.eval_origins), "--eval-batch-size", str(args.eval_batch_size),
                   "--probe-fit-origins", str(args.probe_fit_origins),
                   "--preview-patience", str(args.max_steps + 1), "--patience", str(args.max_steps + 1),
                   "--epochs", str(args.max_steps), "--fused", "true",
                   "--row-stride-multiple", "1", "--row-fraction", "1", "--patch-phase", "fixed"]
        for key, value in sorted(common.items()):
            command += [f"--{key}", str(value)]
        arm["command"] = command
    manifest = {
        "schema": "lejepa-exclusive-fixed-step-campaign-v1", "created_at": utc_now(),
        "campaign": args.campaign, "repository": str(ROOT), "root": str(root),
        "python": str(Path(sys.executable).resolve()), "python_version": sys.version,
        "assets": assets, "data_dir": str(data_dir), "environment": environment,
        "common_config": common, "max_steps": args.max_steps, "schedule_budget": args.max_steps,
        "seed": args.seed, "batch_size": args.batch_size, "eval_every": args.eval_every,
        "eval_origins": args.eval_origins, "eval_batch_size": args.eval_batch_size,
        "probe_fit_origins": args.probe_fit_origins,
        "arm_timeout_seconds": args.arm_timeout_seconds,
        "campaign_timeout_seconds": args.campaign_timeout_seconds,
        "queue": {"max_parallel_runs": 1, "max_attempts": 1, "priority": 0}, "arms": arms,
        "sampling": "trainer-authenticated full eligible training pool; deterministic bounded ticker/date panel; train-only frozen probe fit; final common source after all training labels",
        "completion": "every arm independently exits zero, completes exactly N steps, writes post-probe authenticated endpoint manifest and readable binary forecast report; no early stopping or test split",
        "data_identity_at_plan": "supplied authenticated contract; actual corpus and sample identities must match at execution; planning does not scan bars",
    }
    path = root / "plan.json"
    write_new(path, manifest)
    path.chmod(0o444)
    digest = sha256(path)
    write_new(root / "plan-identity.json", {"plan_sha256": digest})
    print(f"Planned only, no training submitted: {path}\nPlan SHA-256: {digest}")
    print(f"Six arms, one seed, N=schedule={args.max_steps}, batch={args.batch_size}; cadence={args.eval_every}; exclusive mlq; independent watchdog={args.arm_timeout_seconds}s")
    return 0


def load_plan(path, expected_digest=None):
    path = path.resolve(strict=True)
    digest = sha256(path)
    recorded = read_json(path.parent / "plan-identity.json")["plan_sha256"]
    if digest != recorded or (expected_digest is not None and digest != expected_digest):
        raise ValueError("immutable campaign plan authentication failed")
    manifest = read_json(path)
    if manifest["schema"] != "lejepa-exclusive-fixed-step-campaign-v1":
        raise ValueError("unsupported campaign plan schema")
    for asset in manifest["assets"].values():
        if sha256(asset["path"]) != asset["sha256"]:
            raise ValueError(f"pinned asset changed: {asset['path']}")
    return path, manifest, digest


def submit(args):
    path, manifest, digest = load_plan(args.plan)
    receipt = path.parent / "job.json"
    if receipt.exists():
        raise ValueError(f"campaign was already submitted; use follow: {receipt}")
    command = ["mlq", "submit", "--json", "--name", manifest["campaign"],
               "--idempotency-key", f"lejepa-{digest}", "--max-parallel-runs", "1",
               "--max-attempts", "1", "--time-limit", f"{manifest['campaign_timeout_seconds']}s",
               "--cwd", manifest["repository"], "--env", f"LEJEPA_QUEUED_PLAN={digest}"]
    for key, value in sorted(manifest["environment"].items()):
        command += ["--env", f"{key}={value}"]
    command += ["--", manifest["python"], manifest["assets"]["driver"]["path"],
                "_run", "--plan", str(path), "--plan-sha256", digest]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    queued = json.loads(result.stdout)
    write_new(receipt, {"id": queued["id"], "submitted_at": utc_now(), "plan_sha256": digest})
    print(f"Queued {queued['id']}: exclusive, normal priority, max-attempts=1, watchdog={manifest['campaign_timeout_seconds']}s")
    return 0


def follow(args):
    path, _, _ = load_plan(args.plan)
    job = read_json(path.parent / "job.json")
    return subprocess.run(["mlq", "follow", str(job["id"]), "--timeout", args.timeout]).returncode


def endpoint_evidence(plan, arm):
    root = Path(arm["run_root"])
    checkpoint_path = root / "weights/jepa-manifest.json"
    checkpoint = read_json(checkpoint_path)
    expected = {"schema": "causal-patch-temporal-jepa-fixed-endpoint-v1",
                "completed_steps": plan["max_steps"], "schedule_budget": plan["max_steps"],
                "seed": plan["seed"], "batch_size": plan["batch_size"],
                "validation_rows": plan["eval_origins"], "probe_fit_rows": plan["probe_fit_origins"],
                "requested_tickers": []}
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{arm['name']}: no completed matched fixed-step endpoint")
    if checkpoint.get("executable_sha256") != plan["assets"]["executable"]["sha256"]:
        raise ValueError(f"{arm['name']}: trainer executable identity differs")
    if checkpoint["model"].get("jepa_mode", "off") != arm["jepa_mode"]:
        raise ValueError(f"{arm['name']}: objective arm differs")
    if checkpoint["model"].get("future_calendar") is not False:
        raise ValueError(f"{arm['name']}: forecast readout conditions on observed future calendar")
    data = read_json(plan["assets"]["data_contract"]["path"])
    if checkpoint["data"] != data:
        raise ValueError(f"{arm['name']}: actual complete corpus contract differs from pinned expected data; never rewrite a manifest to force compatibility")
    sample_path = root / "research-sample-plan.json"
    if sha256(sample_path) != checkpoint["sample_plan_sha256"]:
        raise ValueError(f"{arm['name']}: sample plan authentication failed")
    samples = read_json(sample_path)
    if (samples["seed"] != plan["seed"]
            or samples["validation"]["requested_rows"] != plan["eval_origins"]
            or samples["probe_fit"]["requested_rows"] != plan["probe_fit_origins"]):
        raise ValueError(f"{arm['name']}: sample budget or seed differs")
    if any(row["common_source_ms"] <= samples["training_last_target_ms"]
           for row in samples["validation"]["origins"]):
        raise ValueError(f"{arm['name']}: validation decisions precede training label reach")
    weights = root / "weights/jepa.safetensors"
    if sha256(weights) != checkpoint["weights_sha256"]:
        raise ValueError(f"{arm['name']}: final weight authentication failed")
    report = subprocess.run([plan["assets"]["report_cli"]["path"], "0", "timexer_segment_jepa_forecast",
                             "--run-root", str(root)],
                            check=True, text=True, capture_output=True, cwd=plan["repository"])
    finite = False
    for line in report.stdout.splitlines():
        fields = line.split("\t")
        if not fields or fields[0] != str(plan["max_steps"]):
            continue
        for field in fields[1:]:
            _, separator, value = field.partition("=")
            if separator:
                finite |= math.isfinite(float(value))
    if not finite:
        raise ValueError(f"{arm['name']}: binary forecast report has no finite endpoint measurement")
    print(f"{arm['name']} endpoint binary report:\n{report.stdout}", flush=True)
    reports = sorted(root.glob("gens/0/*.report.bin"))
    if not reports:
        raise ValueError(f"{arm['name']}: missing generation-zero binary reports")
    model = checkpoint["model"].copy()
    model.pop("jepa_mode", None)
    identity = {"model_without_mode": model, "data": checkpoint["data"],
                "sample_plan_sha256": checkpoint["sample_plan_sha256"],
                "cross_section_sha256": checkpoint["cross_section_sha256"],
                "protocol": {key: checkpoint[key] for key in ("seed", "batch_size", "schedule_budget", "base_learning_rate", "optimizer", "scalar_lr_mult", "mlp_down_lr")}}
    evidence = {"checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
                "weights_sha256": checkpoint["weights_sha256"],
                "sample_plan_sha256": checkpoint["sample_plan_sha256"],
                "training_origins_sha256": samples["training_origins_sha256"],
                "validation_origins_sha256": samples["validation"]["origins_sha256"],
                "probe_fit_origins_sha256": samples["probe_fit"]["origins_sha256"],
                "cross_section_sha256": checkpoint["cross_section_sha256"],
                "reports": [{"path": str(path), "sha256": sha256(path)} for path in reports]}
    return identity, evidence


def run_campaign(args):
    path, manifest, digest = load_plan(args.plan, args.plan_sha256)
    if os.environ.get("LEJEPA_QUEUED_PLAN") != digest:
        raise ValueError("internal driver must be submitted through this plan's exclusive mlq job")
    state_dir = path.parent / "arms"
    state_dir.mkdir()
    reference = None
    for arm in manifest["arms"]:
        load_plan(path, digest)
        if Path(arm["run_root"]).exists():
            raise ValueError(f"refusing to overwrite existing run: {arm['run_root']}")
        started = utc_now()
        clock = time.monotonic()
        status = "failed"
        failure = None
        evidence = None
        write_new(state_dir / f"{arm['name']}-started.json", {"started_at": started, "command": arm["command"], "plan_sha256": digest})
        child = subprocess.Popen(arm["command"], cwd=manifest["repository"])
        try:
            try:
                code = child.wait(timeout=manifest["arm_timeout_seconds"])
            except subprocess.TimeoutExpired:
                status = "timed-out"
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise ValueError(f"{arm['name']}: independent wall-clock watchdog expired")
            if code != 0:
                raise ValueError(f"{arm['name']}: trainer exited {code}")
            identity, evidence = endpoint_evidence(manifest, arm)
            if reference is not None and identity != reference:
                raise ValueError(f"{arm['name']}: backbone/data/sample/schedule identity differs from forecasting baseline")
            reference = identity
            status = "complete"
        except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
            failure = str(error)
        finally:
            elapsed = time.monotonic() - clock
            write_new(state_dir / f"{arm['name']}-finished.json", {
                "status": status, "started_at": started, "finished_at": utc_now(),
                "wall_seconds": elapsed, "failure": failure, "evidence": evidence,
            })
            print(f"{arm['name']}: {status}, independently timed wall={elapsed:.3f}s", flush=True)
        if status != "complete":
            raise ValueError(failure)
    write_new(path.parent / "complete.json", {"plan_sha256": digest, "finished_at": utc_now(), "arms": [arm["name"] for arm in manifest["arms"]]})
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    planning = commands.add_parser("plan", help="snapshot immutable inputs without submitting or loading data")
    planning.add_argument("--executable", type=Path, required=True)
    planning.add_argument("--report-cli", type=Path, required=True)
    planning.add_argument("--data-dir", type=Path, required=True)
    planning.add_argument("--data-contract", type=Path, required=True)
    planning.add_argument("--output-root", type=Path, required=True)
    planning.add_argument("--campaign", type=name, required=True)
    planning.add_argument("--max-steps", type=positive, required=True)
    planning.add_argument("--seed", type=int, required=True)
    planning.add_argument("--batch-size", type=positive, required=True)
    planning.add_argument("--eval-every", type=positive, required=True)
    planning.add_argument("--eval-origins", type=positive, default=2048)
    planning.add_argument("--probe-fit-origins", type=positive, default=2048)
    planning.add_argument("--eval-batch-size", type=positive, default=64)
    planning.add_argument("--arm-timeout-seconds", type=positive, default=1800)
    planning.add_argument("--campaign-timeout-seconds", type=positive, default=12000)
    planning.add_argument("--common-config", type=Path, help="JSON mapping shared CLI knob names to values")
    planning.add_argument("--arm", action="append", help="NAME=MODE; repeat for all six modes in protocol order")
    planning.add_argument("--env", action="append", default=[], help="explicit nonsecret runtime NAME=VALUE passed through mlq")
    submitting = commands.add_parser("submit", help="queue the pinned driver once, normal priority, exclusive")
    submitting.add_argument("--plan", type=Path, required=True)
    following = commands.add_parser("follow", help="follow the existing job; never resubmit on observer timeout")
    following.add_argument("--plan", type=Path, required=True)
    following.add_argument("--timeout", default="55m")
    running = commands.add_parser("_run", help=argparse.SUPPRESS)
    running.add_argument("--plan", type=Path, required=True)
    running.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.action == "plan" and not 0 <= args.seed < 2**64:
        parser.error("seed must fit an unsigned 64-bit integer")
    return {"plan": plan, "submit": submit, "follow": follow, "_run": run_campaign}[args.action](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"LeJEPA campaign stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
