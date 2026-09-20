#!/usr/bin/env python3
"""Plan, submit, and follow matched fixed-step CausalPatch/LeJEPA comparisons.

Each selected model has one exclusive, normal-priority mlq job. A lightweight
collector depends on every selected model succeeding. Planning never loads bars
or launches training. Metrics live only in trainer .report.bin files, read with
the pinned report_cli; JSON receipts contain configuration, identity and lifecycle.
"""

import argparse
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "lejepa-per-model-fixed-step-campaign-v2"
LEGACY_SCHEMA = "lejepa-exclusive-fixed-step-campaign-v1"
BASELINE = "decoupled-lattice-forecast"
MODES = ("off", "latent-one", "latent-multi", "anchored", "anchored-no-sigreg", "anchored-reconstruct")
OBJECTIVE_NAMES = (BASELINE, "latent-one", "latent-multi", "anchored", "no-sigreg", "reconstruct")
PROJECTED_NAMES = ("projected", "projected-no-sigreg")
CONDITIONAL_NAMES = ("conditional", "conditional-full-none")
MOMENT_FIELDS = ("decision_mse_weight", "temporal_moment_weight")
MOMENT_WEIGHTS = {
    "decision-mse": (0.125, 0.0),
    "moment": (0.0, 0.125),
    "moment-strong": (0.0, 0.5),
    "moment-plus-mse": (0.125, 0.125),
}
MODEL_TREATMENTS = ("jepa_mode", "scale_coupling", "horizon_decimation", *MOMENT_FIELDS)
SUITES = {
    "objective-comparison": OBJECTIVE_NAMES,
    "forecasting-controls": (BASELINE, "full-none-forecast"),
    "matched-comparison": (*OBJECTIVE_NAMES, "full-none-forecast"),
    "sigreg-placement": (BASELINE, "anchored", "no-sigreg", *PROJECTED_NAMES),
    "temporal-conditional": (BASELINE, "full-none-forecast", *CONDITIONAL_NAMES),
    "sigreg-dimensionality": (BASELINE, "projected", "projected-small"),
    "temporal-moments": (BASELINE, "full-none-forecast", *MOMENT_WEIGHTS),
}
RECIPES = {
    "decoupled-lattice": {"scale-coupling": "decoupled", "horizon-decimation": "lattice"},
    "full-none": {"scale-coupling": "full", "horizon-decimation": "none"},
}
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
PROTOCOL_KEYS = (
    "max_steps", "schedule_budget", "seed", "batch_size", "eval_every",
    "eval_origins", "eval_batch_size", "probe_fit_origins", "common_config",
)
RUNTIME_ENV = (
    "PATH", "HOME", "LD_LIBRARY_PATH", "PYO3_PYTHON", "PYTHONPATH", "VIRTUAL_ENV",
    "PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF", "CUDA_VISIBLE_DEVICES",
    "CUDA_MODULE_LOADING", "TORCH_CUDA_ARCH_LIST", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
)
MLQ_BASELINE_ENV = ("PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "TMPDIR")


def utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def identity_digest(value):
    return hashlib.sha256(json_bytes(value)).hexdigest()


def write_new(path, value):
    """Publish a durable, complete receipt without ever replacing an existing file."""
    path = Path(path)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
            os.link(temporary, path)
        finally:
            temporary.unlink()
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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


def arm_specs(suite, selected):
    names = selected or SUITES[suite]
    if len(set(names)) != len(names) or set(names) - set(SUITES[suite]):
        raise ValueError("--select names must be unique members of the named suite")
    modes = dict(zip(OBJECTIVE_NAMES, MODES))
    modes.update(zip(PROJECTED_NAMES, ("anchored-projected", "anchored-projected-no-sigreg")))
    modes.update({name: "anchored-conditional" for name in CONDITIONAL_NAMES})
    modes["projected-small"] = "anchored-projected-small"
    return [
        {"name": label, "jepa_mode": modes.get(label, "off"),
         "recipe": "full-none" if label in ("full-none-forecast", "conditional-full-none", *MOMENT_WEIGHTS) else "decoupled-lattice",
         **dict(zip(MOMENT_FIELDS, MOMENT_WEIGHTS.get(label, (0.0, 0.0))))}
        for label in SUITES[suite] if label in names
    ]


def moment_weights(arm):
    """Legacy omissions are zero; named treatments have exact, authenticated weights."""
    values = tuple(arm.get(field, 0.0) for field in MOMENT_FIELDS)
    if (any(not isinstance(value, (int, float)) or isinstance(value, bool)
            or not math.isfinite(value) or value < 0 for value in values)
            or values != MOMENT_WEIGHTS.get(arm["name"], (0.0, 0.0))):
        raise ValueError(f"{arm['name']}: weights differ from the declared temporal moment treatment")
    return dict(zip(MOMENT_FIELDS, values))


def recipe_for(plan, arm):
    recipe = arm.get("recipe", "decoupled-lattice")
    if recipe not in RECIPES or (recipe != "decoupled-lattice" and arm["jepa_mode"] not in ("off", "anchored-conditional")):
        raise ValueError(f"{arm['name']}: unsupported objective/recipe treatment")
    if any(moment_weights(arm).values()) and (arm["jepa_mode"] != "off" or recipe != "full-none"):
        raise ValueError(f"{arm['name']}: temporal moments require forecast-only full-none")
    if plan["schema"] == LEGACY_SCHEMA and any(
            plan["common_config"].get(key) != value for key, value in RECIPES[recipe].items()):
        raise ValueError("legacy reference is not the declared decoupled+lattice baseline")
    return recipe


def treatment_key(plan, arm):
    return (arm["jepa_mode"], recipe_for(plan, arm),
            *(arm.get(field, 0.0) for field in MOMENT_FIELDS))


def load_plan(path, expected_digest=None):
    path = path.resolve(strict=True)
    digest = sha256(path)
    recorded = read_json(path.parent / "plan-identity.json")["plan_sha256"]
    if digest != recorded or (expected_digest is not None and digest != expected_digest):
        raise ValueError("immutable campaign plan authentication failed")
    manifest = read_json(path)
    if manifest["schema"] not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError("unsupported campaign plan schema")
    for asset in manifest["assets"].values():
        if sha256(asset["path"]) != asset["sha256"]:
            raise ValueError(f"pinned asset changed: {asset['path']}")
    return path, manifest, digest


def recovered_reference(path):
    """Authenticate a saved endpoint without changing its failed validator lifecycle."""
    path = Path(path).resolve(strict=True)
    certificate = read_json(path)
    source_path = Path(certificate["source_plan"])
    if not source_path.is_absolute():
        source_path = ROOT / source_path
    source_path, source, digest = load_plan(source_path)
    if digest != certificate["source_plan_sha256"]:
        raise ValueError("recovered reference source plan changed")
    checkpoint = Path(certificate["evidence"]["checkpoint"]).resolve()
    arms = [arm for arm in source["arms"]
            if (Path(arm["run_root"]) / "weights/jepa-manifest.json").resolve() == checkpoint]
    if len(arms) != 1:
        raise ValueError("recovered reference must identify one declared source arm")
    arm = arms[0]
    failed_path = source_path.parent / "arms" / f"{arm['name']}-finished.json"
    failed = read_json(failed_path)
    if (sha256(failed_path) != certificate["original_failed_receipt_sha256"]
            or failed.get("status") != "failed" or failed.get("plan_sha256") != digest
            or failed.get("task") != arm["name"]):
        raise ValueError("recovered reference original failure provenance changed")
    identity, evidence = endpoint_evidence(source, arm)
    if (evidence != certificate["evidence"]
            or identity_digest(identity) != certificate["shared_identity_sha256"]):
        raise ValueError("recovered reference endpoint differs from its revalidation certificate")
    record = {"path": str(path), "sha256": sha256(path), "source_plan": str(source_path),
              "source_plan_sha256": digest, "arm": arm["name"],
              "original_failed_receipt_sha256": sha256(failed_path)}
    return record, source, arm, identity, evidence


def authenticated_recoveries(records):
    recovered = []
    for recorded in records:
        actual = recovered_reference(recorded["path"])
        if actual[0] != recorded:
            raise ValueError("recovered reference certificate or source provenance changed")
        recovered.append(actual)
    return recovered


def completed_reference(path):
    path, manifest, digest = load_plan(path)
    complete_path = path.parent / "complete.json"
    complete = read_json(complete_path)
    if complete["plan_sha256"] != digest or complete["arms"] != [arm["name"] for arm in manifest["arms"]]:
        raise ValueError("reference campaign has no authenticated complete lifecycle")
    receipts = {}
    for arm in manifest["arms"]:
        recipe_for(manifest, arm)
        receipt = path.parent / "arms" / f"{arm['name']}-finished.json"
        if read_json(receipt)["status"] != "complete":
            raise ValueError(f"reference arm did not complete: {arm['name']}")
        receipts[arm["name"]] = sha256(receipt)
    return manifest, {"path": str(path), "plan_sha256": digest,
                      "complete_sha256": sha256(complete_path), "finished_receipts": receipts}


def check_matched_reference(manifest, reference):
    for key in PROTOCOL_KEYS:
        if manifest[key] != reference[key]:
            raise ValueError(f"reference differs in shared {key}; only declared objectives and recipe treatments may differ")
    if read_json(manifest["assets"]["data_contract"]["path"]) != read_json(reference["assets"]["data_contract"]["path"]):
        raise ValueError("reference complete corpus/data contract differs")


def reference_sources(recorded, ancestors=()):
    path = Path(recorded["path"]).resolve()
    if path in ancestors:
        raise ValueError("cyclic campaign reference ancestry")
    manifest, actual = completed_reference(path)
    if actual != recorded:
        raise ValueError("completed reference plan or lifecycle receipts changed")
    sources = []
    if manifest.get("reference"):
        sources = reference_sources(manifest["reference"], ancestors + (path,))
        for _, inherited, _ in sources:
            check_matched_reference(manifest, inherited)
    return sources + [(path, manifest, actual["plan_sha256"])]


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
    required = {"pred-len": 192, **RECIPES["decoupled-lattice"], "target-basis": "cumulative",
                "basis-weight": "uniform", "future-calendar": "false", "x0-lambdas": "disabled"}
    if any(common[key] != value for key, value in required.items()):
        raise ValueError("shared baseline must be causal, uncalibrated 192-bar decoupled+lattice; choose a named suite for recipe treatments")
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
    arms = arm_specs(args.suite, args.select)
    root = args.output_root.resolve() / args.campaign
    if root.exists():
        raise ValueError(f"campaign already exists, never overwrite or relaunch it: {root}")
    data_dir = args.data_dir.resolve(strict=True)
    if not data_dir.is_dir():
        raise ValueError("data-dir must name the actual corpus directory")
    reference, reference_record = completed_reference(args.reference_plan) if args.reference_plan else (None, None)
    environment = (reference["environment"].copy() if reference else
                   {key: os.environ[key] for key in RUNTIME_ENV if key in os.environ})
    for assignment in args.env:
        key, separator, value = assignment.partition("=")
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) or key.startswith("LEJEPA_QUEUED_"):
            raise ValueError("environment must use NAME=VALUE; LEJEPA_QUEUED_* is reserved")
        environment[key] = value
    for key in ("PATH", "LD_LIBRARY_PATH", "PYO3_PYTHON", "PYTHONPATH"):
        if not environment.get(key):
            raise ValueError(f"pin required runtime environment with --env {key}=VALUE")
    if reference and environment != reference["environment"]:
        raise ValueError("matched reference runtime environment differs; reuse its pinned environment")
    manifest = {
        "schema": SCHEMA, "created_at": utc_now(),
        "campaign": args.campaign, "repository": str(ROOT), "root": str(root),
        "python": str(Path(sys.executable).resolve()), "python_version": sys.version,
        "data_dir": str(data_dir), "environment": environment,
        "common_config": common, "max_steps": args.max_steps, "schedule_budget": args.max_steps,
        "seed": args.seed, "batch_size": args.batch_size, "eval_every": args.eval_every,
        "eval_origins": args.eval_origins, "eval_batch_size": args.eval_batch_size,
        "probe_fit_origins": args.probe_fit_origins,
        "arm_timeout_seconds": args.arm_timeout_seconds,
        "arm_queue_timeout_seconds": args.arm_timeout_seconds + 30,
        "collect_timeout_seconds": args.collect_timeout_seconds,
        "queue": {"max_parallel_runs": 1, "max_attempts": 1, "priority": 0},
        "suite": args.suite, "protocol_baseline": BASELINE, "arms": arms,
        "reference": reference_record,
        "allowed_model_differences": list(MODEL_TREATMENTS),
        "sampling": "trainer-authenticated full eligible training pool; deterministic bounded ticker/date panel; train-only frozen probe fit; final common source after all training labels",
        "completion": "every selected arm independently exits zero, completes exactly N steps, writes authenticated endpoint and readable binary forecast report; collection validates selected and reused identities; no early stopping, test split or latent-loss winner selection",
        "data_identity_at_plan": "supplied authenticated contract; actual corpus and sample identities must match at execution; planning does not scan bars",
    }
    inherited_sources = []
    existing = set()
    if reference:
        for key in PROTOCOL_KEYS:
            if manifest[key] != reference[key]:
                raise ValueError(f"reference differs in shared {key}; no silent panel, budget or schedule changes")
        if expected != read_json(reference["assets"]["data_contract"]["path"]):
            raise ValueError("reference complete corpus/data contract differs")
        inherited_sources = reference_sources(reference_record)
        existing = {treatment_key(source, arm)
                    for _, source, _ in inherited_sources for arm in source["arms"]}
    inherited_recoveries = [actual for _, source, _ in inherited_sources
                            for actual in authenticated_recoveries(source.get("recovered_references", []))]
    requested_recoveries = [recovered_reference(path) for path in args.recovered_reference]
    recoveries = []
    seen_recoveries = set()
    for recovered in inherited_recoveries + requested_recoveries:
        record, source, arm, _, _ = recovered
        if record["sha256"] in seen_recoveries:
            continue
        for key in PROTOCOL_KEYS:
            if manifest[key] != source[key]:
                raise ValueError(f"recovered reference differs in shared {key}")
        if expected != read_json(source["assets"]["data_contract"]["path"]):
            raise ValueError("recovered reference complete corpus/data contract differs")
        if source["environment"] != environment:
            raise ValueError("recovered reference runtime environment differs")
        key = treatment_key(source, arm)
        if key in existing:
            raise ValueError("recovered reference duplicates an inherited treatment")
        existing.add(key)
        seen_recoveries.add(record["sha256"])
        recoveries.append(record)
    manifest["recovered_references"] = recoveries
    selected = {treatment_key(manifest, arm) for arm in arms}
    if selected & existing:
        raise ValueError("selected model already exists in reference; use --select for only missing treatments")
    available = selected | existing
    if ("off", "decoupled-lattice", 0.0, 0.0) not in available:
        raise ValueError(f"include {BASELINE}, or --reference-plan containing its completed matched endpoint")
    if args.suite == "temporal-moments" and ("off", "full-none", 0.0, 0.0) not in available:
        raise ValueError("temporal-moments requires its matched full-none-forecast control, selected or inherited")
    for arm in arms:
        arm["run_name"] = f"{args.campaign}-{arm['name']}"
        arm["run_root"] = str(ROOT / "training/runs" / arm["run_name"])
        if Path(arm["run_root"]).exists():
            raise ValueError(f"run already exists: {arm['run_root']}")
    root.mkdir(parents=True)
    pinned = root / "pinned"
    pinned.mkdir()
    assets = {
        "executable": snapshot(args.executable, pinned / "trading_bot_0", True),
        "report_cli": snapshot(args.report_cli, pinned / "report_cli", True),
        "driver": snapshot(Path(__file__), pinned / "lejepa_campaign.py"),
        "data_contract": snapshot(args.data_contract, pinned / "expected-data-contract.json"),
    }
    manifest["assets"] = assets
    for arm in arms:
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
        for key, value in sorted((common | RECIPES[arm["recipe"]]).items()):
            command += [f"--{key}", str(value)]
        for field, value in moment_weights(arm).items():
            command += [f"--{field.replace('_', '-')}", str(value)]
        arm["command"] = command
    path = root / "plan.json"
    write_new(path, manifest)
    path.chmod(0o444)
    digest = sha256(path)
    write_new(root / "plan-identity.json", {"plan_sha256": digest})
    print(f"Planned only, no training submitted: {path}\nPlan SHA-256: {digest}")
    print(f"Suite={args.suite}; protocol baseline={BASELINE}; N=schedule={args.max_steps}, batch={args.batch_size}, seed={args.seed}, cadence={args.eval_every}")
    for arm in arms:
        print(f"Model job: {arm['name']}; objective={arm['jepa_mode']}; recipe={arm['recipe']}; weights={moment_weights(arm)}; watchdog={args.arm_timeout_seconds}s (+30s queue grace)")
    if reference:
        print(f"Reuse {sum(len(source['arms']) for _, source, _ in inherited_sources)} completed reference endpoints, no retraining: {reference_record['path']}")
    print(f"Collection job: after-success all {len(arms)} model jobs; watchdog={args.collect_timeout_seconds}s; all jobs exclusive, normal priority, max-attempts=1")
    return 0


def job_key(digest, label):
    return f"lejepa-{digest}-{label}"


def job_receipt(path, digest, label):
    receipt = read_json(path.parent / "jobs" / f"{label}.json")
    if receipt["plan_sha256"] != digest or receipt["idempotency_key"] != job_key(digest, label) or receipt["task"] != label:
        raise ValueError(f"{label}: queue receipt identity differs")
    request_path = path.parent / "jobs" / f"{label}-request.json"
    if sha256(request_path) != receipt["request_sha256"]:
        raise ValueError(f"{label}: durable queue request authentication failed")
    return receipt


def submit_job(path, manifest, digest, label, dependencies):
    jobs = path.parent / "jobs"
    receipt_path = jobs / f"{label}.json"
    if receipt_path.exists():
        receipt = job_receipt(path, digest, label)
        print(f"Existing {receipt['id']}: {label}; observing/recovering, never relaunching", flush=True)
        return receipt
    limit = manifest["collect_timeout_seconds"] if label == "collect" else manifest["arm_queue_timeout_seconds"]
    command = ["mlq", "submit", "--json", "--name", f"{manifest['campaign']}-{label}",
               "--idempotency-key", job_key(digest, label), "--max-parallel-runs", "1",
               "--max-attempts", "1", "--time-limit", f"{limit}s", "--cwd", manifest["repository"],
               "--env", f"LEJEPA_QUEUED_PLAN={digest}", "--env", f"LEJEPA_QUEUED_TASK={label}"]
    for key, value in sorted(manifest["environment"].items()):
        command += ["--env", f"{key}={value}"]
    for dependency in dependencies:
        command += ["--after-success", str(dependency)]
    command += ["--", manifest["python"], manifest["assets"]["driver"]["path"],
                "_collect" if label == "collect" else "_run-arm", "--plan", str(path), "--plan-sha256", digest]
    if label != "collect":
        command += ["--arm", label]
    request_path = jobs / f"{label}-request.json"
    request = {"plan_sha256": digest, "task": label, "idempotency_key": job_key(digest, label),
               "after_success": dependencies, "command": command}
    if request_path.exists():
        recorded = read_json(request_path)
        client_environment = recorded.get("client_baseline_environment")
        if client_environment is None:
            raise ValueError(f"{label}: interrupted submission lacks pinned client environment; recover its receipt from mlq before continuing")
    else:
        client_environment = {key: os.environ[key] for key in MLQ_BASELINE_ENV if key in os.environ}
    request["client_baseline_environment"] = client_environment
    if request_path.exists():
        if recorded != request:
            raise ValueError(f"{label}: durable queue request changed during recovery")
    else:
        write_new(request_path, request)
    environment = {key: value for key, value in os.environ.items() if key not in MLQ_BASELINE_ENV}
    environment.update(client_environment)
    result = subprocess.run(command, check=True, text=True, capture_output=True, env=environment)
    queued = json.loads(result.stdout)
    receipt = {"id": queued["id"], "task": label, "submitted_at": utc_now(),
               "plan_sha256": digest, "idempotency_key": job_key(digest, label),
               "request_sha256": sha256(request_path), "after_success": dependencies}
    write_new(receipt_path, receipt)
    print(f"Queued/recovered {queued['id']}: {label}; exclusive, normal priority, max-attempts=1, watchdog={limit}s", flush=True)
    return receipt


def submit(args):
    path, manifest, digest = load_plan(args.plan)
    if manifest["schema"] != SCHEMA:
        raise ValueError("legacy campaign is read-only; follow it or create a new per-model plan")
    if manifest["reference"]:
        reference, actual = completed_reference(Path(manifest["reference"]["path"]))
        if actual != manifest["reference"]:
            raise ValueError("completed reference plan or lifecycle receipts changed")
        check_matched_reference(manifest, reference)
    (path.parent / "jobs").mkdir(exist_ok=True)
    with (path.parent / ".submit.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        # max_parallel_runs=1 provides exclusive admission; independent arms need
        # no artificial dependency chain and retain normal FIFO queue fairness.
        jobs = [submit_job(path, manifest, digest, arm["name"], []) for arm in manifest["arms"]]
        submit_job(path, manifest, digest, "collect", [job["id"] for job in jobs])
    return 0


def follow(args):
    path, manifest, digest = load_plan(args.plan)
    if manifest["schema"] == LEGACY_SCHEMA:
        job = read_json(path.parent / "job.json")
        if job["plan_sha256"] != digest:
            raise ValueError("legacy queue receipt identity differs")
        return subprocess.run(["mlq", "follow", str(job["id"]), "--timeout", args.timeout]).returncode
    labels = [arm["name"] for arm in manifest["arms"]] + ["collect"]
    if any(not (path.parent / "jobs" / f"{label}.json").exists() for label in labels):
        raise ValueError("submission is incomplete; rerun submit to recover the same idempotency keys, not a new plan")
    jobs = [job_receipt(path, digest, label) for label in labels]
    failure = 0
    for job in jobs:
        if job["task"] == "collect" and failure:
            print("Collection cannot complete unless every model succeeds; no jobs were resubmitted.")
            return failure
        print(f"Following {job['task']}: {job['id']}", flush=True)
        result = subprocess.run(["mlq", "follow", str(job["id"]), "--timeout", args.timeout])
        if result.returncode == 124:
            return 124
        failure = failure or result.returncode
    if not failure:
        complete = read_json(path.parent / "complete.json")
        if complete["plan_sha256"] != digest:
            raise ValueError("collection receipt identity differs")
        print(f"Complete matched collection: {path.parent / 'complete.json'}")
    return failure


def endpoint_evidence(plan, arm):
    root = Path(arm["run_root"])
    checkpoint_path = root / "weights/jepa-manifest.json"
    checkpoint = read_json(checkpoint_path)
    expected = {"schema": "causal-patch-temporal-jepa-fixed-endpoint-v1",
                "completed_steps": plan["max_steps"], "schedule_budget": plan["schedule_budget"],
                "seed": plan["seed"], "batch_size": plan["batch_size"],
                "validation_rows": plan["eval_origins"], "probe_fit_rows": plan["probe_fit_origins"],
                "requested_tickers": []}
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{arm['name']}: no completed matched fixed-step endpoint")
    if checkpoint.get("executable_sha256") != plan["assets"]["executable"]["sha256"]:
        raise ValueError(f"{arm['name']}: trainer executable identity differs from its own pinned plan")
    model = checkpoint["model"]
    if model.get("jepa_mode", "off") != arm["jepa_mode"]:
        raise ValueError(f"{arm['name']}: objective arm differs")
    recipe = recipe_for(plan, arm)
    for field, value in moment_weights(arm).items():
        actual = model.get(field, 0.0)
        if (not isinstance(actual, (int, float)) or isinstance(actual, bool)
                or not math.isfinite(actual) or actual != value):
            raise ValueError(f"{arm['name']}: actual {field} differs from its declared treatment")
    for key, value in RECIPES[recipe].items():
        if model.get(key.replace("-", "_"), {"scale-coupling": "full", "horizon-decimation": "none"}[key]) != value:
            raise ValueError(f"{arm['name']}: actual {key} differs from declared {recipe} treatment")
    if model.get("future_calendar") is not False or model.get("x0_lambdas") != "disabled":
        raise ValueError(f"{arm['name']}: endpoint is not causal and uncalibrated")
    for key in ("seq-len", "pred-len", "patch-len", "layers", "d-model", "heads", "ffn",
                "dropout", "min-history", "horizon-loss", "horizon-mean"):
        if model.get(key.replace("-", "_")) != plan["common_config"][key]:
            raise ValueError(f"{arm['name']}: actual shared model setting {key} differs")
    for key in ("optimizer", "scalar-lr-mult", "mlp-down-lr"):
        if checkpoint.get(key.replace("-", "_")) != plan["common_config"][key]:
            raise ValueError(f"{arm['name']}: actual optimizer setting {key} differs")
    if "learning-rate" in plan["common_config"] and checkpoint.get("base_learning_rate") != plan["common_config"]["learning-rate"]:
        raise ValueError(f"{arm['name']}: actual base learning rate differs")
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
                             "--run-root", str(root)], check=True, text=True, capture_output=True,
                            cwd=plan["repository"], env=plan["environment"], timeout=30)
    endpoint_rows = []
    for line in report.stdout.splitlines():
        fields = line.split("\t")
        if not fields or fields[0] != str(plan["max_steps"]):
            continue
        finite_fields = []
        for field in fields[1:]:
            _, separator, value = field.partition("=")
            if separator and math.isfinite(float(value)):
                finite_fields.append(field)
        if finite_fields:
            endpoint_rows.append("\t".join([fields[0], *finite_fields]))
    if not endpoint_rows:
        raise ValueError(f"{arm['name']}: binary forecast report has no finite endpoint measurement")
    print(f"{arm['name']} endpoint binary report:", flush=True)
    for row in endpoint_rows:
        print(row, flush=True)
    reports = sorted(root.glob("gens/0/*.report.bin"))
    if not reports:
        raise ValueError(f"{arm['name']}: missing generation-zero binary reports")
    shared_model = {key: value for key, value in model.items()
                    if key not in MODEL_TREATMENTS}
    identity = {"model_without_declared_treatments": shared_model,
                "data_sha256": identity_digest(checkpoint["data"]),
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


def queued_plan(args, task):
    path, manifest, digest = load_plan(args.plan, args.plan_sha256)
    if manifest["schema"] != SCHEMA:
        raise ValueError("internal per-model runner requires a v2 plan")
    if os.environ.get("LEJEPA_QUEUED_PLAN") != digest or os.environ.get("LEJEPA_QUEUED_TASK") != task:
        raise ValueError("internal driver must be submitted through this task's exclusive mlq job")
    if any(os.environ.get(key) != value for key, value in manifest["environment"].items()):
        raise ValueError("queue runtime environment differs from the pinned plan")
    return path, manifest, digest


def interrupted(signum, frame):
    raise InterruptedError(f"queue runner received signal {signum}")


def stop_child(child):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()


def run_arm(args):
    path, manifest, digest = queued_plan(args, args.arm)
    matches = [arm for arm in manifest["arms"] if arm["name"] == args.arm]
    if len(matches) != 1:
        raise ValueError("selected arm is not in the immutable plan")
    arm = matches[0]
    state_dir = path.parent / "arms"
    state_dir.mkdir(exist_ok=True)
    if Path(arm["run_root"]).exists():
        raise ValueError(f"refusing to overwrite existing run: {arm['run_root']}")
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    started = utc_now()
    clock = time.monotonic()
    status, failure, evidence, identity, child = "failed", None, None, None, None
    write_new(state_dir / f"{arm['name']}-started.json", {
        "started_at": started, "command": arm["command"], "plan_sha256": digest,
        "task": arm["name"], "idempotency_key": job_key(digest, arm["name"]),
    })
    try:
        child = subprocess.Popen(arm["command"], cwd=manifest["repository"], env=manifest["environment"])
        try:
            code = child.wait(timeout=manifest["arm_timeout_seconds"])
        except subprocess.TimeoutExpired:
            status = "timed-out"
            raise ValueError(f"{arm['name']}: wall-clock watchdog expired; fixed-step endpoint is incomplete")
        if code != 0:
            raise ValueError(f"{arm['name']}: trainer exited {code}")
        identity, evidence = endpoint_evidence(manifest, arm)
        status = "complete"
    except (Exception, KeyboardInterrupt) as error:
        failure = str(error)
    finally:
        stop_child(child)
        elapsed = time.monotonic() - clock
        write_new(state_dir / f"{arm['name']}-finished.json", {
            "status": status, "started_at": started, "finished_at": utc_now(),
            "plan_sha256": digest, "task": arm["name"],
            "idempotency_key": job_key(digest, arm["name"]),
            "wall_seconds": elapsed, "failure": failure, "evidence": evidence,
            "shared_identity_sha256": identity_digest(identity) if identity else None,
        })
        print(f"{arm['name']}: {status}, independently timed wall={elapsed:.3f}s", flush=True)
    if status != "complete":
        raise ValueError(failure)
    return 0


def collect(args):
    path, manifest, digest = queued_plan(args, "collect")
    state_dir = path.parent / "arms"
    state_dir.mkdir(exist_ok=True)
    started = utc_now()
    write_new(state_dir / "collect-started.json", {"plan_sha256": digest, "started_at": started})
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGINT, interrupted)
    failure = None
    try:
        sources = reference_sources(manifest["reference"]) if manifest["reference"] else []
        for _, reference, _ in sources:
            check_matched_reference(manifest, reference)
        sources.append((path, manifest, digest))
        shared_identity = None
        endpoints = []
        baseline = None
        for source_path, source, source_digest in sources:
            for arm in source["arms"]:
                finished_path = source_path.parent / "arms" / f"{arm['name']}-finished.json"
                finished = read_json(finished_path)
                if finished["status"] != "complete":
                    raise ValueError(f"{arm['name']}: no successful independent lifecycle receipt")
                if source["schema"] == SCHEMA and finished["plan_sha256"] != source_digest:
                    raise ValueError(f"{arm['name']}: lifecycle plan identity differs")
                identity, evidence = endpoint_evidence(source, arm)
                if evidence != finished["evidence"]:
                    raise ValueError(f"{arm['name']}: endpoint changed after its completed lifecycle receipt")
                if source["schema"] == SCHEMA and identity_digest(identity) != finished["shared_identity_sha256"]:
                    raise ValueError(f"{arm['name']}: recorded shared identity changed")
                if shared_identity is not None and identity != shared_identity:
                    raise ValueError(f"{arm['name']}: undeclared backbone/data/sample/schedule difference; only named objective and recipe treatments may differ")
                shared_identity = identity
                recipe = recipe_for(source, arm)
                endpoint = {"name": arm["name"], "jepa_mode": arm["jepa_mode"], "recipe": recipe,
                            **moment_weights(arm),
                            "run_root": arm["run_root"], "source_plan": str(source_path),
                            "source_plan_sha256": source_digest,
                            "trainer_sha256": source["assets"]["executable"]["sha256"],
                            "reused": source_digest != digest, "finished_receipt_sha256": sha256(finished_path),
                            "evidence": evidence}
                endpoints.append(endpoint)
                if treatment_key(source, arm) == ("off", "decoupled-lattice", 0.0, 0.0):
                    baseline = {"name": BASELINE, "source_arm": arm["name"], "run_root": arm["run_root"]}
        for record, source, arm, identity, evidence in authenticated_recoveries(
                manifest.get("recovered_references", [])):
            check_matched_reference(manifest, source)
            if shared_identity is not None and identity != shared_identity:
                raise ValueError(f"{arm['name']}: recovered endpoint has undeclared shared differences")
            shared_identity = identity
            endpoints.append({
                "name": arm["name"], "jepa_mode": arm["jepa_mode"], "recipe": recipe_for(source, arm),
                **moment_weights(arm), "run_root": arm["run_root"],
                "source_plan": record["source_plan"], "source_plan_sha256": record["source_plan_sha256"],
                "trainer_sha256": source["assets"]["executable"]["sha256"], "reused": True,
                "recovered_after_validator_failure": True,
                "original_lifecycle_status": "failed", "revalidation_certificate": record["path"],
                "revalidation_certificate_sha256": record["sha256"],
                "finished_receipt_sha256": record["original_failed_receipt_sha256"], "evidence": evidence,
            })
            if treatment_key(source, arm) == ("off", "decoupled-lattice", 0.0, 0.0):
                baseline = {
                    "name": BASELINE, "source_arm": arm["name"], "run_root": arm["run_root"],
                    "recovered_after_validator_failure": True, "original_lifecycle_status": "failed",
                    "revalidation_certificate": record["path"],
                    "revalidation_certificate_sha256": record["sha256"],
                    "original_failed_receipt_sha256": record["original_failed_receipt_sha256"],
                }
        if baseline is None:
            raise ValueError("collection lacks the declared decoupled+lattice forecasting baseline")
        write_new(path.parent / "complete.json", {
            "schema": "lejepa-matched-comparison-complete-v2", "plan_sha256": digest,
            "finished_at": utc_now(), "arms": [arm["name"] for arm in manifest["arms"]],
            "protocol_baseline": baseline, "shared_identity_sha256": identity_digest(shared_identity),
            "allowed_model_differences": manifest["allowed_model_differences"], "endpoints": endpoints,
            "interpretation": "matched causal uncalibrated forecast endpoints; objective and recipe contrasts are declared separately; latent losses are diagnostic only; no automatic winner",
        })
    except (Exception, KeyboardInterrupt) as error:
        failure = str(error)
    finally:
        write_new(state_dir / "collect-finished.json", {
            "plan_sha256": digest, "started_at": started, "finished_at": utc_now(),
            "status": "failed" if failure else "complete", "failure": failure,
        })
    if failure:
        raise ValueError(failure)
    print(f"Complete matched collection: {path.parent / 'complete.json'}", flush=True)
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
    planning.add_argument("--arm-timeout-seconds", type=positive, default=420, help="error watchdog only; never replaces the fixed update budget")
    planning.add_argument("--collect-timeout-seconds", type=positive, default=120)
    planning.add_argument("--common-config", type=Path, help="JSON shared CLI configuration; recipe contrasts come only from the named suite")
    planning.add_argument("--suite", choices=SUITES, default="objective-comparison")
    planning.add_argument("--select", action="append", choices=sorted({name for suite in SUITES.values() for name in suite}), help="repeat to submit only selected suite members")
    planning.add_argument("--reference-plan", type=Path, help="reuse completed matched arms and their pinned runtime environment without modifying or retraining them")
    planning.add_argument("--recovered-reference", type=Path, action="append", default=[],
                          help="authenticate an existing endpoint revalidation certificate while preserving its original failed lifecycle")
    planning.add_argument("--env", action="append", default=[], help="pin nonsecret runtime NAME=VALUE; defaults to captured runtime allowlist or reference environment")
    submitting = commands.add_parser("submit", help="idempotently queue/recover one exclusive job per model and an after-success collector")
    submitting.add_argument("--plan", type=Path, required=True)
    following = commands.add_parser("follow", help="observe existing model/collection jobs; never submit or retry")
    following.add_argument("--plan", type=Path, required=True)
    following.add_argument("--timeout", default="55m", help="observation timeout per job; a timeout never cancels or resubmits")
    running = commands.add_parser("_run-arm", help=argparse.SUPPRESS)
    running.add_argument("--plan", type=Path, required=True)
    running.add_argument("--plan-sha256", required=True)
    running.add_argument("--arm", required=True)
    collecting = commands.add_parser("_collect", help=argparse.SUPPRESS)
    collecting.add_argument("--plan", type=Path, required=True)
    collecting.add_argument("--plan-sha256", required=True)
    args = parser.parse_args()
    if args.action == "plan" and not 0 <= args.seed < 2**64:
        parser.error("seed must fit an unsigned 64-bit integer")
    return {"plan": plan, "submit": submit, "follow": follow, "_run-arm": run_arm, "_collect": collect}[args.action](args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f"LeJEPA campaign stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
