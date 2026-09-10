#!/usr/bin/env python3
"""Queue complete-corpus TimeXer comparisons and a validation-selected final run."""

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SEED = 20260905
COMMON_CONTEXT = 6000
BATCH_SIZE = 256
PRED_LEN = 192
FEATURE_NAMES = ("time_of_day", "day_of_week", "session_gap", "volume", "market", "spy")
VARIANTS = (("c2048", 2048, False), ("c6000", 6000, False), ("c6000-features", 6000, True))


def feature_set(enabled):
    return {name: enabled for name in FEATURE_NAMES}


def run_name(value):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise argparse.ArgumentTypeError("expected a simple run name")
    return value


def training_command(name, context, features, epochs, data_dir):
    command = [str(ROOT / "trading_bots/run-release-cuda.sh"), "train-timexer-segment",
               "--run", name, "--seq-len", str(context), "--common-context", str(COMMON_CONTEXT),
               "--pred-len", str(PRED_LEN), "--patch-len", "16", "--batch-size", str(BATCH_SIZE),
               "--seed", str(SEED), "--epochs", str(epochs), "--patience", "3", "--fused", "true", "--optimizer", "polar-express",
               "--features", "all" if features else "none"]
    if data_dir:
        command += ["--data-dir", str(data_dir.resolve())]
    return command


def submit_job(name, predecessor, limit, command):
    queued = subprocess.run(["mlq", "submit", "--json", "--name", name,
                             "--idempotency-key", f"{name}-complete-corpus-v1",
                             "--max-parallel-runs", "1", "--max-attempts", "1",
                             "--time-limit", limit, "--after-success", str(predecessor),
                             "--cwd", str(ROOT), "--", *command],
                            cwd=ROOT, check=True, text=True, capture_output=True)
    result = json.loads(queued.stdout)
    identifier = result["id"]
    print(f"Queued {identifier}: {name}; after-success {predecessor}; exclusive; limit {limit}", flush=True)
    return identifier


def submit(args):
    reference = json.loads(subprocess.run(["mlq", "show", str(args.reference_job), "--json"],
                                         check=True, text=True, capture_output=True).stdout)
    command = reference["args"]
    if "--run" not in command or command[command.index("--run") + 1] != args.reference_run:
        raise ValueError("reference job does not produce the configured reference run")
    predecessor = args.reference_job
    for suffix, context, features in VARIANTS:
        name = f"{args.campaign}-{suffix}"
        predecessor = submit_job(name, predecessor, "3h", training_command(name, context, features, 1, args.data_dir))
    launch = [sys.executable, str(Path(__file__).resolve()), "--launch-final",
              "--campaign", args.campaign, "--reference-run", args.reference_run]
    if args.data_dir:
        launch += ["--data-dir", str(args.data_dir.resolve())]
    submit_job(f"{args.campaign}-final", predecessor, "12h", launch)
    print("Each comparison consumes one complete corpus epoch. Final training uses ten epochs with patience three.")


def contract_identity(contract):
    identity = copy.deepcopy(contract)
    for key in ("context", "features", "auxiliary_schema", "spy_fingerprint"):
        identity.pop(key)
    identity["excluded_tickers"].sort(key=lambda ticker: (ticker["ticker"], ticker["reason"]))
    tickers = identity["tickers"]
    if not tickers or len({ticker["ticker"] for ticker in tickers}) != len(tickers):
        raise ValueError("corpus must contain unique ticker identities")
    for ticker in tickers:
        if not re.fullmatch(r"[0-9a-f]{64}", ticker["fingerprint"]):
            raise ValueError("invalid authenticated source fingerprint")
        ticker.pop("context")
    return identity


def read_evidence(name, context, features):
    directory = ROOT / "training/runs" / name
    contract = json.loads((directory / "timexer-segment-data-contract.json").read_text())
    checkpoint = directory / "weights/epoch-0001"
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    if manifest["data"] != contract:
        raise ValueError(f"{name}: checkpoint and corpus contracts differ")
    expected = {"epoch": 1, "epoch_complete": True, "planned_epochs": 1, "seed": SEED,
                "batch_size": BATCH_SIZE, "validation_is_full": True, "fused": True, "optimizer": "polar-express"}
    if any(manifest[key] != value for key, value in expected.items()):
        raise ValueError(f"{name}: not a completed matched one-epoch comparison")
    if manifest["completed_target_bars"] != contract["train_target_bars"]:
        raise ValueError(f"{name}: incomplete training target coverage")
    if (contract["context"], contract["common_context"], contract["pred_len"], contract["features"]) != (
            context, COMMON_CONTEXT, PRED_LEN, feature_set(features)):
        raise ValueError(f"{name}: unexpected context, horizons, or exogenous features")
    if contract["purge"] < PRED_LEN:
        raise ValueError(f"{name}: insufficient chronological purge")
    model = manifest["model"].copy()
    if model.pop("seq_len") != context or model.pop("features") != feature_set(features):
        raise ValueError(f"{name}: model and data feature contracts differ")
    if model["pred_len"] != PRED_LEN or model["patch_len"] != 16:
        raise ValueError(f"{name}: unexpected forecast or patch layout")
    with (checkpoint / "model.safetensors").open("rb") as weights:
        if hashlib.file_digest(weights, "sha256").hexdigest() != manifest["weights_sha256"]:
            raise ValueError(f"{name}: checkpoint weight authentication failed")
    result = subprocess.run([str(ROOT / "target/release/report_cli"), "1", "timexer_segment_validation",
                             "--run", name], cwd=ROOT, check=True, text=True, capture_output=True)
    values = {}
    for line in result.stdout.splitlines():
        fields = line.split("\t")
        if not fields or fields[0] != str(manifest["step"]):
            continue
        for field in fields[1:]:
            label, separator, raw = field.partition("=")
            if separator:
                values[label] = float(raw)
    mse = values.get("full validation", math.nan)
    persistence = values.get("full-validation persistence", math.nan)
    if not math.isfinite(mse) or mse < 0 or not math.isfinite(persistence) or persistence <= 0:
        raise ValueError(f"{name}: missing finite full-validation report at completed epoch step")
    # Losses come exclusively from the shared report reader, never checkpoint JSON.
    return {"identity": contract_identity(contract), "model": model, "mse": mse,
            "persistence": persistence, "validation_origins": manifest["eval_origins"],
            "completed_origins": manifest["completed_origins"],
            # `scalar_lr_mult` and `base_learning_rate` belong here for the same reason the
            # recipe string does: two arms that trained the scalar banks at different rates are
            # NOT a matched comparison, and this dict is what refuses one.
            "protocol": {key: manifest[key] for key in (
                "format", "objective", "numerics", "learning_rate", "base_learning_rate",
                "scalar_lr_mult", "eval_every", "optimizer", "optimizer_recipe")}}


def launch_final(args):
    specifications = [(args.reference_run, 96, False)] + [
        (f"{args.campaign}-{suffix}", context, features) for suffix, context, features in VARIANTS]
    evidence = [read_evidence(*specification) for specification in specifications]
    for candidate in evidence[1:]:
        for key in ("identity", "model", "validation_origins", "completed_origins", "persistence", "protocol"):
            if candidate[key] != evidence[0][key]:
                raise ValueError(f"comparison {key} differs; refusing unmatched selection")
    features = evidence[3]["mse"] < evidence[2]["mse"]
    print(f"Authenticated four complete-corpus comparisons. Final context remains 6000; "
          f"exogenous variates {'enabled' if features else 'disabled'} by matched full-validation MSE. "
          "Comparison values remain in each run's timexer_segment_validation.report.bin.", flush=True)
    status = subprocess.run(training_command(f"{args.campaign}-final", COMMON_CONTEXT, features, 10, args.data_dir),
                            cwd=ROOT).returncode
    return status if status >= 0 else 128 - status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--submit", action="store_true")
    mode.add_argument("--launch-final", action="store_true")
    parser.add_argument("--campaign", type=run_name, default="timexer-universe-20260905")
    parser.add_argument("--reference-job", type=int, default=4987)
    parser.add_argument("--reference-run", type=run_name, default="timexer-universe-c96-20260905")
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    if args.reference_job <= 0:
        parser.error("reference job must be positive")
    if args.submit:
        submit(args)
        return 0
    return launch_final(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(f"TimeXer campaign stopped: {error}", file=sys.stderr)
        raise SystemExit(1)
