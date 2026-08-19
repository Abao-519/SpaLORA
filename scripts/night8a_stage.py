#!/usr/bin/env python3
"""Run preregistered Night-8A training and label-free transforms."""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PYTHON = Path("/root/miniconda3/envs/SpaLORA/bin/python")
TRAINER = REPO / "scripts/night8a_train.py"
TRANSFORMER = REPO / "scripts/night8a_transform.py"
REGISTRY = REPO / "protocols/night8a/SpaLORA_Night8A_MFSPC_Registry_2026-08-20.json"
SOURCE_INDEX = REPO / "outputs/night7b_handoff/source_unit_index.csv"
RAW = Path("/root/autodl-fs/night8a_raw_runs_20260820")
OUT = REPO / "outputs/night8a_handoff"

sys.path.insert(0, str(REPO))
from SpaLORA.night8a_mfspc import canonical_json_sha, file_sha, resolve_modules, select_family  # noqa: E402


ASSAY_AUTHORITY = {
    "a1": {"assays": ["RNA", "ADT"], "source": "locked human lymph A1 protocol"},
    "tonsil": {"assays": ["RNA", "protein"], "source": "locked tonsil assay contract"},
    "d1": {"assays": ["RNA", "ADT"], "source": "locked human lymph D1 protocol"},
    "p22": {"assays": ["RNA", "ATAC"], "source": "locked P22 assay contract"},
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def load_units() -> list[dict]:
    rows = list(csv.DictReader(SOURCE_INDEX.open(newline="", encoding="utf-8")))
    if len(rows) != 30:
        raise RuntimeError("expected 30 locked source units")
    result = []
    for row in rows:
        dataset = row["dataset"]
        metadata = ASSAY_AUTHORITY[dataset]
        family = select_family(metadata)
        worker = Path(row["worker_input"])
        if file_sha(worker) != row["worker_input_sha256"]:
            raise RuntimeError(f"worker input SHA mismatch: {row['unit_id']}")
        result.append({**row, "seed": int(row["seed"]), "K": int(row["K"]),
                       "assay_metadata": metadata, "family": family})
    return result


def configs(stage: str, registry: dict) -> list[dict]:
    if stage == "R1":
        return registry["R1_configs"]
    if stage == "R2":
        return registry["R2_configs"]
    shortlist_path = OUT / "r2_shortlist_ids.json"
    if not shortlist_path.exists():
        raise RuntimeError("R3 shortlist missing")
    shortlist = json.loads(shortlist_path.read_text())
    if set(shortlist) != {"status", "finalist_ids", "source_metrics_sha256", "label_window_closed"}:
        raise RuntimeError("R3 shortlist schema contains evaluator leakage")
    if shortlist["status"] != "LOCKED" or not shortlist["label_window_closed"]:
        raise RuntimeError("R3 shortlist not locked after label closure")
    index = {x["id"]: x for x in registry["R2_configs"]}
    ids = shortlist["finalist_ids"]
    if len(ids) > 3 or len(set(ids)) != len(ids) or any(x not in index for x in ids):
        raise RuntimeError("invalid R3 finalists")
    return [index[x] for x in ids]


def selected_units(stage: str, units: list[dict], registry: dict) -> list[dict]:
    if stage == "R1":
        seeds = registry["fixed_seeds"]["R1_mechanism_probe"]
    elif stage == "R2":
        seeds = registry["fixed_seeds"]["R2_pilot"]
    else:
        seeds = registry["fixed_seeds"]["R3_extension_if_selected"]
    return [u for u in units if u["seed"] in set(map(int, seeds[u["dataset"]]))]


def r02_reference(unit_id: str) -> Path:
    pattern = f"/root/autodl-fs/night7b_score_rnd_20260818/adapter_stage/*/formal/R02/{unit_id}/attempt_001/worker/embedding.npy"
    matches = [Path(x) for x in glob.glob(pattern)]
    if len(matches) != 1:
        raise RuntimeError(f"R02 reference ambiguity for {unit_id}: {matches}")
    manifest = matches[0].parent / "training_manifest.json"
    if not manifest.exists():
        raise RuntimeError(f"R02 manifest missing for {unit_id}")
    return matches[0]


def family_reference(unit: dict) -> Path:
    worker = json.loads(Path(unit["worker_input"]).read_text())
    if unit["family"] == "RNA_PROTEIN":
        return Path(worker["g04_views"])
    return r02_reference(unit["unit_id"])


def alias_target(stage: str, config_id: str, unit: dict) -> str | None:
    if unit["family"] != "RNA_PROTEIN":
        return None
    if stage == "R1" and config_id == "A05_RNA_ANCHOR":
        return "A00_FAMILY_REFERENCE"
    if stage in {"R2", "R3"} and config_id == "B05_SP_RR10_PROTO_RNAANCHOR":
        return "B04_SP_RR10_PROTO"
    return None


def cell_root(stage: str, config_id: str, unit_id: str) -> Path:
    return RAW / stage / config_id / unit_id / "attempt_001"


def make_config(stage: str, registered: dict, unit: dict, root: Path) -> tuple[Path, dict]:
    reference = family_reference(unit)
    config = {
        "schema_version": "night8a-cell-config-v1", "stage": stage,
        "config_id": registered["id"], "unit_id": unit["unit_id"], "seed": unit["seed"],
        "K": unit["K"], "assay_metadata": unit["assay_metadata"],
        "registered_modules": registered["modules"], "worker_input": unit["worker_input"],
        "family_reference": str(reference), "output_dir": str(root / "training"),
        "epochs": 160, "learning_rate": .001, "weight_decay": .00001,
    }
    if "RNA_ANCHOR" in registered["modules"]:
        config["rna_anchor_spatial_support"] = str(RAW / "locked_inputs/p22_spatial_support.npz")
    path = root / "cell_config.json"; atomic_json(path, config)
    return path, config


def run_logged(command: list[str], log: Path, timeout: int, env: dict | None = None) -> tuple[int, float, str | None]:
    started = time.perf_counter(); log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        try:
            result = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                    timeout=timeout, env=env, check=False)
            return result.returncode, time.perf_counter() - started, None
        except subprocess.TimeoutExpired:
            return 124, time.perf_counter() - started, "RUNTIME_TIMEOUT"


def verify_existing_training(root: Path) -> bool:
    manifest_path = root / "training/training_manifest.json"
    reload_path = root / "training/reload_audit.json"
    if not manifest_path.exists() or not reload_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text()); reload = json.loads(reload_path.read_text())
    if manifest.get("status") != "SUCCESS_PRE_LABEL" or reload.get("status") != "PASS":
        return False
    for value in manifest["artifacts"].values():
        if file_sha(value["path"]) != value["sha256"]:
            raise RuntimeError("existing training artifact SHA mismatch")
    return True


def train_cell(stage: str, registered: dict, unit: dict) -> dict:
    root = cell_root(stage, registered["id"], unit["unit_id"])
    alias = alias_target(stage, registered["id"], unit)
    if alias is not None:
        target = cell_root(stage, alias, unit["unit_id"])
        target_manifest = target / "training/training_manifest.json"
        target_reload = target / "training/reload_audit.json"
        if not target_manifest.exists() or not target_reload.exists():
            raise RuntimeError("alias target is not locked")
        root.mkdir(parents=True, exist_ok=True)
        active = resolve_modules(registered["modules"], unit["family"])
        alias_row = {
            "status": "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL", "stage": stage,
            "config_id": registered["id"], "unit_id": unit["unit_id"], "seed": unit["seed"],
            "family": unit["family"], "registered_config_sha256": canonical_json_sha(registered),
            "resolved_runtime_sha256": canonical_json_sha({"family": unit["family"], "modules": active,
                                                             "K": unit["K"],
                                                             "input_dim": 64, "fused_dim": 64}),
            "alias_target_config_id": alias, "alias_target_training_manifest": str(target_manifest),
            "alias_target_training_manifest_sha256": file_sha(target_manifest),
            "alias_target_reload_audit_sha256": file_sha(target_reload), "label_access": False,
            "scientific_training": False,
        }
        atomic_json(root / "alias_manifest.json", alias_row); return {**alias_row, "dataset": unit["dataset"], "root": str(root)}
    config_path, config = make_config(stage, registered, unit, root)
    if verify_existing_training(root):
        manifest = json.loads((root / "training/training_manifest.json").read_text())
        return {"status": "SUCCESS_PRE_LABEL", "stage": stage, "config_id": registered["id"],
                "unit_id": unit["unit_id"], "seed": unit["seed"], "dataset": unit["dataset"],
                "family": unit["family"], "root": str(root), "training_manifest": manifest,
                "resumed_verified": True}
    training_dir = root / "training"
    if training_dir.exists() and any(training_dir.iterdir()):
        raise RuntimeError(f"partial training attempt requires manual preservation: {root}")
    rc, seconds, timeout = run_logged([str(PYTHON), str(TRAINER), "train", "--config", str(config_path)],
                                      root / "train.log", timeout=1800)
    if rc != 0:
        failure = {"status": timeout or "SCIENTIFIC_NUMERICAL_FAILURE_NO_RETRY", "stage": stage,
                   "config_id": registered["id"], "unit_id": unit["unit_id"], "seed": unit["seed"],
                   "dataset": unit["dataset"], "family": unit["family"], "returncode": rc,
                   "runtime_seconds": seconds, "scientific_retry": 0, "fallback": 0, "label_access": False,
                   "log_sha256": file_sha(root / "train.log")}
        atomic_json(root / "failure_manifest.json", failure); return {**failure, "root": str(root)}
    rc2, reload_seconds, timeout2 = run_logged([str(PYTHON), str(TRAINER), "reload", "--config", str(config_path),
                                                "--output", str(training_dir)], root / "reload.log", timeout=600)
    if rc2 != 0:
        failure = {"status": timeout2 or "CHECKPOINT_ROUNDTRIP_FAILURE", "stage": stage,
                   "config_id": registered["id"], "unit_id": unit["unit_id"], "seed": unit["seed"],
                   "dataset": unit["dataset"], "family": unit["family"], "returncode": rc2,
                   "runtime_seconds": seconds + reload_seconds, "scientific_retry": 0,
                   "fallback": 0, "label_access": False, "log_sha256": file_sha(root / "reload.log")}
        atomic_json(root / "failure_manifest.json", failure); return {**failure, "root": str(root)}
    manifest = json.loads((training_dir / "training_manifest.json").read_text())
    return {"status": "SUCCESS_PRE_LABEL", "stage": stage, "config_id": registered["id"],
            "unit_id": unit["unit_id"], "seed": unit["seed"], "dataset": unit["dataset"],
            "family": unit["family"], "root": str(root), "training_manifest": manifest,
            "wrapper_runtime_seconds": seconds + reload_seconds}


def transform_cell(row: dict) -> dict:
    if row["status"] == "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL":
        target = cell_root(row["stage"], row["alias_target_config_id"], row["unit_id"])
        source = target / "transform/transform_manifest.json"
        if not source.exists():
            raise RuntimeError("alias transform target missing")
        manifest = json.loads(source.read_text())
        alias = {**row, "status": "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL",
                 "alias_target_transform_manifest": str(source),
                 "alias_target_transform_manifest_sha256": file_sha(source),
                 "alias_target_cluster_file_sha256": manifest["cluster_file_sha256"],
                 "transform_dir": str(target / "transform")}
        atomic_json(Path(row["root"]) / "transform_alias_manifest.json", alias)
        return alias
    if row["status"] != "SUCCESS_PRE_LABEL":
        return {**row, "transform_status": "SKIPPED_UPSTREAM_TRAINING_FAILURE"}
    root = Path(row["root"]); output = root / "transform"
    manifest_path = output / "transform_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if file_sha(output / "clusters.csv") != manifest["cluster_file_sha256"]:
            raise RuntimeError("existing cluster SHA mismatch")
        return {**row, "transform_status": manifest["status"], "transform_manifest": manifest,
                "transform_dir": str(output), "resumed_transform_verified": True}
    env = os.environ.copy()
    env.update({"OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3",
                "NUMEXPR_NUM_THREADS": "3"})
    rc, seconds, timeout = run_logged([str(PYTHON), str(TRANSFORMER), "--config", str(root / "cell_config.json"),
                                       "--training-output", str(root / "training"), "--output", str(output)],
                                      root / "transform.log", timeout=2700, env=env)
    if rc != 0:
        failure = {**row, "transform_status": timeout or "SCIENTIFIC_NUMERICAL_FAILURE_NO_RETRY",
                   "transform_returncode": rc, "transform_runtime_seconds": seconds,
                   "transform_scientific_retry": 0, "fallback": 0,
                   "transform_log_sha256": file_sha(root / "transform.log")}
        atomic_json(root / "transform_failure_manifest.json", failure); return failure
    manifest = json.loads(manifest_path.read_text())
    return {**row, "transform_status": manifest["status"], "transform_manifest": manifest,
            "transform_dir": str(output)}


def run_stage(stage: str) -> None:
    registry = json.loads(REGISTRY.read_text()); units = load_units()
    registered = configs(stage, registry); chosen = selected_units(stage, units, registry)
    rows = []; planned = len(registered) * len(chosen)
    for cfg in registered:
        for unit in chosen:
            row = train_cell(stage, cfg, unit); rows.append(row)
            print(json.dumps({"event": "training_cell_terminal", "stage": stage,
                              "complete": len(rows), "planned": planned, "config_id": cfg["id"],
                              "unit_id": unit["unit_id"], "status": row["status"]}, sort_keys=True), flush=True)
    actual_training = sum(r["status"] != "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL" for r in rows)
    prior_training = 0
    for prior in ("R1", "R2", "R3"):
        path = OUT / f"{prior.lower()}_stage_manifest.json"
        if path.exists() and prior != stage:
            prior_training += json.loads(path.read_text()).get("actual_training_attempts", 0)
    if prior_training + actual_training > int(registry["runtime"]["science_training_hard_cap"]):
        raise RuntimeError("science training hard cap exceeded")
    transformed = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(transform_cell, row) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            transformed.append(future.result())
            print(json.dumps({"event": "transform_cell_terminal", "stage": stage,
                              "complete": len(transformed), "planned": planned,
                              "status": transformed[-1].get("transform_status", transformed[-1]["status"])},
                             sort_keys=True), flush=True)
    order = {(cfg["id"], unit["unit_id"]): i for i, (cfg, unit) in enumerate(
        (pair for cfg in registered for pair in ((cfg, u) for u in chosen)))}
    transformed.sort(key=lambda x: order[(x["config_id"], x["unit_id"])])
    successes = sum((r.get("transform_status") == "SUCCESS_PRE_LABEL") or
                    (r["status"] == "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL") for r in transformed)
    manifest = {
        "schema_version": "night8a-stage-manifest-v1", "stage": stage,
        "status": "LOCKED_PRE_LABEL", "locked_before_label_access": True,
        "planned_registered_cells": planned, "actual_training_attempts": actual_training,
        "aliases": planned - actual_training, "terminal_cells": len(transformed),
        "successful_or_aliased_cells": successes,
        "scientific_retry_count": 0, "fallback_count": 0,
        "cpu_transform_workers": 3, "threads_per_worker": 3,
        "transform_timeout_seconds": 2700, "label_access": False,
        "registry_sha256": file_sha(REGISTRY), "cells": transformed,
    }
    atomic_json(OUT / f"{stage.lower()}_stage_manifest.json", manifest)
    if len(transformed) != planned:
        raise RuntimeError("stage coverage incomplete")
    print(json.dumps({"event": "stage_locked_pre_label", "stage": stage,
                      "planned": planned, "actual_training": actual_training,
                      "success_or_alias": successes}, sort_keys=True))


def main() -> None:
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", choices=("R1", "R2", "R3"), required=True)
    run_stage(ap.parse_args().stage)


if __name__ == "__main__":
    main()
