#!/usr/bin/env python3
"""Run and pre-label-lock the registered Night-7B R1/R2 adapter stages."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

os.environ.setdefault("R_HOME", "/root/miniconda3/envs/SpaLORA/lib/R")
os.environ["LD_LIBRARY_PATH"] = "/root/miniconda3/envs/SpaLORA/lib/R/lib:" + os.environ.get("LD_LIBRARY_PATH", "")

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha, sparse_sha  # noqa: E402
from SpaLORA.night7a_consensus import (  # noqa: E402
    atomic_json, atomic_sparse, canonical_partition, sha256_file,
)
from SpaLORA.night7b_adaptive import (  # noqa: E402
    row_sparse_strict, run_partition, self_tuning_affinity, sym_zero,
)

OUT = REPO / "outputs/night7b_handoff"
RAW = Path("/root/autodl-fs/night7b_score_rnd_20260818")
SOURCE = RAW / "source"
ADAPTER = RAW / "adapter_stage"
INPUTS = RAW / "adapter_inputs"
CONFIGS = RAW / "adapter_configs"
HEAD_RAW = RAW / "head_stage" / "formal"
REG = REPO / "protocols/night7b/SpaLORA_Night7B_Adaptive_Relational_Fusion_Registry_2026-08-18.json"
TRAINER = REPO / "scripts/night7b_train.py"
PYTHON = Path(sys.executable)
DATASETS = ("a1", "tonsil", "d1", "p22")


def canonical_json_sha(value: dict) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_npy(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npy")
    np.save(tmp, value, allow_pickle=False)
    os.replace(tmp, path)


def save_clusters(path: Path, ids, labels) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    pd.DataFrame({"observation_id": ids, "cluster": np.asarray(labels, dtype=np.int64)}).to_csv(tmp, index=False)
    os.replace(tmp, path)


def load_registry() -> dict:
    value = json.loads(REG.read_text())
    if value["adapter_recipe_order"] != ["R%02d" % i for i in range(10)]:
        raise RuntimeError("recipe order mismatch")
    return value


def load_units() -> list[dict]:
    rows = list(csv.DictReader((OUT / "source_unit_index.csv").open(newline="")))
    if len(rows) != 30:
        raise RuntimeError("source unit cardinality mismatch")
    return rows


def prepare_inputs() -> dict:
    contract = json.loads((OUT / "H_to_R1_contract.json").read_text())
    heads = contract["promoted_head_ids"]
    if len(heads) != 2:
        raise RuntimeError("H promotion cardinality mismatch")
    pseudo_head = heads[0]
    records = []
    for unit in load_units():
        unit_id = unit["unit_id"]
        source_contract = json.loads((SOURCE / unit_id / "worker_input.json").read_text())
        transform_dir = HEAD_RAW / unit_id / pseudo_head / "attempt_001"
        manifest = json.loads((transform_dir / "transform_manifest.json").read_text())
        if manifest["status"] != "success":
            raise RuntimeError("promoted pseudo head is incomplete: %s/%s" % (unit_id, pseudo_head))
        table = pd.read_csv(transform_dir / "clusters.csv")
        ids = [x.strip() for x in Path(source_contract["observation_ids"]).read_text().splitlines() if x.strip()]
        if table["observation_id"].astype(str).tolist() != ids:
            raise RuntimeError("pseudo partition order mismatch: %s" % unit_id)
        pseudo = canonical_partition(table["cluster"].to_numpy())
        unit_dir = INPUTS / unit_id
        unit_dir.mkdir(parents=True, exist_ok=False)
        pseudo_path = unit_dir / "pseudo_partition.npy"
        atomic_npy(pseudo_path, pseudo.astype(np.int64))
        source_contract["pseudo_partition"] = str(pseudo_path)
        source_contract["pseudo_affinity"] = str(transform_dir / "affinity.npz")
        atomic_json(unit_dir / "worker_input.json", source_contract)
        records.append({
            "unit_id": unit_id,
            "worker_input_path": str(unit_dir / "worker_input.json"),
            "worker_input_sha256": sha256_file(unit_dir / "worker_input.json"),
            "pseudo_head": pseudo_head,
            "pseudo_partition_sha256": array_sha(pseudo.astype(np.int64)),
            "pseudo_affinity_sha256": sparse_sha(sp.load_npz(transform_dir / "affinity.npz")),
            "label_access": False,
        })
    result = {
        "status": "LOCKED_PRE_LABEL", "units": records,
        "promoted_head_ids": heads, "pseudo_head_id": pseudo_head,
        "H_contract_sha256": sha256_file(OUT / "H_to_R1_contract.json"),
        "label_access": False,
    }
    atomic_json(OUT / "adapter_input_lock.json", result)
    return result


def recipe_map(registry: dict) -> dict:
    return {x["id"]: x for x in registry["adapter_recipes"]}


def make_config(stage: str, recipe: dict, unit: dict) -> Path:
    config = {
        "recipe_id": recipe["id"], "losses": recipe["losses"],
        "fusion": recipe["fusion"], "seed": int(unit["seed"]),
        "epochs": 160,
    }
    path = CONFIGS / stage / recipe["id"] / (unit["unit_id"] + ".json")
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise RuntimeError("immutable config conflict: %s" % path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json(path, config)
    return path


def run_command(command: list[str], log_path: Path) -> tuple[int, float]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with log_path.open("wb") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT,
                                   env={**os.environ, "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1",
                                        "OPENBLAS_NUM_THREADS":"1", "NUMEXPR_NUM_THREADS":"1"})
    return int(completed.returncode), time.perf_counter() - start


def training_cell(stage: str, recipe: dict, unit: dict) -> dict:
    recipe_id, unit_id = recipe["id"], unit["unit_id"]
    target = ADAPTER / stage / "formal" / recipe_id / unit_id / "attempt_001"
    wrapper_path = target / "cell_manifest.json"
    if wrapper_path.exists():
        value = json.loads(wrapper_path.read_text())
        if value.get("stage") == stage and value.get("recipe_id") == recipe_id:
            return value
        raise RuntimeError("existing training cell manifest mismatch: %s" % target)
    if target.exists():
        raise RuntimeError("partial formal training cell requires explicit infrastructure audit: %s" % target)
    target.mkdir(parents=True)
    worker_output = target / "worker"
    config_path = make_config(stage, recipe, unit)
    start = time.perf_counter()
    train_rc, train_seconds = run_command([
        str(PYTHON), str(TRAINER), "train", "--unit-dir", str(INPUTS / unit_id),
        "--config", str(config_path), "--output", str(worker_output),
    ], target / "train.log")
    row = {
        "stage": stage, "unit_id": unit_id, "dataset": unit["dataset"],
        "seed": int(unit["seed"]), "recipe_id": recipe_id, "attempt": 1,
        "config_path": str(config_path), "config_sha256": sha256_file(config_path),
        "worker_input_sha256": sha256_file(INPUTS / unit_id / "worker_input.json"),
        "train_returncode": train_rc, "train_runtime_seconds": train_seconds,
        "scientific_training": True, "label_access": False,
        "fallback": False, "retry": False,
    }
    if train_rc == 0 and (worker_output / "training_manifest.json").exists():
        reload_rc, reload_seconds = run_command([
            str(PYTHON), str(TRAINER), "reload", "--unit-dir", str(INPUTS / unit_id),
            "--config", str(config_path), "--output", str(worker_output),
        ], target / "reload.log")
        row["reload_returncode"] = reload_rc
        row["reload_runtime_seconds"] = reload_seconds
        if reload_rc == 0:
            train_manifest = json.loads((worker_output / "training_manifest.json").read_text())
            reload_audit = json.loads((worker_output / "reload_forward_audit.json").read_text())
            row.update({
                "status": "success", "training_manifest": train_manifest,
                "training_manifest_sha256": sha256_file(worker_output / "training_manifest.json"),
                "reload_audit": reload_audit,
                "reload_audit_sha256": sha256_file(worker_output / "reload_forward_audit.json"),
            })
        else:
            row.update({"status":"scientific_numerical_failure", "failure_type":"checkpoint_reload_failure"})
    else:
        row.update({"status":"scientific_numerical_failure", "failure_type":"training_process_failure"})
    row["wrapper_runtime_seconds"] = time.perf_counter() - start
    atomic_json(wrapper_path, row)
    return row


def endpoint_affinity(endpoint: str, embedding: np.ndarray, c06: sp.csr_matrix,
                      ids: list[str]) -> sp.csr_matrix:
    az = self_tuning_affinity(embedding, 10, ids)
    if endpoint == "E0_ADAPTER_ONLY":
        return az
    if endpoint == "E1_ADAPTER_C06_MEAN":
        return sym_zero(row_sparse_strict(az) * .5 + row_sparse_strict(c06) * .5)
    raise RuntimeError("unknown endpoint")


def transform_cell(stage: str, training: dict, unit: dict, endpoint: str,
                   head_id: str, resolution: list[float]) -> dict:
    recipe_id, unit_id = training["recipe_id"], unit["unit_id"]
    target = ADAPTER / stage / "formal" / recipe_id / unit_id / "attempt_001" / "transforms" / endpoint / head_id
    manifest_path = target / "transform_manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    target.mkdir(parents=True, exist_ok=False)
    base = {
        "stage":stage, "unit_id":unit_id, "dataset":unit["dataset"], "seed":int(unit["seed"]),
        "recipe_id":recipe_id, "endpoint":endpoint, "head_id":head_id,
        "config_id":"%s__%s__%s" % (recipe_id, endpoint, head_id),
        "training_cell_status":training["status"], "fallback":False, "retry":False,
        "label_access":False,
    }
    if training["status"] != "success":
        base.update({"status":"upstream_training_failure", "failure_type":training.get("failure_type")})
        atomic_json(manifest_path, base)
        return base
    start = time.perf_counter()
    try:
        worker = ADAPTER / stage / "formal" / recipe_id / unit_id / "attempt_001" / "worker"
        embedding = np.load(worker / "embedding.npy", allow_pickle=False)
        ids = [x.strip() for x in (SOURCE / unit_id / "observation_ids.txt").read_text().splitlines() if x.strip()]
        c06 = sp.load_npz(SOURCE / unit_id / "c06_affinity.npz")
        affinity = endpoint_affinity(endpoint, embedding, c06, ids)
        affinity_path = target / "affinity.npz"; atomic_sparse(affinity_path, affinity)
        labels, partition = run_partition(head_id, affinity, int(unit["K"]), resolution)
        clusters_path = target / "clusters.csv"; save_clusters(clusters_path, ids, labels)
        base.update({
            "status":"success", "partition":partition,
            "embedding_sha256":array_sha(embedding),
            "canonical_affinity_sha256":sparse_sha(affinity),
            "affinity_file_sha256":sha256_file(affinity_path),
            "canonical_partition_sha256":array_sha(canonical_partition(labels)),
            "clusters_file_sha256":sha256_file(clusters_path),
            "cluster_count":int(len(np.unique(labels))),
        })
    except Exception as exc:
        trace = target / "failure_trace.txt"; trace.write_text(traceback.format_exc(), encoding="utf-8")
        base.update({"status":"scientific_numerical_failure", "failure_type":type(exc).__name__,
                     "failure_message":str(exc), "failure_trace_sha256":sha256_file(trace)})
    base["runtime_seconds"] = time.perf_counter() - start
    base["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    atomic_json(manifest_path, base)
    return base


def verify_transform(stage: str, training: dict, unit: dict, transform: dict,
                     resolution: list[float]) -> dict:
    if transform["status"] != "success":
        return {"status":"NOT_APPLICABLE", "fresh_process":True}
    recipe_id, unit_id = training["recipe_id"], unit["unit_id"]
    target = ADAPTER / stage / "formal" / recipe_id / unit_id / "attempt_001" / "transforms" / transform["endpoint"] / transform["head_id"]
    worker = ADAPTER / stage / "formal" / recipe_id / unit_id / "attempt_001" / "worker"
    embedding = np.load(worker / "embedding.npy", allow_pickle=False)
    ids = [x.strip() for x in (SOURCE / unit_id / "observation_ids.txt").read_text().splitlines() if x.strip()]
    affinity = endpoint_affinity(transform["endpoint"], embedding,
                                 sp.load_npz(SOURCE / unit_id / "c06_affinity.npz"), ids)
    labels, _ = run_partition(transform["head_id"], affinity, int(unit["K"]), resolution)
    saved = pd.read_csv(target / "clusters.csv")["cluster"].to_numpy()
    audit = {
        "status":"PASS",
        "affinity_exact":sparse_sha(affinity) == transform["canonical_affinity_sha256"],
        "partition_exact":array_sha(canonical_partition(labels)) == array_sha(canonical_partition(saved)),
        "fresh_process":True, "label_access":False,
    }
    if not audit["affinity_exact"] or not audit["partition_exact"]:
        audit["status"] = "FAIL"
    atomic_json(target / "fresh_transform_reload_audit.json", audit)
    if audit["status"] != "PASS":
        raise RuntimeError("fresh transform reload mismatch: %s" % target)
    return audit


def run_stage(stage: str) -> None:
    registry = load_registry(); recipes = recipe_map(registry); all_units = load_units()
    adapter_lock = json.loads((OUT / "adapter_input_lock.json").read_text())
    if adapter_lock["status"] != "LOCKED_PRE_LABEL" or adapter_lock["label_access"]:
        raise RuntimeError("adapter input lock invalid")
    heads = adapter_lock["promoted_head_ids"]
    resolution = registry["partition_heads"]["LEIDEN_EXACT_K"]["resolution_grid"]
    if stage == "R1":
        recipe_ids = registry["adapter_recipe_order"]
        units = [x for x in all_units if int(x["seed"]) in (0, 1)]
        configs = None
    else:
        contract = json.loads((OUT / "R1_to_R2_contract.json").read_text())
        configs = contract["promoted_config_ids"]
        recipe_ids = []
        for config_id in configs:
            recipe_id = config_id.split("__", 1)[0]
            if recipe_id not in recipe_ids:
                recipe_ids.append(recipe_id)
        units = [x for x in all_units if int(x["seed"]) >= 2]
    training_rows = []
    for recipe_id in recipe_ids:
        for unit in units:
            value = training_cell(stage, recipes[recipe_id], unit)
            training_rows.append(value)
            print("%s_TRAIN" % stage, recipe_id, unit["unit_id"], value["status"], flush=True)
    transform_rows = []
    if stage == "R1":
        requested = [(r, endpoint, head) for r in recipe_ids
                     for endpoint in ("E0_ADAPTER_ONLY", "E1_ADAPTER_C06_MEAN") for head in heads]
    else:
        requested = []
        for config_id in configs:
            recipe_id, endpoint, head = config_id.split("__")
            requested.append((recipe_id, endpoint, head))
    lookup = {(x["recipe_id"], x["unit_id"]):x for x in training_rows}
    for recipe_id, endpoint, head in requested:
        for unit in units:
            training = lookup[(recipe_id, unit["unit_id"])]
            value = transform_cell(stage, training, unit, endpoint, head, resolution)
            transform_rows.append(value)
            print("%s_TRANSFORM" % stage, value["config_id"], unit["unit_id"], value["status"], flush=True)
            if value["status"] == "success":
                command = [str(PYTHON), str(Path(__file__).resolve()), "verify-cell", "--stage", stage,
                           "--recipe", recipe_id, "--unit", unit["unit_id"], "--endpoint", endpoint,
                           "--head", head]
                target = ADAPTER / stage / "formal" / recipe_id / unit["unit_id"] / "attempt_001" / "transforms" / endpoint / head
                rc, _ = run_command(command, target / "fresh_transform_reload.log")
                if rc != 0:
                    raise RuntimeError("fresh transform verifier failed: %s" % target)
                audit_path = target / "fresh_transform_reload_audit.json"
                value["fresh_transform_reload_audit"] = json.loads(audit_path.read_text())
                value["fresh_transform_reload_audit_sha256"] = sha256_file(audit_path)
                atomic_json(target / "transform_manifest.json", value)
    expected_training = 80 if stage == "R1" else len(recipe_ids) * 22
    expected_transforms = 320 if stage == "R1" else len(configs) * 22
    if len(training_rows) != expected_training or len(transform_rows) != expected_transforms:
        raise RuntimeError("stage matrix cardinality mismatch")
    reload_pass = 0
    for value in training_rows:
        if value["status"] == "success" and value["reload_audit"]["status"] == "PASS":
            reload_pass += 1
    lock = {
        "schema_version":1, "stage":stage, "status":"LOCKED_PRE_LABEL",
        "planned_training":expected_training, "training_attempts":len(training_rows),
        "successful_training":sum(x["status"] == "success" for x in training_rows),
        "checkpoint_reload_pass":reload_pass,
        "planned_transforms":expected_transforms, "transform_attempts":len(transform_rows),
        "successful_transforms":sum(x["status"] == "success" for x in transform_rows),
        "scientific_retry":0, "fallback_count":0, "label_access":False,
        "recipe_ids":recipe_ids, "config_ids":configs,
        "training_cells":training_rows, "transforms":transform_rows,
    }
    atomic_json(OUT / ("locked_%s_manifest.json" % stage), lock)


def verify_cell(args) -> None:
    registry = load_registry(); units = {x["unit_id"]:x for x in load_units()}
    wrapper = ADAPTER / args.stage / "formal" / args.recipe / args.unit / "attempt_001" / "cell_manifest.json"
    training = json.loads(wrapper.read_text())
    transform_path = ADAPTER / args.stage / "formal" / args.recipe / args.unit / "attempt_001" / "transforms" / args.endpoint / args.head / "transform_manifest.json"
    transform = json.loads(transform_path.read_text())
    verify_transform(args.stage, training, units[args.unit], transform,
                     registry["partition_heads"]["LEIDEN_EXACT_K"]["resolution_grid"])


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare-inputs")
    run = sub.add_parser("run"); run.add_argument("--stage", choices=("R1","R2"), required=True)
    verify = sub.add_parser("verify-cell")
    verify.add_argument("--stage", choices=("R1","R2"), required=True)
    verify.add_argument("--recipe", required=True); verify.add_argument("--unit", required=True)
    verify.add_argument("--endpoint", required=True); verify.add_argument("--head", required=True)
    args = parser.parse_args()
    if args.command == "prepare-inputs": prepare_inputs()
    elif args.command == "run": run_stage(args.stage)
    else: verify_cell(args)


if __name__ == "__main__":
    main()
