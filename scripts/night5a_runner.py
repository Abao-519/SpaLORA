#!/usr/bin/env python3
"""Execute one locked Night-5A funnel stage without semantic-label access."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch


os.environ.setdefault("R_HOME", "/opt/R/4.0.3/lib/R")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night3af_cache import load_cache
from SpaLORA.night3a_ige import input_sha256
from SpaLORA.night3ar_protocol import ScientificWindow, assert_training_payload_label_free, ground_truth_csv_paths
from SpaLORA.night5a_rnd import (
    LEGACY_DELEGATES, Night5ATrainer, load_label_free_artifacts, load_registry,
    registry_contracts, run_identity, sha256_file,
)
from scripts.night3a_runner import cluster_exact, validate_attention


CONFIG_PATH = REPO / "configs/night5a_metric_rnd.json"
REQUIRED = ("embedding.npz", "attention.npz", "clusters.csv", "observation_ids.csv",
            "loss_trajectory.csv", "coefficient_probe.json", "run_manifest.json")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def write_csv(path: Path, rows: list) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())


def training_cfg(dataset: dict) -> dict:
    return {key: dataset[key] for key in ("embedding_dim", "epochs", "loss_factors", "locked_m_bad_expected")}


def run_path(raw_root: Path, dataset: str, candidate_id: str, seed: int) -> Path:
    return raw_root / dataset / candidate_id / ("seed_%d" % seed)


def existing_valid(directory: Path, identity: dict) -> bool:
    if not all((directory / name).is_file() for name in REQUIRED) or (directory / "failure.json").exists():
        return False
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if any(manifest.get(key) != value for key, value in identity.items()):
        return False
    return all(sha256_file(directory / name) == expected
               for name, expected in manifest.get("artifact_sha256", {}).items())


def stage_cells(stage: str, candidates: list, seeds: list) -> list:
    rows, ordinal = [], 0
    for dataset in ("a1", "placenta"):
        for candidate_id in candidates:
            for seed in seeds:
                ordinal += 1
                rows.append({"ordinal": ordinal, "stage": stage, "dataset": dataset,
                             "candidate_id": candidate_id, "seed": int(seed)})
    return rows


def run_one(config, contracts, prepared, artifact, dataset, candidate_id, seed,
            ordinal, stage, raw_root, output):
    cfg = training_cfg(config["datasets"][dataset])
    candidate = contracts[candidate_id]
    identity = run_identity(dataset, candidate_id, seed, candidate["config_sha256"],
                            artifact["manifest_sha256"])
    identity.update({"stage_first_executed": stage, "config_file_sha256": sha256_file(CONFIG_PATH)})
    directory = run_path(raw_root, dataset, candidate_id, seed)
    directory.mkdir(parents=True, exist_ok=True)
    if existing_valid(directory, identity):
        print("SKIP_VALID %s %s seed=%d" % (dataset, candidate_id, seed), flush=True)
        return directory / "run_manifest.json"
    existing = list(directory.iterdir())
    if existing:
        raise RuntimeError("Partial/mismatched run cannot be overwritten: %s" % directory)
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        trainer = Night5ATrainer(prepared.data, cfg, candidate, seed, torch.device("cuda:0"), artifact,
                                 config["ige_epsilon"])
        locked_input = input_sha256(prepared.data, prepared.obs_names, prepared.data["selected_gene_names"])
        result = trainer.train(); torch.cuda.synchronize()
        train_seconds = time.perf_counter() - started
        attention_deviation = validate_attention(result.output, len(prepared.obs_names))
        predicted = cluster_exact(np.asarray(result.output["SpaLORA"], np.float32),
                                  cfg["n_clusters"] if "n_clusters" in cfg else config["datasets"][dataset]["n_clusters"],
                                  config["clustering"]["random_seed"])
        np.savez_compressed(directory / "embedding.npz", SpaLORA=np.asarray(result.output["SpaLORA"], np.float32))
        np.savez_compressed(directory / "attention.npz",
                            alpha=np.asarray(result.output["alpha"], np.float32),
                            alpha_omics1=np.asarray(result.output["alpha_omics1"], np.float32),
                            alpha_omics2=np.asarray(result.output["alpha_omics2"], np.float32))
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str)}).to_csv(directory / "observation_ids.csv", index=False)
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str), "cluster": predicted}).to_csv(
            directory / "clusters.csv", index=False
        )
        write_csv(directory / "loss_trajectory.csv", list(result.logs))
        probe = result.probe or {}
        auxiliary = getattr(result, "auxiliary", {})
        atomic_json(directory / "coefficient_probe.json", {
            "candidate_id": candidate_id, "active_loss_mask": {
                name: bool(float(result.coefficients[name]) != 0.0) for name in result.coefficients
            },
            "raw_initial_losses": result.initial_losses, "raw_rms_gradients": probe.get("gradients", {}),
            "frozen_coefficients": result.coefficients,
            "active_coefficient_sum": float(sum(value for value in result.coefficients.values() if value != 0.0)),
            "corr2_objective_contribution_exact_zero": bool(result.coefficients.get("L_corr2_raw") == 0.0),
            "initial_state_sha256": result.initial_state_sha256, "auxiliary": auxiliary,
            "semantic_label_access": False,
        })
        artifact_sha = {name: sha256_file(directory / name) for name in REQUIRED if name != "run_manifest.json"}
        manifest = dict(identity)
        manifest.update({
            "schema_version": 1, "run_order_ordinal_within_stage": int(ordinal),
            "locked_input_sha256": locked_input, "epochs": int(cfg["epochs"]),
            "initial_state_sha256": result.initial_state_sha256,
            "final_state_sha256": result.final_state_sha256,
            "frozen_coefficients": result.coefficients,
            "artifact_sha256": artifact_sha, "semantic_label_access": False,
            "attention_max_row_sum_deviation": attention_deviation,
            "timings": {"training_seconds": float(train_seconds), "total_seconds": float(time.perf_counter() - started)},
            "resources": {
                "gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated() / 1024 ** 2),
                "gpu_peak_reserved_mib": float(torch.cuda.max_memory_reserved() / 1024 ** 2),
                "process_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
            },
        })
        atomic_json(directory / "run_manifest.json", manifest)
        print("DONE %s %s seed=%d train=%.1fs" % (dataset, candidate_id, seed, train_seconds), flush=True)
        return directory / "run_manifest.json"
    except Exception as exc:
        atomic_json(directory / "failure.json", {
            "dataset": dataset, "candidate_id": candidate_id, "seed": seed, "stage": stage,
            "error": repr(exc), "traceback": traceback.format_exc(),
        })
        raise
    finally:
        gc.collect(); torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("R1", "R2", "R3"), required=True)
    parser.add_argument("--candidates", nargs="*")
    args = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    gate = json.loads((output / "night5a_gate_status.json").read_text(encoding="utf-8"))
    if not (gate.get("p0_git_pass") and gate.get("p0_protect_pass") and gate.get("p0_arch_pass")):
        raise RuntimeError("Night-5A P0 gates did not authorize training")
    registry = load_registry(REPO / config["candidate_registry"])
    contracts = registry_contracts(registry)
    if args.stage == "R1":
        candidates = list(contracts)
        seeds = [0]
        if args.candidates:
            raise RuntimeError("R1 candidate set is fixed and cannot be overridden")
    else:
        decision_path = output / (("r1_decision.json" if args.stage == "R2" else "r2_decision.json"))
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        authorized = list(decision["advanced_candidates"])
        candidates = ["C00_FULL_IGE"] + authorized
        if args.candidates and list(args.candidates) != authorized:
            raise RuntimeError("Requested candidates differ from locked prior-stage decision")
        seeds = [1, 2] if args.stage == "R2" else [3, 4]
    if any(value not in contracts for value in candidates):
        raise RuntimeError("Unregistered candidate in stage")
    candidate_runs = len([value for value in candidates if value != "C00_FULL_IGE"]) * 2 * len(seeds)
    if candidate_runs > config["budgets"][args.stage.lower() + "_candidate"]:
        raise RuntimeError("Stage candidate budget exceeded")

    rows = stage_cells(args.stage, candidates, seeds)
    order_path = output / ("%s_run_order.json" % args.stage.lower())
    order_payload = {"schema_version": 1, "stage": args.stage, "locked_before_training": True,
                     "run_count": len(rows), "candidate_run_count": candidate_runs, "runs": rows}
    order_payload["canonical_sha256"] = __import__("SpaLORA.night5a_rnd", fromlist=["canonical_sha256"]).canonical_sha256(order_payload)
    if order_path.exists():
        existing = json.loads(order_path.read_text(encoding="utf-8"))
        if existing != order_payload:
            raise RuntimeError("Locked stage order changed")
    else:
        atomic_json(order_path, order_payload)

    index = json.loads(Path(config["paths"]["cache_manifest"]).read_text(encoding="utf-8"))
    prepared, artifacts = {}, {}
    forbidden = ground_truth_csv_paths(config)
    for dataset in ("a1", "placenta"):
        row = index["datasets"][dataset]
        directory = Path(config["paths"]["night3af_root"]) / row["directory"]
        prepared[dataset] = load_cache(directory, row["manifest_sha256"])
        cfg = training_cfg(config["datasets"][dataset])
        assert_training_payload_label_free(prepared[dataset].data, cfg, forbidden)
        artifacts[dataset] = load_label_free_artifacts(Path(config["paths"]["label_free_artifacts"]) / dataset)

    raw_root = Path(config["paths"]["raw_runs"])
    raw_root.mkdir(parents=True, exist_ok=True)
    window = ScientificWindow(config, output, "training_%s" % args.stage.lower()).install()
    manifests = []
    try:
        for row in rows:
            try:
                path = run_one(config, contracts, prepared[row["dataset"]], artifacts[row["dataset"]],
                               row["dataset"], row["candidate_id"], int(row["seed"]), int(row["ordinal"]),
                               args.stage, raw_root, output)
                manifests.append({"status": "success", "path": path})
            except Exception as exc:
                directory = run_path(raw_root, row["dataset"], row["candidate_id"], int(row["seed"]))
                failure = directory / "failure.json"
                manifests.append({"status": "failure", "path": failure, "error": repr(exc)})
                print("FAILED_RETAINED %s %s seed=%d %r" %
                      (row["dataset"], row["candidate_id"], int(row["seed"]), exc), flush=True)
        locked_rows = []
        for row, record in zip(rows, manifests):
            locked_rows.append({
                "dataset": row["dataset"], "candidate_id": row["candidate_id"], "seed": int(row["seed"]),
                "status": record["status"], "record_path": str(record["path"]),
                "record_sha256": sha256_file(record["path"]),
            })
        failure_count = sum(row["status"] == "failure" for row in manifests)
        lock_path = output / ("%s_training_manifest.json" % args.stage.lower())
        atomic_json(lock_path, {"schema_version": 1, "stage": args.stage,
                               "locked_before_semantic_label_access": True,
                               "run_count": len(rows), "success_count": len(rows) - failure_count,
                               "failure_count": failure_count,
                               "run_order_sha256": sha256_file(order_path), "runs": locked_rows})
        firewall = window.close(passed=True)
        if not firewall["passed"]:
            raise RuntimeError("Label firewall failed")
        atomic_json(output / ("%s_training_complete.json" % args.stage.lower()), {
            "schema_version": 1, "stage": args.stage, "run_count": len(rows),
            "success_count": len(rows) - failure_count, "failure_count": failure_count,
            "training_manifest_sha256": sha256_file(lock_path), "semantic_label_access": False,
        })
        print("%s_TRAINING_LOCKED success=%d failure=%d total=%d" %
              (args.stage, len(rows) - failure_count, failure_count, len(rows)), flush=True)
    except Exception:
        try:
            window.close(passed=False)
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()
