#!/usr/bin/env python3
"""Fixed-order fresh Night-6C scientific training runner."""
from __future__ import annotations
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night3a_ige import model_state_sha256
from SpaLORA.night3af_cache import load_cache, sha256_file
from SpaLORA.night6c_pipeline import (
    BASE_C04, DATASET_CFG, atomic_json, atomic_torch_save, canonical_json_sha,
    file_row, forward_model, h00, load_graph_data, make_trainer,
    observation_sha, parse_registry, runtime_resources, save_views,
)

OUT = REPO / "outputs/night6c_handoff"
RAW = Path("/root/autodl-fs/night6c_raw_runs_20260817")
CACHE = Path("/root/autodl-fs/night6c_cache_20260817")
REG_PATH = REPO / "protocols/night6c/SpaLORA_Night6B_Candidate_Registry_2026-08-17.json"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def versions() -> dict:
    import scipy, sklearn
    return {"python": platform.python_version(), "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda, "numpy": np.__version__,
            "scipy": scipy.__version__, "sklearn": sklearn.__version__}


def paths(dataset: str, graph_id: str, seed: int) -> tuple[Path, Path]:
    return CACHE / "base" / dataset, CACHE / "graphs" / dataset / graph_id


def prior_success(root: Path, dataset: str, graph_id: str, seed: int):
    successes = []
    for path in sorted(root.glob("attempt_*/run_manifest.json")):
        row = json.loads(path.read_text())
        if row.get("status") == "success":
            if row["dataset"] != dataset or row["graph_id"] != graph_id or int(row["seed"]) != int(seed):
                raise RuntimeError("existing run identity mismatch")
            if sha256_file(path.parent / "model_final.pt") != row["checkpoint_file_sha256"]:
                raise RuntimeError("existing successful checkpoint SHA mismatch")
            successes.append((path, row))
    if len(successes) > 1:
        raise RuntimeError("multiple successful attempts for one scientific cell")
    if successes:
        path, row = successes[0]
        return {**row, "run_dir": str(path.parent),
                "run_manifest_sha256": sha256_file(path)}
    return None


def one(stage: str, dataset: str, graph_id: str, seed: int, ordinal: int,
        graph_contract: dict, code_commit: str) -> dict:
    root = RAW / stage.lower() / graph_id / dataset / f"seed_{seed}"
    existing = prior_success(root, dataset, graph_id, seed)
    if existing is not None:
        return existing
    attempts = sorted(root.glob("attempt_*")) if root.exists() else []
    attempt = len(attempts) + 1
    if attempt > 13:
        raise RuntimeError("per-cell retry safety bound exceeded")
    run = root / f"attempt_{attempt:03d}"
    run.mkdir(parents=True)
    base_dir, graph_dir = paths(dataset, graph_id, seed)
    base_manifest_sha = sha256_file(base_dir / "manifest.json")
    prepared = load_cache(base_dir, base_manifest_sha)
    data, graph_manifest = load_graph_data(prepared, graph_dir)
    device = torch.device("cuda:0")
    started = time.perf_counter()
    try:
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(device)
        trainer = make_trainer(data, dataset, seed, device)
        result = trainer.train()
        views = forward_model(result.model, data, device)
        view_rows = save_views(run / "views.npz", views, prepared.obs_names.astype(str))
        h00_result = h00(views["SpaLORA_fused"], int(DATASET_CFG[dataset]["n_clusters"]))
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str),
                      "cluster": h00_result["labels"]}).to_csv(run / "h00_clusters.csv", index=False)
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str)}).to_csv(run / "observation_ids.csv", index=False)
        pd.DataFrame(result.logs).to_csv(run / "loss_trajectory.csv", index=False)
        probe_diagnostics = {k: v for k, v in result.probe.items() if k != "initial_state"}
        atomic_json(run / "coefficient_probe.json", {
            "candidate_id": "C04_SHRINK25",
            "frozen_coefficients": result.coefficients,
            "active_coefficient_sum": float(sum(result.coefficients.values())),
            "raw_initial_losses": result.initial_losses,
            "raw_rms_gradients": result.probe["gradients"],
            "initial_state_sha256": result.initial_state_sha256,
            "active_losses": result.auxiliary.get("active_loss_names"),
            "corr2_objective_contribution_exact_zero": result.coefficients["L_corr2_raw"] == 0.0,
            "probe_diagnostics_without_tensor_state": probe_diagnostics,
            "semantic_label_access": False,
        })
        state_sha = model_state_sha256(result.model)
        if state_sha != result.final_state_sha256:
            raise RuntimeError("training result state SHA mismatch")
        cfg = {
            "encoder_id": "E00C_C04_B01_CLEAN", "candidate": BASE_C04,
            "dataset_config": DATASET_CFG[dataset], "optimizer": "Adam",
            "learning_rate": 1e-4, "weight_decay": 0.0, "scheduler": None,
            "checkpoint_policy": "fixed_final_epoch", "seed": int(seed),
            "dataset_id": dataset, "graph_candidate": graph_contract,
            "deterministic_seed_function": "SpaLORA.preprocess.fix_seed",
        }
        cache_identity = canonical_json_sha({
            "base_manifest_sha256": base_manifest_sha,
            "graph_cache_sha256": graph_manifest["canonical_graph_cache_sha256"],
            "observation_sha256": observation_sha(prepared.obs_names.astype(str)),
        })
        payload = {
            "model_state_dict": {k: v.detach().cpu() for k, v in result.model.state_dict().items()},
            "canonical_tensor_state_sha256": state_sha,
            "canonical_training_config": cfg,
            "dataset_id": dataset, "graph_candidate_id": graph_id, "seed": int(seed),
            "input_and_cache_sha256": cache_identity, "code_commit": code_commit,
            "software_versions": versions(),
        }
        atomic_torch_save(run / "model_final.pt", payload)
        atomic_json(run / "reload_spec.json", {
            "dataset": dataset, "graph_id": graph_id, "seed": int(seed),
            "base_cache_dir": str(base_dir), "base_cache_manifest_sha256": base_manifest_sha,
            "graph_cache_dir": str(graph_dir),
            "graph_cache_sha256": graph_manifest["canonical_graph_cache_sha256"],
        })
        subprocess.run([sys.executable, str(REPO / "scripts/night6c_reload.py"),
                        "--run-dir", str(run)], cwd=REPO, check=True)
        reload_audit = json.loads((run / "checkpoint_reload_audit.json").read_text())
        resources = runtime_resources(started)
        artifacts = {}
        for path in sorted(run.iterdir()):
            if path.is_file() and path.name != "run_manifest.json":
                artifacts[path.name] = file_row(path)
        manifest = {
            "schema_version": 1, "stage": stage, "ordinal": int(ordinal),
            "dataset": dataset, "graph_id": graph_id, "seed": int(seed),
            "status": "success", "fresh_scientific_training": True,
            "attempt": attempt, "canonical_training_config": cfg,
            "canonical_training_config_sha256": canonical_json_sha(cfg),
            "base_cache_manifest_sha256": base_manifest_sha,
            "graph_cache_sha256": graph_manifest["canonical_graph_cache_sha256"],
            "canonical_input_and_cache_sha256": cache_identity,
            "ordered_observation_sha256": observation_sha(prepared.obs_names.astype(str)),
            "initial_state_sha256": result.initial_state_sha256,
            "final_tensor_state_sha256": state_sha,
            "checkpoint_file_sha256": sha256_file(run / "model_final.pt"),
            "checkpoint_round_trip_pass": reload_audit["status"] == "PASS",
            "h00_cluster_reload_exact": reload_audit["h00_clusters_exact"],
            "view_contracts": view_rows, "coefficients": result.coefficients,
            "h00_selected_model": h00_result["selected_model"],
            "code_commit": code_commit, "artifacts": artifacts,
            "label_values_deserialized": False, "label_values_used": False,
            **resources,
        }
        atomic_json(run / "run_manifest.json", manifest)
        return {**manifest, "run_dir": str(run),
                "run_manifest_sha256": sha256_file(run / "run_manifest.json")}
    except Exception as exc:
        atomic_json(run / "failure.json", {
            "stage": stage, "dataset": dataset, "graph_id": graph_id,
            "seed": int(seed), "ordinal": int(ordinal), "attempt": attempt,
            "status": "implementation_or_infrastructure_failure",
            "exception_type": type(exc).__name__, "message": str(exc),
            "label_values_deserialized": False,
        })
        raise


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("R1", "R2"), required=True)
    args = ap.parse_args()
    registry = json.loads(REG_PATH.read_text(encoding="utf-8"))
    graphs, _ = parse_registry(registry)
    graph_order = list(graphs)
    if args.stage == "R1":
        selected = graph_order; seeds = [0, 1]; planned = 36
    else:
        decision = json.loads((OUT / "r1_decision.json").read_text(encoding="utf-8"))
        selected = [graph_order[0]] + list(decision["advanced_graphs"])
        if len(selected) > 5 or len(set(selected)) != len(selected):
            raise RuntimeError("R2 graph selection contract invalid")
        seeds = [2, 3, 4]; planned = len(selected) * 2 * 3
    commit = git("rev-parse", "HEAD")
    if git("status", "--porcelain", "--", "SpaLORA", "scripts", "tests"):
        raise RuntimeError("scientific code must be committed before training")
    rows = []; ordinal = 0
    for graph_id in selected:
        for dataset in ("a1", "tonsil"):
            for seed in seeds:
                ordinal += 1
                print(json.dumps({"event": "training_start", "stage": args.stage,
                                  "ordinal": ordinal, "planned": planned,
                                  "dataset": dataset, "graph_id": graph_id,
                                  "seed": seed}, sort_keys=True), flush=True)
                rows.append(one(args.stage, dataset, graph_id, seed, ordinal,
                                graphs[graph_id], commit))
    failed_attempts = sum(1 for graph_id in selected for dataset in ("a1", "tonsil")
                          for seed in seeds for _ in
                          (RAW / args.stage.lower() / graph_id / dataset / f"seed_{seed}").glob("attempt_*/failure.json"))
    if failed_attempts > 12:
        raise RuntimeError("implementation/infrastructure retry budget exceeded")
    aggregate = {
        "schema_version": 1, "stage": args.stage, "status": "LOCKED",
        "locked_before_label_access": True, "planned_units": planned,
        "attempted_units": len(rows), "success_count": len(rows),
        "failure_count": failed_attempts, "scientific_training_units": len(rows),
        "implementation_retries": failed_attempts,
        "total_training_attempts": len(rows) + failed_attempts,
        "fixed_order": "graph_dataset_seed",
        "runs": rows,
    }
    atomic_json(OUT / f"{args.stage.lower()}_training_manifest.json", aggregate)
    print(json.dumps({"event": "training_stage_locked", "stage": args.stage,
                      "success": len(rows), "planned": planned}, sort_keys=True))


if __name__ == "__main__":
    main()
