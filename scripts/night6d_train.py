#!/usr/bin/env python3
"""Fixed 40-unit Night-6D fresh training runner."""
from __future__ import annotations

import json
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
from SpaLORA.night6d_pipeline import (
    BASE_C04, DATASET_CFG, GRAPHS, atomic_json, atomic_torch_save,
    canonical_json_sha, file_row, forward_model, h00, load_graph_data,
    make_trainer, observation_sha, runtime_resources, save_views,
)

OUT = REPO / "outputs/night6d_handoff"
RAW = Path("/root/autodl-fs/night6d_raw_runs_20260817")
CACHE = Path("/root/autodl-fs/night6d_cache_20260817")
BASE = {
    "d1": CACHE / "base/d1",
    "p22": Path("/root/autodl-fs/night3af_p0d_builds_20260810/process_a/p22"),
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def versions() -> dict:
    import scipy
    import sklearn
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
    }


def prior_success(root: Path, dataset: str, graph_id: str, seed: int):
    success = []
    for path in sorted(root.glob("attempt_*/run_manifest.json")):
        row = json.loads(path.read_text())
        if row.get("status") == "success":
            if (row["dataset"], row["graph_id"], int(row["seed"])) != (dataset, graph_id, int(seed)):
                raise RuntimeError("existing run identity mismatch")
            if sha256_file(path.parent / "model_final.pt") != row["checkpoint_file_sha256"]:
                raise RuntimeError("existing successful checkpoint SHA mismatch")
            success.append((path, row))
    if len(success) > 1:
        raise RuntimeError("multiple successful attempts for one scientific cell")
    if success:
        path, row = success[0]
        return {**row, "run_dir": str(path.parent), "run_manifest_sha256": sha256_file(path)}
    return None


def one(dataset: str, graph_id: str, seed: int, ordinal: int, code_commit: str) -> dict:
    root = RAW / graph_id / dataset / f"seed_{seed}"
    existing = prior_success(root, dataset, graph_id, seed)
    if existing is not None:
        return existing
    attempts = sorted(root.glob("attempt_*")) if root.exists() else []
    attempt = len(attempts) + 1
    if attempt > 9:
        raise RuntimeError("per-cell retry safety bound exceeded")
    run = root / f"attempt_{attempt:03d}"
    run.mkdir(parents=True)
    base_dir = BASE[dataset]
    graph_dir = CACHE / "graphs" / dataset / graph_id
    base_manifest_sha = sha256_file(base_dir / "manifest.json")
    prepared = load_cache(base_dir, base_manifest_sha)
    data, graph_manifest = load_graph_data(prepared, graph_dir)
    device = torch.device("cuda:0")
    started = time.perf_counter()
    try:
        torch.cuda.set_device(device)
        torch.cuda.init()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        trainer = make_trainer(data, dataset, seed, device)
        result = trainer.train()
        views = forward_model(result.model, data, device)
        view_rows = save_views(run / "views.npz", views, prepared.obs_names.astype(str))
        h00_result = h00(views["SpaLORA_fused"], DATASET_CFG[dataset]["n_clusters"])
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str),
                      "cluster": h00_result["labels"]}).to_csv(run / "h00_clusters.csv", index=False)
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str)}).to_csv(
            run / "observation_ids.csv", index=False)
        pd.DataFrame(result.logs).to_csv(run / "loss_trajectory.csv", index=False)
        probe = {k: v for k, v in result.probe.items() if k != "initial_state"}
        atomic_json(run / "coefficient_probe.json", {
            "candidate_id": "C04_SHRINK25",
            "frozen_coefficients": result.coefficients,
            "active_coefficient_sum": float(sum(result.coefficients.values())),
            "raw_initial_losses": result.initial_losses,
            "raw_rms_gradients": result.probe["gradients"],
            "initial_state_sha256": result.initial_state_sha256,
            "active_losses": result.auxiliary.get("active_loss_names"),
            "corr2_objective_contribution_exact_zero": result.coefficients["L_corr2_raw"] == 0.0,
            "probe_diagnostics_without_tensor_state": probe,
            "semantic_label_access": False,
        })
        state_sha = model_state_sha256(result.model)
        if state_sha != result.final_state_sha256:
            raise RuntimeError("training result state SHA mismatch")
        config = {
            "encoder_id": "E00C_C04_B01_CLEAN",
            "candidate": BASE_C04,
            "dataset_config": DATASET_CFG[dataset],
            "optimizer": "Adam",
            "learning_rate": 1e-4,
            "weight_decay": 0.0,
            "scheduler": None,
            "checkpoint_policy": "fixed_final_epoch",
            "seed": int(seed),
            "dataset_id": dataset,
            "graph_candidate": GRAPHS[graph_id],
            "deterministic_seed_function": "SpaLORA.preprocess.fix_seed",
        }
        cache_identity = canonical_json_sha({
            "base_manifest_sha256": base_manifest_sha,
            "graph_cache_sha256": graph_manifest["canonical_graph_cache_sha256"],
            "observation_sha256": observation_sha(prepared.obs_names.astype(str)),
        })
        atomic_torch_save(run / "model_final.pt", {
            "model_state_dict": {k: v.detach().cpu() for k, v in result.model.state_dict().items()},
            "canonical_tensor_state_sha256": state_sha,
            "canonical_training_config": config,
            "dataset_id": dataset,
            "graph_candidate_id": graph_id,
            "seed": int(seed),
            "input_and_cache_sha256": cache_identity,
            "code_commit": code_commit,
            "software_versions": versions(),
        })
        atomic_json(run / "reload_spec.json", {
            "dataset": dataset,
            "graph_id": graph_id,
            "seed": int(seed),
            "base_cache_dir": str(base_dir),
            "base_cache_manifest_sha256": base_manifest_sha,
            "graph_cache_dir": str(graph_dir),
            "graph_cache_sha256": graph_manifest["canonical_graph_cache_sha256"],
        })
        subprocess.run([sys.executable, str(REPO / "scripts/night6d_reload.py"),
                        "--run-dir", str(run)], cwd=REPO, check=True)
        reload_audit = json.loads((run / "checkpoint_reload_audit.json").read_text())
        resources = runtime_resources(started)
        artifacts = {path.name: file_row(path) for path in sorted(run.iterdir())
                     if path.is_file() and path.name != "run_manifest.json"}
        manifest = {
            "schema_version": 1,
            "ordinal": ordinal,
            "dataset": dataset,
            "graph_id": graph_id,
            "seed": int(seed),
            "status": "success",
            "fresh_scientific_training": True,
            "attempt": attempt,
            "canonical_training_config": config,
            "canonical_training_config_sha256": canonical_json_sha(config),
            "base_cache_manifest_sha256": base_manifest_sha,
            "graph_cache_sha256": graph_manifest["canonical_graph_cache_sha256"],
            "canonical_input_and_cache_sha256": cache_identity,
            "ordered_observation_sha256": observation_sha(prepared.obs_names.astype(str)),
            "initial_state_sha256": result.initial_state_sha256,
            "final_tensor_state_sha256": state_sha,
            "checkpoint_file_sha256": sha256_file(run / "model_final.pt"),
            "checkpoint_round_trip_pass": reload_audit["status"] == "PASS",
            "h00_cluster_reload_exact": reload_audit["h00_clusters_exact"],
            "view_contracts": view_rows,
            "coefficients": result.coefficients,
            "h00_selected_model": h00_result["selected_model"],
            "code_commit": code_commit,
            "artifacts": artifacts,
            "label_values_deserialized": False,
            "label_values_used": False,
            **resources,
        }
        atomic_json(run / "run_manifest.json", manifest)
        return {**manifest, "run_dir": str(run),
                "run_manifest_sha256": sha256_file(run / "run_manifest.json")}
    except Exception as exc:
        atomic_json(run / "failure.json", {
            "dataset": dataset, "graph_id": graph_id, "seed": int(seed),
            "ordinal": ordinal, "attempt": attempt,
            "status": "implementation_or_infrastructure_failure",
            "exception_type": type(exc).__name__, "message": str(exc),
            "label_values_deserialized": False,
        })
        raise


def main() -> None:
    if json.loads((OUT / "p0_semantic_contract.json").read_text())["status"] != "P0_SEMANTIC_PASS":
        raise RuntimeError("P0 semantic gate not passed")
    if git("status", "--porcelain", "--", "SpaLORA", "scripts", "tests"):
        raise RuntimeError("scientific code must be committed before training")
    code_commit = git("rev-parse", "HEAD")
    rows = []
    ordinal = 0
    for dataset in ("d1", "p22"):
        for graph_id in GRAPHS:
            for seed in range(10):
                ordinal += 1
                print(json.dumps({"event": "training_start", "ordinal": ordinal,
                                  "planned": 40, "dataset": dataset,
                                  "graph_id": graph_id, "seed": seed}, sort_keys=True), flush=True)
                rows.append(one(dataset, graph_id, seed, ordinal, code_commit))
    failures = sum(1 for path in RAW.glob("**/attempt_*/failure.json")
                   if json.loads(path.read_text()).get("status") == "implementation_or_infrastructure_failure")
    if failures > 8:
        raise RuntimeError("training correction budget exceeded")
    if len(rows) != 40 or any(not row["checkpoint_round_trip_pass"] or
                              not row["h00_cluster_reload_exact"] for row in rows):
        raise RuntimeError("40-unit checkpoint lock incomplete")
    atomic_json(OUT / "locked_training_manifest.json", {
        "schema_version": 1,
        "status": "LOCKED",
        "locked_before_label_access": True,
        "planned_units": 40,
        "scientific_training_units": len(rows),
        "success_count": len(rows),
        "implementation_or_infrastructure_retries": failures,
        "total_training_attempts": len(rows) + failures,
        "fixed_order": "dataset_graph_seed",
        "runs": rows,
    })
    print(json.dumps({"event": "training_total_lock", "success": len(rows),
                      "planned": 40, "retries": failures}, sort_keys=True))


if __name__ == "__main__":
    main()

