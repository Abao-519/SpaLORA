#!/usr/bin/env python3
"""Night-6C P0 protection, data reuse, baseline specification, and graph caches."""
from __future__ import annotations
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night1_pipeline import prepare_corrected
from SpaLORA.night3af_cache import load_cache, save_cache, sha256_file, verify_cache
from SpaLORA.night6c_firewall import FirewallViolation, guard_path, reject_transform_payload
from SpaLORA.night6c_pipeline import (
    BASE_C04, DATASET_CFG, atomic_json, build_graph_data, canonical_json_sha,
    make_trainer, observation_sha, parse_registry,
)

OUT = REPO / "outputs/night6c_handoff"
CACHE = Path("/root/autodl-fs/night6c_cache_20260817")
LABEL_FREE = Path("/root/autodl-fs/night6b_data_20260817/tonsil_label_free")
A1_SOURCE_CACHE = REPO / "outputs/night3af_handoff/preprocessing_cache/a1"
REG_PATH = REPO / "protocols/night6c/SpaLORA_Night6B_Candidate_Registry_2026-08-17.json"
AUTH_PATH = REPO / "protocols/night6c/SpaLORA_Night6C_Clean_Baseline_Authority_2026-08-17.json"


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def inspect_label_free(path: Path) -> dict:
    obj = ad.read_h5ad(path, backed="r")
    row = {"path": str(path), "sha256": sha256_file(path),
           "size_bytes": path.stat().st_size, "shape": list(map(int, obj.shape)),
           "obs_columns": list(map(str, obj.obs.columns)),
           "spatial_shape": list(map(int, obj.obsm["spatial"].shape)),
           "ordered_observation_sha256": observation_sha(obj.obs_names.astype(str))}
    obj.file.close()
    if row["obs_columns"]:
        raise RuntimeError("label-free H5AD unexpectedly contains obs columns")
    return row


def build_base_caches() -> dict:
    base = CACHE / "base"
    a1_target = base / "a1"
    if a1_target.exists():
        verify_cache(a1_target)
    else:
        shutil.copytree(A1_SOURCE_CACHE, a1_target)
        verify_cache(a1_target)
    tonsil_target = base / "tonsil"
    if tonsil_target.exists():
        verify_cache(tonsil_target)
    else:
        cfg = {"rna": str(LABEL_FREE / "tonsil_s1_rna_label_free.h5ad"),
               "modality2": str(LABEL_FREE / "tonsil_s1_adt_label_free.h5ad"),
               "hvg": 3000, "spatial_neighbors": 18}
        pre = {"min_cells": 10, "rna_target_sum": 10000.0,
               "feature_graph": {"k": 20, "metric": "correlation"},
               "pca_svd_solver": "randomized", "pca_random_state": 0,
               "deterministic_pca": True, "alpha": 1.0,
               "rescue_non_hvg": 1000, "moran_shrinkage_tau": 20.0}
        prepared = prepare_corrected("tonsil", cfg, pre, "corrected_unweighted")
        save_cache(tonsil_target, "tonsil", prepared,
                   {"pca_svd_solver": "randomized", "pca_random_state": 0})
    return {dataset: {"directory": str(base / dataset),
                      "manifest_sha256": sha256_file(base / dataset / "manifest.json"),
                      "manifest": verify_cache(base / dataset)}
            for dataset in ("a1", "tonsil")}


def historical_baseline_contract(base_caches: dict) -> dict:
    night5_root = Path("/root/autodl-fs/SpaLORA-night5a")
    manifests = []
    for seed in range(5):
        path = Path("/root/autodl-fs/night5a_raw_runs_20260813/a1/C04_SHRINK25") / f"seed_{seed}/run_manifest.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        manifests.append({"seed": seed, "path": str(path), "sha256": sha256_file(path),
                          "candidate_id": row["candidate_id"], "config_sha256": row["config_sha256"],
                          "coefficients": row["frozen_coefficients"],
                          "initial_state_sha256": row["initial_state_sha256"],
                          "final_state_sha256": row["final_state_sha256"],
                          "epochs": row["epochs"], "locked_input_sha256": row["locked_input_sha256"]})
    if {x["candidate_id"] for x in manifests} != {"C04_SHRINK25"}:
        raise RuntimeError("historical C04 manifest identity drift")
    if len({x["config_sha256"] for x in manifests}) != 1 or len({x["epochs"] for x in manifests}) != 1:
        raise RuntimeError("historical C04 training semantics conflict")
    if {x["locked_input_sha256"] for x in manifests} != {"045cc0ac20ce5de4bb32d43b08cbdce957c04ad223ade1bfd6d4a4627d1b5829"}:
        raise RuntimeError("historical C04 A1 input drift")
    a1 = load_cache(Path(base_caches["a1"]["directory"]), base_caches["a1"]["manifest_sha256"])
    tonsil = load_cache(Path(base_caches["tonsil"]["directory"]), base_caches["tonsil"]["manifest_sha256"])
    models = {}
    for dataset, prepared in (("a1", a1), ("tonsil", tonsil)):
        trainer = make_trainer(prepared.data, dataset, 0, torch.device("cpu"))
        model = trainer.new_model()
        models[dataset] = {"input_dims": [int(prepared.data["features_omics1"].shape[1]),
                                            int(prepared.data["features_omics2"].shape[1])],
                           "latent_dims": [64, 64],
                           "parameter_count": int(sum(p.numel() for p in model.parameters())),
                           "parameter_names": [name for name, _ in model.named_parameters()]}
    source_files = [night5_root / "scripts/night5a_runner.py", night5_root / "SpaLORA/night5a_rnd.py",
                    night5_root / "configs/night5a_metric_rnd.json"]
    contract = {
        "status": "PASS", "encoder_id": "E00C_C04_B01_CLEAN",
        "semantic_source": "Night-5 B01/C04 registry, runner, five manifests",
        "architecture": "Night5AModel with EncoderOverallCorrected; two private encoders/decoders and three learned attentions",
        "models": models, "active_losses": ["rna_reconstruction", "modality2_reconstruction", "corr1"],
        "corr2_objective_enabled": False,
        "loss_coefficients": {"policy": "active_set_IGE frozen after initial no-label gradient probe",
                              "active_coefficient_sum": 4.0,
                              "historical_seed_values": [{"seed": x["seed"], "coefficients": x["coefficients"]} for x in manifests]},
        "attention": {"policy": "shrink_to_uniform", "learned_fraction": 0.25},
        "optimizer": {"name": "Adam", "learning_rate": 1e-4, "weight_decay": 0.0,
                      "scheduler": None, "epochs": 200},
        "preprocessing": {"variant": "corrected_unweighted", "hvg": 3000,
                          "spatial_reference_k": 18, "feature_reference_k": 20,
                          "feature_reference_metric": "correlation"},
        "initialization": "SpaLORA.preprocess.fix_seed(seed)",
        "determinism_flags": {"fixed_seeds": [0, 1, 2, 3, 4], "fixed_final_epoch": True,
                              "label_early_stop": False, "seed_search": False},
        "checkpoint_policy": "new Night-6C model_final.pt at fixed final epoch; mandatory fresh-process reload",
        "only_varying_fields": ["dataset_id", "graph_candidate_id", "seed", "content_addressed_input_and_graph_cache"],
        "historical_manifests": manifests,
        "source_files": [{"path": str(p), "sha256": sha256_file(p)} for p in source_files],
    }
    contract["canonical_baseline_spec_sha256"] = canonical_json_sha(contract)
    return contract


def graph_caches(registry: dict, base_caches: dict) -> list:
    graphs, _ = parse_registry(registry)
    rows = []
    for dataset in ("a1", "tonsil"):
        prepared = load_cache(Path(base_caches[dataset]["directory"]), base_caches[dataset]["manifest_sha256"])
        for graph_id, contract in graphs.items():
            target = CACHE / "graphs" / dataset / graph_id
            if target.exists():
                raise RuntimeError(f"refusing to overwrite graph cache: {target}")
            _, manifest = build_graph_data(prepared, contract, target, base_caches[dataset]["manifest_sha256"])
            rows.append({"dataset": dataset, "graph_id": graph_id,
                         "directory": str(target), "manifest_sha256": sha256_file(target / "manifest.json"),
                         "canonical_graph_cache_sha256": manifest["canonical_graph_cache_sha256"],
                         "graphs": manifest["graphs"]})
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    registry = json.loads(REG_PATH.read_text(encoding="utf-8"))
    graphs, heads = parse_registry(registry)
    authority = json.loads(AUTH_PATH.read_text(encoding="utf-8"))
    if git("rev-parse", "night6b-final-20260817^{commit}") != "301f49be2ddd15d2d36823c194df3549874729a4":
        raise RuntimeError("upstream final tag mismatch")
    if git("rev-parse", "baseline/pre-night6c-clean-baseline-graph-rescue-20260817^{commit}") != "301f49be2ddd15d2d36823c194df3549874729a4":
        raise RuntimeError("protection tag mismatch")
    expected = {
        "SpaLORA_Night6B_Independent_Planner_Audit_and_Night6C_Decision_2026-08-17.md": "258f5a432547484e924f43d72f41a0e997d4d102cb5d48165a6eac0d17ed1180",
        "SpaLORA_Night6C_Clean_Baseline_Authority_2026-08-17.json": "fa1e3a1e810d849aa13dbe75ba81e57508a40255441318c161dd03555c41c82a",
        "SpaLORA_Night6B_Candidate_Registry_2026-08-17.json": "fde443719dc7ae80ff7cb7a853044ee3df4b1790ac4d7b8fe4d0ec5cee66f030",
        "SpaLORA_Night6C_Clean_Baseline_Graph_and_Clustering_Rescue_Taskbook_2026-08-17.md": "261c9373b51b0193431b0bcf872f0767f760a82fc656a92aad228cae1aa6b07c",
    }
    authority_rows = []
    for name, digest in expected.items():
        path = REPO / "protocols/night6c" / name
        actual = sha256_file(path)
        if actual != digest: raise RuntimeError(f"authority SHA mismatch: {name}")
        authority_rows.append({"file": name, "expected_sha256": digest, "actual_sha256": actual, "match": True})
    protect = {
        "status": "P0_PROTECT_PASS", "upstream_commit": "301f49be2ddd15d2d36823c194df3549874729a4",
        "current_branch": git("branch", "--show-current"), "protection_tag_verified": True,
        "authority_files": authority_rows,
        "registry_graph_count": len(graphs), "registry_head_count": len(heads),
        "fixed_seeds": [0, 1, 2, 3, 4],
        "budgets": authority["budgets_override"],
        "protected_accesses": 0, "night6a_metric_or_raw_use": False,
    }
    atomic_json(OUT / "p0_protect_audit.json", protect)

    rna = inspect_label_free(LABEL_FREE / "tonsil_s1_rna_label_free.h5ad")
    adt = inspect_label_free(LABEL_FREE / "tonsil_s1_adt_label_free.h5ad")
    if rna["sha256"] != authority["valid_night6b_reuse"]["tonsil_label_free_rna_sha256"]:
        raise RuntimeError("tonsil label-free RNA SHA mismatch")
    if adt["sha256"] != authority["valid_night6b_reuse"]["tonsil_label_free_adt_sha256"]:
        raise RuntimeError("tonsil label-free ADT SHA mismatch")
    if rna["ordered_observation_sha256"] != adt["ordered_observation_sha256"]:
        raise RuntimeError("tonsil label-free modalities are not paired")
    base = build_base_caches()
    data_audit = {
        "status": "P0_DATA_REUSE_PASS", "known_k": 4, "target_column": "final_annot",
        "target_label_vector_sha256_not_opened": authority["valid_night6b_reuse"]["target_label_vector_sha256"],
        "label_free": {"rna": rna, "adt": adt}, "base_caches": base,
        "data_steward": {"deserialized_into_memory": True, "explicitly_indexed_or_observed": False,
                         "used_for_training_or_selection": False, "authorized_role": "data_steward",
                         "per_spot_labels_read_this_round": False},
        "trainer_transformer": {"deserialized_original_obs_into_memory": False,
                                "explicitly_indexed_or_observed": False,
                                "used_for_training_or_selection": False,
                                "authorized_role": "trainer_transformer"},
    }
    atomic_json(OUT / "p0_data_reuse_audit.json", data_audit)
    baseline = historical_baseline_contract(base)
    atomic_json(OUT / "p0_baseline_spec.json", baseline)
    rows = graph_caches(registry, base)
    atomic_json(OUT / "graph_cache_manifest_index.json", {
        "status": "LOCKED", "entry_count": len(rows), "expected_entry_count": 18,
        "all_candidate_specific": True, "entries": rows,
    })
    resolved = dict(registry)
    resolved["night6c_authority_override"] = authority
    resolved["tonsil_known_k_resolved"] = 4
    resolved["fresh_encoder_reference"] = baseline
    resolved["resolved_registry_sha256"] = canonical_json_sha(resolved)
    atomic_json(OUT / "candidate_registry_resolved_night6c.json", resolved)

    # Positive and negative firewall probes without opening protected data.
    firewall_rows = []
    safe = LABEL_FREE / "tonsil_s1_rna_label_free.h5ad"
    firewall_rows.append({"probe": "trainer_label_free", "pass": bool(guard_path(safe, role="trainer_transformer", operation="read"))})
    negatives = [
        ("original_tonsil", "/root/autodl-fs/datasets/human_tonsil_official/section1/s1_adata_rna.h5ad"),
        ("p22", "/root/autodl-fs/P22/mouse_brain.h5ad"),
        ("d1", "/root/autodl-fs/Human lymph node/D1/data.h5ad"),
        ("gse198353", "/root/autodl-fs/GSE198353/data.h5ad"),
        ("night4b", "/root/autodl-fs/night4b_raw/results.json"),
        ("night5d", "/root/autodl-fs/night5d_metric/results.json"),
        ("night6a", "/root/autodl-fs/night6a_raw_runs_20260814/results.json"),
    ]
    for name, path in negatives:
        try: guard_path(path, role="trainer_transformer", operation="read")
        except FirewallViolation: firewall_rows.append({"probe": name, "pass": True})
        else: raise RuntimeError(f"firewall negative probe failed: {name}")
    try: reject_transform_payload(labels=np.array([0, 1]))
    except FirewallViolation: firewall_rows.append({"probe": "label_payload", "pass": True})
    else: raise RuntimeError("label payload negative probe failed")
    atomic_json(OUT / "firewall/data_role_and_access_audit.json", {
        "status": "PASS", "probes": firewall_rows, "original_tonsil_annotation_accesses": 0,
        "protected_dataset_accesses": 0,
    })
    print(json.dumps({"status": "P0_DATA_BASELINE_GRAPH_PASS", "graph_caches": len(rows),
                      "baseline_spec_sha256": baseline["canonical_baseline_spec_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
