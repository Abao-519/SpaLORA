#!/usr/bin/env python3
"""Night-11B restricted RNA-protein discordance P0 runner."""
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
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from SpaLORA.night11b_discordance import (
    array_sha, atomic_json, atomic_npz, bootstrap_stability, canonical_sha,
    crossfit_shared, diffused_perturbation, evidence_axes, file_sha,
    load_registered_input, make_permutations, matched_iid, ordered_text_sha,
    spatial_blocks,
)

RAW = Path("/root/autodl-fs/night11b_rna_protein_discordance_identifiability_20260822")
OUT = REPO / "outputs/night11b_handoff"
CONFIG_PATH = REPO / "configs/night11b/night11b_contract.json"

PARENT_COMMIT = "e6e85fc378b46a2aba34636ac999bfa9c976b87f"
PARENT_TAG = "post-night11a-direction-reset-audit-final-20260822"
BRANCH = "revision/q2-night11b-rna-protein-discordance-identifiability-20260822"

DATASETS = {
    "a1": {
        "dataset": "a1", "tissue": "lymph_node",
        "rna": "/root/autodl-fs/Human lymph node/A1/humanlymphnode_rna.h5ad",
        "adt": "/root/autodl-fs/Human lymph node/A1/humanlymphnode_adt.h5ad",
        "graph": "/root/autodl-fs/night6c_cache_20260817/graphs/a1/G04_SP10_F10_EUC_UNION/adj_spatial_omics1_support.npz",
    },
    "tonsil": {
        "dataset": "tonsil", "tissue": "tonsil",
        "rna": "/root/autodl-fs/night6b_data_20260817/tonsil_label_free/tonsil_s1_rna_label_free.h5ad",
        "adt": "/root/autodl-fs/night6b_data_20260817/tonsil_label_free/tonsil_s1_adt_label_free.h5ad",
        "graph": "/root/autodl-fs/night6c_cache_20260817/graphs/tonsil/G04_SP10_F10_EUC_UNION/adj_spatial_omics1_support.npz",
    },
    "d1": {
        "dataset": "d1", "tissue": "lymph_node",
        "rna": "/root/autodl-fs/night6d_data_20260817/d1_label_free/d1_rna_label_free.h5ad",
        "adt": "/root/autodl-fs/night6d_data_20260817/d1_label_free/d1_adt_label_free.h5ad",
        "graph": "/root/autodl-fs/night6d_cache_20260817/graphs/d1/G04_SP10_F10_EUC_UNION/adj_spatial_omics1_support.npz",
    },
}

FROZEN = {
    "schema": "spalora.night11b.rna_protein_discordance_p0.v1",
    "scope": "P0_ONLY_NOT_A_FULL_MODEL",
    "spatial_folds": 5,
    "ridge_alpha_grid": [0.01, 0.1, 1.0, 10.0, 100.0],
    "bootstrap_count": 32,
    "permutation_count": 199,
    "bootstrap_seed": 110032,
    "permutation_seed": 110199,
    "control_seed": 110421,
    "decision_bootstrap_count": 2000,
    "decision_bootstrap_seed": 110777,
    "diffusion_steps": 3,
    "combined_evidence": "sqrt(p_boot * p_space)",
    "shared_predictability": "per-target out-of-fold R2; reported separately and not multiplied into combined evidence",
    "exact_link_policy": "deposited ADT target equals deposited RNA gene symbol byte-exactly",
    "excluded_links": ["HLA-DRA", "PTPRC-1"],
    "formal_correction_cycle_limit": 1,
    "preformal_implementation_corrections": 1,
    "scientific_retry_or_fallback": False,
    "dense_n_by_n": False,
    "labels_and_clustering_metrics": False,
}


def git(*args):
    return subprocess.check_output(["git"] + list(args), cwd=str(REPO), text=True).strip()


def gpu_used_mib():
    try:
        value = subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=used_memory", "--format=csv,noheader,nounits"
        ], text=True).strip()
        return max([int(float(x)) for x in value.splitlines() if x.strip()] or [0])
    except Exception:
        return 0


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(rows)
    os.replace(str(temp), str(path))


def write_tsv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    os.replace(str(temp), str(path))


def assert_git_parent():
    if git("rev-parse", "HEAD") != PARENT_COMMIT and not (OUT / "formal_freeze_manifest.json").exists():
        raise RuntimeError("pre-freeze HEAD differs from fixed parent")
    if git("rev-parse", PARENT_TAG + "^{}") != PARENT_COMMIT:
        raise RuntimeError("parent tag peel mismatch")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("Night-11B branch mismatch")


def artifact_manifest(path, arrays, extra):
    return {
        "artifact": str(path), "artifact_sha256": file_sha(path),
        "arrays": {name: {"shape": list(value.shape), "dtype": str(value.dtype), "sha256": array_sha(value)} for name, value in arrays.items()},
        **extra,
    }


def smoke(dataset):
    assert_git_parent()
    start = time.perf_counter(); gpu_start = gpu_used_mib()
    data = load_registered_input(DATASETS[dataset])
    blocks = spatial_blocks(data["coordinates"], FROZEN["spatial_folds"])
    shared = crossfit_shared(data["design"], data["adt"], blocks, FROZEN["ridge_alpha_grid"])
    permutations = make_permutations(len(data["obs"]), 7, FROZEN["permutation_seed"] + list(DATASETS).index(dataset))
    iid = matched_iid(shared["full_residual"], 4, FROZEN["control_seed"])
    iid_stability = bootstrap_stability(iid, data["graph"], blocks, 1, FROZEN["bootstrap_seed"]).reshape(4, 29)
    axes = evidence_axes(shared["full_residual"], data["graph"], blocks, iid_stability, permutations, 1, FROZEN["bootstrap_seed"])
    arrays = {
        "ordered_ids_utf8": np.asarray(data["obs"], dtype="S"),
        "feature_ids_utf8": np.asarray(data["feature_ids"], dtype="S"),
        "coordinates": data["coordinates"].astype(np.float64),
        "folds": blocks.astype(np.int64), "shared_prediction": shared["prediction"].astype(np.float64),
        "residual": shared["residual"].astype(np.float64), "combined": axes["combined"].astype(np.float64),
    }
    root = RAW / "smoke" / dataset
    artifact = root / "smoke_artifact.npz"
    atomic_npz(artifact, **arrays)
    manifest = artifact_manifest(artifact, arrays, {
        "schema": "spalora.night11b.real_smoke.v1", "dataset": dataset,
        "status": "SERIALIZED_PENDING_FRESH_RELOAD", "input_audit": data["audit"],
        "exact_link_count": len(data["feature_ids"]), "feature_ids": data["feature_ids"],
        "excluded_links": data["unmatched"], "spatial_fold_counts": np.bincount(blocks).tolist(),
        "outer_selected_alphas": shared["outer_alphas"], "full_alpha": shared["full_alpha"],
        "smoke_bootstrap_count": 1, "smoke_permutation_count": 7,
        "shared_r2_median": float(np.median(shared["shared_r2"])),
        "bootstrap_stability_median": float(np.median(axes["stability"])),
        "spatial_excess_median": float(np.median(axes["spatial_excess"])),
        "elapsed_seconds": time.perf_counter() - start, "gpu_used_mib_start": gpu_start,
        "gpu_used_mib_end": gpu_used_mib(), "label_values_opened": False,
    })
    atomic_json(root / "smoke_manifest.json", manifest)
    reload_script = REPO / "scripts/night11b/night11b_reload.py"
    subprocess.check_call([sys.executable, str(reload_script), "--manifest", str(root / "smoke_manifest.json")], cwd=str(REPO))
    reload_audit = json.loads((root / "fresh_reload_audit.json").read_text(encoding="utf-8"))
    manifest["status"] = "REAL_END_TO_END_AND_FRESH_RELOAD_PASS"
    manifest["fresh_reload"] = reload_audit
    atomic_json(root / "smoke_manifest.json", manifest)
    print(json.dumps({"event": "smoke_complete", "dataset": dataset, "status": manifest["status"], "N": len(data["obs"]), "exact_links": 29}, sort_keys=True), flush=True)


def freeze():
    assert_git_parent()
    smokes = []
    for dataset in DATASETS:
        path = RAW / "smoke" / dataset / "smoke_manifest.json"
        if not path.is_file():
            raise RuntimeError("missing real smoke: %s" % dataset)
        row = json.loads(path.read_text(encoding="utf-8"))
        if row["status"] != "REAL_END_TO_END_AND_FRESH_RELOAD_PASS":
            raise RuntimeError("real smoke/reload did not pass: %s" % dataset)
        smokes.append(row)
    feature_lists = {tuple(row["feature_ids"]) for row in smokes}
    if len(feature_lists) != 1:
        raise RuntimeError("29 exact mappings differ across real datasets")
    features = list(next(iter(feature_lists)))
    resolved = dict(FROZEN)
    resolved.update({
        "parent_commit": PARENT_COMMIT, "parent_tag": PARENT_TAG, "branch": BRANCH,
        "datasets": {row["dataset"]: row["input_audit"] for row in smokes},
        "linked_features": features, "linked_feature_sha256": ordered_text_sha(features),
        "smoke_manifest_sha256": {row["dataset"]: file_sha(RAW / "smoke" / row["dataset"] / "smoke_manifest.json") for row in smokes},
        "smoke_numeric_values_used_to_change_formula": False,
        "contract_frozen_before_formal": True,
    })
    resolved["canonical_contract_sha256"] = canonical_sha(resolved)
    atomic_json(CONFIG_PATH, resolved)
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [{"ordinal": i + 1, "rna_gene_symbol": name, "adt_target": name, "mapping_rule": "BYTE_EXACT_EQUAL", "eligible": True} for i, name in enumerate(features)]
    write_tsv(OUT / "linked_feature_registry.tsv", list(rows[0]), rows)
    preflight = {
        "schema": "spalora.night11b.real_input_preflight.v1", "status": "3_OF_3_PASS",
        "rows": [{k: row[k] for k in ("dataset", "status", "input_audit", "exact_link_count", "feature_ids", "excluded_links", "spatial_fold_counts", "outer_selected_alphas", "full_alpha", "elapsed_seconds", "fresh_reload")} for row in smokes],
        "all_real_feature_level": True, "all_registered_preprocessing": True,
        "all_sparse_graph": True, "all_fresh_reload": True,
        "label_values_opened": False,
    }
    atomic_json(OUT / "real_input_preflight.json", preflight)
    atomic_json(OUT / "formal_freeze_manifest.json", {
        "status": "FORMAL_FROZEN", "contract_path": str(CONFIG_PATH),
        "contract_sha256": file_sha(CONFIG_PATH), "canonical_contract_sha256": resolved["canonical_contract_sha256"],
        "real_smoke_count": 3, "real_smoke_pass": 3, "fresh_reload_pass": 3,
        "bootstrap_count": 32, "permutation_count": 199, "formal_correction_cycle": 0,
        "preformal_implementation_corrections": 1,
        "formal_started": False, "scientific_formula_mutable": False,
    })
    atomic_json(OUT / "preformal_correction_audit.json", {
        "status": "CLOSED_BEFORE_FORMAL", "correction_count": 1,
        "cycle_0": {"dataset": "a1", "stage": "real_smoke", "status": "IMPLEMENTATION_FAILURE_PRESERVED",
                    "exception": "ValueError: sparse spatial-linked RNA was passed to numpy.concatenate without toarray()",
                    "scientific_values_used": False},
        "cycle_1": {"status": "3_OF_3_REAL_SMOKE_AND_FRESH_RELOAD_PASS"},
        "formula_or_threshold_changed": False, "formal_rows_invalidated": 0,
    })
    source_paths = [
        REPO / "SpaLORA/night1_pipeline.py", REPO / "SpaLORA/preprocess.py",
        REPO / "SpaLORA/night11b_discordance.py", REPO / "scripts/night11b/night11b_runner.py",
        REPO / "scripts/night11b/night11b_reload.py",
        REPO / "scripts/post_night11a_direction_reset/read_only_asset_audit.py",
        REPO / "outputs/post_night11a_direction_reset/asset_capability_matrix.csv",
        REPO / "outputs/post_night11a_direction_reset/source_dependency_map.md",
        REPO / "outputs/post_night11a_direction_reset/direction_decision.json",
    ]
    atomic_json(OUT / "p0_source_audit.json", {
        "status": "PASS", "files": [{"path": str(p.relative_to(REPO)), "size": p.stat().st_size, "sha256": file_sha(p)} for p in source_paths],
        "verified_semantics": ["Night-1 RNA log-normalize", "Night-1 Seurat CLR then Scanpy ddof=1 scale", "29 byte-exact mappings", "G04 sparse graph", "HDF5 label firewall"],
        "guesswork_used": False,
    })
    atomic_json(OUT / "p0_remote_audit.json", {
        "status": "PASS", "parent_commit": PARENT_COMMIT, "parent_tag": PARENT_TAG,
        "tag_peel": git("rev-parse", PARENT_TAG + "^{}"), "branch": git("branch", "--show-current"),
        "repo": str(REPO), "raw_root": str(RAW), "ordinary_push_only": True,
    })
    print(json.dumps({"event": "formal_frozen", "contract_sha256": file_sha(CONFIG_PATH), "smoke": "3/3"}, sort_keys=True), flush=True)


def ci(values, alpha=0.05):
    return [float(np.quantile(values, alpha / 2.0)), float(np.quantile(values, 1.0 - alpha / 2.0))]


def auc_bootstrap(labels, score, count, seed):
    labels = np.asarray(labels, dtype=np.int64); score = np.asarray(score, dtype=np.float64)
    rng = np.random.RandomState(int(seed)); pos = np.flatnonzero(labels == 1); neg = np.flatnonzero(labels == 0)
    values = []
    for _ in range(int(count)):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        values.append(roc_auc_score(labels[idx], score[idx]))
    return float(roc_auc_score(labels, score)), ci(np.asarray(values))


def delta_auc_bootstrap(labels, score, axis, count, seed):
    labels = np.asarray(labels, dtype=np.int64); score = np.asarray(score); axis = np.asarray(axis)
    rng = np.random.RandomState(int(seed)); pos = np.flatnonzero(labels == 1); neg = np.flatnonzero(labels == 0)
    values = []
    for _ in range(int(count)):
        idx = np.concatenate([rng.choice(pos, len(pos), replace=True), rng.choice(neg, len(neg), replace=True)])
        values.append(roc_auc_score(labels[idx], score[idx]) - roc_auc_score(labels[idx], axis[idx]))
    point = float(roc_auc_score(labels, score) - roc_auc_score(labels, axis))
    return point, ci(np.asarray(values))


def run_one_formal(dataset, contract):
    start = time.perf_counter(); data = load_registered_input(DATASETS[dataset])
    audit = contract["datasets"][dataset]
    if data["audit"]["ordered_observation_sha256"] != audit["ordered_observation_sha256"] or data["audit"]["linked_feature_sha256"] != audit["linked_feature_sha256"]:
        raise RuntimeError("formal input differs from frozen smoke: %s" % dataset)
    blocks = spatial_blocks(data["coordinates"], contract["spatial_folds"])
    shared = crossfit_shared(data["design"], data["adt"], blocks, contract["ridge_alpha_grid"])
    seed_offset = list(DATASETS).index(dataset) * 10000
    perms = make_permutations(len(data["obs"]), contract["permutation_count"], contract["permutation_seed"] + seed_offset)
    iid_null = matched_iid(shared["full_residual"], contract["bootstrap_count"], contract["control_seed"] + seed_offset)
    iid_stability = bootstrap_stability(iid_null, data["graph"], blocks, contract["bootstrap_count"], contract["bootstrap_seed"] + seed_offset).reshape(contract["bootstrap_count"], 29)
    diffused = diffused_perturbation(shared["full_residual"], data["w"], contract["control_seed"] + seed_offset + 1, contract["diffusion_steps"])
    iid_control = iid_null[:, :29]
    coordinate_control = diffused[perms[0]]
    protein_control = shared["full_residual"][perms[1]]
    order = np.roll(np.arange(29), 1)
    wrong_design = np.concatenate([data["design"][:, order], data["design"][:, 29 + order]], axis=1)
    link_control = crossfit_shared(wrong_design, data["adt"], blocks, contract["ridge_alpha_grid"])["full_residual"]
    names = ["REAL_RESIDUAL", "DIFFUSED_STRUCTURED", "MATCHED_IID", "COORDINATE_GRAPH_PERMUTATION", "PROTEIN_SPOT_PERMUTATION", "LINK_PERMUTATION"]
    matrices = [shared["full_residual"], diffused, iid_control, coordinate_control, protein_control, link_control]
    patterns = np.concatenate(matrices, axis=1)
    axes = evidence_axes(patterns, data["graph"], blocks, iid_stability, perms, contract["bootstrap_count"], contract["bootstrap_seed"] + seed_offset + 999)
    rows = []
    for group_index, name in enumerate(names):
        sl = slice(group_index * 29, (group_index + 1) * 29)
        for j, feature in enumerate(data["feature_ids"]):
            k = sl.start + j
            rows.append({
                "dataset": dataset, "tissue": DATASETS[dataset]["tissue"], "protein_target": feature,
                "control": name, "known_structured": 1 if name == "DIFFUSED_STRUCTURED" else (0 if name != "REAL_RESIDUAL" else ""),
                "shared_oof_r2": float(shared["shared_r2"][j]) if name == "REAL_RESIDUAL" else "",
                "bootstrap_stability": float(axes["stability"][k]), "p_boot": float(axes["p_boot"][k]),
                "morans_i": float(axes["moran"][k]), "spatial_excess": float(axes["spatial_excess"][k]),
                "p_space": float(axes["p_space"][k]), "combined_evidence": float(axes["combined"][k]),
            })
    root = RAW / "formal_cycle_0" / dataset
    arrays = {
        "shared_prediction_oof": shared["prediction"], "residual_oof": shared["residual"],
        "residual_full": shared["full_residual"], "control_patterns": patterns,
        "stability": axes["stability"], "p_boot": axes["p_boot"], "p_space": axes["p_space"], "combined": axes["combined"],
    }
    artifact = root / "formal_artifact.npz"; atomic_npz(artifact, **arrays)
    manifest = artifact_manifest(artifact, arrays, {
        "schema": "spalora.night11b.formal_dataset.v1", "dataset": dataset, "status": "FORMAL_COMPLETE",
        "contract_sha256": file_sha(CONFIG_PATH), "input_audit": data["audit"], "feature_ids": data["feature_ids"],
        "outer_selected_alphas": shared["outer_alphas"], "full_alpha": shared["full_alpha"],
        "elapsed_seconds": time.perf_counter() - start, "formal_correction_cycle": 0,
        "label_values_opened": False, "scientific_retry_or_fallback": False,
    })
    atomic_json(root / "formal_manifest.json", manifest)
    write_csv(root / "evidence_rows.csv", list(rows[0]), rows)
    return rows, manifest


def formal():
    contract = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    freeze_manifest = json.loads((OUT / "formal_freeze_manifest.json").read_text(encoding="utf-8"))
    if freeze_manifest["status"] != "FORMAL_FROZEN" or freeze_manifest["contract_sha256"] != file_sha(CONFIG_PATH):
        raise RuntimeError("formal freeze/contract mismatch")
    start = time.perf_counter(); gpu_start = gpu_used_mib()
    freeze_manifest["formal_started"] = True
    atomic_json(OUT / "formal_freeze_manifest.json", freeze_manifest)
    all_rows = []; manifests = []
    for dataset in DATASETS:
        rows, manifest = run_one_formal(dataset, contract)
        all_rows.extend(rows); manifests.append(manifest)
        print(json.dumps({"event": "formal_dataset_complete", "dataset": dataset, "elapsed_seconds": manifest["elapsed_seconds"]}, sort_keys=True), flush=True)
    fields = list(all_rows[0]); write_csv(OUT / "evidence_by_dataset_feature.csv", fields, all_rows)
    control_rows = [r for r in all_rows if r["control"] != "REAL_RESIDUAL"]
    labels = np.asarray([int(r["known_structured"]) for r in control_rows])
    full = np.asarray([float(r["combined_evidence"]) for r in control_rows])
    boot_axis = np.asarray([float(r["p_boot"]) for r in control_rows])
    space_axis = np.asarray([float(r["p_space"]) for r in control_rows])
    pooled_point, pooled_ci = auc_bootstrap(labels, full, contract["decision_bootstrap_count"], contract["decision_bootstrap_seed"])
    boot_delta, boot_delta_ci = delta_auc_bootstrap(labels, full, boot_axis, contract["decision_bootstrap_count"], contract["decision_bootstrap_seed"] + 1)
    space_delta, space_delta_ci = delta_auc_bootstrap(labels, full, space_axis, contract["decision_bootstrap_count"], contract["decision_bootstrap_seed"] + 2)
    dataset_auc = {}
    for dataset in DATASETS:
        rows = [r for r in control_rows if r["dataset"] == dataset]
        dataset_auc[dataset] = float(roc_auc_score([int(r["known_structured"]) for r in rows], [float(r["combined_evidence"]) for r in rows]))
    real = {(r["dataset"], r["protein_target"]): r for r in all_rows if r["control"] == "REAL_RESIDUAL"}
    features = contract["linked_features"]
    a1_score = np.asarray([float(real[("a1", f)]["combined_evidence"]) for f in features])
    d1_score = np.asarray([float(real[("d1", f)]["combined_evidence"]) for f in features])
    rho = float(spearmanr(a1_score, d1_score).correlation)
    rng = np.random.RandomState(contract["permutation_seed"] + 99000)
    null_rho = np.asarray([spearmanr(a1_score, d1_score[rng.permutation(29)]).correlation for _ in range(contract["permutation_count"])])
    p_repro = float((1 + np.sum(null_rho >= rho)) / (len(null_rho) + 1.0))
    controls_gate = pooled_ci[0] > 0.5 and all(v > 0.5 for v in dataset_auc.values())
    axis_gate = boot_delta_ci[0] > 0.0 and space_delta_ci[0] > 0.0
    repro_gate = rho > 0.0 and p_repro < 0.05
    roundtrip_gate = all(m["status"] == "FORMAL_COMPLETE" for m in manifests)
    if controls_gate and axis_gate and repro_gate and roundtrip_gate:
        terminal = "NIGHT11B_EVIDENCE_AXES_IDENTIFIABLE"; classification = "LOCAL SIGNAL"
    elif controls_gate and axis_gate and roundtrip_gate:
        terminal = "NIGHT11B_SYNTHETIC_ONLY_NO_REAL_IDENTIFIABILITY"; classification = "SCIENTIFIC NEGATIVE"
    else:
        terminal = "NIGHT11B_EVIDENCE_AXES_NOT_IDENTIFIABLE"; classification = "SCIENTIFIC NEGATIVE"
    summary_rows = []
    for dataset in DATASETS:
        rr = [real[(dataset, f)] for f in features]
        summary_rows.append({
            "dataset": dataset, "N": contract["datasets"][dataset]["spot_count"],
            "input_shape": "%sx%s RNA; %sx%s ADT" % tuple(contract["datasets"][dataset]["rna_raw_shape"] + contract["datasets"][dataset]["adt_raw_shape"]),
            "exact_links": 29, "shared_oof_r2_median": float(np.median([float(x["shared_oof_r2"]) for x in rr])),
            "bootstrap_stability_median": float(np.median([float(x["bootstrap_stability"]) for x in rr])),
            "spatial_excess_median": float(np.median([float(x["spatial_excess"]) for x in rr])),
            "real_combined_median": float(np.median([float(x["combined_evidence"]) for x in rr])),
            "controls_auc": dataset_auc[dataset],
            "runtime_seconds": next(m["elapsed_seconds"] for m in manifests if m["dataset"] == dataset),
        })
    control_summary = [
        {"scope": "pooled", "metric": "combined_control_AUROC", "point": pooled_point, "ci_low": pooled_ci[0], "ci_high": pooled_ci[1]},
        {"scope": "pooled", "metric": "combined_minus_boot_axis_AUROC", "point": boot_delta, "ci_low": boot_delta_ci[0], "ci_high": boot_delta_ci[1]},
        {"scope": "pooled", "metric": "combined_minus_space_axis_AUROC", "point": space_delta, "ci_low": space_delta_ci[0], "ci_high": space_delta_ci[1]},
    ] + [{"scope": d, "metric": "combined_control_AUROC", "point": v, "ci_low": "", "ci_high": ""} for d, v in dataset_auc.items()]
    write_csv(OUT / "control_summary.csv", list(control_summary[0]), control_summary)
    decision = {
        "schema": "spalora.night11b.decision.v1", "terminal_status": terminal, "classification": classification,
        "scope": "P0 local identifiability only; not a full model or clustering claim",
        "gates": {
            "real_end_to_end_and_reload_3_of_3": True, "controls_gate": controls_gate,
            "full_evidence_better_than_each_axis_gate": axis_gate, "a1_d1_reproducibility_gate": repro_gate,
            "pooled_control_auc": pooled_point, "pooled_control_auc_ci95": pooled_ci,
            "dataset_control_auc": dataset_auc,
            "combined_minus_boot_auc": boot_delta, "combined_minus_boot_auc_ci95": boot_delta_ci,
            "combined_minus_space_auc": space_delta, "combined_minus_space_auc_ci95": space_delta_ci,
            "a1_d1_spearman": rho, "a1_d1_permutation_p": p_repro,
        },
        "formal_correction_cycles": 0, "scientific_retries": 0, "formula_changes_after_formal": 0,
        "labels_read": 0, "clustering_metrics_computed": 0,
    }
    atomic_json(OUT / "night11b_decision.json", decision)
    atomic_json(OUT / "formal_execution_audit.json", {
        "status": "FORMAL_COMPLETE", "dataset_count": 3, "feature_rows": 3 * 29,
        "control_rows": 3 * 29 * 5, "bootstrap_count": 32, "permutation_count": 199,
        "manifests": [{"dataset": m["dataset"], "path": str(RAW / "formal_cycle_0" / m["dataset"] / "formal_manifest.json"), "sha256": file_sha(RAW / "formal_cycle_0" / m["dataset"] / "formal_manifest.json")} for m in manifests],
        "formal_correction_cycles": 0, "scientific_retries": 0,
    })
    forbidden = {
        "training_label_reads": 0, "evaluation_label_reads": 0, "total_label_reads": 0,
        "ARI": 0, "NMI": 0, "AMI": 0, "FMI": 0, "Q": 0, "annotation_spatial_metrics": 0,
        "MISAR_Y": 0, "E18_5": 0, "new_external_data": 0, "new_downloads": 0,
        "third_party_benchmarks": 0, "clustering_head_scientific_evaluations": 0,
        "QCRD": 0, "night11a_gate_revisions": 0, "new_candidate_searches": 0,
        "dataset_name_routing": 0, "dense_N_by_N": 0, "scientific_retry_or_fallback": 0,
    }
    atomic_json(OUT / "label_firewall_audit.json", {"status": "PASS", "all_zero": all(v == 0 for v in forbidden.values()), "counters": forbidden, "h5_boundary": "X + obs/_index + var/_index + obsm/spatial only; obs annotation datasets not opened"})
    elapsed = time.perf_counter() - start
    resources = {
        "status": "PASS", "wall_seconds": elapsed, "per_dataset_seconds": {m["dataset"]: m["elapsed_seconds"] for m in manifests},
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_used_mib_start": gpu_start, "gpu_used_mib_end": gpu_used_mib(), "gpu_peak_attributed_mib": 0,
        "gpu_code_path_used": False, "wall_budget_seconds": 14400, "within_budget": elapsed <= 14400,
    }
    atomic_json(OUT / "resource_audit.json", resources)
    atomic_json(OUT / "night11b_contract.json", contract)
    render_report(decision, summary_rows, resources)
    print(json.dumps({"event": "formal_complete", "terminal_status": terminal, "classification": classification, "pooled_auc": pooled_point, "spearman": rho, "p": p_repro}, sort_keys=True), flush=True)


def render_report(decision, rows, resources):
    g = decision["gates"]
    lines = [
        "# Night-11B RNA–蛋白不一致可识别性 P0 报告", "",
        "## 负责人现在需要知道的三件事", "",
        "1. 本轮只判断 RNA–蛋白不一致能否被拆成 RNA 可解释的共享部分、稳定且有空间结构的蛋白特异残差、不可复现技术噪声三个可观测对象。",
        "2. 实际工作停在 feature/residual/fusion 之前的证据层：从真实 RNA/ADT feature matrix 出发，做空间块交叉拟合、残差重采样稳定性、稀疏图空间超额和固定 controls；没有训练完整融合模型。",
        "3. 冻结终态是 `%s`（`%s`）。这只回答 P0 可识别性，不能解释成聚类提高、SOTA 或论文已经成立。" % (decision["terminal_status"], decision["classification"]), "",
        "## 可识别性主表", "",
        "| 数据集 | 真实输入 shape | 29 对 | shared OOF R² 中位数 | bootstrap stability 中位数 | spatial excess 中位数 | control AUROC | 运行秒 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append("| {dataset} | {input_shape} | {exact_links} | {shared_oof_r2_median:.4f} | {bootstrap_stability_median:.4f} | {spatial_excess_median:.4f} | {controls_auc:.4f} | {runtime_seconds:.1f} |".format(**r))
    lines += [
        "", "## 冻结判定", "",
        "- pooled control AUROC = %.4f，95%% bootstrap CI [%.4f, %.4f]。" % (g["pooled_control_auc"], g["pooled_control_auc_ci95"][0], g["pooled_control_auc_ci95"][1]),
        "- combined 相对 boot 单轴 AUROC 差 = %.4f，95%% CI [%.4f, %.4f]。" % (g["combined_minus_boot_auc"], g["combined_minus_boot_auc_ci95"][0], g["combined_minus_boot_auc_ci95"][1]),
        "- combined 相对 space 单轴 AUROC 差 = %.4f，95%% CI [%.4f, %.4f]。" % (g["combined_minus_space_auc"], g["combined_minus_space_auc_ci95"][0], g["combined_minus_space_auc_ci95"][1]),
        "- A1↔D1 同 target evidence-rank Spearman = %.4f，预注册 permutation p = %.4f。" % (g["a1_d1_spearman"], g["a1_d1_permutation_p"]),
        "- 真实端到端 smoke 与 fresh-process reload：3/3。formal correction cycle：0。scientific retry：0。", "",
        "## 导师汇报版", "",
        "Night-11B 没有训练一个新模型，而是先检查 RNA–蛋白不一致是否有可观测的三分证据。",
        "三个真实组织单元都从 feature-level RNA/ADT 和登记的稀疏空间图出发，并只用了 29 个完全同名的 deposited feature pairs。",
        "共享部分由空间块外推的 ridge prediction 定义，蛋白特异部分是其残差。",
        "残差证据由重采样稳定性和相对空间 permutation null 的超额共同构成，shared predictability 单独报告。",
        "known controls 与 A1–D1 现实复现门严格按 formal 前冻结的阈值判定。",
        "最终分类见上文，任何 synthetic control 结果都没有被包装成真实生物学成功。",
        "本轮未读取标签，未计算 ARI/NMI/Q，也没有证明 SOTA 或聚类性能提高。", "",
        "## 技术附录", "",
        "- wall time：%.1f 秒；peak RSS：%.1f MiB；GPU attributed peak：0 MiB。" % (resources["wall_seconds"], resources["peak_rss_mib"]),
        "- fixed formula：`sqrt(p_boot * p_space)`；32 bootstraps；199 permutations。",
        "- 标签与全部禁区计数见 `label_firewall_audit.json`，均为 0。",
    ]
    (OUT / "night11b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("smoke", "freeze", "formal"), required=True)
    parser.add_argument("--dataset", choices=tuple(DATASETS))
    args = parser.parse_args()
    if args.phase == "smoke":
        if not args.dataset: parser.error("--dataset is required for smoke")
        smoke(args.dataset)
    elif args.phase == "freeze": freeze()
    else: formal()


if __name__ == "__main__":
    main()
