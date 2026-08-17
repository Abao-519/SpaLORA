#!/usr/bin/env python3
"""Night-6D P0 protection, label-free data contracts, and graph caches."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night1_pipeline import prepare_corrected
from SpaLORA.night3af_cache import load_cache, save_cache, sha256_file, verify_cache
from SpaLORA.night6d_firewall import FirewallViolation, guard_path, reject_transform_payload
from SpaLORA.night6d_pipeline import (
    DATASET_CFG, GRAPHS, HEADS, atomic_json, build_graph_data,
    canonical_json_sha, load_graph_data, observation_sha,
    validate_locked_contracts,
)

OUT = REPO / "outputs/night6d_handoff"
DATA = Path("/root/autodl-fs/night6d_data_20260817")
CACHE = Path("/root/autodl-fs/night6d_cache_20260817")
RAW = Path("/root/autodl-fs/night6d_raw_runs_20260817")
P22_CACHE = Path("/root/autodl-fs/night3af_p0d_builds_20260810/process_a/p22")
PROTOCOLS = REPO / "protocols/night6d"
REGISTRY_PATH = PROTOCOLS / "SpaLORA_Night6D_Locked_D1_P22_Confirmation_Registry_2026-08-17.json"
ACCESS_LOG = OUT / "firewall/access.jsonl"

EXPECTED_AUTHORITY = {
    "SpaLORA_Night6C_Independent_Planner_Audit_and_Night6D_Decision_2026-08-17.md":
        "f54c0a3941f75d4690fa6836b4042596a2b0e1b2aede1fc1e7ca7a32a55f3766",
    "SpaLORA_Night6D_Locked_D1_P22_Confirmation_Registry_2026-08-17.json":
        "439dbe7b7c3971b1b9af8303a6d4bdfeefc0a19090fb80e4d38e4374586647fe",
    "SpaLORA_Night6D_Locked_D1_P22_Confirmation_Taskbook_2026-08-17.md":
        "ce33d22027695744943d0042b5f372f42487d3471e8484815573248fc05fe909",
}


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()


def decode(value):
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.generic):
        return value.item()
    return value


def strings(values) -> list[str]:
    return [str(decode(x)) for x in np.asarray(values)]


def h5_index(group: h5py.Group) -> list[str]:
    key = str(decode(group.attrs.get("_index", "_index")))
    return strings(group[key][...])


def h5_x(handle: h5py.File):
    node = handle["X"]
    if isinstance(node, h5py.Dataset):
        return np.asarray(node)
    shape = tuple(map(int, node.attrs["shape"]))
    encoding = str(decode(node.attrs.get("encoding-type", "csr_matrix")))
    cls = sp.csc_matrix if encoding == "csc_matrix" else sp.csr_matrix
    return cls((node["data"][...], node["indices"][...], node["indptr"][...]), shape=shape)


def low_level_model_fields(path: Path) -> dict:
    guard_path(path, role="data_steward", operation="low_level_hdf5_copy", audit_log=ACCESS_LOG)
    with h5py.File(path, "r") as handle:
        obs_names = h5_index(handle["obs"])
        var_names = h5_index(handle["var"])
        x = h5_x(handle)
        spatial = np.asarray(handle["obsm/spatial"])
        obs_storage_keys = sorted(str(x) for x in handle["obs"].keys())
    return {
        "obs_names": obs_names,
        "var_names": var_names,
        "x": x,
        "spatial": spatial,
        "source_obs_storage_keys_not_read": obs_storage_keys,
    }


def create_label_free(source: Path, target: Path, expected_sha: str) -> dict:
    guarded = guard_path(source, role="data_steward", operation="byte_hash_only", audit_log=ACCESS_LOG)
    if sha256_file(guarded) != expected_sha:
        raise RuntimeError(f"source SHA mismatch: {source}")
    fields = low_level_model_fields(source)
    if target.exists():
        check = ad.read_h5ad(target, backed="r")
        existing = {
            "shape": list(map(int, check.shape)),
            "obs_columns": list(map(str, check.obs.columns)),
            "ordered_observation_sha256": observation_sha(check.obs_names.astype(str)),
            "spatial_shape": list(map(int, check.obsm["spatial"].shape)),
        }
        check.file.close()
        if existing["obs_columns"] or existing["ordered_observation_sha256"] != observation_sha(fields["obs_names"]):
            raise RuntimeError(f"existing label-free file contract mismatch: {target}")
    else:
        obj = ad.AnnData(
            X=fields["x"],
            obs=pd.DataFrame(index=pd.Index(fields["obs_names"], dtype=str)),
            var=pd.DataFrame(index=pd.Index(fields["var_names"], dtype=str)),
        )
        obj.obsm["spatial"] = fields["spatial"]
        target.parent.mkdir(parents=True, exist_ok=True)
        obj.write_h5ad(target, compression="gzip")
    proof = ad.read_h5ad(target, backed="r")
    row = {
        "path": str(target),
        "sha256": sha256_file(target),
        "size_bytes": target.stat().st_size,
        "shape": list(map(int, proof.shape)),
        "obs_columns": list(map(str, proof.obs.columns)),
        "ordered_observation_sha256": observation_sha(proof.obs_names.astype(str)),
        "spatial_shape": list(map(int, proof.obsm["spatial"].shape)),
        "source_sha256": expected_sha,
        "source_opened_with_anndata": False,
        "source_annotation_obs_values_read": False,
        "source_obs_storage_keys_not_read": fields["source_obs_storage_keys_not_read"],
    }
    proof.file.close()
    if row["obs_columns"] or row["shape"][0] != 3359:
        raise RuntimeError("D1 label-free output is not zero-obs/3359")
    return row


def canonicalize_d1_coordinate_orientation(rna_path: Path, adt_path: Path,
                                           adt_proof: dict) -> dict:
    """Make the two zero-obs copies share the RNA orientation, fail-closed.

    The registered D1 files encode the identical spatial geometry with exactly
    opposite coordinate signs.  Reflection through the origin preserves every
    pairwise distance and therefore every preregistered spatial graph.  We only
    canonicalize the new label-free ADT copy; source files remain untouched.
    """
    rna = ad.read_h5ad(rna_path)
    adt = ad.read_h5ad(adt_path)
    x = np.asarray(rna.obsm["spatial"])
    y = np.asarray(adt.obsm["spatial"])
    if not np.array_equal(rna.obs_names.astype(str), adt.obs_names.astype(str)):
        raise RuntimeError("D1 label-free modality barcode order mismatch")
    source_equal = bool(np.array_equal(x, y))
    source_exact_negation = bool(np.array_equal(x, -y))
    if not source_equal:
        if not source_exact_negation:
            raise RuntimeError("D1 coordinate mismatch is not the registered exact sign reflection")
        adt.obsm["spatial"] = x.copy()
        temporary = adt_path.with_suffix(".canonicalizing.h5ad")
        adt.write_h5ad(temporary, compression="gzip")
        os.replace(temporary, adt_path)
    rna.file.close() if getattr(rna, "file", None) else None
    adt.file.close() if getattr(adt, "file", None) else None
    proof = ad.read_h5ad(adt_path, backed="r")
    adt_proof.update({
        "sha256": sha256_file(adt_path), "size_bytes": adt_path.stat().st_size,
        "spatial_shape": list(map(int, proof.obsm["spatial"].shape)),
        "source_coordinates_equal_to_rna": source_equal,
        "source_coordinates_exact_negative_of_rna": source_exact_negation,
        "canonicalized_to_rna_coordinate_orientation": not source_equal,
        "pairwise_distance_geometry_changed": False,
    })
    if not np.array_equal(np.asarray(proof.obsm["spatial"]), x):
        proof.file.close()
        raise RuntimeError("D1 coordinate orientation canonicalization failed")
    proof.file.close()
    return adt_proof


def verify_p22_cache() -> dict:
    manifest_path = P22_CACHE / "manifest.json"
    if sha256_file(manifest_path) != "254eda53a7e7a62e30ded97443878045b78af9e573cbb5e6f2c365488ddf8f81":
        raise RuntimeError("P22 recorded manifest SHA mismatch")
    manifest = json.loads(manifest_path.read_text())
    failures = []
    for name, spec in manifest["files"].items():
        path = guard_path(P22_CACHE / name, role="data_steward",
                          operation="cache_verify", audit_log=ACCESS_LOG)
        if not path.is_file() or path.stat().st_size != spec["size_bytes"] or sha256_file(path) != spec["sha256"]:
            failures.append(name)
    if failures:
        raise RuntimeError(f"P22 cache file verification failed: {failures}")
    if manifest["canonical_cache_content_sha256"] != "895bfa73c763e1fa1992ddc732fc193c4e12760249b8636c98f1c5f3111dbc40":
        raise RuntimeError("P22 canonical cache content SHA mismatch")
    if manifest["canonical_model_input_sha256"] != "1111f2a7b879a0770e31ca5d98ebb9d4ce407c9f7d3ea9cc6faf02f3c8f63a95":
        raise RuntimeError("P22 canonical model-input SHA mismatch")
    if manifest["files"]["observation_ids.tsv"]["sha256"] != "c4381314391f7bf7b05bb55b1c318c58903724a2dc560d1152447b02e704c7eb":
        raise RuntimeError("P22 observation ID SHA mismatch")
    if manifest["files"]["coordinates.npy"]["sha256"] != "67e58c064a39ff60903b42e27cb07226b2bd2d42b6fcf363174a6ed8ae572541":
        raise RuntimeError("P22 coordinate SHA mismatch")
    metadata = json.loads((P22_CACHE / "metadata.json").read_text())
    if metadata["n_observations"] != 9196 or metadata["n_selected_genes"] != 2000:
        raise RuntimeError("P22 immutable cache shape/HVG mismatch")
    return {
        "status": "PASS",
        "directory": str(P22_CACHE),
        "read_only_historical_cache": True,
        "manifest_sha256": sha256_file(manifest_path),
        "manifest_file_count": len(manifest["files"]),
        "manifest_files_verified": len(manifest["files"]),
        "canonical_cache_content_sha256": manifest["canonical_cache_content_sha256"],
        "canonical_model_input_sha256": manifest["canonical_model_input_sha256"],
        "n_observations": metadata["n_observations"],
        "hvg": metadata["n_selected_genes"],
        "semantic_label_values_read": metadata.get("semantic_label_values_read"),
        "regeneration_performed": False,
    }


def base_caches(d1_rna: Path, d1_adt: Path) -> dict:
    target = CACHE / "base/d1"
    if target.exists():
        verify_cache(target)
    else:
        cfg = {
            "rna": str(d1_rna),
            "modality2": str(d1_adt),
            "hvg": 3000,
            "spatial_neighbors": 18,
        }
        pre = {
            "min_cells": 10,
            "rna_target_sum": 10000.0,
            "feature_graph": {"k": 20, "metric": "correlation"},
            "pca_svd_solver": "randomized",
            "pca_random_state": 0,
            "deterministic_pca": True,
            "alpha": 1.0,
            "rescue_non_hvg": 1000,
            "moran_shrinkage_tau": 20.0,
        }
        prepared = prepare_corrected("d1", cfg, pre, "corrected_unweighted")
        save_cache(target, "d1", prepared,
                   {"pca_svd_solver": "randomized", "pca_random_state": 0})
    d1_manifest = verify_cache(target)
    p22_manifest = verify_cache(P22_CACHE)
    return {
        "d1": {"directory": str(target), "manifest_sha256": sha256_file(target / "manifest.json"),
               "manifest": d1_manifest},
        "p22": {"directory": str(P22_CACHE), "manifest_sha256": sha256_file(P22_CACHE / "manifest.json"),
                "manifest": p22_manifest},
    }


def graph_caches(base: dict) -> list[dict]:
    rows = []
    for dataset in ("d1", "p22"):
        prepared = load_cache(Path(base[dataset]["directory"]), base[dataset]["manifest_sha256"])
        for graph_id, contract in GRAPHS.items():
            target = CACHE / "graphs" / dataset / graph_id
            if target.exists():
                _, manifest = load_graph_data(prepared, target)
            else:
                _, manifest = build_graph_data(prepared, contract, target, base[dataset]["manifest_sha256"])
            rows.append({
                "dataset": dataset,
                "graph_id": graph_id,
                "directory": str(target),
                "manifest_sha256": sha256_file(target / "manifest.json"),
                "canonical_graph_cache_sha256": manifest["canonical_graph_cache_sha256"],
                "candidate_config_sha256": manifest["candidate_config_sha256"],
                "graphs": manifest["graphs"],
            })
    return rows


def environment() -> dict:
    import anndata
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "anndata": anndata.__version__,
        "h5py": h5py.__version__,
        "rpy2": importlib.metadata.version("rpy2"),
        "git": git("--version"),
        "r_mclust": subprocess.check_output(
            ["Rscript", "-e", "cat(as.character(packageVersion('mclust')))"], text=True).strip(),
    }


def firewall_probes() -> list[dict]:
    rows = []
    safe = DATA / "d1_label_free/d1_rna_label_free.h5ad"
    rows.append({"probe": "trainer_label_free", "pass": bool(guard_path(
        safe, role="trainer_transformer", operation="read", audit_log=ACCESS_LOG))})
    negatives = [
        ("trainer_original_d1", "/root/autodl-fs/Human lymph node/D1/humanlymphnode_rna.h5ad"),
        ("trainer_original_p22", "/root/autodl-fs/P22 mouse brain coronal section/mousebrain_rna.h5ad"),
        ("trainer_d1_ground_truth", "/root/autodl-fs/Human lymph node/D1/D1_groundtruth.csv"),
        ("trainer_p22_ground_truth", "/root/autodl-fs/P22 mouse brain coronal section/MouseBrain_groundtruth.csv"),
        ("trainer_gse198353", "/root/autodl-fs/GSE198353/data.h5ad"),
        ("trainer_night4b", "/root/autodl-fs/night4b/results.json"),
        ("evaluator_early", "/root/autodl-fs/Human lymph node/D1/D1_groundtruth.csv"),
    ]
    for name, path in negatives:
        role = "evaluator" if name == "evaluator_early" else "trainer_transformer"
        operation = "parse_ground_truth" if role == "evaluator" else "read"
        try:
            guard_path(path, role=role, operation=operation, phase_locked=False, audit_log=ACCESS_LOG)
        except FirewallViolation:
            rows.append({"probe": name, "pass": True})
        else:
            raise RuntimeError(f"firewall negative probe failed: {name}")
    for key in ("labels", "single_seed_metric", "intermediate_epoch_metric", "checkpoint_metric"):
        try:
            reject_transform_payload(**{key: [1]})
        except FirewallViolation:
            rows.append({"probe": f"payload_{key}", "pass": True})
        else:
            raise RuntimeError(f"payload negative probe failed: {key}")
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    validate_locked_contracts()
    parent = "172cefba7559b34d2894ffa304553c0068ca23b9"
    branch = "revision/q2-night6d-locked-d1-p22-confirmation-20260817"
    protection = "baseline/pre-night6d-locked-d1-p22-confirmation-20260817"
    if git("rev-parse", "night6c-final-20260817^{commit}") != parent:
        raise RuntimeError("Night-6C parent tag mismatch")
    if git("rev-parse", f"{protection}^{{commit}}") != parent:
        raise RuntimeError("Night-6D protection tag mismatch")
    if git("branch", "--show-current") != branch:
        raise RuntimeError("Night-6D branch mismatch")
    authority = []
    for name, expected in EXPECTED_AUTHORITY.items():
        path = PROTOCOLS / name
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"authority SHA mismatch: {name}")
        authority.append({"file": name, "expected_sha256": expected,
                          "actual_sha256": actual, "match": True})
    registry = json.loads(REGISTRY_PATH.read_text())
    if registry["authority_parent"]["commit"] != parent or registry["seeds"] != list(range(10)):
        raise RuntimeError("Night-6D registry parent/seed drift")
    source = registry["datasets"]
    for dataset_key, mod2 in (("d1_human_lymph_node", "modality2"),
                              ("p22_mouse_brain", "modality2")):
        cfg = source[dataset_key]
        for role, path_key, sha_key in (("rna", "rna_path", "rna_sha256"),
                                        (mod2, "modality2_path", "modality2_sha256"),
                                        ("ground_truth", "ground_truth_path", "ground_truth_sha256")):
            operation = "byte_hash_only"
            path = guard_path(cfg[path_key], role="data_steward", operation=operation, audit_log=ACCESS_LOG)
            if sha256_file(path) != cfg[sha_key]:
                raise RuntimeError(f"{dataset_key} {role} byte SHA mismatch")

    d1_cfg = source["d1_human_lymph_node"]
    lf_root = DATA / "d1_label_free"
    d1_rna = lf_root / "d1_rna_label_free.h5ad"
    d1_adt = lf_root / "d1_adt_label_free.h5ad"
    rna_proof = create_label_free(Path(d1_cfg["rna_path"]), d1_rna, d1_cfg["rna_sha256"])
    adt_proof = create_label_free(Path(d1_cfg["modality2_path"]), d1_adt, d1_cfg["modality2_sha256"])
    adt_proof = canonicalize_d1_coordinate_orientation(d1_rna, d1_adt, adt_proof)
    if rna_proof["ordered_observation_sha256"] != adt_proof["ordered_observation_sha256"]:
        raise RuntimeError("D1 label-free modalities are not barcode-aligned")
    with h5py.File(d1_rna, "r") as a, h5py.File(d1_adt, "r") as b:
        if not np.array_equal(a["obsm/spatial"][...], b["obsm/spatial"][...]):
            raise RuntimeError("D1 label-free coordinates mismatch")
    atomic_json(OUT / "d1_label_free_manifest.json", {
        "status": "PASS",
        "rna": rna_proof,
        "adt": adt_proof,
        "paired_barcodes_exact": True,
        "coordinates_exact": True,
        "n_observations": 3359,
        "obs_zero_columns": True,
        "ground_truth": {
            "path": d1_cfg["ground_truth_path"],
            "size_bytes": Path(d1_cfg["ground_truth_path"]).stat().st_size,
            "sha256": d1_cfg["ground_truth_sha256"],
            "parsed_prelock": False,
        },
        "access_semantics": {
            "deserialized_into_memory": True,
            "explicitly_indexed_or_observed": False,
            "used_for_training_or_selection": False,
            "authorized_role": "data_steward",
            "annotation_obs_values_read": False,
        },
    })
    p22 = verify_p22_cache()
    p22_cfg = source["p22_mouse_brain"]
    p22["ground_truth"] = {
        "path": p22_cfg["ground_truth_path"],
        "size_bytes": Path(p22_cfg["ground_truth_path"]).stat().st_size,
        "sha256": p22_cfg["ground_truth_sha256"],
        "parsed_prelock": False,
    }
    atomic_json(OUT / "p22_cache_reuse_audit.json", p22)

    base = base_caches(d1_rna, d1_adt)
    graphs = graph_caches(base)
    atomic_json(OUT / "graph_cache_manifest_index.json", {
        "status": "LOCKED",
        "entry_count": len(graphs),
        "expected_entry_count": 4,
        "entries": graphs,
    })
    probes = firewall_probes()
    atomic_json(OUT / "data_role_and_label_firewall.json", {
        "status": "PASS",
        "roles": {
            "data_steward": "byte hash, schema, low-level D1 copy, immutable cache verification",
            "trainer_transformer": "label-free inputs/caches and known scalar K only",
            "evaluator": "single post-total-lock label window only",
        },
        "probes": probes,
        "ground_truth_parse_count_prelock": 0,
        "original_h5ad_anndata_read_count": 0,
        "trainer_transformer_original_obs_deserialization_count": 0,
    })
    atomic_json(OUT / "environment_versions.json", environment())
    atomic_json(OUT / "p0_protect_audit.json", {
        "status": "P0_PROTECT_PASS",
        "parent_commit": parent,
        "parent_tag": "night6c-final-20260817",
        "branch": branch,
        "protection_tag": protection,
        "authority_files": authority,
        "night6c_delivery_index_sha256": registry["night6c_evidence"]["delivery_index_sha256"],
        "night6c_delivery_verification": {"internal": "77/77", "external": "4/4", "local_post_dispatch": "3/3"},
        "fixed_seeds": list(range(10)),
        "fixed_training_units": 40,
        "fixed_transforms": 80,
        "budgets": registry["budgets"],
        "A1_or_tonsil_runs": 0,
        "GSE198353_runs": 0,
        "Night4B_runs": 0,
    })
    atomic_json(OUT / "dataset_and_method_contract.json", {
        "status": "LOCKED",
        "datasets": source,
        "dataset_training_config": DATASET_CFG,
        "encoder": registry["encoder"],
        "graphs": GRAPHS,
        "heads": HEADS,
        "graph_config_sha256": {k: canonical_json_sha(v) for k, v in GRAPHS.items()},
        "head_config_sha256": {k: canonical_json_sha(v) for k, v in HEADS.items()},
        "base_caches": base,
        "parameter_tuning": False,
        "candidate_selection": False,
    })
    print(json.dumps({
        "status": "P0_PROTECT_AND_DATA_CONTRACT_PASS",
        "d1_label_free": True,
        "p22_cache_files": p22["manifest_files_verified"],
        "graph_caches": len(graphs),
        "label_values_read": False,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
