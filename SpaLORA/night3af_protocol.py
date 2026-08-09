"""Night-3A-F integrity lock and deterministic cache protocol helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .night3ar_protocol import (
    ScientificWindow, assert_training_payload_label_free, atomic_json,
    ground_truth_csv_paths, integrity_read, record_integrity_reads, sha256_file,
    training_cfg,
)


def preprocessing_config(config: dict) -> dict:
    pre = config["preprocessing"]
    return {
        "min_cells": int(pre["min_cells"]),
        "alpha": float(pre["alpha_compatibility_only"]),
        "rescue_non_hvg": int(pre["rescue_non_hvg_compatibility_only"]),
        "moran_shrinkage_tau": float(pre["moran_shrinkage_tau_compatibility_only"]),
        "feature_graph": {"k": int(pre["feature_graph_k"]), "metric": pre["feature_graph_metric"]},
        "deterministic_pca": True,
        "pca_svd_solver": pre["pca_svd_solver"],
        "pca_random_state": int(pre["pca_random_state"]),
    }


def verify_night3af_lock(repo: Path, config_path: Path, config: dict,
                         lock: dict, output: Path, stage: str) -> List[dict]:
    reads = [integrity_read(config_path, lock["config_sha256"], "locked Night-3A-F configuration")]
    for name, expected in sorted(lock["source_sha256"].items()):
        reads.append(integrity_read(repo / name, expected, "locked source"))
    forbidden = set(ground_truth_csv_paths(config))
    for name, expected in sorted(lock["data_sha256"].items()):
        purpose = "ground-truth integrity only" if str(Path(name).resolve()) in forbidden else "training input integrity"
        reads.append(integrity_read(Path(name), expected, purpose))
    reads.append(integrity_read(repo / config["run_order"]["manifest"], lock["run_order_sha256"],
                                "exact preregistered 60-cell order"))
    cache_manifest = output / "preprocessing_cache_manifest.json"
    reads.append(integrity_read(cache_manifest, lock["preprocessing_cache_manifest_sha256"],
                                "published deterministic cache manifest"))
    if not all(row["match"] for row in reads):
        record_integrity_reads(output, stage, reads)
        raise RuntimeError("Night-3A-F integrity lock mismatch")
    record_integrity_reads(output, stage, reads)
    return reads


def load_cache_index(output: Path) -> dict:
    return json.loads((output / "preprocessing_cache_manifest.json").read_text(encoding="utf-8"))


__all__ = [
    "ScientificWindow", "assert_training_payload_label_free", "atomic_json",
    "ground_truth_csv_paths", "sha256_file", "training_cfg", "preprocessing_config",
    "verify_night3af_lock", "load_cache_index",
]
