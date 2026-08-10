"""Night-3B byte locks, immutable-cache routing, and label firewall helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .night3ar_protocol import (
    ScientificWindow,
    assert_training_payload_label_free,
    atomic_json,
    ground_truth_csv_paths,
    integrity_read,
    record_integrity_reads,
    sha256_file,
    training_cfg,
)


def load_cache_index(config: dict) -> dict:
    return json.loads(Path(config["paths"]["cache_manifest"]).read_text(encoding="utf-8"))


def cache_directory(config: dict, row: dict) -> Path:
    return Path(config["paths"]["night3af_root"]) / row["directory"]


def verify_night3b_lock(repo: Path, config_path: Path, config: dict,
                        lock: dict, output: Path, stage: str) -> List[dict]:
    reads = [integrity_read(config_path, lock["config_sha256"], "locked Night-3B configuration")]
    for name, expected in sorted(lock["source_sha256"].items()):
        reads.append(integrity_read(repo / name, expected, "locked Night-3B source"))
    forbidden = set(ground_truth_csv_paths(config))
    for name, expected in sorted(lock["data_sha256"].items()):
        purpose = "ground-truth integrity only" if str(Path(name).resolve()) in forbidden else "training input integrity"
        reads.append(integrity_read(Path(name), expected, purpose))
    reads.append(integrity_read(
        repo / config["run_order"]["manifest"], lock["run_order_sha256"],
        "exact preregistered 120-cell order",
    ))
    reads.append(integrity_read(
        Path(config["paths"]["cache_manifest"]), lock["cache_manifest_sha256"],
        "published immutable Night-3AF cache manifest",
    ))
    if not all(row["match"] for row in reads):
        record_integrity_reads(output, stage, reads)
        raise RuntimeError("Night-3B integrity lock mismatch")
    record_integrity_reads(output, stage, reads)
    return reads


__all__ = [
    "ScientificWindow", "assert_training_payload_label_free", "atomic_json",
    "ground_truth_csv_paths", "sha256_file", "training_cfg", "load_cache_index",
    "cache_directory", "verify_night3b_lock",
]
