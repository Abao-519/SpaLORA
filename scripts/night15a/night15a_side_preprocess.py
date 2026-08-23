#!/usr/bin/env python3
"""Prepare label-free RNA+protein side-lane inputs for the unified core."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/night13b")]

import night13b_run as n13b  # noqa: E402
from SpaLORA.night13a_runner import load_h5ad_pair  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def run(dataset: str, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    spec = n13b.DATASETS[dataset]
    if spec["adapter"] != "protein":
        raise ValueError("side preprocess is restricted to the RNA+protein lane")
    rna_path = Path(str(spec["rna"]))
    other_path = Path(str(spec["other"]))
    # This loader touches paired feature matrices, identifiers and coordinates;
    # it does not accept or resolve any annotation/ground-truth argument.
    payload = load_h5ad_pair(rna_path, other_path, "protein", observation_ids=None)
    archive = output / "preprocessed_feature_level.npz"
    np.savez_compressed(
        archive,
        x1=np.asarray(payload["view1"], dtype=np.float32),
        x2=np.asarray(payload["view2"], dtype=np.float32),
        coordinates=np.asarray(payload["coordinates"], dtype=np.float64),
        ids=np.asarray(payload["ids"], dtype=str),
    )
    atomic_json(output / "side_input_preflight.json", {
        "dataset": dataset,
        "family": "RNA+protein",
        "rna_path": str(rna_path),
        "other_path": str(other_path),
        "rna_sha256": sha256_file(rna_path),
        "other_sha256": sha256_file(other_path),
        "observations": int(len(payload["ids"])),
        "x1_shape": list(np.asarray(payload["view1"]).shape),
        "x2_shape": list(np.asarray(payload["view2"]).shape),
        "coordinate_shape": list(np.asarray(payload["coordinates"]).shape),
        "labels_read": 0,
        "labels_in_preprocessing": False,
        "archive_sha256": sha256_file(archive),
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=("A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3"),
        required=True,
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.dataset, Path(args.output))


if __name__ == "__main__":
    main()
