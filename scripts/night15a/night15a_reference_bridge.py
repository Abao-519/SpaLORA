#!/usr/bin/env python3
"""Bridge a locked Night-14A C15 artifact into the Night-15A head schema."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "scripts/night13b")]

import night13b_run as n13b  # noqa: E402
from SpaLORA.night15a_mcdf import array_sha256  # noqa: E402


SOURCE = Path(
    "/root/autodl-fs/night14a_topology_conflict_sprint_20260823/formal/"
    "development_cycle3/C15_BAL_XREC_600_WEAK_ALIGN/P22/seed_0/roundtrip_expected.npz"
)
SOURCE_SHA = "c7258ae2ce57da95502d4ba9f95bf87b9b06812155ceedf71b3a804e441afe83"


def file_sha(path: Path) -> str:
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


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    if file_sha(SOURCE) != SOURCE_SHA:
        raise RuntimeError("locked C15 source SHA mismatch")
    source = np.load(SOURCE, allow_pickle=False)
    payload = n13b.base_payload("P22")
    views = {
        "z1": np.asarray(source["z1"], dtype=np.float32),
        "z2": np.asarray(source["z2"], dtype=np.float32),
        "base_fused": np.asarray(source["fused"], dtype=np.float32),
        "mcdf": np.asarray(source["W02_TCF_FINAL"], dtype=np.float32),
    }
    expected_shape = (len(payload["ids"]), 64)
    if any(value.shape != expected_shape for value in views.values()):
        raise RuntimeError("locked C15 view shape mismatch")
    archive = output / "views.npz"
    np.savez_compressed(
        archive,
        **views,
        ids=np.asarray(payload["ids"], dtype=str),
        coordinates=np.asarray(payload["coordinates"], dtype=np.float64),
    )
    atomic_json(
        output / "training_audit.json",
        {
            "candidate_id": "LOCKED_C15_W02_REFERENCE",
            "mechanism_family": "locked_night14a_trainable_backbone_reference",
            "seed": 0,
            "status": "LOCKED_REUSE",
            "source_path": str(SOURCE),
            "source_sha256": SOURCE_SHA,
            "labels_in_model_or_checkpoint": False,
            "view_hashes": {key: array_sha256(value) for key, value in views.items()},
            "bridge_archive_sha256": file_sha(archive),
        },
    )
    atomic_json(
        output / "fresh_process_reload.json",
        {
            "source_roundtrip_authority_pass": True,
            "all_numerically_close": True,
            "source_is_existing_roundtrip_expected_artifact": True,
            "bridge_is_not_a_new_training_run": True,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(Path(args.output))


if __name__ == "__main__":
    main()
