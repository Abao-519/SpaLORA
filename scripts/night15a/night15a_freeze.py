#!/usr/bin/env python3
"""Freeze seed-0-selected candidates and endpoint recipes before unseen seeds."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd


def file_sha256(path: Path) -> str:
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


def run(entries: list[str], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=False)
    registry = []
    for raw in entries:
        bank, candidate, dataset, path_text = raw.split("|", 3)
        path = Path(path_text)
        frame = pd.read_csv(path)
        if set(frame.model_seed.astype(int)) != {0}:
            raise RuntimeError("freeze source is not exclusively development model seed 0")
        for cluster_k, group in frame.groupby("cluster_k"):
            row = group.sort_values(
                ["absolute_ari", "absolute_nmi"], ascending=False
            ).iloc[0]
            frozen_id = f"{bank}__{candidate}__{dataset}__K{int(cluster_k)}"
            config = {
                "frozen_id": frozen_id,
                "preprocess_bank": bank,
                "candidate_id": candidate,
                "dataset": dataset,
                "cluster_k": int(cluster_k),
                "source_view": str(row.source_view),
                "filter_id": str(row.filter_id),
                "pca_dimension": int(row.pca_dimension),
                "coordinate_basis": str(row.coordinate_basis),
                "coordinate_weight": float(row.coordinate_weight),
                "algorithm": str(row.algorithm),
                "refinement_id": "NONE",
                "development_model_seed": 0,
                "development_endpoint_seed": int(row.endpoint_seed),
                "development_absolute_ari": float(row.absolute_ari),
                "development_absolute_nmi": float(row.absolute_nmi),
                "selection_rule": "highest seed0 ARI then NMI within the declared quick/full screen",
                "public_label_role": "cross-run development HPO after partition only",
                "labels_in_model_or_cluster_fit": False,
                "screen_source_path": str(path),
                "screen_source_sha256": file_sha256(path),
                "unseen_backbone_seeds": [3, 4, 5, 6, 7],
            }
            destination = output / f"{frozen_id}.json"
            atomic_json(destination, config)
            registry.append({**config, "config_path": str(destination), "config_sha256": file_sha256(destination)})
    if not 3 <= len(set(row["candidate_id"] for row in registry)) <= 5:
        raise RuntimeError("freeze registry must contain three to five mechanism candidates")
    atomic_json(output / "frozen_candidate_and_head_registry.json", {
        "status": "FORMAL_UNSEEN_SEED_FREEZE",
        "development_seeds_used_for_selection": [0],
        "unseen_backbone_seeds": [3, 4, 5, 6, 7],
        "formula_or_protocol_changes_after_freeze_allowed": False,
        "labels_in_model_loss_gradient_or_checkpoint_selection": False,
        "entries": registry,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entry", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(args.entry, Path(args.output))


if __name__ == "__main__":
    main()
