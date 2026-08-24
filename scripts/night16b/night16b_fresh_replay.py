#!/usr/bin/env python3
"""Fresh-process producer replay and independent benchmark evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

from SpaLORA.night16b_unified_structured_decoder import (
    RepairConfig,
    generic_repair,
    partition_sha256,
)
from scripts.night16b.night16b_candidate_pipeline import (
    LANE_DATASET,
    _feature_bank,
    encode_partition,
    graph_from_archive,
    lane_reference,
    moran_geary,
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def produce(args: argparse.Namespace) -> None:
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    partition_dir = args.output / "partitions"
    partition_dir.mkdir(exist_ok=True)
    rows = []
    for lane, record in registry["lanes"].items():
        started = time.perf_counter()
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False)
        graph = graph_from_archive(data)
        feature_bank, audit = _feature_bank(data, bank, lane, args.morphology)
        partition = np.load(args.frozen / "starts" / f"{lane}.npy", allow_pickle=False)
        if partition_sha256(partition) != record["initial_partition_sha256"]:
            raise RuntimeError(f"{lane}: start artifact SHA mismatch")
        step_diagnostics = []
        for step, raw in enumerate(record["replay_config_chain"]):
            config = RepairConfig(**raw)
            feature = feature_bank.get(config.feature_mode, feature_bank["retained"])
            partition, diagnostics = generic_repair(
                partition, feature, graph, int(record["k"]), config
            )
            step_diagnostics.append({"step": step, **diagnostics})
        digest = partition_sha256(partition)
        if digest != record["expected_partition_sha256"]:
            raise RuntimeError(f"{lane}: fresh producer replay mismatch")
        output_path = partition_dir / f"{lane}.npy"
        np.save(output_path, partition.astype(np.int32), allow_pickle=False)
        rows.append(
            {
                "lane": lane,
                "n": int(record["n"]),
                "k": int(record["k"]),
                "ordered_id_sha256": hashlib.sha256(np.asarray(data["ids"]).astype("U").tobytes()).hexdigest(),
                "initial_partition_sha256": record["initial_partition_sha256"],
                "partition_sha256": digest,
                "partition_file_sha256": file_sha(output_path),
                "graph_nnz": int(graph.nnz),
                "config_chain_sha256": hashlib.sha256(
                    json.dumps(record["replay_config_chain"], sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "wall_seconds": time.perf_counter() - started,
                "labels_opened": 0,
                "dense_n_by_n_count": 0,
                "diagnostics": step_diagnostics,
                "feature_audit": audit,
            }
        )
    manifest = {
        "status": "PASS",
        "fresh_process": True,
        "producer_only": True,
        "labels_opened": 0,
        "dense_n_by_n_count": 0,
        "lanes": rows,
    }
    (args.output / "producer_replay.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def evaluate(args: argparse.Namespace) -> None:
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    rows = []
    for lane, record in registry["lanes"].items():
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        labels, mask = lane_reference(data, lane)
        truth = encode_partition(labels[mask])
        graph = graph_from_archive(data)
        partition = np.load(args.input / "partitions" / f"{lane}.npy", allow_pickle=False)
        if partition_sha256(partition) != record["expected_partition_sha256"]:
            raise RuntimeError(f"{lane}: evaluator partition SHA mismatch")
        observed = partition[mask]
        moran, geary = moran_geary(partition, graph)
        sizes = np.bincount(partition, minlength=int(record["k"]))
        rows.append(
            {
                "lane": lane,
                "n": len(partition),
                "evaluated_observations": int(mask.sum()),
                "k": int(record["k"]),
                "absolute_ari": float(adjusted_rand_score(truth, observed)),
                "absolute_nmi": float(normalized_mutual_info_score(truth, observed)),
                "ami": float(adjusted_mutual_info_score(truth, observed)),
                "fmi": float(fowlkes_mallows_score(truth, observed)),
                "morans_i": moran,
                "gearys_c": geary,
                "min_cluster_size": int(sizes.min()),
                "cluster_sizes": json.dumps(sizes.astype(int).tolist(), separators=(",", ":")),
                "partition_sha256": partition_sha256(partition),
            }
        )
    output = pd.DataFrame(rows)
    output.to_csv(args.output, index=False)
    (args.output.parent / f"{args.output.stem}_audit.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "reference_array_open_events": len(rows),
                "labels_in_producer": 0,
                "labels_used_for_independent_evaluation": True,
                "metrics_sha256": file_sha(args.output),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    for name in ("produce", "evaluate"):
        child = sub.add_parser(name)
        child.add_argument("--registry", type=Path, required=True)
        child.add_argument("--kit", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        if name == "produce":
            child.add_argument("--banks", type=Path, required=True)
            child.add_argument("--morphology", type=Path, required=True)
            child.add_argument("--frozen", type=Path, required=True)
        else:
            child.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "produce":
        produce(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()
