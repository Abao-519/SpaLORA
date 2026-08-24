#!/usr/bin/env python3
"""Other-study family-default producer and independent evaluator.

The default producer does not read the held-out lane's benchmark metrics.  It
selects the most frequent, then lowest-complexity config chain among the
registered donor studies and applies it to the held-out lane's fixed authority
start.  Because those authority starts are themselves public-benchmark HPO
artifacts, this table measures decoder-parameter transfer, not an end-to-end
zero-label deployment claim.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

from SpaLORA.night16b_unified_structured_decoder import RepairConfig, generic_repair, partition_sha256
from scripts.night16b.night16b_candidate_pipeline import (
    LANE_DATASET,
    _feature_bank,
    encode_partition,
    graph_from_archive,
    lane_reference,
    moran_geary,
)


DONORS = {
    "A1": ["D1"],
    "D1": ["A1"],
    "tonsil_s1": ["tonsil_s2", "tonsil_s3"],
    "tonsil_s2": ["tonsil_s1", "tonsil_s3"],
    "tonsil_s3": ["tonsil_s1", "tonsil_s2"],
    "P22": ["MISAR_E15_5_S1"],
    "MISAR_E15_5_S1": ["P22"],
}


def canonical(chain: list[dict[str, object]]) -> str:
    return json.dumps(chain, sort_keys=True, separators=(",", ":"))


def complexity(chain: list[dict[str, object]]) -> tuple[int, int, str]:
    enabled = sum(int(bool(item.get("enabled", False))) for item in chain)
    sweeps = sum(int(item.get("boundary_refine_sweeps", 0)) for item in chain)
    return enabled, sweeps, canonical(chain)


def donor_chain(registry: dict[str, object], donors: list[str]) -> tuple[list[dict[str, object]], list[str]]:
    chains = [registry["lanes"][lane]["replay_config_chain"] for lane in donors]
    encoded = [canonical(chain) for chain in chains]
    counts = Counter(encoded)
    leaders = [chain for chain in chains if counts[canonical(chain)] == max(counts.values())]
    selected = min(leaders, key=complexity)
    return selected, donors


def produce(args: argparse.Namespace) -> None:
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    partition_dir = args.output / "partitions"
    partition_dir.mkdir(exist_ok=True)
    rows = []
    for lane, donors in DONORS.items():
        chain, donor_lanes = donor_chain(registry, donors)
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False)
        graph = graph_from_archive(data)
        features, _ = _feature_bank(data, bank, lane, args.morphology)
        partition = np.load(args.frontier / f"{lane}.npy", allow_pickle=False)
        start_sha = partition_sha256(partition)
        resolved = []
        for raw in chain:
            updated = dict(raw)
            requested = str(updated.get("feature_mode", "retained"))
            if requested not in features:
                if requested.startswith("molecular") and "molecular" in features:
                    updated["feature_mode"] = "molecular"
                else:
                    updated["feature_mode"] = "retained"
            config = RepairConfig(**updated)
            feature = features[config.feature_mode]
            partition, diagnostics = generic_repair(partition, feature, graph, int(registry["lanes"][lane]["k"]), config)
            resolved.append({"requested": raw, "resolved": updated, "diagnostics": diagnostics})
        output_path = partition_dir / f"{lane}.npy"
        np.save(output_path, partition.astype(np.int32), allow_pickle=False)
        rows.append(
            {
                "lane": lane,
                "donor_lanes": donor_lanes,
                "selection": "other-study mode then lower complexity; current-lane metrics unread",
                "start_partition_sha256": start_sha,
                "partition_sha256": partition_sha256(partition),
                "config_chain": chain,
                "resolved_steps": resolved,
                "labels_opened": 0,
                "dense_n_by_n_count": 0,
            }
        )
    (args.output / "family_default_producer.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "scope": "decoder-parameter transfer on fixed label-assisted authority starts",
                "labels_opened": 0,
                "lanes": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def evaluate(args: argparse.Namespace) -> None:
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    producer = json.loads((args.input / "family_default_producer.json").read_text(encoding="utf-8"))
    producer_by_lane = {item["lane"]: item for item in producer["lanes"]}
    rows = []
    for lane in DONORS:
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        labels, mask = lane_reference(data, lane)
        truth = encode_partition(labels[mask])
        graph = graph_from_archive(data)
        partition = np.load(args.input / "partitions" / f"{lane}.npy", allow_pickle=False)
        moran, geary = moran_geary(partition, graph)
        sizes = np.bincount(partition, minlength=int(registry["lanes"][lane]["k"]))
        rows.append(
            {
                "lane": lane,
                "donor_lanes": json.dumps(producer_by_lane[lane]["donor_lanes"], separators=(",", ":")),
                "absolute_ari": float(adjusted_rand_score(truth, partition[mask])),
                "absolute_nmi": float(normalized_mutual_info_score(truth, partition[mask])),
                "ami": float(adjusted_mutual_info_score(truth, partition[mask])),
                "fmi": float(fowlkes_mallows_score(truth, partition[mask])),
                "morans_i": moran,
                "gearys_c": geary,
                "min_cluster_size": int(sizes.min()),
                "cluster_sizes": json.dumps(sizes.astype(int).tolist(), separators=(",", ":")),
                "partition_sha256": partition_sha256(partition),
            }
        )
    pd.DataFrame(rows).to_csv(args.output, index=False)
    (args.output.parent / f"{args.output.stem}_audit.json").write_text(
        json.dumps(
            {
                "status": "PASS",
                "heldout_lane_metrics_read_by_default_producer": 0,
                "public_reference_array_open_events_in_evaluator": len(rows),
                "scope": "incremental decoder-parameter transfer on fixed Night-16A authorities",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="action", required=True)
    producer = sub.add_parser("produce")
    producer.add_argument("--registry", type=Path, required=True)
    producer.add_argument("--kit", type=Path, required=True)
    producer.add_argument("--banks", type=Path, required=True)
    producer.add_argument("--frontier", type=Path, required=True)
    producer.add_argument("--morphology", type=Path, required=True)
    producer.add_argument("--output", type=Path, required=True)
    evaluator = sub.add_parser("evaluate")
    evaluator.add_argument("--registry", type=Path, required=True)
    evaluator.add_argument("--kit", type=Path, required=True)
    evaluator.add_argument("--input", type=Path, required=True)
    evaluator.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    produce(args) if args.action == "produce" else evaluate(args)


if __name__ == "__main__":
    main()
