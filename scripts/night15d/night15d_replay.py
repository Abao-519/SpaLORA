#!/usr/bin/env python3
"""Fresh-process replay of all frozen Night-15D finalists."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from SpaLORA.night15c_cluster_energy import sha256_array
from SpaLORA.night15d_reliability_energy import (
    csr_from_archive,
    edge_similarity,
    multiscale_feature_bank,
    reduce_full,
    reliability_energy_icm,
    reliability_transition,
)
from night15d_arena import (
    DATASET_FOR_LANE,
    evaluate,
    lane_semantics,
    reproduce_night15c_authority,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--night15c-registry", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition-root", type=Path)
    args = parser.parse_args()
    frozen = json.loads(args.registry.read_text(encoding="utf-8"))
    night15c = json.loads(args.night15c_registry.read_text(encoding="utf-8"))[
        "stable_configs"
    ]
    rows = []
    started_all = time.perf_counter()
    for lane, registered in frozen["lanes"].items():
        started = time.perf_counter()
        dataset = DATASET_FOR_LANE[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(
            args.banks / f"{dataset}_selected_partition_bank.npz",
            allow_pickle=False,
            mmap_mode="r",
        )
        k, labels, mask = lane_semantics(data, lane)
        config = registered["config"]
        graph = csr_from_archive(data, "graph")
        authority = reproduce_night15c_authority(data, bank, lane, k, night15c[lane])
        features = multiscale_feature_bank(
            bank[f"{lane}__retained_embedding"], data["view1"], data["view2"], graph
        )
        view1 = reduce_full(data["view1"], min(24, data["view1"].shape[1]))
        view2 = reduce_full(data["view2"], min(24, data["view2"].shape[1]))
        binary, weighted = edge_similarity(
            graph,
            data["view1"],
            data["view2"],
            str(config["edge_mode"]),
            tau=float(config["tau"]),
            dim=16,
        )
        support, support_diag = reliability_transition(
            binary, weighted, str(config["normalization"])
        )
        partition, completed, collapse, unary_diag = reliability_energy_icm(
            authority,
            support,
            k,
            float(config["beta"]),
            int(config["steps"]),
            str(config["unary_mode"]),
            features[str(config["feature"])],
            view1,
            view2,
            margin_temperature=float(config["margin_temperature"]),
            retained_weight=float(config["retained_weight"]),
            switch_penalty=float(config["switch_penalty"]),
        )
        metrics = evaluate(labels, mask, partition)
        row = {
            "lane": lane,
            "status": "PASS",
            "k": int(k),
            "total_observations": int(len(partition)),
            "evaluated_observations": int(mask.sum()),
            "ids_sha256": sha256_array(data["ids"]),
            "authority_partition_sha256": sha256_array(authority),
            "partition_sha256": sha256_array(partition),
            "observed_cardinality": int(len(np.unique(partition))),
            "cluster_sizes": np.bincount(partition, minlength=k).astype(int).tolist(),
            "changed_observations": int(np.sum(partition != authority)),
            "steps_completed": int(completed),
            "collapse_guard_triggered": bool(collapse),
            "config": config,
            "wall_seconds": time.perf_counter() - started,
            **metrics,
            **support_diag,
            **unary_diag,
        }
        expected_metrics = (registered["absolute_ari"], registered["absolute_nmi"])
        if row["partition_sha256"] != registered["partition_sha256"]:
            raise RuntimeError(f"partition mismatch for {lane}")
        if abs(row["absolute_ari"] - expected_metrics[0]) > 1e-12 or abs(
            row["absolute_nmi"] - expected_metrics[1]
        ) > 1e-12:
            raise RuntimeError(f"metric mismatch for {lane}")
        if row["observed_cardinality"] != k:
            raise RuntimeError(f"cardinality mismatch for {lane}")
        if args.partition_root is not None:
            args.partition_root.mkdir(parents=True, exist_ok=True)
            np.save(args.partition_root / f"{lane}.npy", partition, allow_pickle=False)
        rows.append(row)
        print(json.dumps({"lane": lane, "partition_sha256": row["partition_sha256"]}), flush=True)
    result = {
        "status": "PASS",
        "lane_count": len(rows),
        "rows": rows,
        "wall_seconds": time.perf_counter() - started_all,
        "labels_in_model_or_energy": 0,
        "dataset_name_reads_in_model_core": 0,
        "dense_n_by_n_count": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": "PASS", "lanes": len(rows), "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
