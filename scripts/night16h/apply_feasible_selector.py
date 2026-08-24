#!/usr/bin/env python3
"""Apply one frozen Night-16H selector without opening evaluation labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from SpaLORA.night16g_basin_selector import candidate_similarity, partition_sha256
from SpaLORA.night16h_feasible_selector import select_candidate, select_weighted_rank


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def truth(value: object) -> bool:
    return str(value).strip().lower() == "true"


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    feature_path = Path(args.features)
    bank_path = Path(args.partition_bank)
    config_path = Path(args.config)
    records = list(csv.DictReader(feature_path.open(encoding="utf-8")))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    with np.load(bank_path, allow_pickle=False) as archive:
        ids = archive["ids"].astype("U")
        partitions = np.asarray(archive["partitions"], dtype=np.int32)
        candidate_ids = archive["candidate_ids"].astype("U")
    if candidate_ids.tolist() != [row["candidate_id"] for row in records]:
        raise ValueError("feature/bank candidate order mismatch")
    mode = config.get("formal_feasibility_mode", "SMALLEST_SCALE_INTERNAL_EDGE")
    feasible = np.asarray([truth(row[f"feasible_{mode}"]) for row in records])
    selector = config.get("selector", config.get("formal_label_free_selector"))
    if selector == "FROZEN_WEIGHTED_RANK":
        weights = config["selector_weights"]
        result = select_weighted_rank(
            records,
            feasible,
            float(weights["molecular_weight"]),
            float(weights["topology_weight"]),
            float(weights["risk_weight"]),
        )
    else:
        similarity = candidate_similarity(partitions)
        result = select_candidate(records, partitions, feasible, similarity, selector)
    index = result.selected_index
    partition = np.asarray(partitions[index], dtype=np.int32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        ids=ids,
        partition=partition,
        selected_candidate_id=np.asarray(candidate_ids[index]),
    )
    with np.load(output, allow_pickle=False) as replay:
        reload_pass = (
            np.array_equal(ids, replay["ids"].astype("U"))
            and np.array_equal(partition, replay["partition"].astype(np.int32))
            and str(replay["selected_candidate_id"]) == str(candidate_ids[index])
        )
    if not reload_pass:
        raise RuntimeError("artifact reload mismatch")
    manifest = {
        "schema": "night16h-frozen-selector-producer-v1",
        "lane": args.lane,
        "feasibility_mode": mode,
        "selector": selector,
        "selector_diagnostics": dict(result.diagnostics),
        "selected_candidate_id": str(candidate_ids[index]),
        "selected_partition_sha256": partition_sha256(partition),
        "feasible_candidate_count": result.feasible_count,
        "total_candidate_count": len(records),
        "artifact_reload": "PASS",
        "artifact_sha256": sha256(output),
        "feature_sha256": sha256(feature_path),
        "partition_bank_sha256": sha256(bank_path),
        "config_sha256": sha256(config_path),
        "producer_label_reads": 0,
        "dense_observation_by_observation_count": 0,
        "thread_limit": 1,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
    }
    manifest_path = output.with_suffix(".producer.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", required=True)
    parser.add_argument("--features", required=True)
    parser.add_argument("--partition-bank", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    with threadpool_limits(1):
        run(parser.parse_args())


if __name__ == "__main__":
    main()
