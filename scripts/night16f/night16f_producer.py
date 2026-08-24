#!/usr/bin/env python3
"""Materialize locked Night-16F attribution partitions without annotations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import scipy.sparse as sp

from SpaLORA.night15f_multiscale_expansion import prepare_expansion_evidence
from SpaLORA.night16e_tsre import partition_sha256, prepare_tsre_evidence
from SpaLORA.night16f_support_attribution import (
    PRIMARY_ATTRIBUTION_ARMS,
    registered_arms,
    run_attribution_arm,
)
from scripts.night16e.night16e_producer import config_sha256, load_config


def load_csr(archive: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            archive[f"{prefix}__data"],
            archive[f"{prefix}__indices"],
            archive[f"{prefix}__indptr"],
        ),
        shape=tuple(int(x) for x in archive[f"{prefix}__shape"]),
    )


def ordered_id_sha256(ids: np.ndarray) -> str:
    return hashlib.sha256(b"\0".join(str(x).encode("utf-8") for x in ids)).hexdigest()


def select_config(registry: dict[str, object]) -> tuple[dict[str, object], str]:
    matches = [
        candidate
        for candidate in registry["candidates"]
        if candidate["variant"] == "TSRE_FULL"
    ]
    if len(matches) != 1:
        raise ValueError("frozen registry must contain exactly one TSRE_FULL config")
    return matches[0]["config"], str(matches[0]["candidate_id"])


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    registry = json.loads(Path(args.registry).read_text())
    config_value, config_authority = select_config(registry)
    config = load_config(config_value)
    with np.load(args.carrier, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"]).astype("U")
        view1 = np.asarray(archive["view1"], dtype=np.float32)
        view2 = np.asarray(archive["view2"], dtype=np.float32)
        retained = np.asarray(archive["retained"], dtype=np.float32)
        start_ids = np.asarray(archive["start_ids"]).astype("U")
        starts = np.asarray(archive["start_partitions"], dtype=np.int32)
        graphs = tuple(load_csr(archive, f"graph{index}") for index in range(3))
        accessed_keys = sorted(archive.files)
    n = len(ids)
    if not (len(view1) == len(view2) == len(retained) == starts.shape[1] == n):
        raise ValueError("numeric carrier observation mismatch")
    if len(np.unique(ids)) != n:
        raise ValueError("non-unique carrier IDs")
    evidence = prepare_tsre_evidence(
        prepare_expansion_evidence(graphs, retained, view1, view2)
    )
    rows: list[dict[str, object]] = []
    partitions: list[np.ndarray] = []
    primary_for_seed = ("INPUT_START",) + PRIMARY_ATTRIBUTION_ARMS

    for start_index, (start_id, initial) in enumerate(zip(start_ids, starts)):
        arms = registered_arms() if start_index == 0 else primary_for_seed
        for arm in arms:
            candidate_started = time.perf_counter()
            row: dict[str, object] = {
                "candidate_id": f"{start_id}__{arm}",
                "start_id": str(start_id),
                "start_index": int(start_index),
                "start_role": "PRIMARY_AUTHORITY_OR_MEDOID" if start_index == 0 else "ROBUSTNESS_KMEANS_SEED",
                "arm": arm,
                "status": "PASS",
                "failure": "",
            }
            try:
                partition, diagnostics = run_attribution_arm(
                    initial, args.k, evidence, config, ids, arm
                )
                sizes = np.bincount(partition, minlength=int(args.k))
                if len(np.unique(partition)) != int(args.k) or np.any(sizes <= 0):
                    raise RuntimeError("candidate violated exact-K/no-empty-cluster contract")
                row.update(
                    partition_index=len(partitions),
                    partition_sha256=partition_sha256(partition),
                    initial_partition_sha256=partition_sha256(initial),
                    changed_from_initial=int(np.sum(partition != initial)),
                    cluster_sizes_full=[int(x) for x in sizes],
                    min_cluster_size_full=int(sizes.min()),
                    diagnostics=diagnostics,
                )
                partitions.append(np.asarray(partition, dtype=np.int32))
            except Exception as exc:
                row.update(
                    status="FAILED",
                    failure=f"{type(exc).__name__}: {exc}",
                    partition_index=-1,
                )
            row["wall_seconds"] = float(time.perf_counter() - candidate_started)
            rows.append(row)

    if not partitions:
        raise RuntimeError("producer generated no valid partitions")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, ids=ids, partitions=np.stack(partitions).astype(np.int32))
    temporary.replace(output)
    with np.load(output, allow_pickle=False) as replay:
        replay_ids = np.asarray(replay["ids"]).astype("U")
        replay_bank = np.asarray(replay["partitions"], dtype=np.int32)
    if not np.array_equal(replay_ids, ids):
        raise RuntimeError("artifact reload ID mismatch")
    for row in rows:
        if row["status"] == "PASS":
            observed = partition_sha256(replay_bank[int(row["partition_index"])])
            if observed != row["partition_sha256"]:
                raise RuntimeError("artifact reload partition mismatch")
    manifest = {
        "schema": "night16f-locked-attribution-producer-v1",
        "lane": args.lane,
        "data_id": args.data_id,
        "family": "RNA_CHROMATIN",
        "k": int(args.k),
        "n": int(n),
        "view1_shape": list(view1.shape),
        "view2_shape": list(view2.shape),
        "retained_shape": list(retained.shape),
        "graph_shapes_nnz": [
            {"shape": list(graph.shape), "nnz": int(graph.nnz)} for graph in graphs
        ],
        "ordered_id_sha256": ordered_id_sha256(ids),
        "carrier_path": str(Path(args.carrier).resolve()),
        "carrier_keys_accessed": accessed_keys,
        "annotation_keys_accessed": [],
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
        "frozen_config_authority": config_authority,
        "frozen_config_sha256": config_sha256(config_value),
        "arm_contract": list(registered_arms()),
        "artifact_reload": "PASS",
        "rows": rows,
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "gpu_time_seconds": 0.0,
        "peak_gpu_mib": 0.0,
    }
    output.with_suffix(".producer.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--data-id", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
