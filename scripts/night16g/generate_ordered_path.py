#!/usr/bin/env python3
"""Generate an ordered pairwise-strength continuation without annotations."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import resource
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from SpaLORA.night15f_multiscale_expansion import prepare_expansion_evidence
from SpaLORA.night16e_tsre import partition_sha256, prepare_tsre_evidence
from SpaLORA.night16f_support_attribution import run_attribution_arm
from scripts.night16e.night16e_producer import load_config
from scripts.night16f.night16f_producer import load_csr, select_config


def ordered_id_sha256(ids: np.ndarray) -> str:
    return hashlib.sha256(b"\0".join(str(value).encode() for value in ids)).hexdigest()


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    registry = json.loads(Path(args.registry).read_text(encoding="utf-8"))
    value, authority = select_config(registry)
    config = load_config(value)
    lambdas = tuple(float(value) for value in args.lambda_values.split(","))
    if sorted(lambdas) != list(lambdas) or len(set(lambdas)) != len(lambdas):
        raise ValueError("lambda path must be strictly ordered")
    with np.load(args.carrier, allow_pickle=False) as archive:
        ids = np.asarray(archive["ids"]).astype("U")
        view1 = np.asarray(archive["view1"], dtype=np.float32)
        view2 = np.asarray(archive["view2"], dtype=np.float32)
        retained = np.asarray(archive["retained"], dtype=np.float32)
        start_ids = np.asarray(archive["start_ids"]).astype("U")
        starts = np.asarray(archive["start_partitions"], dtype=np.int32)
        graphs = tuple(load_csr(archive, f"graph{index}") for index in range(3))
    evidence = prepare_tsre_evidence(prepare_expansion_evidence(graphs, retained, view1, view2))
    rows: list[dict[str, object]] = []
    partitions: list[np.ndarray] = []
    for start_index, (start_id, initial) in enumerate(zip(start_ids, starts)):
        path_id = f"UNIFORM_BETA_PATH::{start_id}"
        for path_index, multiplier in enumerate(lambdas):
            candidate_started = time.perf_counter()
            candidate_id = f"PATH_UNIFORM__{start_id}__L{path_index:02d}"
            row: dict[str, object] = {
                "candidate_id": candidate_id,
                "start_id": str(start_id),
                "start_index": int(start_index),
                "start_role": "ORDERED_CONTINUATION_START",
                "arm": "UNIFORM_MASS_MATCHED_PATH",
                "path_id": path_id,
                "path_index": int(path_index),
                "path_lambda": float(multiplier),
                "status": "PASS",
                "failure": "",
            }
            try:
                varied = replace(
                    config,
                    base=replace(
                        config.base,
                        pairwise_beta=float(config.base.pairwise_beta) * multiplier,
                    ),
                )
                partition, diagnostics = run_attribution_arm(
                    initial, args.k, evidence, varied, ids, "UNIFORM_MASS_MATCHED"
                )
                sizes = np.bincount(partition, minlength=args.k)
                if len(np.unique(partition)) != args.k or np.any(sizes <= 0):
                    raise RuntimeError("ordered path candidate violates exact K/no-empty")
                row.update(
                    partition_index=len(partitions),
                    partition_sha256=partition_sha256(partition),
                    initial_partition_sha256=partition_sha256(initial),
                    changed_from_initial=int(np.sum(partition != initial)),
                    cluster_sizes_full=[int(value) for value in sizes],
                    min_cluster_size_full=int(sizes.min()),
                    diagnostics=diagnostics,
                )
                partitions.append(np.asarray(partition, dtype=np.int32))
            except Exception as exc:
                row.update(status="FAILED", failure=f"{type(exc).__name__}: {exc}", partition_index=-1)
            row["wall_seconds"] = float(time.perf_counter() - candidate_started)
            rows.append(row)
    if not partitions:
        raise RuntimeError("ordered continuation produced no valid partitions")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    locked_partitions = np.stack(partitions).astype(np.int32)
    np.savez_compressed(output, ids=ids, partitions=locked_partitions)
    with np.load(output, allow_pickle=False) as replay:
        replay_ids = np.asarray(replay["ids"]).astype("U")
        replay_partitions = np.asarray(replay["partitions"], dtype=np.int32)
    if not np.array_equal(replay_ids, ids) or not np.array_equal(
        replay_partitions, locked_partitions
    ):
        raise RuntimeError("ordered continuation artifact reload mismatch")
    manifest = {
        "schema": "night16g-ordered-energy-continuation-v1",
        "lane": args.lane,
        "k": int(args.k),
        "n": int(len(ids)),
        "path_axis": "pairwise_beta_multiplier",
        "path_lambdas": list(lambdas),
        "path_arm": "UNIFORM_MASS_MATCHED",
        "frozen_config_authority": authority,
        "ordered_id_sha256": ordered_id_sha256(ids),
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
        "thread_limit": 1,
        "artifact_reload": "PASS",
        "rows": rows,
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    output.with_suffix(".producer.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({key: manifest[key] for key in ("lane", "n", "k", "path_lambdas", "wall_seconds")}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--lambda-values", default="0,0.25,0.5,0.75,1,1.5,2")
    parser.add_argument("--output", required=True)
    with threadpool_limits(limits=1):
        run(parser.parse_args())


if __name__ == "__main__":
    main()
