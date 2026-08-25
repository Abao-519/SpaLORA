#!/usr/bin/env python3
"""Label-closed Night-18C Stage-A producer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import scipy.sparse as sp

from SpaLORA.night18c_rsp_gtd import (
    common_kmeans_endpoint,
    compose_representation,
    l2_lowpass,
    sha256_array,
    shared_view_basis,
    solve_permuted_private,
    solve_shared_private,
    stage_a_configs,
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_graph(archive: np.lib.npyio.NpzFile, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (archive[f"{prefix}__data"], archive[f"{prefix}__indices"], archive[f"{prefix}__indptr"]),
        shape=tuple(int(x) for x in archive[f"{prefix}__shape"]),
    )


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    carrier_path = Path(args.carrier)
    with np.load(carrier_path, allow_pickle=False) as archive:
        required = {"ids", "view1", "view2", "retained", "graph0__data", "graph0__indices", "graph0__indptr", "graph0__shape"}
        missing = required - set(archive.files)
        if missing:
            raise KeyError(f"carrier missing {sorted(missing)}")
        ids = archive["ids"].astype("U")
        view1 = np.asarray(archive["view1"], dtype=np.float64)
        view2 = np.asarray(archive["view2"], dtype=np.float64)
        retained = np.asarray(archive["retained"], dtype=np.float64)
        graph = load_graph(archive, "graph0")
    if not (len(ids) == len(view1) == len(view2) == len(retained) == graph.shape[0]):
        raise ValueError("carrier observation mismatch")
    if len(np.unique(ids)) != len(ids):
        raise ValueError("carrier IDs are not unique")
    configs = stage_a_configs()
    h1, h2, alignment = shared_view_basis(view1, view2, configs[0].trend_dimension)
    fused = 0.5 * (h1 + h2)
    partitions: list[np.ndarray] = []
    records: list[dict[str, object]] = []

    def add(candidate_id: str, arm: str, config_id: str, representation: np.ndarray, diagnostics: dict[str, object]) -> None:
        partition = common_kmeans_endpoint(representation, args.k)
        records.append({
            "candidate_id": candidate_id,
            "arm": arm,
            "config_id": config_id,
            "representation_sha256": sha256_array(representation),
            "partition_sha256": sha256_array(partition),
            "min_cluster_size": int(np.bincount(partition, minlength=args.k).min()),
            "diagnostics": diagnostics,
        })
        partitions.append(partition)

    retained_representation = compose_representation(retained, None, 0.0)
    add("IDENTITY_RETAINED", "IDENTITY_RETAINED", "IDENTITY", retained_representation, {"solver": "NONE"})
    identity_fused = compose_representation(retained, fused, configs[0].trend_weight)
    add("IDENTITY_FUSED", "IDENTITY_FUSED", "IDENTITY", identity_fused, {"solver": "NONE"})
    cache: dict[tuple[str, str], tuple[np.ndarray, dict[str, object]]] = {}
    for config in configs:
        lowpass = l2_lowpass(fused, graph, strength=config.lambda_tv * 8.0, steps=8)
        add(f"{config.config_id}__L2_LOWPASS", "L2_LOWPASS", config.config_id,
            compose_representation(retained, lowpass, config.trend_weight), {"solver": "JACOBI_L2", "strength": config.lambda_tv * 8.0})
        for arm in ("GRAPH_TV_ONLY", "PRIVATE_ONLY", "FULL"):
            trend, r1, r2, diagnostics = solve_shared_private(h1, h2, graph, config, arm)
            cache[(config.config_id, arm)] = (trend, diagnostics)
            diagnostics = dict(diagnostics)
            diagnostics.update({"r1_sha256": sha256_array(r1), "r2_sha256": sha256_array(r2), "config": config.to_dict()})
            add(f"{config.config_id}__{arm}", arm, config.config_id,
                compose_representation(retained, trend, config.trend_weight), diagnostics)
        permuted, diagnostics = solve_permuted_private(h1, h2, graph, config, ids)
        diagnostics = dict(diagnostics); diagnostics["config"] = config.to_dict()
        add(f"{config.config_id}__PERMUTED_PRIVATE", "PERMUTED_PRIVATE", config.config_id,
            compose_representation(retained, permuted, config.trend_weight), diagnostics)
    if len({record["candidate_id"] for record in records}) != len(records):
        raise RuntimeError("duplicate candidate IDs")
    array = np.stack(partitions).astype(np.int32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, ids=ids, partitions=array, candidate_ids=np.asarray([x["candidate_id"] for x in records]))
    with np.load(output, allow_pickle=False) as replay:
        if not np.array_equal(replay["ids"].astype("U"), ids) or not np.array_equal(replay["partitions"], array):
            raise RuntimeError("artifact reload failed")
    manifest = {
        "schema": "night18c-rsp-gtd-stage-a-producer-v1",
        "lane": args.lane,
        "k": int(args.k),
        "n": len(ids),
        "view1_shape": list(view1.shape),
        "view2_shape": list(view2.shape),
        "retained_shape": list(retained.shape),
        "graph_shape": list(graph.shape),
        "graph_nnz": int(graph.nnz),
        "carrier_sha256": file_sha(carrier_path),
        "ordered_ids_sha256": sha256_array(ids),
        "producer_annotation_arrays_accessed": 0,
        "view_coordinate_alignment": alignment,
        "candidate_count": len(records),
        "records": records,
        "artifact_sha256": file_sha(output),
        "artifact_reload": "PASS",
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_time_seconds": 0.0,
        "peak_gpu_mib": 0.0,
        "thread_limit": 1,
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
