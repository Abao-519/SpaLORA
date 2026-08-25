#!/usr/bin/env python3
"""Materialize label-free Night-18D placenta transfer candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from threadpoolctl import threadpool_limits

from SpaLORA.night15f_multiscale_expansion import prepare_expansion_evidence
from SpaLORA.night18d_placenta_transfer import (
    PRIMARY_ARMS,
    config_from_dict,
    medoid_index,
    run_arm,
    sha256_array,
    structure_feasibility,
)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def graph_from_npz(carrier, index: int) -> sp.csr_matrix:
    graph = sp.csr_matrix(
        (carrier[f"graph{index}__data"], carrier[f"graph{index}__indices"], carrier[f"graph{index}__indptr"]),
        shape=tuple(int(value) for value in carrier[f"graph{index}__shape"]),
    )
    graph.sort_indices()
    return graph


def json_safe(value):
    if isinstance(value, dict): return {str(key): json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(child) for child in value]
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    carrier_path, config_path = Path(args.carrier), Path(args.config)
    frozen = json.loads(config_path.read_text(encoding="utf-8"))
    if frozen["status"] != "FROZEN_BEFORE_PLACENTA_EVALUATION" or frozen["placenta_metrics_read"] != 0:
        raise RuntimeError("family profile was not frozen before placenta evaluation")
    with np.load(carrier_path, allow_pickle=False) as carrier:
        ids = carrier["ids"].astype("U")
        retained = carrier["retained"].astype(np.float32)
        view1 = carrier["view1"].astype(np.float32)
        view2 = carrier["view2"].astype(np.float32)
        start_names = carrier["start_names"].astype("U")
        starts = carrier["start_partitions"].astype(np.int32)
        graphs = tuple(graph_from_npz(carrier, index) for index in range(3))
    if len(ids) != 1662 or starts.shape != (13, 1662) or len(np.unique(start_names)) != 13:
        raise RuntimeError("carrier/start authority mismatch")
    if any(graph.shape != (1662, 1662) for graph in graphs): raise RuntimeError("graph shape mismatch")
    with threadpool_limits(limits=1):
        evidence = prepare_expansion_evidence(graphs, retained, view1, view2)
    if args.mode == "p0":
        profiles = [(frozen["primary_profile_id"], frozen["primary_config"])]
        arm_names, start_indices = ("NO_OP_START", "L2_LOWPASS_MATCHED", "FULL_FROZEN_ENERGY"), range(2)
    else:
        profiles = [(frozen["primary_profile_id"], frozen["primary_config"])] + [(name, value) for name, value in sorted(frozen["sensitivity_profiles"].items())]
        arm_names, start_indices = PRIMARY_ARMS, range(len(starts))
    records, partitions, candidate_ids = [], [], []
    for profile_index, (profile_id, raw_config) in enumerate(profiles):
        config = config_from_dict(raw_config)
        selected_arms = arm_names if profile_index == 0 else ("FULL_FROZEN_ENERGY",)
        for arm in selected_arms:
            for start_index in start_indices:
                candidate_id = f"{profile_id}__{arm}__{start_names[start_index]}"
                with threadpool_limits(limits=1):
                    partition, diagnostics = run_arm(starts[start_index], args.k, evidence, retained, graphs[1], config, arm)
                partition = np.asarray(partition, dtype=np.int32)
                if len(np.unique(partition)) != args.k or not np.isfinite(partition).all():
                    raise RuntimeError(f"invalid partition: {candidate_id}")
                feasibility = structure_feasibility(partition, graphs[0], args.k)
                records.append({
                    "candidate_id": candidate_id, "profile_id": profile_id, "arm": arm,
                    "start_index": int(start_index), "start_name": str(start_names[start_index]),
                    "partition_sha256": sha256_array(partition), "input_start_sha256": sha256_array(starts[start_index]),
                    "changed_observations": int(np.sum(partition != starts[start_index])),
                    "feasibility": feasibility, "diagnostics": json_safe(diagnostics),
                })
                candidate_ids.append(candidate_id); partitions.append(partition)
    partitions = np.stack(partitions).astype(np.int32)
    if len(set(candidate_ids)) != len(candidate_ids): raise RuntimeError("duplicate candidate ID")
    selections = {}
    if args.mode == "formal":
        primary = [index for index, row in enumerate(records) if row["profile_id"] == frozen["primary_profile_id"]]
        for arm in PRIMARY_ARMS:
            indices = [index for index in primary if records[index]["arm"] == arm]
            feasible = [index for index in indices if records[index]["feasibility"]["feasible"]]
            selections[f"{arm}__STRUCTURE_FEASIBLE_MEDOID"] = medoid_index(partitions, feasible)
            selections[f"{arm}__PLAIN_MEDOID"] = medoid_index(partitions, indices)
        for profile_id, _ in profiles[1:]:
            indices = [index for index, row in enumerate(records) if row["profile_id"] == profile_id]
            feasible = [index for index in indices if records[index]["feasibility"]["feasible"]]
            selections[f"{profile_id}__FULL__STRUCTURE_FEASIBLE_MEDOID"] = medoid_index(partitions, feasible)
    artifact = Path(args.output); artifact.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact, ids=ids, candidate_ids=np.asarray(candidate_ids, dtype="U"), partitions=partitions,
        selection_names=np.asarray(list(selections), dtype="U"), selection_indices=np.asarray(list(selections.values()), dtype=np.int32),
    )
    with np.load(artifact, allow_pickle=False) as replay:
        if not np.array_equal(replay["ids"], ids) or not np.array_equal(replay["partitions"], partitions):
            raise RuntimeError("artifact reload mismatch")
    manifest = {
        "status": "P0_PRODUCER_LOCKED" if args.mode == "p0" else "FORMAL_PRODUCER_LOCKED_BEFORE_LABEL_EVALUATION",
        "schema": "night18d-placenta-transfer-producer-v1", "mode": args.mode,
        "carrier_sha256": file_sha(carrier_path), "family_config_sha256": file_sha(config_path),
        "ordered_ids_sha256": sha256_array(ids), "candidate_count": len(records), "partition_matrix_sha256": sha256_array(partitions),
        "candidate_ids_sha256": sha256_array(np.asarray(candidate_ids, dtype="U")), "records": records, "selections": selections,
        "annotation_columns_accessed": 0, "labels_read": 0, "dense_n_by_n_created": 0,
        "artifact_reload": "PASS", "artifact_sha256": file_sha(artifact), "wall_seconds": time.perf_counter() - started,
    }
    artifact.with_suffix(".json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--carrier", required=True); parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("p0", "formal"), required=True); parser.add_argument("--k", type=int, default=10); parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__": main()
