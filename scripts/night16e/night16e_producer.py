#!/usr/bin/env python3
"""Materialize Night-16E candidate partitions without reading annotations."""

from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig, csr_from_archive
from SpaLORA.night15f_multiscale_expansion import ExpansionEnergyConfig, prepare_expansion_evidence
from SpaLORA.night16e_tsre import (
    TSREConfig,
    direct_energy_control,
    partition_sha256,
    prepare_tsre_evidence,
    tsre_expansion,
)


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def config_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def load_config(value: dict[str, object]) -> TSREConfig:
    base_value = dict(value["base"])
    local = ContinuousEnergyConfig(**dict(base_value.pop("local")))
    base = ExpansionEnergyConfig(local=local, **base_value)
    extra = {field.name: value[field.name] for field in fields(TSREConfig) if field.name != "base"}
    return TSREConfig(base=base, **extra)


def load_numeric_inputs(
    kit_root: Path,
    retained_root: Path,
    starts_root: Path,
    data_id: str,
    lane: str,
    k: int,
) -> dict[str, object]:
    archive_path = kit_root / f"{data_id}.npz"
    retained_path = retained_root / f"{data_id}_selected_partition_bank.npz"
    start_path = starts_root / f"{lane}.npy"
    with np.load(archive_path, allow_pickle=False) as archive:
        # Explicit allow-list: annotation arrays in the compute kit are never accessed.
        accessed = [
            "ids",
            "view1",
            "view2",
            "operator4__data",
            "operator4__indices",
            "operator4__indptr",
            "operator4__shape",
            "graph__data",
            "graph__indices",
            "graph__indptr",
            "graph__shape",
            "operator18__data",
            "operator18__indices",
            "operator18__indptr",
            "operator18__shape",
        ]
        ids = np.asarray(archive["ids"])
        view1 = np.asarray(archive["view1"], dtype=np.float32)
        view2 = np.asarray(archive["view2"], dtype=np.float32)
        graphs = tuple(csr_from_archive(archive, prefix) for prefix in ("operator4", "graph", "operator18"))
    with np.load(retained_path, allow_pickle=False) as retained_archive:
        retained_key = f"{lane}__retained_embedding"
        retained_id_key = f"{lane}__retained_embedding_id"
        retained = np.asarray(retained_archive[retained_key], dtype=np.float32)
        retained_id = str(np.asarray(retained_archive[retained_id_key]).reshape(-1)[0])
    initial = np.asarray(np.load(start_path, allow_pickle=False), dtype=np.int32)
    n = len(initial)
    if not (len(ids) == len(view1) == len(view2) == len(retained) == n):
        raise ValueError("numeric input observation mismatch")
    if len(np.unique(initial)) != int(k):
        raise ValueError("strong-start K mismatch")
    id_digest = hashlib.sha256(np.ascontiguousarray(ids).tobytes()).hexdigest()
    return {
        "ids": ids,
        "view1": view1,
        "view2": view2,
        "graphs": graphs,
        "retained": retained,
        "retained_id": retained_id,
        "initial": initial,
        "accessed_keys": accessed,
        "id_sha256": id_digest,
    }


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    registry = json.loads(Path(args.registry).read_text())
    inputs = load_numeric_inputs(
        Path(args.kit_root),
        Path(args.retained_root),
        Path(args.starts_root),
        args.data_id,
        args.lane,
        args.k,
    )
    base_evidence = prepare_expansion_evidence(
        inputs["graphs"], inputs["retained"], inputs["view1"], inputs["view2"]
    )
    evidence = prepare_tsre_evidence(base_evidence)
    partitions: list[np.ndarray] = []
    rows: list[dict[str, object]] = []

    for candidate in registry["candidates"]:
        candidate_started = time.perf_counter()
        row = {
            "candidate_id": candidate["candidate_id"],
            "profile_id": candidate["profile_id"],
            "base_id": candidate["base_id"],
            "variant": candidate["variant"],
            "config_sha256": config_sha256(candidate.get("config")),
            "status": "PASS",
            "failure": "",
        }
        try:
            if candidate["variant"] == "INPUT_STRONG_START":
                partition = np.asarray(inputs["initial"], dtype=np.int32).copy()
                diagnostics = {
                    "control": "BYTE_EXACT_INPUT_STRONG_START",
                    "producer_label_reads": 0,
                    "dense_n_by_n_count": 0,
                    "changed_observations": 0,
                }
            else:
                config = load_config(candidate["config"])
                if candidate["variant"] == "NIGHT15F_DIRECT":
                    partition, diagnostics = direct_energy_control(
                        inputs["initial"], args.k, evidence, config
                    )
                else:
                    partition, diagnostics = tsre_expansion(
                        inputs["initial"], args.k, evidence, config
                    )
            row["partition_index"] = len(partitions)
            row["partition_sha256"] = partition_sha256(partition)
            row["changed_from_strong_start"] = int(np.sum(partition != inputs["initial"]))
            row["cluster_sizes_full"] = [int(x) for x in np.bincount(partition, minlength=args.k)]
            row["diagnostics"] = diagnostics
            partitions.append(np.asarray(partition, dtype=np.int32))
        except Exception as exc:  # retained as an auditable failed row
            row["status"] = "FAILED"
            row["failure"] = f"{type(exc).__name__}: {exc}"
            row["partition_index"] = -1
        row["wall_seconds"] = float(time.perf_counter() - candidate_started)
        rows.append(row)

    if not partitions:
        raise RuntimeError("producer generated no valid partitions")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, partitions=np.stack(partitions).astype(np.int32))
    temporary.replace(output)
    with np.load(output, allow_pickle=False) as replay:
        replay_bank = np.asarray(replay["partitions"], dtype=np.int32)
    for row in rows:
        if row["status"] == "PASS":
            observed = partition_sha256(replay_bank[int(row["partition_index"])])
            if observed != row["partition_sha256"]:
                raise RuntimeError("artifact reload partition mismatch")
    manifest = {
        "schema": "night16e-producer-v1",
        "lane": args.lane,
        "data_id": args.data_id,
        "family": registry["family"],
        "k": int(args.k),
        "n": int(len(inputs["initial"])),
        "view1_shape": list(inputs["view1"].shape),
        "view2_shape": list(inputs["view2"].shape),
        "retained_shape": list(inputs["retained"].shape),
        "retained_id": inputs["retained_id"],
        "graph_shapes_nnz": [
            {"shape": list(graph.shape), "nnz": int(graph.nnz)} for graph in inputs["graphs"]
        ],
        "ordered_id_sha256": inputs["id_sha256"],
        "accessed_archive_keys": inputs["accessed_keys"],
        "annotation_keys_accessed": [],
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
        "artifact_reload": "PASS",
        "rows": rows,
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    output.with_suffix(".producer.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit-root", required=True)
    parser.add_argument("--retained-root", required=True)
    parser.add_argument("--starts-root", required=True)
    parser.add_argument("--data-id", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--k", required=True, type=int)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()

