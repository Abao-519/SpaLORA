#!/usr/bin/env python3
"""Label-free producer for Night-18E CCSR candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from SpaLORA.night15e_continuous_reliability_energy import ContinuousEnergyConfig
from SpaLORA.night15f_multiscale_expansion import (
    ExpansionEnergyConfig,
    continuous_multiscale_expansion,
    prepare_expansion_evidence,
)
from SpaLORA.night18d_placenta_transfer import l2_lowpass_partition
from SpaLORA.night18e_ccsr import CertificateConfig, ccsr_expansion


def sha_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def execution_contract_hash(*hashes: str) -> str:
    """Hash the exact source/registry hashes that define one producer run."""
    payload = json.dumps(list(hashes), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def load_csr(archive, prefix: str) -> sp.csr_matrix:
    return sp.csr_matrix(
        (
            np.asarray(archive[f"{prefix}__data"]),
            np.asarray(archive[f"{prefix}__indices"]),
            np.asarray(archive[f"{prefix}__indptr"]),
        ),
        shape=tuple(int(x) for x in archive[f"{prefix}__shape"]),
    )


def energy_config(raw: dict) -> ExpansionEnergyConfig:
    return ExpansionEnergyConfig(**{**raw, "local": ContinuousEnergyConfig(**raw["local"])})


def json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def run(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    base_registry = json.loads(Path(args.base_registry).read_text())
    cert_registry = json.loads(Path(args.certificate_registry).read_text())
    config = energy_config(base_registry["lanes"][args.lane])
    all_cert = {row["config_id"]: row for row in cert_registry["configs"]}
    requested = (
        [cert_registry["p0_config_id"]]
        if args.mode == "p0"
        else [row["config_id"] for row in cert_registry["configs"]]
    )
    if len(set(requested)) != len(requested) or any(key not in all_cert for key in requested):
        raise ValueError("certificate registry selection invalid")
    with np.load(args.carrier, allow_pickle=False) as carrier:
        allowed = {
            "ids", "view1", "view2", "retained", "start_ids", "start_names",
            "start_partitions",
            *{f"graph{i}__{part}" for i in range(3) for part in ("data", "indices", "indptr", "shape")},
        }
        accessed = [key for key in carrier.files if key in allowed]
        ids = np.asarray(carrier["ids"]).astype("U")
        view1 = np.asarray(carrier["view1"], dtype=np.float32)
        view2 = np.asarray(carrier["view2"], dtype=np.float32)
        retained = np.asarray(carrier["retained"], dtype=np.float32)
        name_key = "start_ids" if "start_ids" in carrier else "start_names"
        start_ids = np.asarray(carrier[name_key]).astype("U")
        starts = np.asarray(carrier["start_partitions"], dtype=np.int32)
        graphs = [load_csr(carrier, f"graph{i}") for i in range(3)]
    if len(np.unique(ids)) != len(ids) or starts.shape[1] != len(ids):
        raise ValueError("carrier ID/start contract invalid")
    if args.start_indices:
        start_indices = [int(x) for x in args.start_indices.split(",")]
    else:
        start_indices = [0] if args.mode == "p0" else list(range(min(3, len(starts))))
    if len(set(start_indices)) != len(start_indices) or any(
        index < 0 or index >= len(starts) for index in start_indices
    ):
        raise ValueError("start indices invalid")
    evidence = prepare_expansion_evidence(graphs, retained, view1, view2)
    partitions: list[np.ndarray] = []
    candidate_ids: list[str] = []
    rows: list[dict[str, object]] = []

    def append(
        start_index: int,
        arm: str,
        cert_id: str,
        partition: np.ndarray,
        diagnostics: dict[str, object],
        wall: float,
        status: str = "PASS",
        failure: str = "",
    ) -> None:
        candidate_id = f"{start_ids[start_index]}__{cert_id}__{arm}"
        if candidate_id in candidate_ids:
            raise RuntimeError("duplicate candidate ID")
        partition = np.asarray(partition, dtype=np.int32)
        if status == "PASS" and len(np.unique(partition)) != int(args.k):
            raise RuntimeError("producer candidate violated exact K")
        candidate_ids.append(candidate_id)
        partitions.append(partition)
        sizes = np.bincount(partition, minlength=int(args.k))
        rows.append(
            {
                "candidate_id": candidate_id,
                "start_id": str(start_ids[start_index]),
                "start_index": int(start_index),
                "arm": arm,
                "certificate_config_id": cert_id,
                "status": status,
                "failure": failure,
                "partition_index": len(partitions) - 1,
                "partition_sha256": sha_array(partition),
                "initial_partition_sha256": sha_array(starts[start_index]),
                "changed_from_initial": int(np.sum(partition != starts[start_index])),
                "cluster_sizes_full": [int(x) for x in sizes],
                "min_cluster_size_full": int(np.min(sizes)),
                "wall_seconds": float(wall),
                "diagnostics": json_safe(diagnostics),
            }
        )

    for start_index in start_indices:
        initial = starts[start_index]
        if len(np.unique(initial)) != int(args.k):
            raise ValueError("input start violates exact K")
        begin = time.perf_counter()
        append(start_index, "NO_OP_STRONG_START", "BASE", initial.copy(), {"solver": "NO_OP"}, time.perf_counter()-begin)
        begin = time.perf_counter()
        low = l2_lowpass_partition(retained, graphs[1], initial, args.k, config.expansion_cycles)
        append(start_index, "L2_LOWPASS_MATCHED", "BASE", low, {"solver": "L2_LOWPASS_PROTOTYPE_KMEANS"}, time.perf_counter()-begin)
        begin = time.perf_counter()
        original, original_diag = continuous_multiscale_expansion(initial, args.k, evidence, config)
        append(start_index, "NIGHT15F_ORIGINAL_SELF_RETURN", "BASE", original, original_diag, time.perf_counter()-begin)
        for cert_id in requested:
            spec = all_cert[cert_id]
            certificate = CertificateConfig(
                margin_rank_quantile=float(spec["margin_rank_quantile"]),
                minimum_view_support=int(spec["minimum_view_support"]),
                keep_untrusted_original_self_return=bool(spec["keep_untrusted_original_self_return"]),
                epsilon_relative=float(spec["epsilon_relative"]),
            )
            for arm in (
                "CCSR_FULL", "CCSR_CERTIFICATE_DISABLED",
                "CCSR_RANDOM_MASK_COUNT_MATCHED", "CCSR_UNARY_MARGIN_ONLY",
                "CCSR_VIEW_SUPPORT_ONLY", "CCSR_PROTECT_ALL",
            ):
                begin = time.perf_counter()
                try:
                    partition, diagnostics = ccsr_expansion(
                        initial, ids, args.k, evidence, config, certificate, arm=arm
                    )
                    append(start_index, arm, cert_id, partition, diagnostics, time.perf_counter()-begin)
                except Exception as exc:
                    append(
                        start_index, arm, cert_id, initial.copy(), {}, time.perf_counter()-begin,
                        status="FAILED", failure=f"{type(exc).__name__}: {exc}",
                    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = np.stack(partitions).astype(np.int32)
    np.savez_compressed(
        output,
        ids=ids,
        candidate_ids=np.asarray(candidate_ids).astype("U"),
        partitions=matrix,
    )
    with np.load(output, allow_pickle=False) as replay:
        if not np.array_equal(replay["ids"].astype("U"), ids):
            raise RuntimeError("artifact ID reload mismatch")
        if not np.array_equal(replay["candidate_ids"].astype("U"), np.asarray(candidate_ids)):
            raise RuntimeError("artifact candidate reload mismatch")
        if not np.array_equal(replay["partitions"], matrix):
            raise RuntimeError("artifact partition reload mismatch")
    producer_path = Path(__file__).resolve()
    core_path = producer_path.parents[2] / "SpaLORA" / "night18e_ccsr.py"
    producer_sha256 = sha_file(producer_path)
    core_sha256 = sha_file(core_path)
    base_registry_sha256 = sha_file(args.base_registry)
    certificate_registry_sha256 = sha_file(args.certificate_registry)
    manifest = {
        "schema": "night18e-label-free-ccsr-producer-v2",
        "lane": args.lane,
        "data_id": args.data_id,
        "mode": args.mode,
        "n": int(len(ids)),
        "k": int(args.k),
        "view1_shape": list(view1.shape),
        "view2_shape": list(view2.shape),
        "retained_shape": list(retained.shape),
        "graph_shapes_nnz": [
            {"shape": list(graph.shape), "nnz": int(graph.nnz), "dtype": str(graph.data.dtype)}
            for graph in graphs
        ],
        "start_indices": start_indices,
        "candidate_count": len(rows),
        "pass_count": sum(row["status"] == "PASS" for row in rows),
        "failed_count": sum(row["status"] != "PASS" for row in rows),
        "carrier_path": str(Path(args.carrier).resolve()),
        "carrier_sha256": sha_file(args.carrier),
        "base_registry_sha256": base_registry_sha256,
        "certificate_registry_sha256": certificate_registry_sha256,
        "producer_source_path": str(producer_path),
        "producer_source_sha256": producer_sha256,
        "ccsr_core_source_path": str(core_path),
        "ccsr_core_source_sha256": core_sha256,
        "execution_contract_sha256": execution_contract_hash(
            core_sha256,
            producer_sha256,
            base_registry_sha256,
            certificate_registry_sha256,
        ),
        "ordered_ids_sha256": sha_array(ids),
        "candidate_ids_sha256": sha_array(np.asarray(candidate_ids).astype("U")),
        "partition_matrix_sha256": sha_array(matrix),
        "artifact_sha256": sha_file(output),
        "artifact_reload": "PASS",
        "carrier_keys_accessed": sorted(accessed),
        "annotation_columns_accessed": 0,
        "producer_label_reads": 0,
        "dense_n_by_n_created": 0,
        "wall_seconds": time.perf_counter() - started,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "gpu_time_seconds": 0.0,
        "rows": rows,
    }
    output.with_suffix(".producer.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--carrier", required=True)
    parser.add_argument("--base-registry", required=True)
    parser.add_argument("--certificate-registry", required=True)
    parser.add_argument("--lane", required=True)
    parser.add_argument("--data-id", required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--mode", choices=("p0", "development", "confirmation"), required=True)
    parser.add_argument("--start-indices", default="")
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
