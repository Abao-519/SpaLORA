#!/usr/bin/env python3
"""Real, label-free Night-16C P0 for newly registered physical units.

The runner is deliberately separated from the CMBF-TPR producer.  It may map
an authority manifest to normal modality adapters and one of two frozen family
configs, but the producer itself receives only numeric views, a sparse graph,
a start bank and a numeric config.  No annotation array is loaded here.

Producer mode reads real registered matrices, performs deterministic reduction,
builds a sparse graph and a common start bank, materializes a checkpoint, then
runs CMBF-TPR.  Replay mode starts from that checkpoint in a fresh process and
requires an exact partition hash match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from SpaLORA.night16c_cmbf_tpr import (
    CMBFTPRConfig,
    array_sha256,
    cmbf_tpr,
    partition_sha256,
    sparse_graph_from_csr_arrays,
)


def _json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_real_input(spec: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path = Path(str(spec["input_path"]))
    with np.load(path, allow_pickle=False) as payload:
        if spec["input_schema"] == "registered_matrices":
            view1 = np.asarray(payload["rna"], dtype=np.float32)
            view2 = np.asarray(payload["target"], dtype=np.float32)
            coords = np.asarray(payload["coordinates"], dtype=np.float64)
            ids = np.asarray(payload["observation_ids"]).astype(str)
        elif spec["input_schema"] == "night13a_roundtrip":
            view1 = np.asarray(payload["view1"], dtype=np.float32)
            view2 = np.asarray(payload["view2"], dtype=np.float32)
            coords = np.asarray(payload["coordinates"], dtype=np.float64)
            ids = np.asarray(payload["ordered_ids"]).astype(str)
        else:
            raise ValueError(f"unsupported input schema: {spec['input_schema']}")
    if not (len(view1) == len(view2) == len(coords) == len(ids)):
        raise ValueError("real input observation counts do not match")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("ordered observation IDs are not unique")
    if not (np.isfinite(view1).all() and np.isfinite(view2).all() and np.isfinite(coords).all()):
        raise ValueError("real input contains non-finite values")
    return view1, view2, coords, ids


def _reduce(value: np.ndarray, n_components: int = 30) -> tuple[np.ndarray, dict[str, object], np.ndarray, np.ndarray]:
    scaler = StandardScaler(with_mean=True, with_std=True)
    scaled = scaler.fit_transform(np.asarray(value, dtype=np.float64))
    dimension = int(min(n_components, scaled.shape[1], max(1, scaled.shape[0] - 1)))
    pca = PCA(n_components=dimension, svd_solver="full")
    reduced = pca.fit_transform(scaled).astype(np.float32)
    reconstructed = pca.inverse_transform(reduced)
    reconstruction_loss = float(np.mean((scaled - reconstructed) ** 2))
    audit = {
        "input_shape": [int(x) for x in value.shape],
        "output_shape": [int(x) for x in reduced.shape],
        "input_dtype": str(value.dtype),
        "output_dtype": str(reduced.dtype),
        "explained_variance_sum": float(np.sum(pca.explained_variance_ratio_)),
        "finite_reconstruction_loss": reconstruction_loss,
        "optimizer_steps": 0,
        "trainable_parameters": 0,
    }
    return reduced, audit, scaler.mean_.astype(np.float64), pca.components_.astype(np.float64)


def _spatial_graph(coords: np.ndarray, neighbours: int = 6) -> sp.csr_matrix:
    k = min(int(neighbours) + 1, len(coords))
    model = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(coords)
    distance, index = model.kneighbors(coords)
    row = np.repeat(np.arange(len(coords), dtype=np.int64), k - 1)
    col = index[:, 1:].reshape(-1).astype(np.int64)
    dist = distance[:, 1:].reshape(-1)
    positive = dist[dist > 0]
    scale = float(np.median(positive)) if len(positive) else 1.0
    data = np.exp(-dist / max(scale, 1e-12))
    graph = sp.csr_matrix((data, (row, col)), shape=(len(coords), len(coords)))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0)
    graph.eliminate_zeros()
    graph.sort_indices()
    return graph


def _select_guarded_central_start(bank: np.ndarray, k: int) -> tuple[int, list[float], list[int], int, np.ndarray]:
    centrality: list[float] = []
    minimum_sizes: list[int] = []
    for i in range(len(bank)):
        centrality.append(float(np.mean([adjusted_rand_score(bank[i], bank[j]) for j in range(len(bank)) if j != i])))
        minimum_sizes.append(int(np.bincount(bank[i], minlength=k).min()))
    minimum_threshold = max(5, int(np.ceil(0.01 * len(bank[0]) / k)))
    valid = np.asarray(minimum_sizes) >= minimum_threshold
    if np.any(valid):
        score = np.where(valid, np.asarray(centrality), -np.inf)
        selected = int(np.argmax(score))
    else:
        selected = int(np.argmax(np.asarray(minimum_sizes)))
    return selected, centrality, minimum_sizes, minimum_threshold, valid


def _common_start_bank(view1: np.ndarray, view2: np.ndarray, k: int, seed: int) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    fused = np.concatenate([StandardScaler().fit_transform(view1), StandardScaler().fit_transform(view2)], axis=1)
    inputs = [view1, view2, fused]
    starts = []
    for matrix in inputs:
        starts.append(
            KMeans(n_clusters=k, random_state=seed, n_init=5, max_iter=150, algorithm="lloyd")
            .fit_predict(matrix)
            .astype(np.int32)
        )
    bank = np.stack(starts, axis=0)
    # A partition with a singleton can be central to two other starts while
    # still violating the producer's registered exact-K/minimum-size contract.
    # Apply the same generic, label-free guard before centrality selection.
    selected, centrality, minimum_sizes, minimum_threshold, valid = _select_guarded_central_start(bank, k)
    audit = {
        "generator": "common KMeans bank on view1/view2/fused",
        "seed": int(seed),
        "n_init": 5,
        "start_partition_sha256": [partition_sha256(x) for x in bank],
        "partition_centrality": centrality,
        "minimum_cluster_sizes": minimum_sizes,
        "minimum_cluster_threshold": int(minimum_threshold),
        "valid_start_mask": [bool(x) for x in valid],
        "selected_index": selected,
        "selection_rule": "maximum label-free mean partition-to-partition ARI centrality among exact-K starts satisfying the generic minimum-cluster guard",
    }
    return bank, bank[selected].copy(), audit


def _knn_overlap(view1: np.ndarray, view2: np.ndarray, neighbours: int = 10) -> float:
    k = min(int(neighbours) + 1, len(view1))
    first = NearestNeighbors(n_neighbors=k).fit(view1).kneighbors(return_distance=False)[:, 1:]
    second = NearestNeighbors(n_neighbors=k).fit(view2).kneighbors(return_distance=False)[:, 1:]
    total = 0.0
    for a, b in zip(first, second):
        total += len(set(a.tolist()).intersection(b.tolist())) / float(k - 1)
    return float(total / len(view1))


def _label_free_metrics(partition: np.ndarray, view1: np.ndarray, view2: np.ndarray, graph: sp.csr_matrix) -> dict[str, object]:
    fused = np.concatenate([view1, view2], axis=1)
    sample_size = min(2000, len(fused))
    rng = np.random.default_rng(20260824)
    sample = np.sort(rng.choice(len(fused), size=sample_size, replace=False))
    row = np.repeat(np.arange(graph.shape[0], dtype=np.int64), np.diff(graph.indptr))
    agreement = float(np.average(partition[row] == partition[graph.indices], weights=graph.data))
    sizes = np.bincount(partition)
    return {
        "silhouette_sampled": float(silhouette_score(fused[sample], partition[sample], metric="euclidean")),
        "calinski_harabasz": float(calinski_harabasz_score(fused, partition)),
        "davies_bouldin": float(davies_bouldin_score(fused, partition)),
        "spatial_edge_agreement": agreement,
        "cross_modal_knn_overlap": _knn_overlap(view1, view2),
        "cluster_sizes": [int(x) for x in sizes],
        "min_cluster_size": int(sizes.min()),
    }


def _checkpoint_path(unit_dir: Path) -> Path:
    return unit_dir / "p0_checkpoint.npz"


def produce(spec: dict[str, object], family_configs: dict[str, object], output_root: Path) -> dict[str, object]:
    started = time.perf_counter()
    unit = str(spec["unit_id"])
    unit_dir = output_root / unit
    unit_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(str(spec["input_path"]))
    before = source_path.stat()
    raw1, raw2, coords, ids = _load_real_input(spec)
    view1, audit1, mean1, components1 = _reduce(raw1)
    view2, audit2, mean2, components2 = _reduce(raw2)
    graph = _spatial_graph(coords)
    k = int(spec["engineering_k"])
    bank, initial, start_audit = _common_start_bank(view1, view2, k, int(spec["seed"]))
    config_value = family_configs[str(spec["family"])]
    config = CMBFTPRConfig(**config_value)
    checkpoint = _checkpoint_path(unit_dir)
    np.savez_compressed(
        checkpoint,
        view1=view1,
        view2=view2,
        coordinates=coords,
        ordered_ids=ids,
        graph_data=graph.data,
        graph_indices=graph.indices,
        graph_indptr=graph.indptr,
        graph_shape=np.asarray(graph.shape, dtype=np.int64),
        start_bank=bank,
        initial=initial,
        scaler_mean1=mean1,
        scaler_mean2=mean2,
        pca_components1=components1,
        pca_components2=components2,
        config_json=np.asarray(json.dumps(config_value, sort_keys=True)),
    )
    with np.load(checkpoint, allow_pickle=False) as strict:
        strict_graph = sparse_graph_from_csr_arrays(strict["graph_data"], strict["graph_indices"], strict["graph_indptr"], strict["graph_shape"])
        partition, evidence, diagnostics = cmbf_tpr(
            strict["initial"], strict["view1"], strict["view2"], strict_graph, strict["start_bank"], config
        )
        metrics = _label_free_metrics(partition, strict["view1"], strict["view2"], strict_graph)
        id_sha = array_sha256(strict["ordered_ids"])
    np.save(unit_dir / "partition.npy", partition, allow_pickle=False)
    after = source_path.stat()
    record = {
        "schema": "night16c-new-unit-real-p0-v1",
        "unit_id": unit,
        "family": str(spec["family"]),
        "input_path": str(source_path),
        "input_schema": str(spec["input_schema"]),
        "input_file_sha256": _file_sha256(source_path),
        "input_size": int(before.st_size),
        "input_mtime_ns_before": int(before.st_mtime_ns),
        "input_mtime_ns_after": int(after.st_mtime_ns),
        "input_immutable": bool(before.st_size == after.st_size and before.st_mtime_ns == after.st_mtime_ns),
        "ordered_id_sha256": id_sha,
        "observation_count": int(len(ids)),
        "coordinates_shape": [int(x) for x in coords.shape],
        "coordinates_dtype": str(coords.dtype),
        "view1": audit1,
        "view2": audit2,
        "sparse_graph_shape": [int(x) for x in graph.shape],
        "sparse_graph_nnz": int(graph.nnz),
        "dense_n_by_n_count": 0,
        "engineering_k": k,
        "engineering_k_source": str(spec["engineering_k_source"]),
        "start_bank": start_audit,
        "family_config": config_value,
        "family_config_id": str(spec["family_config_id"]),
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": _file_sha256(checkpoint),
        "checkpoint_strict_reload": True,
        "partition_sha256": partition_sha256(partition),
        "cmbf_tpr": diagnostics,
        "label_free_metrics": metrics,
        "training_labels_read": 0,
        "evaluation_labels_read": 0,
        "total_labels_read": 0,
        "optimizer_steps": 0,
        "optimizer_step_applicability": "not applicable: deterministic, non-trainable CMBF-TPR producer",
        "wall_seconds": float(time.perf_counter() - started),
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "peak_gpu_mib": 0,
        "status": "PASS",
    }
    _json_dump(unit_dir / "producer_record.json", record)
    return record


def replay_unit(unit_dir: Path) -> dict[str, object]:
    record = json.loads((unit_dir / "producer_record.json").read_text(encoding="utf-8"))
    checkpoint = _checkpoint_path(unit_dir)
    with np.load(checkpoint, allow_pickle=False) as payload:
        graph = sparse_graph_from_csr_arrays(payload["graph_data"], payload["graph_indices"], payload["graph_indptr"], payload["graph_shape"])
        config = CMBFTPRConfig(**json.loads(str(payload["config_json"])))
        partition, _, diagnostics = cmbf_tpr(
            payload["initial"], payload["view1"], payload["view2"], graph, payload["start_bank"], config
        )
        ids_sha = array_sha256(payload["ordered_ids"])
    observed = partition_sha256(partition)
    result = {
        "unit_id": record["unit_id"],
        "fresh_process_partition_sha256": observed,
        "producer_partition_sha256": record["partition_sha256"],
        "partition_exact": bool(observed == record["partition_sha256"]),
        "ordered_id_sha256": ids_sha,
        "ordered_id_exact": bool(ids_sha == record["ordered_id_sha256"]),
        "cluster_sizes": diagnostics["cluster_sizes"],
        "dense_n_by_n_count": 0,
        "label_reads": 0,
        "status": "PASS" if observed == record["partition_sha256"] and ids_sha == record["ordered_id_sha256"] else "FAIL",
    }
    _json_dump(unit_dir / "fresh_process_replay.json", result)
    if result["status"] != "PASS":
        raise RuntimeError(f"fresh-process replay mismatch for {record['unit_id']}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=("producer", "replay"), required=True)
    args = parser.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    output = Path(args.output)
    if args.mode == "producer":
        rows = [produce(spec, manifest["family_configs"], output) for spec in manifest["units"]]
        summary = {
            "schema": "night16c-new-unit-real-p0-summary-v1",
            "mode": "producer",
            "passed": int(sum(x["status"] == "PASS" for x in rows)),
            "expected": len(rows),
            "units": rows,
            "new_download_count": 0,
            "label_reads": 0,
            "dense_n_by_n_count": 0,
        }
        _json_dump(output / "producer_summary.json", summary)
    else:
        rows = [replay_unit(output / str(spec["unit_id"])) for spec in manifest["units"]]
        summary = {
            "schema": "night16c-new-unit-real-p0-replay-summary-v1",
            "mode": "fresh_process_replay",
            "passed": int(sum(x["status"] == "PASS" for x in rows)),
            "expected": len(rows),
            "units": rows,
            "label_reads": 0,
            "dense_n_by_n_count": 0,
        }
        _json_dump(output / "fresh_process_replay_summary.json", summary)


if __name__ == "__main__":
    main()
