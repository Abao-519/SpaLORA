#!/usr/bin/env python3
"""Night-7A authority, source-artifact, and real H05 semantic preflight."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night7a_consensus import (  # noqa: E402
    CANDIDATE_ORDER, DATASETS, G00, G04, K_BY_DATASET, VIEWS,
    affinity_audit, array_sha, atomic_json, atomic_sparse,
    build_base_affinities, candidate_affinity, canonical_csr,
    canonical_partition, parse_registry, partition_sha, run_spectral,
    sha256_file, sparse_sha,
)

OUT = REPO / "outputs/night7a_handoff"
RAW = Path("/root/autodl-fs/night7a_consensus_20260818")
REG = REPO / "protocols/night7a/SpaLORA_Night7A_Consensus_Registry_2026-08-18.json"
PARENT = "e8a49fb874209b2bd4474691ee5d03ee7639c0c7"
AUTHORITY = {
    "SpaLORA_Night6D_Independent_Planner_Audit_and_Night7A_Decision_2026-08-18.md":
        "7f708473bb0663134cc07de2af5bd0bcab979ef2bb41ff0025188937e4116209",
    "SpaLORA_Night7A_Consensus_Registry_2026-08-18.json":
        "70a719bc8b34318a5b6b043d959fab7b425ffcf7987faa3f060055a71a919767",
    "SpaLORA_Night7A_CPU_Consensus_and_Benchmark_Preflight_Taskbook_2026-08-18.md":
        "ef21f9b7934f7137740119fd20d3db6a7e3f2a53ff61927525b8450f75c8a6ac",
}
SOURCE_SPECS = (
    {
        "source": "night6c",
        "raw_manifest": Path("/root/autodl-fs/SpaLORA-night6c/outputs/night6c_handoff/raw_artifact_manifest.csv"),
        "transform_manifests": (
            Path("/root/autodl-fs/SpaLORA-night6c/outputs/night6c_handoff/r1_transform_manifest.json"),
            Path("/root/autodl-fs/SpaLORA-night6c/outputs/night6c_handoff/r2_transform_manifest.json"),
        ),
        "datasets": {"a1": range(5), "tonsil": range(5)},
        "stage": lambda seed: "R1" if seed < 2 else "R2",
    },
    {
        "source": "night6d",
        "raw_manifest": Path("/root/autodl-fs/SpaLORA-night6d/outputs/night6d_handoff/raw_artifact_manifest.csv"),
        "transform_manifests": (
            Path("/root/autodl-fs/SpaLORA-night6d/outputs/night6d_handoff/locked_transform_manifest.json"),
        ),
        "datasets": {"d1": range(10), "p22": range(10)},
        "stage": lambda seed: None,
    },
)


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def verify_file(path: Path, size: int, digest: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing source artifact: {path}")
    actual_size = path.stat().st_size
    actual_sha = sha256_file(path)
    if actual_size != int(size) or actual_sha != str(digest):
        raise RuntimeError(
            f"source artifact mismatch: {path}; expected {size}/{digest}, "
            f"observed {actual_size}/{actual_sha}"
        )


def load_ids(path: Path) -> list[str]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return [str(row["observation_id"]) for row in rows]


def observation_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def array_contract(value: np.ndarray) -> dict:
    return {
        "shape": list(value.shape), "dtype": str(value.dtype),
        "finite": bool(np.isfinite(value).all()),
        "canonical_array_sha256": array_sha(value),
    }


def resolve_source_views() -> tuple[list[dict], dict]:
    records, resolved = [], {}
    base_seen = {}
    for spec in SOURCE_SPECS:
        rows = list(csv.DictReader(spec["raw_manifest"].open(newline="")))
        index = defaultdict(list)
        for row in rows:
            index[(row.get("stage") or None, row["dataset"], row["graph_id"],
                   int(row["seed"]), row["artifact"])].append(row)
        for dataset, seeds in spec["datasets"].items():
            for seed in seeds:
                stage = spec["stage"](seed)
                for graph_id in (G00, G04):
                    selected = {}
                    for artifact in ("views.npz", "observation_ids.csv",
                                     "run_manifest.json", "reload_spec.json"):
                        rows_for_key = index[(stage, dataset, graph_id, seed, artifact)]
                        if len(rows_for_key) != 1:
                            raise RuntimeError(
                                f"source cardinality {spec['source']}/{stage}/{dataset}/"
                                f"{graph_id}/{seed}/{artifact}: {len(rows_for_key)}"
                            )
                        row = rows_for_key[0]
                        verify_file(Path(row["absolute_path"]), int(row["size_bytes"]),
                                    row["sha256"])
                        selected[artifact] = row
                    views_path = Path(selected["views.npz"]["absolute_path"])
                    ids_path = Path(selected["observation_ids.csv"]["absolute_path"])
                    run_path = Path(selected["run_manifest.json"]["absolute_path"])
                    reload_path = Path(selected["reload_spec.json"]["absolute_path"])
                    run = json.loads(run_path.read_text())
                    reload_spec = json.loads(reload_path.read_text())
                    if (run["dataset"] != dataset or run["graph_id"] != graph_id or
                            int(run["seed"]) != seed or run["status"] != "success"):
                        raise RuntimeError(f"source run identity mismatch: {run_path}")
                    if (run.get("label_values_deserialized") is not False or
                            run.get("label_values_used") is not False):
                        raise RuntimeError(f"source label flag mismatch: {run_path}")
                    ids = load_ids(ids_path)
                    ordered_sha = observation_sha(ids)
                    if ordered_sha != run["ordered_observation_sha256"]:
                        raise RuntimeError(f"observation SHA mismatch: {ids_path}")
                    with np.load(views_path, allow_pickle=False) as payload:
                        view_values = {key: np.asarray(payload[key]) for key in VIEWS}
                    contracts = {}
                    for key, value in view_values.items():
                        observed = array_contract(value)
                        expected = run["view_contracts"][key]
                        if (observed["shape"] != expected["shape"] or
                                observed["dtype"] != expected["dtype"] or
                                not observed["finite"] or
                                observed["canonical_array_sha256"] !=
                                expected["canonical_array_sha256"] or
                                expected["ordered_observation_sha256"] != ordered_sha):
                            raise RuntimeError(f"view contract mismatch: {views_path}/{key}")
                        if value.ndim != 2 or value.shape[0] != len(ids):
                            raise RuntimeError(f"view observation mismatch: {views_path}/{key}")
                        contracts[key] = observed
                    base_dir = Path(reload_spec["base_cache_dir"])
                    base_manifest_path = base_dir / "manifest.json"
                    base_manifest_sha = sha256_file(base_manifest_path)
                    if (base_manifest_sha != run["base_cache_manifest_sha256"] or
                            base_manifest_sha != reload_spec["base_cache_manifest_sha256"]):
                        raise RuntimeError(f"base cache manifest mismatch: {base_manifest_path}")
                    base_manifest = json.loads(base_manifest_path.read_text())
                    coordinates_path = base_dir / "coordinates.npy"
                    base_ids_path = base_dir / "observation_ids.tsv"
                    for filename, path in (("coordinates.npy", coordinates_path),
                                           ("observation_ids.tsv", base_ids_path)):
                        meta = base_manifest["files"][filename]
                        verify_file(path, meta["size_bytes"], meta["sha256"])
                    if ids_path.read_bytes() != base_ids_path.read_bytes():
                        raise RuntimeError(f"source/base observation bytes differ: {ids_path}")
                    coords = np.load(coordinates_path, allow_pickle=False)
                    if (coords.ndim != 2 or coords.shape[0] != len(ids) or
                            not np.isfinite(coords).all()):
                        raise RuntimeError(f"coordinate contract mismatch: {coordinates_path}")
                    base_key = (spec["source"], dataset)
                    value = (str(base_dir), base_manifest_sha,
                             sha256_file(coordinates_path), ordered_sha)
                    if base_key in base_seen and base_seen[base_key] != value:
                        raise RuntimeError(f"base cache changed within dataset: {base_key}")
                    base_seen[base_key] = value
                    record = {
                        "source": spec["source"], "stage": stage,
                        "dataset": dataset, "graph_id": graph_id, "seed": seed,
                        "attempt": int(run["attempt"]),
                        "views_path": str(views_path),
                        "views_size_bytes": views_path.stat().st_size,
                        "views_sha256": sha256_file(views_path),
                        "run_manifest_path": str(run_path),
                        "run_manifest_sha256": sha256_file(run_path),
                        "observation_ids_path": str(ids_path),
                        "observation_ids_sha256": sha256_file(ids_path),
                        "ordered_observation_sha256": ordered_sha,
                        "observation_count": len(ids),
                        "base_cache_dir": str(base_dir),
                        "base_cache_manifest_sha256": base_manifest_sha,
                        "coordinates_path": str(coordinates_path),
                        "coordinates_sha256": sha256_file(coordinates_path),
                        "view_contracts": contracts,
                    }
                    records.append(record)
                    resolved[(dataset, seed, graph_id)] = {
                        "record": record, "ids": ids, "views": view_values,
                        "coords": coords,
                    }
    if len(records) != 60 or len(resolved) != 60:
        raise RuntimeError(f"source views are not 60/60: {len(records)}/{len(resolved)}")
    for dataset in DATASETS:
        seeds = range(5) if dataset in {"a1", "tonsil"} else range(10)
        for seed in seeds:
            left, right = resolved[(dataset, seed, G00)], resolved[(dataset, seed, G04)]
            if (left["ids"] != right["ids"] or
                    not np.array_equal(left["coords"], right["coords"])):
                raise RuntimeError(f"G00/G04 order or coordinate mismatch: {dataset}/{seed}")
    return records, resolved


def resolve_predictions() -> tuple[list[dict], dict]:
    records, resolved = [], {}
    base_ids = {}
    for spec in SOURCE_SPECS:
        rows = []
        for path in spec["transform_manifests"]:
            manifest = json.loads(path.read_text())
            if not manifest.get("locked_before_label_access"):
                raise RuntimeError(f"source transform not pre-label locked: {path}")
            rows.extend(manifest["transforms"])
        index = defaultdict(list)
        for row in rows:
            index[(row["dataset"], int(row["seed"]), row["graph_id"],
                   row["head_id"])].append(row)
        for dataset, seeds in spec["datasets"].items():
            if dataset not in base_ids:
                sample = next(iter(seeds))
                sample_row = next(x for x in rows if x["dataset"] == dataset and
                                  int(x["seed"]) == sample and x["graph_id"] == G00)
                run = json.loads((Path(sample_row["run_dir"]) / "run_manifest.json").read_text())
                reload_spec = json.loads((Path(sample_row["run_dir"]) / "reload_spec.json").read_text())
                base_ids[dataset] = load_ids(Path(reload_spec["base_cache_dir"]) /
                                             "observation_ids.tsv")
                if observation_sha(base_ids[dataset]) != run["ordered_observation_sha256"]:
                    raise RuntimeError(f"base prediction order mismatch: {dataset}")
            for seed in seeds:
                for graph_id in (G00, G04):
                    for head_id in ("H00_FUSED_PCA20_MCLUST_EEE",
                                    "H05_EQUAL3_AFFINITY_SPECTRAL"):
                        matches = index[(dataset, seed, graph_id, head_id)]
                        if len(matches) != 1:
                            raise RuntimeError(
                                f"prediction cardinality {dataset}/{seed}/{graph_id}/{head_id}: {len(matches)}"
                            )
                        row = matches[0]
                        if (row["status"] != "success" or row["fallback"] is not False or
                                row["label_values_deserialized"] is not False or
                                row["label_values_used"] is not False):
                            raise RuntimeError(f"prediction flags mismatch: {dataset}/{seed}/{graph_id}/{head_id}")
                        artifact = row["artifacts"]["clusters.csv"]
                        path = Path(artifact["path"])
                        verify_file(path, artifact["size_bytes"], artifact["sha256"])
                        table = pd.read_csv(path)
                        ids = table["observation_id"].astype(str).tolist()
                        if ids != base_ids[dataset]:
                            raise RuntimeError(f"prediction observation order mismatch: {path}")
                        labels = table["cluster"].to_numpy()
                        record = {
                            "source": spec["source"], "dataset": dataset,
                            "seed": seed, "graph_id": graph_id, "head_id": head_id,
                            "path": str(path), "size_bytes": path.stat().st_size,
                            "sha256": sha256_file(path),
                            "observation_count": len(ids),
                            "canonical_partition_sha256": partition_sha(labels),
                            "cluster_count": int(len(np.unique(labels))),
                        }
                        records.append(record)
                        resolved[(dataset, seed, graph_id, head_id)] = labels
    if len(records) != 120 or len(resolved) != 120:
        raise RuntimeError(f"prediction source is not 120/120: {len(records)}/{len(resolved)}")
    return records, resolved


def save_base(dataset: str, seed: int, base: dict, ids: list[str],
              source_pair: list[dict]) -> dict:
    target = RAW / "base" / dataset / f"seed_{seed}"
    target.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for graph_id in (G00, G04):
        for key, matrix in zip(VIEWS, base["affinities"][graph_id]):
            filename = f"A_{graph_id}_{key}.npz"
            path = target / filename
            atomic_sparse(path, matrix)
            artifacts[filename] = {
                "path": str(path), "size_bytes": path.stat().st_size,
                "file_sha256": sha256_file(path),
                "canonical_sparse_sha256": sparse_sha(matrix),
            }
        for key, values in zip(VIEWS, base["neighbor_sets"][graph_id]):
            filename = f"neighbors_{graph_id}_{key}.npy"
            path = target / filename
            np.save(path, np.asarray(values, dtype=np.int64), allow_pickle=False)
            artifacts[filename] = {
                "path": str(path), "size_bytes": path.stat().st_size,
                "file_sha256": sha256_file(path),
                "canonical_array_sha256": array_sha(np.asarray(values, dtype=np.int64)),
            }
        filename = f"reliability_{graph_id}.npy"
        path = target / filename
        values = np.asarray(base["reliability"][graph_id], dtype=np.float64)
        np.save(path, values, allow_pickle=False)
        artifacts[filename] = {
            "path": str(path), "size_bytes": path.stat().st_size,
            "file_sha256": sha256_file(path),
            "canonical_array_sha256": array_sha(values),
        }
    for key in ("S_G00", "S_G04", "T_spatial"):
        path = target / f"{key}.npz"
        atomic_sparse(path, base[key])
        artifacts[path.name] = {
            "path": str(path), "size_bytes": path.stat().st_size,
            "file_sha256": sha256_file(path),
            "canonical_sparse_sha256": sparse_sha(base[key]),
        }
    meta = {
        "dataset": dataset, "seed": seed,
        "ids": ids, "ordered_observation_sha256": observation_sha(ids),
        "source_pair": source_pair, "artifacts": artifacts,
        "label_access": False, "gpu_allocation_mib": 0.0,
    }
    atomic_json(target / "base_manifest.json", meta)
    meta["base_manifest_path"] = str(target / "base_manifest.json")
    meta["base_manifest_sha256"] = sha256_file(target / "base_manifest.json")
    return {key: value for key, value in meta.items() if key != "ids"}


def run_semantic_cell(dataset: str, seed: int) -> None:
    """Run one real-data parity cell in an isolated process.

    The CPU-only instance has a 2 GiB cgroup limit.  Keeping all 30 cells in
    one Python allocator caused an infrastructure SIGKILL after 21 successful
    cells, so every cell is now executed in a fresh process.  The scientific
    functions, inputs, solver, tolerances, and fixed order are unchanged.
    """
    if dataset not in DATASETS:
        raise RuntimeError(f"unknown P0 dataset: {dataset}")
    allowed = range(5) if dataset in {"a1", "tonsil"} else range(10)
    if seed not in allowed:
        raise RuntimeError(f"invalid P0 seed for {dataset}: {seed}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the empty string")
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise RuntimeError("Night-7A requires a CPU-only visible runtime")

    view_index = OUT / "source_views_index.csv"
    prediction_index = OUT / "source_prediction_index.csv"
    if not view_index.is_file() or not prediction_index.is_file():
        raise RuntimeError("P0-SOURCE indexes must be locked before semantic cells")
    view_rows = list(csv.DictReader(view_index.open(newline="")))
    prediction_rows = list(csv.DictReader(prediction_index.open(newline="")))
    selected = {}
    for graph_id in (G00, G04):
        matches = [
            row for row in view_rows
            if row["dataset"] == dataset and int(row["seed"]) == seed
            and row["graph_id"] == graph_id
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"P0 view-index cardinality {dataset}/{seed}/{graph_id}: {len(matches)}"
            )
        row = matches[0]
        views_path = Path(row["views_path"])
        ids_path = Path(row["observation_ids_path"])
        coords_path = Path(row["coordinates_path"])
        verify_file(views_path, int(row["views_size_bytes"]), row["views_sha256"])
        if sha256_file(ids_path) != row["observation_ids_sha256"]:
            raise RuntimeError(f"P0 observation-ID SHA mismatch: {ids_path}")
        if sha256_file(coords_path) != row["coordinates_sha256"]:
            raise RuntimeError(f"P0 coordinate SHA mismatch: {coords_path}")
        ids = load_ids(ids_path)
        if observation_sha(ids) != row["ordered_observation_sha256"]:
            raise RuntimeError(f"P0 observation order mismatch: {ids_path}")
        with np.load(views_path, allow_pickle=False) as payload:
            view_values = {key: np.asarray(payload[key]) for key in VIEWS}
        coords = np.load(coords_path, allow_pickle=False)
        if (
            any(value.ndim != 2 or value.shape[0] != len(ids)
                or not np.isfinite(value).all() for value in view_values.values())
            or coords.ndim != 2 or coords.shape[0] != len(ids)
            or not np.isfinite(coords).all()
        ):
            raise RuntimeError(f"P0 cell input contract failed: {dataset}/{seed}/{graph_id}")
        selected[graph_id] = {
            "record": row, "ids": ids, "views": view_values, "coords": coords,
        }

    left, right = selected[G00], selected[G04]
    if left["ids"] != right["ids"] or not np.array_equal(left["coords"], right["coords"]):
        raise RuntimeError(f"P0 paired source mismatch: {dataset}/{seed}")

    historical = {}
    for graph_id in (G00, G04):
        matches = [
            row for row in prediction_rows
            if row["dataset"] == dataset and int(row["seed"]) == seed
            and row["graph_id"] == graph_id
            and row["head_id"] == "H05_EQUAL3_AFFINITY_SPECTRAL"
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"P0 prediction-index cardinality {dataset}/{seed}/{graph_id}: "
                f"{len(matches)}"
            )
        row = matches[0]
        path = Path(row["path"])
        verify_file(path, int(row["size_bytes"]), row["sha256"])
        table = pd.read_csv(path)
        if table["observation_id"].astype(str).tolist() != left["ids"]:
            raise RuntimeError(f"P0 historical prediction order mismatch: {path}")
        labels = table["cluster"].to_numpy()
        if partition_sha(labels) != row["canonical_partition_sha256"]:
            raise RuntimeError(f"P0 historical partition SHA mismatch: {path}")
        historical[graph_id] = labels

    started = time.perf_counter()
    base = build_base_affinities(
        left["views"], right["views"], left["ids"], left["coords"]
    )
    six = list(base["affinities"][G00]) + list(base["affinities"][G04])
    six_mean = canonical_csr(sum(six[1:], six[0]) * (1.0 / 6.0))
    c02, _ = candidate_affinity("C02_DUAL_ARITHMETIC_MEAN", base, left["ids"])
    diff = canonical_csr(c02 - six_mean)
    c02_error = float(np.max(np.abs(diff.data))) if diff.nnz else 0.0
    if c02_error > 1e-12:
        raise RuntimeError(f"C02 six-view identity failed: {dataset}/{seed}/{c02_error}")

    semantic_rows, repeats = [], []
    candidate_by_graph = {
        G00: "C01_G00_H05",
        G04: "C00_G04_H05_CONFIRMED",
    }
    for graph_id, matrix in ((G00, base["S_G00"]), (G04, base["S_G04"])):
        consensus1, _ = candidate_affinity(
            candidate_by_graph[graph_id], base, left["ids"]
        )
        consensus2, _ = candidate_affinity(
            candidate_by_graph[graph_id], base, left["ids"]
        )
        first_sha = sparse_sha(consensus1)
        second_sha = sparse_sha(consensus2)
        source_sha = sparse_sha(matrix)
        if first_sha != second_sha or first_sha != source_sha:
            raise RuntimeError(
                f"H05 consensus repeat failed: {dataset}/{seed}/{graph_id}"
            )
        labels1, runtime1, peak1 = run_spectral(consensus1, dataset)
        labels2, runtime2, peak2 = run_spectral(consensus2, dataset)
        exact = np.array_equal(
            canonical_partition(labels1),
            canonical_partition(historical[graph_id]),
        )
        deterministic = partition_sha(labels1) == partition_sha(labels2)
        if not exact or not deterministic:
            raise RuntimeError(
                f"H05 real parity failed: {dataset}/{seed}/{graph_id}; "
                f"exact={exact}; deterministic={deterministic}"
            )
        repeats.append({
            "dataset": dataset, "seed": seed,
            "candidate_id": candidate_by_graph[graph_id],
            "first_canonical_sparse_sha256": first_sha,
            "repeat_canonical_sparse_sha256": second_sha,
            "source_canonical_sparse_sha256": source_sha,
            "exact_repeat": True,
        })
        semantic_rows.append({
            "dataset": dataset, "seed": seed, "graph_id": graph_id,
            "authoritative_partition_sha256": partition_sha(historical[graph_id]),
            "observed_partition_sha256": partition_sha(labels1),
            "repeat_partition_sha256": partition_sha(labels2),
            "exact_partition_parity": exact,
            "repeat_deterministic": deterministic,
            "runtime_seconds": runtime1, "repeat_runtime_seconds": runtime2,
            "peak_rss_mib": max(peak1, peak2),
            "affinity": affinity_audit(consensus1),
        })
    base_manifest = save_base(
        dataset, seed, base, left["ids"], [left["record"], right["record"]]
    )
    target = RAW / "p0_cells" / dataset / f"seed_{seed}.json"
    atomic_json(target, {
        "status": "PASS", "dataset": dataset, "seed": seed,
        "c02_six_view_identity_max_error": c02_error,
        "semantic_rows": semantic_rows,
        "consensus_repeat_sparse_sha_cells": repeats,
        "base_manifest": base_manifest,
        "runtime_seconds": time.perf_counter() - started,
        "label_access": False, "formal_transform_attempts": 0,
        "scientific_training": 0, "checkpoint_forward": 0,
        "gpu_allocation_mib": 0.0,
        "thread_environment": {
            key: os.environ.get(key)
            for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
        },
    })
    print(json.dumps({
        "status": "PASS", "dataset": dataset, "seed": seed,
        "parity": len(semantic_rows), "base_manifest": str(base_manifest["base_manifest_path"]),
    }, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell-dataset")
    parser.add_argument("--cell-seed", type=int)
    args = parser.parse_args()
    if args.cell_dataset is not None:
        if args.cell_seed is None:
            raise RuntimeError("--cell-seed is required with --cell-dataset")
        run_semantic_cell(args.cell_dataset, args.cell_seed)
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("CUDA_VISIBLE_DEVICES must be the empty string")
    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise RuntimeError("Night-7A requires a CPU-only visible runtime")
    OUT.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)
    authority_rows = []
    for filename, expected in AUTHORITY.items():
        path = REPO / "protocols/night7a" / filename
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"authority SHA mismatch: {filename}: {actual}")
        authority_rows.append({"path": str(path), "sha256": actual,
                               "size_bytes": path.stat().st_size})
    registry = json.loads(REG.read_text())
    parse_registry(registry)
    head = git("rev-parse", "HEAD")
    if git("rev-parse", "baseline/pre-night7a-cpu-consensus-preflight-20260818^{}") != PARENT:
        raise RuntimeError("protection tag does not peel to Night-6D parent")
    if subprocess.call(["git", "-C", str(REPO), "merge-base", "--is-ancestor", PARENT, head]) != 0:
        raise RuntimeError("Night-6D parent is not an ancestor of the active worktree")
    p0_authority = {
        "status": "PASS", "message": "P0-AUTHORITY PASS; BEGIN P0-SOURCE",
        "authority": authority_rows, "parent_commit": PARENT,
        "active_commit": head, "branch": git("branch", "--show-current"),
        "protection_tag_peeled_commit": PARENT,
        "cpu": platform.processor(), "python": platform.python_version(),
        "torch": torch.__version__, "numpy": np.__version__,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_visible_device_count": torch.cuda.device_count(),
        "budgets": {"scientific_training": "0/0", "checkpoint_forward": "0/0",
                    "diffusion": "0/0", "formal_consensus_transforms_max": 360,
                    "correction_transforms_max": 12, "total_attempts_max": 372,
                    "formal_benchmark_runs": "0/0", "fresh_label_reads": "0/0"},
    }
    atomic_json(OUT / "p0_authority_and_budget.json", p0_authority)

    source_records, views = resolve_source_views()
    prediction_records, predictions = resolve_predictions()
    pd.DataFrame([{key: value for key, value in row.items() if key != "view_contracts"}
                  for row in source_records]).to_csv(OUT / "source_views_index.csv", index=False)
    pd.DataFrame(prediction_records).to_csv(OUT / "source_prediction_index.csv", index=False)
    atomic_json(OUT / "source_reuse_manifest.json", {
        "status": "PASS", "views": "60/60", "predictions": "120/120",
        "g00_g04_order_and_coordinates": "30/30", "label_access": False,
        "training": 0, "checkpoint_forward": 0, "gpu_allocation_mib": 0.0,
        "source_views_index_sha256": sha256_file(OUT / "source_views_index.csv"),
        "source_prediction_index_sha256": sha256_file(OUT / "source_prediction_index.csv"),
    })

    del views, predictions
    gc.collect()
    semantic_rows, base_manifests, c02_errors, repeats = [], [], [], []
    start = time.perf_counter()
    for dataset in DATASETS:
        seeds = range(5) if dataset in {"a1", "tonsil"} else range(10)
        for seed in seeds:
            result_path = RAW / "p0_cells" / dataset / f"seed_{seed}.json"
            if result_path.exists():
                raise RuntimeError(f"refusing to reuse a pre-existing P0 cell: {result_path}")
            subprocess.check_call([
                sys.executable, str(Path(__file__).resolve()),
                "--cell-dataset", dataset, "--cell-seed", str(seed),
            ], env=os.environ.copy())
            result = json.loads(result_path.read_text())
            if result.get("status") != "PASS":
                raise RuntimeError(f"P0 semantic cell did not pass: {result_path}")
            semantic_rows.extend(result["semantic_rows"])
            base_manifests.append(result["base_manifest"])
            c02_errors.append(float(result["c02_six_view_identity_max_error"]))
            repeats.extend(result["consensus_repeat_sparse_sha_cells"])
    if len(semantic_rows) != 60:
        raise RuntimeError(f"H05 parity is not 30x2: {len(semantic_rows)}")
    if len(repeats) != 60 or not all(row["exact_repeat"] for row in repeats):
        raise RuntimeError(f"H05 consensus repeat SHA is not 60/60: {len(repeats)}")
    atomic_json(OUT / "p0_semantic_contract.json", {
        "status": "PASS", "h05_exact_partition_parity": "30/30 x 2",
        "c02_six_view_identity_max_error": max(c02_errors, default=0.0),
        "consensus_repeat_sparse_sha_all_equal": True,
        "consensus_repeat_sparse_sha_cells": repeats,
        "spectral": {"affinity": "precomputed", "assign_labels": "discretize",
                     "n_init": 20, "random_state": 2020},
        "source_function": "SpaLORA.night6c_pipeline.self_tuning_affinity/spectral",
        "source_code_sha256": sha256_file(REPO / "SpaLORA/night6c_pipeline.py"),
        "registry_sha256": sha256_file(REG), "label_access": False,
        "scientific_training": 0, "checkpoint_forward": 0,
        "gpu_allocation_mib": 0.0,
        "process_isolation": {
            "reason": "2 GiB cgroup; prevent allocator accumulation across real cells",
            "one_fresh_process_per_dataset_seed": True,
            "scientific_semantics_changed": False,
            "solver_changed": False, "tolerance_changed": False,
        },
        "runtime_seconds": time.perf_counter() - start,
        "cells": semantic_rows,
        "base_manifests": base_manifests,
    })
    atomic_json(OUT / "budget_and_access_audit_prelock.json", {
        "status": "PASS", "scientific_training": 0, "checkpoint_forward": 0,
        "diffusion": 0, "formal_transforms": 0, "corrections": 0,
        "pre_science_infrastructure_attempts_preserved": 2,
        "total_transform_attempts": 0, "formal_benchmark_runs": 0,
        "development_per_spot_ground_truth_deserializations": 0,
        "prior_per_seed_metric_table_parses": 0,
        "fresh_external_per_spot_label_reads": 0,
        "gpu_allocation_mib": 0.0,
    })
    print(json.dumps({"status": "PASS", "views": len(source_records),
                      "predictions": len(prediction_records),
                      "h05_parity": len(semantic_rows), "base_units": len(base_manifests)},
                     sort_keys=True))


if __name__ == "__main__":
    main()
