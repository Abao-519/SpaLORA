#!/usr/bin/env python3
"""Local-first Night-15G optional morphology development search.

Reference labels are consulted only after each candidate partition has been
created.  They provide known K, public-benchmark metrics and cross-run HPO.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from typing import Mapping, Sequence

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

try:
    import resource as _resource

    def peak_rss_mib() -> float:
        return _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss / 1024.0
except ImportError:  # Windows local-first runner
    def peak_rss_mib() -> float:
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        if not ctypes.windll.psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return float("nan")
        return counters.PeakWorkingSetSize / (1024.0 * 1024.0)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15e_continuous_reliability_energy import (  # noqa: E402
    ContinuousEnergyConfig,
    csr_from_archive,
)
from SpaLORA.night15f_multiscale_expansion import (  # noqa: E402
    ExpansionEnergyConfig,
)
from SpaLORA.night15g_optional_morphology_energy import (  # noqa: E402
    OptionalMorphologyConfig,
    optional_morphology_expansion,
    prepare_optional_morphology_evidence,
)


MORPH_FILE = {
    "A1": "A1_morphology_views.npz",
    "D1": "D1_morphology_views.npz",
    "tonsil_s3": "tonsil_s3_morphology_views.npz",
}


def array_sha256(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def encode(value: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(value, dtype=str), return_inverse=True)[1].astype(np.int32)


def mean_binary_moran(partition: np.ndarray, graph: sp.spmatrix) -> float:
    labels = np.asarray(partition, dtype=np.int32)
    weight = sp.csr_matrix(graph, dtype=np.float64)
    total = float(weight.sum())
    values = []
    for group in np.unique(labels):
        x = (labels == group).astype(np.float64)
        centered = x - x.mean()
        denominator = float(centered @ centered)
        if denominator > 0:
            values.append(len(x) * float(centered @ (weight @ centered)) / (total * denominator))
    return float(np.mean(values)) if values else 0.0


def mean_binary_geary(partition: np.ndarray, graph: sp.spmatrix) -> float:
    labels = np.asarray(partition, dtype=np.int32)
    weight = sp.csr_matrix(graph, dtype=np.float64).tocoo()
    total = float(weight.data.sum())
    values = []
    for group in np.unique(labels):
        x = (labels == group).astype(np.float64)
        denominator = float(np.sum((x - x.mean()) ** 2))
        if denominator > 0:
            numerator = float(np.sum(weight.data * (x[weight.row] - x[weight.col]) ** 2))
            values.append((len(x) - 1.0) * numerator / (2.0 * total * denominator))
    return float(np.mean(values)) if values else 0.0


def evaluate(labels, mask, partition, graph) -> dict:
    truth = encode(np.asarray(labels)[mask])
    observed = np.asarray(partition, dtype=np.int32)[mask]
    return {
        "absolute_ari": float(adjusted_rand_score(truth, observed)),
        "absolute_nmi": float(normalized_mutual_info_score(truth, observed)),
        "ami": float(adjusted_mutual_info_score(truth, observed)),
        "fmi": float(fowlkes_mallows_score(truth, observed)),
        "morans_i": mean_binary_moran(partition, graph),
        "gearys_c": mean_binary_geary(partition, graph),
    }


def standardize(value: np.ndarray) -> np.ndarray:
    return StandardScaler(copy=True).fit_transform(np.asarray(value, dtype=np.float32)).astype(np.float32)


def reduce_feature(value: np.ndarray, dim: int) -> np.ndarray:
    value = standardize(value)
    dim = min(int(dim), value.shape[1], value.shape[0] - 1)
    if dim < value.shape[1]:
        value = PCA(n_components=dim, svd_solver="full").fit_transform(value)
    return standardize(value)


def parse_base(record: Mapping[str, object]) -> ExpansionEnergyConfig:
    payload = dict(record)
    local = ContinuousEnergyConfig(**payload.pop("local"))
    return ExpansionEnergyConfig(local=local, **payload)


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def selection_key(row: Mapping[str, object]) -> tuple:
    if row.get("status") != "PASS":
        return (-1, -math.inf, -math.inf, -math.inf)
    da, dn = float(row["delta_ari"]), float(row["delta_nmi"])
    return (int(da > 1e-12 and dn > 1e-12), min(da, dn), da + dn, float(row["absolute_ari"]))


def candidate_id(variant: str, config: OptionalMorphologyConfig, prefix: str) -> str:
    payload = json.dumps({"variant": variant, "config": asdict(config)}, sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:14]}"


def load_lane(args, lane: str):
    data = np.load(args.kit / f"{lane}.npz", allow_pickle=False, mmap_mode="r")
    bank = np.load(args.banks / f"{lane}_selected_partition_bank.npz", allow_pickle=False, mmap_mode="r")
    morph = np.load(args.morphology / MORPH_FILE[lane], allow_pickle=False, mmap_mode="r")
    if not np.array_equal(np.asarray(data["ids"], dtype=str), np.asarray(morph["ordered_ids"], dtype=str)):
        raise RuntimeError(f"{lane}: morphology/compute-kit ordered IDs differ")
    graph = csr_from_archive(data, "graph")
    graphs = (csr_from_archive(data, "operator4"), graph, csr_from_archive(data, "operator18"))
    retained = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
    partition = np.load(args.partitions / f"{lane}.npy", allow_pickle=False).astype(np.int32)
    labels = np.asarray(data["labels_primary"], dtype=str)
    mask = np.asarray(data["label_mask"], dtype=bool)
    k = int(data["k_primary"][0])
    return data, graph, graphs, retained, partition, labels, mask, k, morph


def feature_variants(morph) -> dict[str, np.ndarray]:
    handcrafted64 = reduce_feature(np.asarray(morph["handcrafted"], dtype=np.float32), 64)
    resnet64 = reduce_feature(np.asarray(morph["resnet18"], dtype=np.float32), 64)
    combined64 = reduce_feature(np.concatenate((handcrafted64, resnet64), axis=1), 64)
    output = {}
    for name, maximum in (
        ("HANDCRAFTED", handcrafted64),
        ("RESNET18", resnet64),
        ("COMBINED", combined64),
    ):
        for dim in (16, 32, 64):
            output[f"{name}_D{dim}"] = reduce_feature(maximum, dim)
    return output


def designed_configs(base: ExpansionEnergyConfig) -> list[OptionalMorphologyConfig]:
    output = []
    for unary, edge in (
        (0.01, 0.0),
        (0.03, 0.0),
        (0.10, 0.0),
        (0.30, 0.0),
        (0.0, 0.01),
        (0.0, 0.03),
        (0.0, 0.10),
        (0.0, 0.30),
        (0.01, 0.01),
        (0.03, 0.03),
        (0.10, 0.10),
        (0.30, 0.30),
        (0.10, 0.03),
        (0.03, 0.10),
    ):
        for reliability_mix in (0.0, 1.0):
            output.append(
                OptionalMorphologyConfig(
                    base=base,
                    morphology_unary_weight=unary,
                    morphology_edge_weight=edge,
                    reliability_mix=reliability_mix,
                    reliability_bias=0.0,
                    reliability_consistency_weight=1.0,
                    reliability_conflict_weight=1.0,
                    reliability_margin_weight=1.0,
                    reliability_temperature=1.0,
                )
            )
    return output


def random_configs(base: ExpansionEnergyConfig, count: int, seed: int) -> list[OptionalMorphologyConfig]:
    rng = np.random.default_rng(int(seed))
    output = []
    for index in range(int(count)):
        unary = 0.0 if index % 13 == 0 else float(np.exp(rng.uniform(np.log(0.002), np.log(3.0))))
        edge = 0.0 if index % 11 == 0 else float(np.exp(rng.uniform(np.log(0.002), np.log(3.0))))
        output.append(
            OptionalMorphologyConfig(
                base=base,
                morphology_unary_weight=unary,
                morphology_edge_weight=edge,
                reliability_mix=float(rng.uniform(0.0, 1.0)),
                reliability_bias=float(rng.uniform(-2.5, 2.5)),
                reliability_consistency_weight=float(rng.uniform(0.0, 2.5)),
                reliability_conflict_weight=float(rng.uniform(0.0, 2.5)),
                reliability_margin_weight=float(rng.uniform(0.0, 2.5)),
                reliability_temperature=float(np.exp(rng.uniform(np.log(0.2), np.log(2.5)))),
            )
        )
    return output


def row_for_partition(
    lane,
    stage,
    algorithm,
    identifier,
    variant,
    config_json,
    partition,
    initial,
    labels,
    mask,
    graph,
    k,
    authority_metrics,
    wall_seconds,
    diagnostics=None,
):
    metrics = evaluate(labels, mask, partition, graph)
    diagnostics = diagnostics or {}
    return {
        "lane": lane,
        "stage": stage,
        "algorithm": algorithm,
        "config_id": identifier,
        "feature_variant": variant,
        "config_json": config_json,
        "status": "PASS",
        "failure": "",
        "k": k,
        "total_observations": len(partition),
        "evaluated_observations": int(mask.sum()),
        "initial_partition_sha256": array_sha256(initial),
        "partition_sha256": array_sha256(partition),
        "cluster_sizes": json.dumps(np.bincount(partition, minlength=k).astype(int).tolist(), separators=(",", ":")),
        "night15f_ari": authority_metrics["absolute_ari"],
        "night15f_nmi": authority_metrics["absolute_nmi"],
        "delta_ari": metrics["absolute_ari"] - authority_metrics["absolute_ari"],
        "delta_nmi": metrics["absolute_nmi"] - authority_metrics["absolute_nmi"],
        "wall_seconds": wall_seconds,
        "peak_rss_mib": peak_rss_mib(),
        **metrics,
        **diagnostics,
    }


def run_direct_controls(lane, retained, variants, initial, labels, mask, graph, k, authority):
    rows = []
    retained_block = reduce_feature(retained, min(64, retained.shape[1]))
    for variant, feature in variants.items():
        for seed in range(3):
            for algorithm in ("MORPHOLOGY_ONLY_KMEANS", "MORPHOLOGY_ONLY_DIAG_GMM"):
                started = time.perf_counter()
                if algorithm.endswith("KMEANS"):
                    partition = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(feature)
                else:
                    partition = GaussianMixture(
                        n_components=k, covariance_type="diag", random_state=seed, n_init=3, max_iter=300
                    ).fit_predict(feature)
                rows.append(
                    row_for_partition(
                        lane, "DIRECT_CONTROL", algorithm, f"{variant}_S{seed}", variant,
                        json.dumps({"seed": seed}, separators=(",", ":")), partition, initial,
                        labels, mask, graph, k, authority, time.perf_counter() - started,
                    )
                )
        # Each block is standardized/reduced exactly once.  Apply the relative
        # block weight only afterwards: a second per-column StandardScaler would
        # algebraically erase it (the superseded A1 probe exposed that bug).
        retained_equalized = retained_block / math.sqrt(max(retained_block.shape[1], 1))
        morphology_equalized = feature / math.sqrt(max(feature.shape[1], 1))
        for weight in (0.01, 0.03, 0.10, 0.30, 1.0, 3.0, 10.0):
            direct = np.concatenate((retained_equalized, weight * morphology_equalized), axis=1).astype(
                np.float32
            )
            centered_pca = PCA(
                n_components=min(64, direct.shape[1], direct.shape[0] - 1),
                svd_solver="full",
            ).fit_transform(direct).astype(np.float32)
            for representation_name, merged in (
                ("BLOCK_WEIGHTED_DIRECT", direct),
                ("BLOCK_WEIGHTED_PCA", centered_pca),
            ):
                for seed in range(5):
                    started = time.perf_counter()
                    partition = KMeans(
                        n_clusters=k,
                        random_state=seed,
                        n_init=20,
                    ).fit_predict(merged)
                    rows.append(
                        row_for_partition(
                            lane,
                            "DIRECT_CONTROL_REV2",
                            f"{representation_name}_KMEANS",
                            f"{variant}_{representation_name}_W{weight:g}_S{seed}",
                            variant,
                            json.dumps(
                                {
                                    "weight": weight,
                                    "seed": seed,
                                    "retained_block_dim": int(retained_block.shape[1]),
                                    "morphology_block_dim": int(feature.shape[1]),
                                    "block_dimension_equalization": True,
                                    "post_weight_column_standardization": False,
                                },
                                separators=(",", ":"),
                            ),
                            partition,
                            initial,
                            labels,
                            mask,
                            graph,
                            k,
                            authority,
                            time.perf_counter() - started,
                        )
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--night15f-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", nargs="+", default=["A1", "D1", "tonsil_s3"])
    parser.add_argument("--random-count", type=int, default=90)
    args = parser.parse_args()
    registry = json.loads(args.night15f_registry.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    summary = {}
    canonical_lane_seed_offset = {"A1": 0, "D1": 1, "tonsil_s3": 2}
    for lane in args.lanes:
        if lane not in canonical_lane_seed_offset:
            raise ValueError(f"no registered random-search seed offset for lane {lane!r}")
        lane_index = canonical_lane_seed_offset[lane]
        data, graph, graphs, retained, initial, labels, mask, k, morph = load_lane(args, lane)
        authority = evaluate(labels, mask, initial, graph)
        registered = registry["lanes"][lane]
        if abs(authority["absolute_ari"] - float(registered["absolute_ari"])) > 1e-12 or abs(
            authority["absolute_nmi"] - float(registered["absolute_nmi"])
        ) > 1e-12:
            raise RuntimeError(f"{lane}: Night-15F authority metrics mismatch")
        base = parse_base(registered["config"])
        variants = feature_variants(morph)
        rows = [
            row_for_partition(
                lane, "AUTHORITY", "NIGHT15F_MOLECULAR_ONLY", "NIGHT15F_AUTHORITY", "NONE", "{}",
                initial, initial, labels, mask, graph, k, authority, 0.0,
                {"optional_view_present": 0, "missing_view_exact_fallback": 1},
            )
        ]
        rows.extend(run_direct_controls(lane, retained, variants, initial, labels, mask, graph, k, authority))
        configs = designed_configs(base) + random_configs(base, args.random_count, 20260824 + lane_index)
        best_partition = initial.copy()
        for variant_index, (variant, feature) in enumerate(variants.items()):
            evidence = prepare_optional_morphology_evidence(
                graphs,
                retained,
                [np.asarray(data["view1"]), np.asarray(data["view2"])],
                [feature],
                [np.ones(len(feature), dtype=np.float32)],
                optional_dim=feature.shape[1],
                optional_edge_dim=min(24, feature.shape[1]),
            )
            for config_index, config in enumerate(configs):
                identifier = candidate_id(variant, config, f"O{variant_index:02d}_{config_index:04d}")
                started = time.perf_counter()
                try:
                    partition, diagnostics = optional_morphology_expansion(initial, k, evidence, config)
                    row = row_for_partition(
                        lane, "OPTIONAL_ENERGY_SEARCH", "OPTIONAL_MORPHOLOGY_ALPHA_EXPANSION",
                        identifier, variant, json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
                        partition, initial, labels, mask, graph, k, authority,
                        time.perf_counter() - started, diagnostics,
                    )
                    if selection_key(row) > selection_key(max(rows, key=selection_key)):
                        best_partition = partition.copy()
                    rows.append(row)
                except Exception as error:
                    rows.append(
                        {
                            "lane": lane,
                            "stage": "OPTIONAL_ENERGY_SEARCH",
                            "algorithm": "OPTIONAL_MORPHOLOGY_ALPHA_EXPANSION",
                            "config_id": identifier,
                            "feature_variant": variant,
                            "config_json": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
                            "status": "FAILED",
                            "failure": repr(error),
                            "wall_seconds": time.perf_counter() - started,
                        }
                    )
        write_rows(args.output / f"{lane}_all_run_ledger.csv", rows)
        balanced = max(rows, key=selection_key)
        max_ari = max((row for row in rows if row.get("status") == "PASS"), key=lambda row: float(row["absolute_ari"]))
        max_nmi = max((row for row in rows if row.get("status") == "PASS"), key=lambda row: float(row["absolute_nmi"]))
        summary[lane] = {"balanced": balanced, "max_ari": max_ari, "max_nmi": max_nmi, "rows": len(rows)}
        print(json.dumps({"lane": lane, "balanced": balanced, "max_ari": max_ari, "max_nmi": max_nmi}, indent=2))
    (args.output / "search_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
