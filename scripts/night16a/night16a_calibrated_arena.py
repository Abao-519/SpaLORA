#!/usr/bin/env python3
"""Internal arena for the compact Night-16A self-calibrating formula.

Candidate partitions are produced from observable data and frozen constants.
Public references are read only after all candidates for a lane have been
materialized.  The resulting metrics are an internal discovery ledger; a
separate script freezes the cross-study automatic output.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)
from sklearn.preprocessing import StandardScaler

from SpaLORA.night15e_continuous_reliability_energy import csr_from_archive, reduce_full
from SpaLORA.night16a_self_calibrating_energy import (
    CalibrationConstants,
    calibrate_from_statistics,
    observable_statistics,
    run_calibrated_energy,
)
from SpaLORA.night15f_multiscale_expansion import prepare_expansion_evidence
from SpaLORA.night15g_optional_morphology_energy import prepare_optional_morphology_evidence


LANE_DATASET = {
    "A1": "A1", "D1": "D1", "tonsil_s1": "tonsil_s1", "tonsil_s2": "tonsil_s2",
    "tonsil_s3": "tonsil_s3", "P22": "P22", "P22_3DOT_K18": "P22",
    "MISAR_E15_5_S1": "MISAR_E15_5_S1", "MISAR_E15_5_S1_K12": "MISAR_E15_5_S1",
}


def sha(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def encode(value):
    return np.unique(np.asarray(value, dtype=str), return_inverse=True)[1].astype(np.int32)


def lane_semantics(data, lane):
    if lane == "P22_3DOT_K18":
        return 18, np.asarray(data["labels_k18_author_assignment"]), np.ones(len(data["ids"]), dtype=bool)
    if lane == "MISAR_E15_5_S1_K12":
        return 12, np.asarray(data["labels_primary"]), np.asarray(data["label_mask"], dtype=bool)
    return int(data["k_primary"][0]), np.asarray(data["labels_primary"]), np.asarray(data["label_mask"], dtype=bool)


def evaluate(labels, mask, partition):
    truth = encode(labels[mask])
    observed = np.asarray(partition, dtype=np.int32)[mask]
    return {
        "absolute_ari": float(adjusted_rand_score(truth, observed)),
        "absolute_nmi": float(normalized_mutual_info_score(truth, observed)),
        "ami": float(adjusted_mutual_info_score(truth, observed)),
        "fmi": float(fowlkes_mallows_score(truth, observed)),
    }


def standardize(value):
    return StandardScaler().fit_transform(np.asarray(value, dtype=np.float32)).astype(np.float32)


def optional_view(path):
    if not path.exists():
        return None, None
    data = np.load(path, allow_pickle=False, mmap_mode="r")
    hand = reduce_full(data["handcrafted"], 32)
    resnet = reduce_full(data["resnet18"], 32)
    combined = reduce_full(np.concatenate((hand, resnet), axis=1), 48)
    return combined, np.asarray(data["presence_mask"], dtype=np.float32)


def start_bank(npz, ledger_rows, lane, max_heuristic=5):
    available = []
    index = 0
    while f"{lane}__p{index:03d}" in npz:
        name = str(npz[f"{lane}__n{index:03d}"][0])
        partition = np.asarray(npz[f"{lane}__p{index:03d}"], dtype=np.int32)
        row = ledger_rows[(lane, name)]
        available.append((name, partition, row))
        index += 1
    mandatory = []
    patterns = (
        "authority::night15f", "bank::medoid", "bank::consensus",
        "night15g::A1__balanced", "night15g::D1__nonmicro_balanced",
        "night15g::tonsil_s3__balanced",
    )
    for item in available:
        if item[0] in patterns:
            mandatory.append(item)
    if int(max_heuristic) <= 0:
        return [(name, partition) for name, partition, _ in mandatory]
    # A fixed label-free score broadens the bank without using the evaluator.
    matrix = []
    keys = ("graph_same_fraction", "partition_centrality_median_ari", "normalized_cluster_entropy",
            "min_cluster_relative_to_equal", "separation__retained", "separation__view1", "separation__view2")
    for _, _, row in available:
        matrix.append([float(row.get(key, 0.0) or 0.0) for key in keys])
    matrix = np.asarray(matrix, dtype=np.float64)
    median = np.median(matrix, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(matrix - median), axis=0), np.std(matrix, axis=0))
    z = np.clip((matrix - median) / np.maximum(scale, 1e-8), -5.0, 5.0)
    heuristic = 0.25 * z[:, 0] + 0.15 * z[:, 1] + 0.10 * z[:, 2] + 0.20 * z[:, 3] + 0.10 * z[:, 4] + 0.10 * z[:, 5] + 0.10 * z[:, 6]
    selected = list(mandatory)
    for idx in np.argsort(-heuristic):
        item = available[int(idx)]
        if item[0] not in {name for name, _, _ in selected}:
            selected.append(item)
        if len(selected) >= len(mandatory) + max_heuristic:
            break
    return [(name, partition) for name, partition, _ in selected]


def generated_start_bank(npz, ledger_rows, lane, maximum=10):
    """Return starts reconstructed from the fixed generic generators only.

    Historical authority, teacher, consensus and Night-15G profile partitions
    are intentionally excluded because their configuration/profile selection
    used the current benchmark's public reference during earlier development.
    Ranking below is computed solely from within-lane observable descriptors.
    """

    available = []
    index = 0
    while f"{lane}__p{index:03d}" in npz:
        name = str(npz[f"{lane}__n{index:03d}"][0])
        if name.startswith(("kmeans::", "gmm_diag::", "gmm_tied::")):
            row = ledger_rows[(lane, name)]
            if float(row["min_cluster_relative_to_equal"]) >= 0.02:
                available.append((name, np.asarray(npz[f"{lane}__p{index:03d}"], dtype=np.int32), row))
        index += 1
    if not available:
        raise RuntimeError(f"{lane}: no structurally eligible generic starts")
    keys = (
        "graph_same_fraction",
        "partition_centrality_median_ari",
        "normalized_cluster_entropy",
        "min_cluster_relative_to_equal",
        "separation__retained",
        "separation__view1",
        "separation__view2",
        "margin__retained",
        "margin__view1",
        "margin__view2",
    )
    matrix = np.asarray(
        [[float(row.get(key, 0.0) or 0.0) for key in keys] for _, _, row in available],
        dtype=np.float64,
    )
    median = np.median(matrix, axis=0)
    scale = np.maximum(1.4826 * np.median(np.abs(matrix - median), axis=0), np.std(matrix, axis=0))
    z = np.clip((matrix - median) / np.maximum(scale, 1e-8), -5.0, 5.0)
    # Fixed data-derived ranking: structure and centrality are tempered by
    # balance and agreement with every molecular representation.
    weight = np.asarray([0.18, 0.18, 0.12, 0.12, 0.10, 0.07, 0.07, 0.06, 0.05, 0.05])
    order = np.argsort(-(z @ weight), kind="mergesort")[: max(1, int(maximum))]
    return [(available[int(i)][0], available[int(i)][1]) for i in order]


def fixed_start_bank(npz, lane, name="kmeans::retained::s0"):
    index = 0
    while f"{lane}__p{index:03d}" in npz:
        observed = str(npz[f"{lane}__n{index:03d}"][0])
        if observed == name:
            return [(observed, np.asarray(npz[f"{lane}__p{index:03d}"], dtype=np.int32))]
        index += 1
    raise RuntimeError(f"{lane}: fixed start {name!r} is unavailable")


def graph_same_fraction(graph, partition):
    coo = sp.triu(sp.csr_matrix(graph), k=1, format="coo")
    if not coo.nnz:
        return 0.0
    return float(np.average(partition[coo.row] == partition[coo.col], weights=np.maximum(coo.data, 1e-8)))


def partition_descriptors(partition, k, graph, blocks):
    partition = np.asarray(partition, dtype=np.int32)
    sizes = np.bincount(partition, minlength=int(k)).astype(np.float64)
    probability = sizes / max(float(sizes.sum()), 1.0)
    positive = probability[probability > 0]
    output = {
        "selector_graph_same_fraction": graph_same_fraction(graph, partition),
        "selector_normalized_cluster_entropy": float(-np.sum(positive * np.log(positive)) / max(np.log(k), 1e-8)),
        "selector_min_cluster_relative_to_equal": float(sizes.min() / max(len(partition) / k, 1.0)),
    }
    for name, feature in blocks.items():
        centers = np.stack([feature[partition == group].mean(axis=0) for group in range(k)])
        distance = np.mean((feature[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        assigned = distance[np.arange(len(partition)), partition]
        global_total = np.mean(np.sum((feature - feature.mean(axis=0)) ** 2, axis=1))
        nearest = np.partition(distance, 1, axis=1)[:, :2]
        output[f"selector_separation__{name}"] = float(
            1.0 - np.mean(assigned) / max(global_total / max(feature.shape[1], 1), 1e-8)
        )
        output[f"selector_margin__{name}"] = float(
            np.mean((nearest[:, 1] - nearest[:, 0]) / np.maximum(nearest[:, 1], 1e-6))
        )
    return output


def constants_bank(count, seed):
    designed = [
        CalibrationConstants(),
        CalibrationConstants(pairwise_factor=2.0, self_return_factor=1.0, size_factor=0.05),
        CalibrationConstants(pairwise_factor=4.0, self_return_factor=2.0, size_factor=0.10),
        CalibrationConstants(pairwise_factor=12.0, self_return_factor=6.0, size_factor=0.20),
        CalibrationConstants(pairwise_factor=20.0, self_return_factor=10.0, size_factor=0.30),
        CalibrationConstants(pairwise_factor=8.0, self_return_factor=0.0, size_factor=0.15),
        CalibrationConstants(pairwise_factor=8.0, self_return_factor=4.0, size_factor=0.0),
        CalibrationConstants(pairwise_factor=0.0, self_return_factor=0.0, size_factor=0.0),
    ]
    rng = np.random.default_rng(seed)
    while len(designed) < count:
        designed.append(
            CalibrationConstants(
                pairwise_factor=float(np.exp(rng.uniform(np.log(0.5), np.log(30.0)))),
                self_return_factor=float(np.exp(rng.uniform(np.log(0.25), np.log(16.0)))),
                size_factor=float(rng.uniform(0.0, 0.40)),
                retained_prior=float(np.exp(rng.uniform(np.log(0.25), np.log(4.0)))),
                optional_factor=float(np.exp(rng.uniform(np.log(0.10), np.log(2.0)))),
                scale_temperature=float(np.exp(rng.uniform(np.log(0.05), np.log(1.0)))),
                expansion_cycles=int(rng.integers(1, 3)),
            )
        )
    return designed[:count]


def config_id(constants):
    payload = json.dumps(asdict(constants), sort_keys=True, separators=(",", ":"))
    return "SC_" + hashlib.sha256(payload.encode()).hexdigest()[:14]


def write_csv(path, rows):
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--start-probe", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", nargs="*", default=list(LANE_DATASET))
    parser.add_argument("--constant-count", type=int, default=32)
    parser.add_argument("--max-heuristic-starts", type=int, default=5)
    parser.add_argument(
        "--start-policy",
        choices=("development_authority", "generated_only", "fixed_kmeans_retained_s0"),
        default="development_authority",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    probe_npz = np.load(args.start_probe / "start_bank_partitions.npz", allow_pickle=False)
    with (args.start_probe / "start_bank_probe_ledger.csv").open(encoding="utf-8", newline="") as handle:
        probe_rows = list(csv.DictReader(handle))
    lookup = {(row["lane"], row["candidate"]): row for row in probe_rows}
    constants = constants_bank(args.constant_count, 20260824)
    all_rows = []
    partition_payload = {}
    calibration_registry = {}
    lane_summary = {}
    for lane in args.lanes:
        started = time.perf_counter()
        dataset = LANE_DATASET[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(args.banks / f"{dataset}_selected_partition_bank.npz", allow_pickle=False, mmap_mode="r")
        k, labels, mask = lane_semantics(data, lane)
        graph = csr_from_archive(data, "graph")
        graphs = (csr_from_archive(data, "operator4"), graph, csr_from_archive(data, "operator18"))
        retained = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
        if args.start_policy == "fixed_kmeans_retained_s0":
            starts = fixed_start_bank(probe_npz, lane)
        elif args.start_policy == "generated_only":
            starts = generated_start_bank(probe_npz, lookup, lane, args.max_heuristic_starts)
        else:
            starts = start_bank(probe_npz, lookup, lane, args.max_heuristic_starts)
        morph_path = args.morphology / f"{dataset}_morphology_views.npz"
        morphology, presence = optional_view(morph_path)
        selector_blocks = {
            "retained": reduce_full(retained, min(32, retained.shape[1])),
            "view1": reduce_full(data["view1"], min(24, data["view1"].shape[1])),
            "view2": reduce_full(data["view2"], min(24, data["view2"].shape[1])),
        }
        if morphology is not None:
            selector_blocks["optional"] = morphology
            optional_cached = prepare_optional_morphology_evidence(
                graphs,
                retained,
                (data["view1"], data["view2"]),
                (morphology,),
                (presence,),
            )
        else:
            optional_cached = None
        evidence = prepare_expansion_evidence(graphs, retained, data["view1"], data["view2"])
        statistics = observable_statistics(
            evidence,
            retained,
            data["view1"],
            data["view2"],
            [item[1] for item in starts],
            k,
            morphology,
        )
        generated = []
        for constants_index, constant in enumerate(constants):
            calibrated = calibrate_from_statistics(statistics, constant)
            identifier = config_id(constant)
            calibration_registry[f"{lane}::{identifier}"] = {
                "constants": asdict(constant),
                "statistics_and_parameters": calibrated.statistics,
            }
            for start_name, initial in starts:
                run_started = time.perf_counter()
                try:
                    partition, diagnostics = run_calibrated_energy(
                        initial, k, evidence, calibrated, graphs, retained, data["view1"], data["view2"],
                        morphology, presence, optional_cached,
                    )
                    status, failure = "PASS", ""
                except Exception as exc:
                    partition = initial.copy()
                    diagnostics = {}
                    status, failure = "FAILED", f"{type(exc).__name__}: {exc}"
                generated.append(
                    (
                        identifier,
                        constants_index,
                        constant,
                        calibrated,
                        start_name,
                        partition,
                        diagnostics,
                        status,
                        failure,
                        time.perf_counter() - run_started,
                    )
                )
        # Only now does this development runner expose the public reference.
        for (
            identifier,
            constants_index,
            constant,
            calibrated,
            start_name,
            partition,
            diagnostics,
            status,
            failure,
            wall,
        ) in generated:
            sizes = np.bincount(partition, minlength=k)
            partition_key = f"{lane}__{identifier}__{sha(partition)[:12]}"
            metrics = evaluate(labels, mask, partition) if status == "PASS" else {"absolute_ari": np.nan, "absolute_nmi": np.nan, "ami": np.nan, "fmi": np.nan}
            row = {
                "lane": lane,
                "candidate_key": f"{lane}::{identifier}::{start_name}::{sha(partition)[:16]}",
                "config_id": identifier,
                "constants_index": constants_index,
                "start_name": start_name,
                "status": status,
                "failure": failure,
                "partition_sha256": sha(partition),
                "partition_key": partition_key,
                "cluster_sizes": json.dumps(sizes.tolist(), separators=(",", ":")),
                "min_cluster_size": int(sizes.min()),
                "min_cluster_relative_to_equal": float(sizes.min() / max(len(partition) / k, 1.0)),
                "total_observations": len(partition),
                "evaluated_observations": int(mask.sum()),
                "k": k,
                "wall_seconds": wall,
                **metrics,
                **partition_descriptors(partition, k, graph, selector_blocks),
                **{
                    f"calibration_stat__{key}": value
                    for key, value in calibrated.statistics.items()
                    if isinstance(value, (int, float))
                },
                **{
                    f"calibration_constant__{key}": value
                    for key, value in asdict(constant).items()
                    if isinstance(value, (int, float))
                },
                **{key: value for key, value in diagnostics.items() if isinstance(value, (int, float, str))},
            }
            all_rows.append(row)
            partition_payload[partition_key] = partition.astype(np.int32)
        lane_summary[lane] = {
            "tensor_shapes": {
                "retained": list(retained.shape), "view1": list(data["view1"].shape), "view2": list(data["view2"].shape),
                "optional": None if morphology is None else list(morphology.shape),
            },
            "graph_shapes": [[int(g.shape[0]), int(g.shape[1]), int(g.nnz)] for g in graphs],
            "start_count": len(starts),
            "starts": [name for name, _ in starts],
            "start_policy": args.start_policy,
            "constant_count": len(constants),
            "wall_seconds": time.perf_counter() - started,
        }
    write_csv(args.output / "calibrated_arena_ledger.csv", all_rows)
    np.savez_compressed(args.output / "calibrated_arena_partitions.npz", **partition_payload)
    (args.output / "calibration_registry.json").write_text(json.dumps(calibration_registry, indent=2) + "\n", encoding="utf-8")
    (args.output / "arena_summary.json").write_text(json.dumps({"lanes": lane_summary, "rows": len(all_rows)}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
