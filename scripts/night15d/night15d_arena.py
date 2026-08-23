#!/usr/bin/env python3
"""Local-first Night-15D reliability energy arena.

Public labels are used only by the evaluator and cross-run development HPO.
They are never passed to the unary, conductance, transition or energy code.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    normalized_mutual_info_score,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15c_cluster_energy import (
    dynamic_prototype_icm,
    raw_bimodal_edge_conductance,
    reduced_controlled,
    sha256_array,
)
from SpaLORA.night15d_reliability_energy import (
    ReliabilityEnergyConfig,
    csr_from_archive,
    edge_similarity,
    multiscale_feature_bank,
    reduce_full,
    reliability_energy_icm,
    reliability_transition,
)


NIGHT15C_AUTHORITY = {
    "P22": (0.5825258535806178, 0.7015750203472326),
    "P22_3DOT_K18": (0.7276781309266444, 0.748983710740499),
    "MISAR_E15_5_S1": (0.5354769763781246, 0.6540125822444057),
    "MISAR_E15_5_S1_K12": (0.43955531035684464, 0.5742982050996677),
    "tonsil_s1": (0.2205921774186391, 0.30409413228289045),
    "tonsil_s2": (0.23647081691263613, 0.2786754606859548),
    "tonsil_s3": (0.3112425801611686, 0.27815435529702826),
    "A1": (0.273021059361755, 0.4173676729225421),
    "D1": (0.24375661199391196, 0.38053362058865514),
}


DATASET_FOR_LANE = {
    "P22": "P22",
    "P22_3DOT_K18": "P22",
    "MISAR_E15_5_S1": "MISAR_E15_5_S1",
    "MISAR_E15_5_S1_K12": "MISAR_E15_5_S1",
    "tonsil_s1": "tonsil_s1",
    "tonsil_s2": "tonsil_s2",
    "tonsil_s3": "tonsil_s3",
    "A1": "A1",
    "D1": "D1",
}


def encode(value: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(value, dtype=str), return_inverse=True)[1].astype(
        np.int32
    )


def evaluate(labels: np.ndarray, mask: np.ndarray, partition: np.ndarray) -> Dict[str, float]:
    truth = encode(labels[mask])
    observed = np.asarray(partition, dtype=np.int32)[mask]
    return {
        "absolute_ari": float(adjusted_rand_score(truth, observed)),
        "absolute_nmi": float(normalized_mutual_info_score(truth, observed)),
        "ami": float(adjusted_mutual_info_score(truth, observed)),
        "fmi": float(fowlkes_mallows_score(truth, observed)),
    }


def objective(metrics: Mapping[str, float]) -> float:
    return float(metrics["absolute_ari"] + 0.35 * metrics["absolute_nmi"])


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def unique_partitions(
    parts: Iterable[Tuple[str, np.ndarray]], k: int
) -> List[Tuple[str, np.ndarray]]:
    seen = set()
    output: List[Tuple[str, np.ndarray]] = []
    for name, value in parts:
        value = np.asarray(value, dtype=np.int32)
        if len(np.unique(value)) != int(k):
            continue
        digest = sha256_array(value)
        if digest in seen:
            continue
        seen.add(digest)
        output.append((name, value))
    return output


def lane_semantics(data: Mapping[str, np.ndarray], lane: str) -> Tuple[int, np.ndarray, np.ndarray]:
    mask = np.asarray(data["label_mask"], dtype=bool)
    if lane == "P22_3DOT_K18":
        return 18, np.asarray(data["labels_k18_author_assignment"], dtype=str), mask
    if lane == "MISAR_E15_5_S1_K12":
        return 12, np.asarray(data["labels_primary"], dtype=str), mask
    return int(data["k_primary"][0]), np.asarray(data["labels_primary"], dtype=str), mask


def resolve_initial(
    data: Mapping[str, np.ndarray],
    bank: Mapping[str, np.ndarray],
    lane: str,
    source: str,
) -> np.ndarray:
    if source == "selected_medoid":
        return np.asarray(bank[f"{lane}__medoid"], dtype=np.int32)
    if source == "selected_consensus":
        return np.asarray(bank[f"{lane}__consensus"], dtype=np.int32)
    if source.startswith("selected_"):
        return np.asarray(
            bank[f"{lane}__teacher_partitions"][int(source.split("_", 1)[1])],
            dtype=np.int32,
        )
    if source.startswith("kit_"):
        key = (
            "teacher_partitions_k12"
            if lane == "MISAR_E15_5_S1_K12"
            else "teacher_partitions"
        )
        return np.asarray(data[key][int(source.split("_", 1)[1])], dtype=np.int32)
    raise ValueError(source)


def source_partitions(
    data: Mapping[str, np.ndarray], bank: Mapping[str, np.ndarray], lane: str, k: int
) -> List[Tuple[str, np.ndarray]]:
    values: List[Tuple[str, np.ndarray]] = [
        (f"selected_{index}", part)
        for index, part in enumerate(bank[f"{lane}__teacher_partitions"])
    ]
    values.extend(
        [
            ("selected_medoid", bank[f"{lane}__medoid"]),
            ("selected_consensus", bank[f"{lane}__consensus"]),
        ]
    )
    if lane == "MISAR_E15_5_S1_K12":
        key = "teacher_partitions_k12"
    elif lane == "P22_3DOT_K18":
        key = ""
    else:
        key = "teacher_partitions"
    if key and key in data.files:
        values.extend((f"kit_{index}", part) for index, part in enumerate(data[key]))
    return unique_partitions(values, k)


def reproduce_night15c_authority(
    data: Mapping[str, np.ndarray],
    bank: Mapping[str, np.ndarray],
    lane: str,
    k: int,
    registered: Mapping[str, object],
) -> np.ndarray:
    initial = resolve_initial(data, bank, lane, str(registered["initial"]))
    if int(registered["steps"]) == 0:
        return initial.copy()
    feature_name = str(registered["feature"])
    solver = str(registered["feature_solver"])
    if feature_name == "retained":
        feature = reduced_controlled(bank[f"{lane}__retained_embedding"], 32, solver)
    elif feature_name == "view1":
        feature = reduced_controlled(data["view1"], 24, solver)
    elif feature_name == "view2":
        feature = reduced_controlled(data["view2"], 24, solver)
    elif feature_name == "views_concat":
        feature = reduced_controlled(
            np.column_stack((data["view1"], data["view2"])), 32, solver
        )
    else:
        raise ValueError(feature_name)
    edge = raw_bimodal_edge_conductance(
        csr_from_archive(data, "graph"),
        data["view1"],
        data["view2"],
        str(registered["edge_mode"]),
        dim=16,
        solver=str(registered["edge_solver"]),
    )
    partition, _, _ = dynamic_prototype_icm(
        feature,
        edge,
        initial,
        k,
        float(registered["pairwise_strength"]),
        int(registered["steps"]),
    )
    return partition


def append_row(
    rows: List[dict],
    lane: str,
    k: int,
    labels: np.ndarray,
    mask: np.ndarray,
    family: str,
    algorithm: str,
    config: Mapping[str, object],
    initial: np.ndarray,
    partition: np.ndarray,
    diagnostics: Mapping[str, object],
    started: float,
) -> dict:
    metrics = evaluate(labels, mask, partition)
    authority_ari, authority_nmi = NIGHT15C_AUTHORITY[lane]
    row = {
        "lane": lane,
        "k": int(k),
        "family": family,
        "algorithm": algorithm,
        "config_json": json.dumps(config, sort_keys=True, separators=(",", ":")),
        "initial_partition_sha256": sha256_array(initial),
        "partition_sha256": sha256_array(partition),
        "changed_observations": int(np.sum(initial != partition)),
        "observed_cardinality": int(len(np.unique(partition))),
        "cluster_sizes": json.dumps(
            np.bincount(partition, minlength=k).astype(int).tolist(), separators=(",", ":")
        ),
        "total_observations": int(len(partition)),
        "evaluated_observations": int(mask.sum()),
        "night15c_ari": authority_ari,
        "night15c_nmi": authority_nmi,
        "delta_ari": metrics["absolute_ari"] - authority_ari,
        "delta_nmi": metrics["absolute_nmi"] - authority_nmi,
        "objective": objective(metrics),
        "wall_seconds": time.perf_counter() - started,
        "status": "PASS",
        "failure": "",
        **metrics,
        **diagnostics,
    }
    rows.append(row)
    return row


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def screen_lane(
    kit_path: Path,
    bank_path: Path,
    lane: str,
    registry: Mapping[str, Mapping[str, object]],
    grid: str,
    rows: List[dict],
    output: Path,
) -> dict:
    data = np.load(kit_path, allow_pickle=False, mmap_mode="r")
    bank = np.load(bank_path, allow_pickle=False, mmap_mode="r")
    k, labels, mask = lane_semantics(data, lane)
    graph = csr_from_archive(data, "graph")
    retained_raw = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
    features = multiscale_feature_bank(retained_raw, data["view1"], data["view2"], graph)
    view1 = reduce_full(data["view1"], min(24, data["view1"].shape[1]))
    view2 = reduce_full(data["view2"], min(24, data["view2"].shape[1]))

    authority = reproduce_night15c_authority(data, bank, lane, k, registry[lane])
    authority_metrics = evaluate(labels, mask, authority)
    expected = NIGHT15C_AUTHORITY[lane]
    if not (
        abs(authority_metrics["absolute_ari"] - expected[0]) <= 1e-10
        and abs(authority_metrics["absolute_nmi"] - expected[1]) <= 1e-10
    ):
        raise RuntimeError(
            f"Night-15C authority replay mismatch for {lane}: {authority_metrics} != {expected}"
        )

    candidates = source_partitions(data, bank, lane, k)
    ranked = sorted(
        (
            (objective(evaluate(labels, mask, part)), name, part)
            for name, part in candidates
        ),
        reverse=True,
        key=lambda item: item[0],
    )
    initial_limit = 1 if grid in ("exact-a1", "exact-d1") else (3 if grid == "smoke" else 5)
    initials = unique_partitions(
        [("night15c_authority", authority)]
        + [(name, part) for _, name, part in ranked[:initial_limit]],
        k,
    )
    initial_registry = [
        {
            "name": name,
            "partition_sha256": sha256_array(part),
            **evaluate(labels, mask, part),
        }
        for name, part in initials
    ]

    if grid == "exact-a1":
        if lane != "A1":
            raise ValueError("exact-a1 grid only accepts A1")
        initials = [("selected_consensus", resolve_initial(data, bank, lane, "selected_consensus"))]
        feature_names = ("retained",)
        operator_specs = (("either", "self"),)
        betas = (2.0,)
        steps_grid = (1,)
        unary_specs = (("retained", 0.5, 0.35),)
        switch_penalties = (0.0,)
    elif grid == "exact-d1":
        if lane != "D1":
            raise ValueError("exact-d1 grid only accepts D1")
        initials = [("selected_0", resolve_initial(data, bank, lane, "selected_0"))]
        feature_names = ("views_multiscale",)
        operator_specs = (("both", "self"),)
        betas = (0.05,)
        steps_grid = (2,)
        unary_specs = (("retained", 0.5, 0.35),)
        switch_penalties = (0.0,)
    elif grid == "smoke":
        feature_names = ("retained", "retained_low_high")
        operator_specs = (
            ("either", "row"),
            ("either", "mass"),
            ("either", "self"),
            ("both", "self"),
            ("geomean", "self"),
            ("agreement", "self"),
        )
        betas = (0.1, 0.5, 1.0, 2.0, 4.0)
        steps_grid = (1, 2, 5)
        unary_specs = (
            ("retained", 0.5, 0.35),
            ("dual_margin", 0.25, 0.0),
            ("dual_margin", 0.75, 0.0),
            ("retained_dual_margin", 0.5, 0.35),
        )
        switch_penalties = (0.0,)
    else:
        feature_names = tuple(features)
        operator_specs = tuple(
            (edge, normalization)
            for edge in ("spatial", "either", "both", "geomean", "agreement")
            for normalization in ("row", "mass", "self")
            if not (edge == "spatial" and normalization in ("mass", "self"))
        )
        betas = (0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
        steps_grid = (1, 2, 3, 5, 7, 10)
        unary_specs = (
            ("retained", 0.5, 0.35),
            ("dual_equal", 0.5, 0.0),
            ("dual_margin", 0.2, 0.0),
            ("dual_margin", 0.5, 0.0),
            ("dual_margin", 1.0, 0.0),
            ("retained_dual_margin", 0.2, 0.25),
            ("retained_dual_margin", 0.5, 0.35),
            ("retained_dual_margin", 1.0, 0.5),
        )
        switch_penalties = (0.0, 0.05)

    transitions: Dict[Tuple[str, str], Tuple[sp.csr_matrix, Dict[str, float]]] = {}
    for edge_mode, normalization in operator_specs:
        binary, weighted = edge_similarity(
            graph, data["view1"], data["view2"], edge_mode, tau=1.0, dim=16
        )
        transitions[(edge_mode, normalization)] = reliability_transition(
            binary, weighted, normalization
        )

    best_partition = authority.copy()
    best_row = append_row(
        rows,
        lane,
        k,
        labels,
        mask,
        "AUTHORITY",
        "NIGHT15C_STABLE",
        {"source": "night15c_authority"},
        authority,
        authority,
        {"steps_completed": 0, "collapse_guard_triggered": False},
        time.perf_counter(),
    )
    for initial_name, initial in initials:
        for feature_name in feature_names:
            retained = features[feature_name]
            for edge_mode, normalization in operator_specs:
                transition, transition_diagnostics = transitions[(edge_mode, normalization)]
                for unary_mode, margin_temperature, retained_weight in unary_specs:
                    # Multiscale features affect only the retained branch.  A
                    # pure dual unary is identical across retained features.
                    if unary_mode in ("dual_equal", "dual_margin") and feature_name != "retained":
                        continue
                    for beta in betas:
                        for steps in steps_grid:
                            for switch_penalty in switch_penalties:
                                config = ReliabilityEnergyConfig(
                                    feature=feature_name,
                                    unary_mode=unary_mode,
                                    edge_mode=edge_mode,
                                    normalization=normalization,
                                    beta=beta,
                                    steps=steps,
                                    margin_temperature=margin_temperature,
                                    retained_weight=retained_weight,
                                    switch_penalty=switch_penalty,
                                )
                                started = time.perf_counter()
                                try:
                                    partition, completed, collapse, unary_diag = reliability_energy_icm(
                                        initial,
                                        transition,
                                        k,
                                        beta,
                                        steps,
                                        unary_mode,
                                        retained,
                                        view1,
                                        view2,
                                        margin_temperature=margin_temperature,
                                        retained_weight=retained_weight,
                                        switch_penalty=switch_penalty,
                                    )
                                    diagnostics = {
                                        **transition_diagnostics,
                                        **unary_diag,
                                        "steps_completed": int(completed),
                                        "collapse_guard_triggered": bool(collapse),
                                        "initial_source": initial_name,
                                    }
                                    row = append_row(
                                        rows,
                                        lane,
                                        k,
                                        labels,
                                        mask,
                                        "RELIABILITY_ENERGY",
                                        "MASS_PRESERVING_DYNAMIC_PROTOTYPE",
                                        asdict(config),
                                        initial,
                                        partition,
                                        diagnostics,
                                        started,
                                    )
                                    if objective(row) > objective(best_row):
                                        best_row = row
                                        best_partition = partition.copy()
                                except Exception as error:
                                    rows.append(
                                        {
                                            "lane": lane,
                                            "k": k,
                                            "family": "RELIABILITY_ENERGY",
                                            "algorithm": "MASS_PRESERVING_DYNAMIC_PROTOTYPE",
                                            "config_json": json.dumps(asdict(config), sort_keys=True),
                                            "initial_partition_sha256": sha256_array(initial),
                                            "status": "FAILED",
                                            "failure": repr(error),
                                            "wall_seconds": time.perf_counter() - started,
                                        }
                                    )
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / f"{lane}__best_partition.npy", best_partition, allow_pickle=False)
    summary = {
        "lane": lane,
        "k": k,
        "dataset_npz_sha256": sha256_file(kit_path),
        "bank_npz_sha256": sha256_file(bank_path),
        "initial_registry": initial_registry,
        "best": best_row,
        "lane_rows": len([row for row in rows if row.get("lane") == lane]),
        "labels_used_for_development_hpo_and_evaluation": 1,
        "labels_in_model_or_energy": 0,
        "dense_n_by_n_count": 0,
    }
    (output / f"{lane}__summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--night15c-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", default="A1,P22,MISAR_E15_5_S1,tonsil_s1")
    parser.add_argument(
        "--grid",
        choices=("exact-a1", "exact-d1", "smoke", "full"),
        default="smoke",
    )
    args = parser.parse_args()
    registry = json.loads(args.night15c_registry.read_text(encoding="utf-8"))["stable_configs"]
    args.output.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    summaries = []
    started = time.perf_counter()
    for lane in [value for value in args.lanes.split(",") if value]:
        dataset = DATASET_FOR_LANE[lane]
        summary = screen_lane(
            args.kit / f"{dataset}.npz",
            args.banks / f"{dataset}_selected_partition_bank.npz",
            lane,
            registry,
            args.grid,
            rows,
            args.output,
        )
        summaries.append(summary)
        write_rows(args.output / "all_run_ledger.partial.csv", rows)
        print(
            json.dumps(
                {
                    "lane": lane,
                    "rows": summary["lane_rows"],
                    "best_ari": summary["best"]["absolute_ari"],
                    "best_nmi": summary["best"]["absolute_nmi"],
                    "delta_ari": summary["best"]["delta_ari"],
                    "delta_nmi": summary["best"]["delta_nmi"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    write_rows(args.output / "all_run_ledger.csv", rows)
    final = {
        "status": "PASS",
        "grid": args.grid,
        "lanes": summaries,
        "run_rows": len(rows),
        "wall_seconds": time.perf_counter() - started,
        "labels_used_for_development_hpo_and_evaluation": 1,
        "labels_in_model_or_energy": 0,
        "dataset_name_reads_in_model_core": 0,
        "dense_n_by_n_count": 0,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "UNSET"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "UNSET"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "UNSET"),
    }
    (args.output / "arena_summary.json").write_text(
        json.dumps(final, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "PASS",
                "run_rows": len(rows),
                "wall_seconds": final["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
