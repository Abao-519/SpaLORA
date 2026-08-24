"""Local-first Night-16C family search, freeze, replay, and evaluation runner."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    roc_auc_score,
    v_measure_score,
)

from SpaLORA.night16c_cmbf_tpr import (
    CMBFTPRConfig,
    align_partition,
    cmbf_tpr,
    encode_partition,
    partition_sha256,
    prepare_boundary_evidence,
    sparse_graph_from_csr_arrays,
    trusted_prototype_refinement,
)


PRIMARY_LANES = (
    "A1",
    "D1",
    "tonsil_s1",
    "tonsil_s2",
    "tonsil_s3",
    "P22",
    "MISAR_E15_5_S1",
)
FAMILIES = {
    "RNA_PROTEIN": ("A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3"),
    "RNA_CHROMATIN": ("P22", "MISAR_E15_5_S1"),
}
DISCOVERY = {
    "RNA_PROTEIN": ("A1", "tonsil_s1"),
    "RNA_CHROMATIN": ("P22", "MISAR_E15_5_S1"),
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def config_id(config: CMBFTPRConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return "CMBF_" + sha256_bytes(payload.encode())[:16]


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("refusing to write empty CSV")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def load_graph(z: np.lib.npyio.NpzFile) -> sp.csr_matrix:
    return sparse_graph_from_csr_arrays(
        z["graph__data"], z["graph__indices"], z["graph__indptr"], z["graph__shape"]
    )


def load_producer_input(kit_root: Path, start_root: Path, lane: str) -> dict[str, object]:
    with np.load(kit_root / f"{lane}.npz", allow_pickle=False) as z:
        value = {
            "ids": z["ids"].copy(),
            "coordinates": z["coordinates"].astype(np.float32),
            "view1": z["view1"].astype(np.float32),
            "view2": z["view2"].astype(np.float32),
            "k": int(z["k_primary"][0]),
            "start_bank": z["teacher_partitions"].astype(np.int32),
            "graph": load_graph(z),
        }
    initial = np.load(start_root / f"{lane}.npy", allow_pickle=False).astype(np.int32)
    if len(initial) != len(value["ids"]):
        raise ValueError(f"{lane}: start observation mismatch")
    value["initial"] = encode_partition(initial)
    return value


def load_evaluation_input(kit_root: Path, lane: str) -> dict[str, np.ndarray]:
    with np.load(kit_root / f"{lane}.npz", allow_pickle=False) as z:
        return {
            "labels": z["labels_primary"].copy(),
            "mask": z["label_mask"].astype(bool),
            "coordinates": z["coordinates"].astype(np.float64),
            "view1": z["view1"].astype(np.float32),
            "view2": z["view2"].astype(np.float32),
            "graph": load_graph(z),
        }


def candidate_configs(count: int = 64) -> list[CMBFTPRConfig]:
    no_op = CMBFTPRConfig(
        pairwise_strength=0.0,
        self_return_strength=0.0,
        anchor_strength=0.0,
        sweeps=0,
    )
    choices = {
        "directional_mix": [0.0, 0.25, 0.50],
        "conflict_pass": [0.0, 0.10, 0.25],
        "pairwise_strength": [0.03, 0.08, 0.16, 0.32],
        "self_return_strength": [0.02, 0.08, 0.20],
        "trust_threshold": [0.40, 0.55, 0.70],
        "anchor_strength": [0.02, 0.08, 0.20],
        "unary_view1_weight": [0.35, 0.50, 0.65],
        "move_boundary_floor": [0.05, 0.12, 0.22],
        "sweeps": [1, 2],
    }
    rng = np.random.default_rng(20260824)
    configs = [no_op]
    seen = {config_id(no_op)}
    # Deterministic anchors cover weak/medium/strong spatial energy.
    anchors = [
        CMBFTPRConfig(pairwise_strength=0.03, self_return_strength=0.02, trust_threshold=0.70, anchor_strength=0.20, sweeps=1),
        CMBFTPRConfig(pairwise_strength=0.08, self_return_strength=0.08, trust_threshold=0.55, anchor_strength=0.08, sweeps=1),
        CMBFTPRConfig(pairwise_strength=0.16, self_return_strength=0.20, trust_threshold=0.55, anchor_strength=0.08, sweeps=2, directional_mix=0.25),
        CMBFTPRConfig(pairwise_strength=0.32, self_return_strength=0.20, trust_threshold=0.40, anchor_strength=0.02, sweeps=2, directional_mix=0.50, conflict_pass=0.10),
    ]
    for config in anchors:
        if config_id(config) not in seen:
            configs.append(config)
            seen.add(config_id(config))
    while len(configs) < int(count):
        config = CMBFTPRConfig(
            directional_mix=float(rng.choice(choices["directional_mix"])),
            conflict_pass=float(rng.choice(choices["conflict_pass"])),
            pairwise_strength=float(rng.choice(choices["pairwise_strength"])),
            self_return_strength=float(rng.choice(choices["self_return_strength"])),
            trust_threshold=float(rng.choice(choices["trust_threshold"])),
            anchor_strength=float(rng.choice(choices["anchor_strength"])),
            unary_view1_weight=float(rng.choice(choices["unary_view1_weight"])),
            move_boundary_floor=float(rng.choice(choices["move_boundary_floor"])),
            sweeps=int(rng.choice(choices["sweeps"])),
        )
        identifier = config_id(config)
        if identifier not in seen:
            configs.append(config)
            seen.add(identifier)
    return configs


def _evidence_key(config: CMBFTPRConfig) -> tuple[float, float, bool, bool, bool]:
    return (
        float(config.directional_mix),
        float(config.conflict_pass),
        bool(config.boundary_enabled),
        bool(config.conflict_enabled),
        bool(config.directional_enabled),
    )


def produce_search(args: argparse.Namespace) -> None:
    kit_root = Path(args.kit_root)
    start_root = Path(args.start_root)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    if args.config_registry:
        raw = json.loads(Path(args.config_registry).read_text(encoding="utf-8"))
        configs = [CMBFTPRConfig(**item) for item in raw["configs"]]
    else:
        configs = candidate_configs(args.config_count)
    write_json(out / "config_registry.json", {config_id(c): asdict(c) for c in configs})
    descriptor_rows: list[dict[str, object]] = []
    for lane in FAMILIES[args.family]:
        data = load_producer_input(kit_root, start_root, lane)
        partitions = []
        identifiers = []
        cache: dict[tuple[float, float, bool, bool, bool], object] = {}
        for config in configs:
            started = time.perf_counter()
            key = _evidence_key(config)
            if key not in cache:
                cache[key] = prepare_boundary_evidence(
                    data["view1"], data["view2"], data["graph"], data["initial"], data["start_bank"], config
                )
            partition, detail = trusted_prototype_refinement(data["initial"], cache[key], config)
            identifier = config_id(config)
            identifiers.append(identifier)
            partitions.append(partition)
            descriptor_rows.append(
                {
                    "family": args.family,
                    "lane": lane,
                    "candidate_id": identifier,
                    "config_json": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
                    "status": "PASS",
                    "failure": "",
                    "partition_sha256": detail["partition_sha256"],
                    "initial_partition_sha256": detail["initial_partition_sha256"],
                    "changed_observations": detail["changed_observations"],
                    "min_cluster_size": detail["min_cluster_size"],
                    "cluster_sizes": json.dumps(detail["cluster_sizes"], separators=(",", ":")),
                    "support_mean": detail["support_mean"],
                    "boundary_mean": detail["boundary_mean"],
                    "conflict_mean": detail["conflict_mean"],
                    "conductance_mean": detail["conductance_mean"],
                    "rejected_mass_mean": detail["rejected_mass_mean"],
                    "trust_mean": detail["trust_mean"],
                    "wall_seconds": time.perf_counter() - started,
                    "producer_label_reads": 0,
                    "dense_n_by_n_count": 0,
                }
            )
        np.savez_compressed(
            out / f"{lane}_candidate_partitions.npz",
            candidate_ids=np.asarray(identifiers),
            partitions=np.stack(partitions).astype(np.int32),
        )
    write_csv(out / "candidate_descriptor_ledger.csv", descriptor_rows)
    write_json(
        out / "producer_manifest.json",
        {
            "family": args.family,
            "lanes": list(FAMILIES[args.family]),
            "candidate_count_per_lane": len(configs),
            "rows": len(descriptor_rows),
            "producer_label_reads": 0,
            "dense_n_by_n_count": 0,
        },
    )


def _spatial_metrics(partition: np.ndarray, graph: sp.csr_matrix) -> tuple[float, float, float]:
    partition = encode_partition(partition).astype(np.float64)
    graph = sp.triu(graph.maximum(graph.T), k=1, format="coo")
    if graph.nnz == 0:
        return 0.0, 1.0, 0.0
    weights = np.asarray(graph.data, dtype=np.float64)
    centered = partition - partition.mean()
    denom = float(np.sum(centered * centered))
    wsum = float(np.sum(weights))
    moran = float(len(partition) / (2.0 * wsum) * np.sum(weights * centered[graph.row] * centered[graph.col]) / max(denom, 1e-12))
    geary = float((len(partition) - 1) / (2.0 * wsum) * np.sum(weights * (partition[graph.row] - partition[graph.col]) ** 2) / max(denom, 1e-12))
    agreement = float(np.average(partition[graph.row] == partition[graph.col], weights=weights))
    return moran, geary, agreement


def evaluate_partition(partition: np.ndarray, labels: np.ndarray, mask: np.ndarray, graph: sp.csr_matrix) -> dict[str, object]:
    partition = encode_partition(partition)
    labels = np.asarray(labels)[mask]
    prediction = partition[mask]
    moran, geary, agreement = _spatial_metrics(partition, graph)
    sizes = np.bincount(partition)
    return {
        "absolute_ari": float(adjusted_rand_score(labels, prediction)),
        "absolute_nmi": float(normalized_mutual_info_score(labels, prediction)),
        "ami": float(adjusted_mutual_info_score(labels, prediction)),
        "fmi": float(fowlkes_mallows_score(labels, prediction)),
        "homogeneity": float(homogeneity_score(labels, prediction)),
        "v_measure": float(v_measure_score(labels, prediction)),
        "morans_i": moran,
        "gearys_c": geary,
        "neighbor_agreement": agreement,
        "evaluated_observations": int(np.sum(mask)),
        "min_cluster_size": int(sizes.min()),
        "cluster_sizes": json.dumps([int(x) for x in sizes], separators=(",", ":")),
        "partition_sha256": partition_sha256(partition),
    }


def evaluate_search(args: argparse.Namespace) -> None:
    kit_root = Path(args.kit_root)
    source = Path(args.input)
    descriptors = read_csv(source / "candidate_descriptor_ledger.csv")
    descriptor_by_key = {(row["lane"], row["candidate_id"]): row for row in descriptors}
    rows: list[dict[str, object]] = []
    for lane in FAMILIES[args.family]:
        evaluation = load_evaluation_input(kit_root, lane)
        with np.load(source / f"{lane}_candidate_partitions.npz", allow_pickle=False) as z:
            identifiers = z["candidate_ids"]
            partitions = z["partitions"]
        for identifier, partition in zip(identifiers, partitions):
            identifier = str(identifier)
            base = descriptor_by_key[(lane, identifier)].copy()
            base.update(evaluate_partition(partition, evaluation["labels"], evaluation["mask"], evaluation["graph"]))
            base["evaluator_label_reads"] = 1
            rows.append(base)
    write_csv(source / "evaluated_search_ledger.csv", rows)
    write_json(
        source / "evaluation_audit.json",
        {
            "family": args.family,
            "rows": len(rows),
            "candidates_materialized_before_evaluation": True,
            "producer_label_reads": 0,
            "public_annotation_evaluator_reads": len(FAMILIES[args.family]),
        },
    )


def select_family_config(rows: list[dict[str, str]], family: str) -> dict[str, object]:
    discovery = DISCOVERY[family]
    by_lane_candidate = {(r["lane"], r["candidate_id"]): r for r in rows}
    baseline: dict[str, dict[str, str]] = {}
    for lane in discovery:
        candidates = [r for r in rows if r["lane"] == lane and int(float(r["changed_observations"])) == 0]
        if not candidates:
            raise RuntimeError(f"{lane}: no registered no-op baseline")
        baseline[lane] = sorted(candidates, key=lambda r: r["candidate_id"])[0]
    identifiers = sorted({r["candidate_id"] for r in rows})
    ranking = []
    tol = 1e-12
    for identifier in identifiers:
        lane_rows = [by_lane_candidate[(lane, identifier)] for lane in discovery]
        deltas = []
        valid = True
        for lane, row in zip(discovery, lane_rows):
            minimum = max(5, int(np.ceil(0.01 * int(float(row["evaluated_observations"])) / len(json.loads(row["cluster_sizes"])))))
            valid &= int(float(row["min_cluster_size"])) >= minimum
            deltas.append(
                (
                    float(row["absolute_ari"]) - float(baseline[lane]["absolute_ari"]),
                    float(row["absolute_nmi"]) - float(baseline[lane]["absolute_nmi"]),
                )
            )
        dual = sum(da > tol and dn > tol for da, dn in deltas)
        config_json = lane_rows[0]["config_json"]
        complexity = sum(float(v) != 0 for v in json.loads(config_json).values() if isinstance(v, (int, float)))
        ranking.append(
            {
                "candidate_id": identifier,
                "valid": bool(valid),
                "dual_gain_studies": int(dual),
                "worst_delta_ari": float(min(x[0] for x in deltas)),
                "mean_delta_ari": float(np.mean([x[0] for x in deltas])),
                "mean_delta_nmi": float(np.mean([x[1] for x in deltas])),
                "complexity": int(complexity),
                "config": json.loads(config_json),
                "discovery_deltas": {lane: {"delta_ari": da, "delta_nmi": dn} for lane, (da, dn) in zip(discovery, deltas)},
            }
        )
    eligible = [r for r in ranking if r["valid"]]
    selected = max(
        eligible,
        key=lambda r: (
            r["dual_gain_studies"],
            r["worst_delta_ari"],
            r["mean_delta_ari"],
            r["mean_delta_nmi"],
            -r["complexity"],
            r["candidate_id"],
        ),
    )
    return {"family": family, "discovery_lanes": list(discovery), "selected": selected, "ranking": sorted(eligible, key=lambda r: (-r["dual_gain_studies"], -r["worst_delta_ari"], -r["mean_delta_ari"], -r["mean_delta_nmi"], r["complexity"], r["candidate_id"]))}


def freeze(args: argparse.Namespace) -> None:
    root = Path(args.input)
    frozen = {}
    for family in FAMILIES:
        rows = read_csv(root / family / "evaluated_search_ledger.csv")
        result = select_family_config(rows, family)
        frozen[family] = result
        write_csv(root / family / "family_selection_ranking.csv", result["ranking"])
    write_json(Path(args.output), {"schema": "night16c-family-frozen-config-v1", "families": frozen})


def produce_frozen(args: argparse.Namespace) -> None:
    kit_root = Path(args.kit_root)
    start_root = Path(args.start_root)
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))["families"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for family, lanes in FAMILIES.items():
        config = CMBFTPRConfig(**frozen[family]["selected"]["config"])
        for lane in lanes:
            data = load_producer_input(kit_root, start_root, lane)
            started = time.perf_counter()
            partition, evidence, detail = cmbf_tpr(
                data["initial"], data["view1"], data["view2"], data["graph"], data["start_bank"], config
            )
            np.save(out / f"{lane}.npy", partition, allow_pickle=False)
            rows.append(
                {
                    "family": family,
                    "lane": lane,
                    "candidate_id": config_id(config),
                    "partition_sha256": partition_sha256(partition),
                    "initial_partition_sha256": partition_sha256(data["initial"]),
                    "changed_observations": detail["changed_observations"],
                    "min_cluster_size": detail["min_cluster_size"],
                    "cluster_sizes": json.dumps(detail["cluster_sizes"], separators=(",", ":")),
                    "support_mean": detail["support_mean"],
                    "boundary_mean": detail["boundary_mean"],
                    "conflict_mean": detail["conflict_mean"],
                    "conductance_mean": detail["conductance_mean"],
                    "rejected_mass_mean": detail["rejected_mass_mean"],
                    "trust_mean": detail["trust_mean"],
                    "wall_seconds": time.perf_counter() - started,
                    "producer_label_reads": 0,
                    "dense_n_by_n_count": 0,
                }
            )
            np.savez_compressed(
                out / f"{lane}_field_summary.npz",
                support=evidence.support,
                boundary=evidence.boundary,
                conflict=evidence.conflict,
                conductance=evidence.conductance,
                rejected_mass=evidence.rejected_mass,
                node_trust=evidence.node_trust,
            )
    write_csv(out / "producer_ledger.csv", rows)
    write_json(out / "producer_manifest.json", {"lanes": list(PRIMARY_LANES), "producer_label_reads": 0, "dense_n_by_n_count": 0})


def evaluate_frozen(args: argparse.Namespace) -> None:
    kit_root = Path(args.kit_root)
    source = Path(args.input)
    producer_rows = {r["lane"]: r for r in read_csv(source / "producer_ledger.csv")}
    rows = []
    for lane in PRIMARY_LANES:
        evaluation = load_evaluation_input(kit_root, lane)
        partition = np.load(source / f"{lane}.npy", allow_pickle=False)
        row = producer_rows[lane].copy()
        row.update(evaluate_partition(partition, evaluation["labels"], evaluation["mask"], evaluation["graph"]))
        rows.append(row)
    write_csv(source / "metrics.csv", rows)
    write_json(source / "evaluation_audit.json", {"rows": len(rows), "producer_label_reads": 0, "evaluator_label_reads": len(rows)})


def _boundary_nodes(graph: sp.csr_matrix, labels: np.ndarray) -> np.ndarray:
    nodes = np.zeros(graph.shape[0], dtype=bool)
    row = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
    different = labels[row] != labels[graph.indices]
    nodes[row[different]] = True
    nodes[graph.indices[different]] = True
    return nodes


def _distance_to_boundary(graph: sp.csr_matrix, boundary: np.ndarray) -> np.ndarray:
    distance = np.full(graph.shape[0], np.inf)
    frontier = list(np.flatnonzero(boundary))
    distance[frontier] = 0
    head = 0
    while head < len(frontier):
        node = int(frontier[head]); head += 1
        begin, end = graph.indptr[node : node + 2]
        for neighbour in graph.indices[begin:end]:
            if not np.isfinite(distance[neighbour]):
                distance[neighbour] = distance[node] + 1
                frontier.append(int(neighbour))
    return distance


def produce_diagnostics(args: argparse.Namespace) -> None:
    kit_root = Path(args.kit_root)
    start_root = Path(args.start_root)
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))["families"]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    rows = []
    for family, lanes in FAMILIES.items():
        config = CMBFTPRConfig(**frozen[family]["selected"]["config"])
        for lane in lanes:
            data = load_producer_input(kit_root, start_root, lane)
            evidence = prepare_boundary_evidence(data["view1"], data["view2"], data["graph"], data["initial"], data["start_bank"], config)
            np.savez_compressed(
                out / f"{lane}.npz",
                initial=data["initial"],
                edge_support=evidence.support,
                edge_boundary=evidence.boundary,
                edge_conflict=evidence.conflict,
                node_trust=evidence.node_trust,
                graph_indices=evidence.graph.indices,
                graph_indptr=evidence.graph.indptr,
                graph_shape=np.asarray(evidence.graph.shape),
            )
            rows.append({"family": family, "lane": lane, "partition_sha256": partition_sha256(data["initial"]), "field_sha256": sha256_bytes((out / f"{lane}.npz").read_bytes()), "producer_label_reads": 0})
    write_csv(out / "producer_ledger.csv", rows)


def evaluate_diagnostics(args: argparse.Namespace) -> None:
    kit_root = Path(args.kit_root)
    source = Path(args.input)
    rows = []
    for lane in PRIMARY_LANES:
        evaluation = load_evaluation_input(kit_root, lane)
        with np.load(source / f"{lane}.npz", allow_pickle=False) as z:
            initial = z["initial"]
            edge_boundary = z["edge_boundary"]
            edge_support = z["edge_support"]
            edge_conflict = z["edge_conflict"]
            node_trust = z["node_trust"]
        graph = evaluation["graph"]
        labels = evaluation["labels"]
        mask = evaluation["mask"]
        row_index = np.repeat(np.arange(graph.shape[0]), np.diff(graph.indptr))
        valid_edge = mask[row_index] & mask[graph.indices]
        truth_edge = labels[row_index] != labels[graph.indices]
        predicted_edge = initial[row_index] != initial[graph.indices]
        truth = truth_edge[valid_edge]
        pred = predicted_edge[valid_edge]
        tp = int(np.sum(truth & pred)); fp = int(np.sum(~truth & pred)); fn = int(np.sum(truth & ~pred))
        precision = tp / max(tp + fp, 1); recall = tp / max(tp + fn, 1)
        aligned = align_partition(encode_partition(labels[mask]), initial[mask])
        wrong_masked = aligned != encode_partition(labels[mask])
        truth_encoded = encode_partition(labels)
        boundary_nodes = _boundary_nodes(graph, truth_encoded)
        distance = _distance_to_boundary(graph, boundary_nodes)
        wrong_full = np.zeros(len(mask), dtype=bool); wrong_full[np.flatnonzero(mask)] = wrong_masked
        rows.append(
            {
                "lane": lane,
                "boundary_precision": precision,
                "boundary_recall": recall,
                "boundary_f1": 2 * precision * recall / max(precision + recall, 1e-12),
                "cmbf_boundary_auc": float(roc_auc_score(truth.astype(int), edge_boundary[valid_edge])) if len(np.unique(truth)) == 2 else float("nan"),
                "support_inverse_auc": float(roc_auc_score(truth.astype(int), 1.0 - edge_support[valid_edge])) if len(np.unique(truth)) == 2 else float("nan"),
                "conflict_true_boundary_mean": float(np.mean(edge_conflict[valid_edge][truth])) if np.any(truth) else float("nan"),
                "conflict_nonboundary_mean": float(np.mean(edge_conflict[valid_edge][~truth])) if np.any(~truth) else float("nan"),
                "wrong_observations_after_hungarian": int(np.sum(wrong_full)),
                "wrong_within_1_hop_of_true_boundary_fraction": float(np.mean(distance[wrong_full] <= 1)) if np.any(wrong_full) else 1.0,
                "wrong_within_2_hop_of_true_boundary_fraction": float(np.mean(distance[wrong_full] <= 2)) if np.any(wrong_full) else 1.0,
                "trust_correct_mean": float(np.mean(node_trust[~wrong_full])) if np.any(~wrong_full) else float("nan"),
                "trust_wrong_mean": float(np.mean(node_trust[wrong_full])) if np.any(wrong_full) else float("nan"),
            }
        )
    write_csv(source / "boundary_diagnostic_table.csv", rows)


def produce_ablation(args: argparse.Namespace) -> None:
    kit_root = Path(args.kit_root); start_root = Path(args.start_root)
    frozen = json.loads(Path(args.frozen).read_text(encoding="utf-8"))["families"]
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    variants = {
        "STRONG_START_ONLY": lambda c: replace(c, sweeps=0),
        "CMBF_TPR_FULL": lambda c: c,
        "BOUNDARY_STATE_DISABLED": lambda c: replace(c, boundary_enabled=False),
        "CONFLICT_STATE_DISABLED": lambda c: replace(c, conflict_enabled=False),
        "TRUST_GATE_DISABLED": lambda c: replace(c, trust_gate_enabled=False),
        "GENERIC_REPAIR_ONLY": lambda c: replace(c, sweeps=0),
        "DIRECTIONAL_EDGE_ONLY": lambda c: replace(c, trust_gate_enabled=False, self_return_strength=0.0, anchor_strength=0.0),
    }
    rows=[]
    for family, lanes in FAMILIES.items():
        base=CMBFTPRConfig(**frozen[family]["selected"]["config"])
        for lane in lanes:
            data=load_producer_input(kit_root,start_root,lane)
            for name,make in variants.items():
                config=make(base)
                partition,_,detail=cmbf_tpr(data["initial"],data["view1"],data["view2"],data["graph"],data["start_bank"],config)
                np.save(out/f"{lane}__{name}.npy",partition,allow_pickle=False)
                rows.append({"family":family,"lane":lane,"variant":name,"partition_sha256":partition_sha256(partition),"changed_observations":detail["changed_observations"],"producer_label_reads":0})
    write_csv(out/"producer_ledger.csv",rows)


def evaluate_ablation(args: argparse.Namespace) -> None:
    kit_root=Path(args.kit_root); source=Path(args.input)
    rows=[]
    for producer in read_csv(source/"producer_ledger.csv"):
        lane=producer["lane"]; variant=producer["variant"]
        evaluation=load_evaluation_input(kit_root,lane)
        partition=np.load(source/f"{lane}__{variant}.npy",allow_pickle=False)
        row=producer.copy(); row.update(evaluate_partition(partition,evaluation["labels"],evaluation["mask"],evaluation["graph"])); rows.append(row)
    write_csv(source/"minimal_contribution_table.csv",rows)


def main() -> None:
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command",required=True)
    def common(p):
        p.add_argument("--kit-root",required=True); p.add_argument("--start-root",required=True)
    p=sub.add_parser("produce-search"); common(p); p.add_argument("--family",choices=FAMILIES,required=True); p.add_argument("--config-count",type=int,default=64); p.add_argument("--config-registry"); p.add_argument("--output",required=True); p.set_defaults(func=produce_search)
    p=sub.add_parser("evaluate-search"); p.add_argument("--kit-root",required=True); p.add_argument("--family",choices=FAMILIES,required=True); p.add_argument("--input",required=True); p.set_defaults(func=evaluate_search)
    p=sub.add_parser("freeze"); p.add_argument("--input",required=True); p.add_argument("--output",required=True); p.set_defaults(func=freeze)
    p=sub.add_parser("produce-frozen"); common(p); p.add_argument("--frozen",required=True); p.add_argument("--output",required=True); p.set_defaults(func=produce_frozen)
    p=sub.add_parser("evaluate-frozen"); p.add_argument("--kit-root",required=True); p.add_argument("--input",required=True); p.set_defaults(func=evaluate_frozen)
    p=sub.add_parser("produce-diagnostics"); common(p); p.add_argument("--frozen",required=True); p.add_argument("--output",required=True); p.set_defaults(func=produce_diagnostics)
    p=sub.add_parser("evaluate-diagnostics"); p.add_argument("--kit-root",required=True); p.add_argument("--input",required=True); p.set_defaults(func=evaluate_diagnostics)
    p=sub.add_parser("produce-ablation"); common(p); p.add_argument("--frozen",required=True); p.add_argument("--output",required=True); p.set_defaults(func=produce_ablation)
    p=sub.add_parser("evaluate-ablation"); p.add_argument("--kit-root",required=True); p.add_argument("--input",required=True); p.set_defaults(func=evaluate_ablation)
    args=parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
