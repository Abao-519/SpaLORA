#!/usr/bin/env python3
"""Independent public-annotation evaluator and family-config selector."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import (
    adjusted_mutual_info_score,
    adjusted_rand_score,
    fowlkes_mallows_score,
    homogeneity_score,
    normalized_mutual_info_score,
    v_measure_score,
)

from SpaLORA.night16d_cmbf_rl import encode_partition, graph_from_csr_arrays


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("empty output")
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def load_evaluation(kit_root: Path, lane: str) -> dict[str, object]:
    with np.load(kit_root / f"{lane}.npz", allow_pickle=False) as z:
        return {
            "labels": z["labels_primary"].copy(),
            "mask": z["label_mask"].astype(bool),
            "graph": graph_from_csr_arrays(z["graph__data"], z["graph__indices"], z["graph__indptr"], z["graph__shape"]),
        }


def spatial_metrics(partition: np.ndarray, graph: sp.csr_matrix) -> tuple[float, float, float]:
    """Categorical spatial metrics using one-vs-rest indicator macros.

    Cluster identifiers are nominal.  Computing Moran/Geary directly on their
    arbitrary integer codes is not permutation invariant, so each cluster is
    evaluated as a binary indicator and the per-cluster statistics are macro
    averaged.
    """
    value = encode_partition(partition)
    edge = sp.triu(graph.maximum(graph.T), k=1, format="coo")
    weight = np.asarray(edge.data, dtype=np.float64)
    wsum = max(float(np.sum(weight)), 1e-12)
    morans, gearys = [], []
    for group in range(int(value.max()) + 1):
        indicator = (value == group).astype(np.float64)
        centered = indicator - indicator.mean()
        denom = max(float(np.sum(centered * centered)), 1e-12)
        morans.append(float(len(value) / wsum * np.sum(weight * centered[edge.row] * centered[edge.col]) / denom))
        gearys.append(float((len(value) - 1) / (2 * wsum) * np.sum(weight * (indicator[edge.row] - indicator[edge.col]) ** 2) / denom))
    moran = float(np.mean(morans))
    geary = float(np.mean(gearys))
    agreement = float(np.average(value[edge.row] == value[edge.col], weights=weight))
    return moran, geary, agreement


def metrics(partition: np.ndarray, evaluation: dict[str, object]) -> dict[str, object]:
    partition = encode_partition(partition)
    mask = evaluation["mask"]
    labels = evaluation["labels"][mask]
    prediction = partition[mask]
    moran, geary, agreement = spatial_metrics(partition, evaluation["graph"])
    full_sizes = np.bincount(partition)
    eval_sizes = np.bincount(prediction, minlength=len(full_sizes))
    return {
        "absolute_ari": float(adjusted_rand_score(labels, prediction)),
        "absolute_nmi": float(normalized_mutual_info_score(labels, prediction)),
        "ami": float(adjusted_mutual_info_score(labels, prediction)),
        "fmi": float(fowlkes_mallows_score(labels, prediction)),
        "homogeneity": float(homogeneity_score(labels, prediction)),
        "v_measure": float(v_measure_score(labels, prediction)),
        "morans_i_macro": moran,
        "gearys_c_macro": geary,
        "categorical_spatial_metric": "one_vs_rest_macro",
        "neighbor_agreement": agreement,
        "evaluated_observations": int(mask.sum()),
        "min_cluster_size_full": int(full_sizes.min()),
        "cluster_sizes_full": json.dumps([int(x) for x in full_sizes], separators=(",", ":")),
        "min_cluster_size_eval": int(eval_sizes.min()),
        "cluster_sizes_eval": json.dumps([int(x) for x in eval_sizes], separators=(",", ":")),
    }


def evaluate_screen(args: argparse.Namespace) -> None:
    source = Path(args.input)
    producer = read_csv(source / "producer_ledger.csv")
    by_key = {(x["lane"], x["candidate_id"]): x for x in producer}
    rows: list[dict[str, object]] = []
    for lane in [x for x in args.lanes.split(",") if x]:
        evaluation = load_evaluation(Path(args.kit_root), lane)
        with np.load(source / f"{lane}_locked_partitions.npz", allow_pickle=False) as z:
            ids = z["candidate_ids"]
            partitions = z["partitions"]
        for identifier, partition in zip(ids, partitions):
            identifier = str(identifier)
            row: dict[str, object] = dict(by_key[(lane, identifier)])
            row.update(metrics(partition, evaluation))
            row["evaluator_label_reads"] = 1
            rows.append(row)
    write_csv(source / "evaluated_ledger.csv", rows)
    write_json(
        source / "evaluation_audit.json",
        {
            "rows": len(rows),
            "candidate_partitions_locked_before_evaluation": True,
            "producer_label_reads": 0,
            "public_annotation_lane_reads": len(set(x["lane"] for x in rows)),
        },
    )


def evaluate_p0(args: argparse.Namespace) -> None:
    source = Path(args.input)
    partition = np.load(source / "partition.npy", allow_pickle=False)
    row = metrics(partition, load_evaluation(Path(args.kit_root), args.lane))
    row.update({"lane": args.lane, "public_annotation_evaluator_reads": 1, "partition_locked_before_evaluation": True})
    write_json(source / "evaluation.json", row)


def select_family(args: argparse.Namespace) -> None:
    rows = read_csv(Path(args.input) / "evaluated_ledger.csv")
    discovery = [x for x in args.discovery_lanes.split(",") if x]
    rows = [x for x in rows if x["lane"] in discovery and x.get("status") == "PASS"]
    input_baseline: dict[str, dict[str, str]] = {}
    same_head_baseline: dict[str, dict[str, str]] = {}
    for lane in discovery:
        exact = [x for x in rows if x["lane"] == lane and json.loads(x["config_json"])["operation_mode"] == "teacher"]
        headed = [x for x in rows if x["lane"] == lane and json.loads(x["config_json"])["operation_mode"] == "teacher_head"]
        if not exact or not headed:
            raise RuntimeError(f"{lane}: missing exact-input or same-head teacher control")
        input_baseline[lane] = sorted(exact, key=lambda x: x["candidate_id"])[0]
        same_head_baseline[lane] = sorted(headed, key=lambda x: x["candidate_id"])[0]
    candidates: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        config = json.loads(row["config_json"])
        if config["operation_mode"] != "full":
            continue
        key = row["config_sha256"]
        candidates.setdefault(key, []).append(row)
    ranking: list[dict[str, object]] = []
    for key, group in candidates.items():
        by_lane: dict[str, list[dict[str, str]]] = {lane: [] for lane in discovery}
        for row in group:
            by_lane[row["lane"]].append(row)
        if any(not by_lane[lane] for lane in discovery):
            continue
        lane_summary = {}
        dual = 0
        deltas_ari, deltas_nmi = [], []
        for lane in discovery:
            # Family config performance aggregates every registered training seed.
            ari = float(np.mean([float(x["absolute_ari"]) for x in by_lane[lane]]))
            nmi = float(np.mean([float(x["absolute_nmi"]) for x in by_lane[lane]]))
            da = ari - float(input_baseline[lane]["absolute_ari"])
            dn = nmi - float(input_baseline[lane]["absolute_nmi"])
            da_head = ari - float(same_head_baseline[lane]["absolute_ari"])
            dn_head = nmi - float(same_head_baseline[lane]["absolute_nmi"])
            dual += int(da > 1e-12 and dn > 1e-12)
            deltas_ari.append(da); deltas_nmi.append(dn)
            lane_summary[lane] = {
                "mean_ari": ari,
                "mean_nmi": nmi,
                "delta_ari_vs_input_strong_start": da,
                "delta_nmi_vs_input_strong_start": dn,
                "delta_ari_vs_same_head_teacher": da_head,
                "delta_nmi_vs_same_head_teacher": dn_head,
            }
        config = json.loads(group[0]["config_json"])
        complexity = sum(bool(v) for v in config.values() if isinstance(v, (bool, int, float)))
        ranking.append(
            {
                "config_sha256": key,
                "config": config,
                "dual_gain_studies": dual,
                "worst_delta_ari": float(min(deltas_ari)),
                "mean_delta_ari": float(np.mean(deltas_ari)),
                "mean_delta_nmi": float(np.mean(deltas_nmi)),
                "complexity": complexity,
                "lanes": lane_summary,
            }
        )
    if not ranking:
        raise RuntimeError("no valid full configs")
    selected = max(
        ranking,
        key=lambda x: (
            x["dual_gain_studies"], x["worst_delta_ari"], x["mean_delta_ari"],
            x["mean_delta_nmi"], -x["complexity"], x["config_sha256"],
        ),
    )
    output = {
        "family": args.family,
        "discovery_lanes": discovery,
        "selection_baseline": "byte-exact INPUT_STRONG_START; same-head teacher deltas are reported separately",
        "selection_rule": "dual-gain studies vs INPUT_STRONG_START, worst delta ARI, mean delta ARI, mean delta NMI, lower complexity",
        "selected": selected,
        "ranking": sorted(ranking, key=lambda x: (-x["dual_gain_studies"], -x["worst_delta_ari"], -x["mean_delta_ari"], -x["mean_delta_nmi"], x["complexity"])),
        "label_assisted_family_benchmark_hpo": True,
    }
    write_json(Path(args.output), output)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("evaluate-screen")
    p.add_argument("--kit-root", required=True); p.add_argument("--lanes", required=True); p.add_argument("--input", required=True); p.set_defaults(func=evaluate_screen)
    p = sub.add_parser("evaluate-p0")
    p.add_argument("--kit-root", required=True); p.add_argument("--lane", required=True); p.add_argument("--input", required=True); p.set_defaults(func=evaluate_p0)
    p = sub.add_parser("select-family")
    p.add_argument("--family", required=True); p.add_argument("--discovery-lanes", required=True); p.add_argument("--input", required=True); p.add_argument("--output", required=True); p.set_defaults(func=select_family)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
