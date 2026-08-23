#!/usr/bin/env python3
"""Adaptive continuous-parameter search for the Night-15E energy.

The score-ceiling phase performs per-lane public-label development HPO.  The
joint phase optimizes one numeric configuration across all studies and reports
leave-one-study-out incremental transfer on fixed Night-15D authorities.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import qmc

SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_ROOT))

from night15e_continuous_search import (
    FAMILY_FOR_LANE,
    NIGHT15D_AUTHORITY,
    STUDY_FOR_LANE,
    ContinuousEnergyConfig,
    baseline_row,
    build_policy_audit,
    load_lane,
    run_one,
    selection_key,
    sha256_array,
    write_rows,
)


# (field, lower, upper, logarithmic)
SPACE = (
    ("beta", 0.005, 20.0, True),
    ("edge_floor", 0.0001, 0.80, True),
    ("conflict_center", -0.20, 1.00, False),
    ("conflict_temperature", 0.005, 1.00, True),
    ("conflict_union_weight", 0.001, 0.999, False),
    ("conflict_penalty", 0.0001, 2.00, True),
    ("mass_center", -1.50, 1.50, False),
    ("mass_temperature", 0.005, 1.00, True),
    ("neighbor_capacity", 0.01, 1.00, False),
    ("low_weight", 0.0001, 5.00, True),
    ("twohop_weight", 0.0001, 5.00, True),
    ("high_weight", 0.0001, 5.00, True),
    ("unary_temperature", 0.02, 3.00, True),
    ("retained_bias", -4.00, 8.00, False),
    ("view_balance", -4.00, 4.00, False),
    ("trust_scale", 0.0001, 10.00, True),
    ("trust_center", -2.00, 4.00, False),
    ("trust_temperature", 0.005, 2.00, True),
    ("move_threshold", 0.0, 1.00, False),
    ("move_fraction", 0.001, 0.80, True),
    ("sweeps", 1.0, 10.0, False),
)


def encode(config: ContinuousEnergyConfig) -> np.ndarray:
    values = []
    payload = asdict(config)
    for field, low, high, logarithmic in SPACE:
        value = float(payload[field])
        if logarithmic:
            value = (math.log(np.clip(value, low, high)) - math.log(low)) / (math.log(high) - math.log(low))
        else:
            value = (np.clip(value, low, high) - low) / (high - low)
        values.append(float(np.clip(value, 0.0, 1.0)))
    return np.asarray(values, dtype=np.float64)


def decode(value: np.ndarray) -> ContinuousEnergyConfig:
    payload = {}
    for coordinate, (field, low, high, logarithmic) in zip(np.clip(value, 0.0, 1.0), SPACE):
        if logarithmic:
            observed = math.exp(math.log(low) + float(coordinate) * (math.log(high) - math.log(low)))
        else:
            observed = low + float(coordinate) * (high - low)
        payload[field] = int(round(observed)) if field == "sweeps" else float(observed)
    payload["sweeps"] = int(np.clip(payload["sweeps"], 1, 10))
    return ContinuousEnergyConfig(**payload)


def scalar_score(row: Mapping[str, object]) -> float:
    if row.get("status") != "PASS":
        return -1e9
    da = float(row["delta_ari"])
    dn = float(row["delta_nmi"])
    # Strictly rewards joint improvement while retaining absolute-score signal.
    return min(da, dn) + 0.20 * (da + dn) - 0.50 * (max(-da, 0.0) + max(-dn, 0.0))


def config_from_row(row: Mapping[str, str]) -> ContinuousEnergyConfig | None:
    try:
        payload = json.loads(row["config_json"])
        if set(payload) == {field for field, *_ in SPACE}:
            return ContinuousEnergyConfig(**payload)
    except Exception:
        return None
    return None


def ledger_anchors(path: Path, lane: str, limit: int = 10) -> List[ContinuousEnergyConfig]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("lane") == lane and row.get("status") == "PASS"]
    rows.sort(key=lambda row: scalar_score(row), reverse=True)
    output = []
    seen = set()
    for row in rows:
        config = config_from_row(row)
        if config is None:
            continue
        payload = json.dumps(asdict(config), sort_keys=True)
        if payload not in seen:
            output.append(config)
            seen.add(payload)
        if len(output) >= int(limit):
            break
    return output


def semantic_anchor(old: Mapping[str, object]) -> ContinuousEnergyConfig:
    edge_mode = str(old["edge_mode"])
    normalization = str(old["normalization"])
    feature = str(old["feature"])
    unary_mode = str(old["unary_mode"])
    union = {"either": 0.98, "both": 0.02, "geomean": 0.50, "agreement": 0.25, "spatial": 0.50}[edge_mode]
    edge_floor = 0.65 if edge_mode == "spatial" else 0.02
    mass_center = -1.2 if normalization == "row" else (0.25 if normalization == "self" else 0.55)
    capacity = 1.0 if normalization in ("row", "self") else 0.70
    low = 1.0 if "low" in feature or "multiscale" in feature else 0.0001
    twohop = 1.0 if "multiscale" in feature or feature == "views_low_high" else 0.0001
    high = 1.0 if "high" in feature or "multiscale" in feature else 0.0001
    retained_bias = 6.0 if unary_mode == "retained" else (1.2 if unary_mode == "retained_dual_margin" else -1.0)
    return ContinuousEnergyConfig(
        beta=float(old["beta"]),
        edge_floor=edge_floor,
        conflict_center=0.05 if edge_mode in ("either", "both") else 0.55,
        conflict_temperature=0.03,
        conflict_union_weight=union,
        conflict_penalty=0.10,
        mass_center=mass_center,
        mass_temperature=0.12,
        neighbor_capacity=capacity,
        low_weight=low,
        twohop_weight=twohop,
        high_weight=high,
        unary_temperature=float(old.get("margin_temperature", 0.5)),
        retained_bias=retained_bias,
        view_balance=0.0,
        trust_scale=max(float(old.get("switch_penalty", 0.0)), 0.0001),
        trust_center=0.15,
        trust_temperature=0.10,
        move_threshold=0.001,
        move_fraction=0.25,
        sweeps=int(old["steps"]),
    )


def unique_vectors(values: Sequence[np.ndarray]) -> List[np.ndarray]:
    seen = set()
    output = []
    for value in values:
        key = np.round(np.asarray(value), 12).tobytes()
        if key not in seen:
            seen.add(key)
            output.append(np.asarray(value, dtype=np.float64))
    return output


def adaptive_lane(
    args: argparse.Namespace,
    lane: str,
    lane_index: int,
    old_config: Mapping[str, object],
) -> Tuple[List[dict], dict, np.ndarray]:
    data, graph, labels, mask, k, initial, evidence = load_lane(args.kit, args.banks, args.partitions, lane)
    base = baseline_row(lane, graph, labels, mask, k, initial)
    rows = [base]
    best = base
    best_partition = initial.copy()
    rng = np.random.default_rng(20260826 + lane_index * 7919)
    anchors = ledger_anchors(args.previous_ledger, lane, limit=12)
    anchors.append(semantic_anchor(old_config))
    vectors = unique_vectors([encode(config) for config in anchors])
    lhs = qmc.LatinHypercube(d=len(SPACE), seed=20260826 + lane_index).random(args.batch)
    candidates = unique_vectors(vectors + [row for row in lhs])
    elite_vectors = vectors
    for iteration in range(args.iterations):
        if iteration > 0:
            # Cross-entropy sampling from several elites plus broad exploration.
            elite_matrix = np.stack(elite_vectors)
            center = elite_matrix.mean(axis=0)
            spread = np.maximum(elite_matrix.std(axis=0), 0.035) * (0.78 ** (iteration - 1))
            sampled = np.clip(rng.normal(center, spread, size=(args.batch, len(SPACE))), 0.0, 1.0)
            broad = qmc.LatinHypercube(d=len(SPACE), seed=20260826 + lane_index * 101 + iteration).random(max(4, args.batch // 8))
            candidates = unique_vectors([row for row in sampled] + [row for row in broad] + elite_vectors)
        evaluated = []
        for candidate_index, vector in enumerate(candidates):
            config = decode(vector)
            identifier = f"{lane}__CEM_I{iteration:02d}_{candidate_index:03d}"
            row, partition = run_one(
                lane, graph, labels, mask, k, initial, evidence, identifier, config, "ADAPTIVE_SCORE_CEILING"
            )
            row["adaptive_iteration"] = iteration
            rows.append(row)
            evaluated.append((scalar_score(row), vector, row, partition))
            if selection_key(row) > selection_key(best):
                best = row
                best_partition = partition.copy()
        evaluated.sort(key=lambda item: item[0], reverse=True)
        elite_vectors = unique_vectors([item[1] for item in evaluated[: args.elites]])
        print(
            json.dumps(
                {
                    "lane": lane,
                    "iteration": iteration,
                    "rows": len(evaluated),
                    "best_ari": best["absolute_ari"],
                    "best_nmi": best["absolute_nmi"],
                    "delta_ari": best["delta_ari"],
                    "delta_nmi": best["delta_nmi"],
                }
            ),
            flush=True,
        )
    return rows, best, best_partition


def joint_score(rows: Sequence[Mapping[str, object]]) -> float:
    grouped: Dict[str, List[float]] = {}
    regression = 0.0
    for row in rows:
        da, dn = float(row["delta_ari"]), float(row["delta_nmi"])
        grouped.setdefault(str(row["study"]), []).append(da + 0.35 * dn)
        regression += max(-da, 0.0) + max(-dn, 0.0)
    effects = [float(np.mean(values)) for values in grouped.values()]
    return float(np.mean(effects) + min(effects) - 1.5 * regression / len(rows))


def adaptive_joint(args: argparse.Namespace, old_registry: Mapping[str, object]) -> dict:
    lanes = list(NIGHT15D_AUTHORITY)
    loaded = {
        lane: load_lane(args.kit, args.banks, args.partitions, lane)
        for lane in lanes
    }
    rng = np.random.default_rng(20260827)
    anchors = []
    # Candidate anchors from every lane plus the no-move continuous policy.
    for lane in lanes:
        anchors.extend(ledger_anchors(args.previous_ledger, lane, limit=3))
        anchors.append(semantic_anchor(old_registry[lane]["config"]))
    no_move = decode(np.full(len(SPACE), 0.5))
    no_move = ContinuousEnergyConfig(**{**asdict(no_move), "move_threshold": 100.0, "sweeps": 1})
    anchors.append(no_move)
    elite_vectors = unique_vectors([encode(config) for config in anchors])
    all_rows = []
    config_summaries = []
    for iteration in range(args.joint_iterations):
        elite_matrix = np.stack(elite_vectors)
        center = elite_matrix.mean(axis=0)
        spread = np.maximum(elite_matrix.std(axis=0), 0.04) * (0.75 ** iteration)
        samples = np.clip(rng.normal(center, spread, size=(args.joint_batch, len(SPACE))), 0.0, 1.0)
        if iteration == 0:
            samples = np.vstack((elite_matrix, samples, qmc.LatinHypercube(d=len(SPACE), seed=20260827).random(8)))
        candidates = unique_vectors([row for row in samples])
        evaluated = []
        for candidate_index, vector in enumerate(candidates):
            config = decode(vector)
            identifier = f"JOINT_I{iteration:02d}_{candidate_index:03d}"
            candidate_rows = []
            for lane in lanes:
                data, graph, labels, mask, k, initial, evidence = loaded[lane]
                row, _ = run_one(
                    lane, graph, labels, mask, k, initial, evidence, identifier, config, "JOINT_INCREMENTAL_POLICY_SEARCH"
                )
                row["joint_iteration"] = iteration
                all_rows.append(row)
                candidate_rows.append(row)
            score = joint_score(candidate_rows) if all(row["status"] == "PASS" for row in candidate_rows) else -1e9
            evaluated.append((score, vector, identifier, candidate_rows, config))
            config_summaries.append({"config_id": identifier, "score": score, "config": asdict(config)})
        evaluated.sort(key=lambda item: item[0], reverse=True)
        elite_vectors = unique_vectors([item[1] for item in evaluated[: args.joint_elites]])
        print(json.dumps({"joint_iteration": iteration, "best_score": evaluated[0][0], "config_id": evaluated[0][2]}), flush=True)
    complete_ids = {item["config_id"] for item in config_summaries}
    policy = build_policy_audit([row for row in all_rows if row["config_id"] in complete_ids])
    best_summary = max(config_summaries, key=lambda row: row["score"])
    return {
        "status": "PASS",
        "best_joint": best_summary,
        "policy_audit": policy,
        "config_summaries": config_summaries,
        "rows": all_rows,
        "scope_qualification": "incremental policy transfer on fixed Night-15D authorities",
        "full_end_to_end_zero_label_deployability_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--partitions", type=Path, required=True)
    parser.add_argument("--previous-ledger", type=Path, required=True)
    parser.add_argument("--night15d-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lanes", default=",".join(NIGHT15D_AUTHORITY))
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--batch", type=int, default=56)
    parser.add_argument("--elites", type=int, default=10)
    parser.add_argument("--joint-iterations", type=int, default=3)
    parser.add_argument("--joint-batch", type=int, default=20)
    parser.add_argument("--joint-elites", type=int, default=6)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    registry = json.loads(args.night15d_registry.read_text(encoding="utf-8"))["lanes"]
    all_rows = []
    summaries = {}
    started = time.perf_counter()
    for lane_index, lane in enumerate([value for value in args.lanes.split(",") if value]):
        rows, best, partition = adaptive_lane(args, lane, lane_index, registry[lane]["config"])
        all_rows.extend(rows)
        (args.output / "partitions").mkdir(exist_ok=True)
        np.save(args.output / "partitions" / f"{lane}.npy", partition, allow_pickle=False)
        summaries[lane] = {
            "best": best,
            "partition_sha256": sha256_array(partition),
            "public_labels_used_for_per_lane_score_ceiling_hpo": 1,
            "labels_in_model_or_energy": 0,
        }
        write_rows(args.output / "adaptive_score_ceiling_ledger.partial.csv", all_rows)
    write_rows(args.output / "adaptive_score_ceiling_ledger.csv", all_rows)
    joint = adaptive_joint(args, registry)
    write_rows(args.output / "joint_policy_search_ledger.csv", joint.pop("rows"))
    (args.output / "joint_policy_search.json").write_text(
        json.dumps(joint, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    result = {
        "status": "PASS",
        "lanes": summaries,
        "score_ceiling_rows": len(all_rows),
        "wall_seconds": time.perf_counter() - started,
        "labels_used_for_public_development_hpo": 1,
        "labels_in_model_or_energy": 0,
        "dataset_name_reads_in_model_core": 0,
        "dense_n_by_n_count": 0,
    }
    (args.output / "adaptive_search_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "rows": len(all_rows), "wall_seconds": result["wall_seconds"]}))


if __name__ == "__main__":
    main()
