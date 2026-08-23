#!/usr/bin/env python3
"""Night-15F local-first multiscale alpha-expansion development arena.

Reference labels are passed only to ``evaluate`` after a candidate partition
has been produced.  The energy core receives numeric feature arrays, sparse
registered graphs, K, an initial partition, and numeric controls only.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from SpaLORA.night15e_continuous_reliability_energy import (  # noqa: E402
    ContinuousEnergyConfig,
    csr_from_archive,
)
from SpaLORA.night15f_multiscale_expansion import (  # noqa: E402
    ExpansionEnergyConfig,
    PreparedExpansionEvidence,
    continuous_multiscale_expansion,
    prepare_expansion_evidence,
)
from scripts.night15e.night15e_continuous_search import (  # noqa: E402
    DATASET_FOR_LANE,
    FAMILY_FOR_LANE,
    STUDY_FOR_LANE,
    evaluate,
    load_lane,
    sha256_array,
)


NIGHT15E_AUTHORITY = {
    "A1": (0.2752927992564834, 0.4199077568017935),
    "D1": (0.2546579651152784, 0.3890444821916162),
    "MISAR_E15_5_S1": (0.540931814030495, 0.6666169188694607),
    "MISAR_E15_5_S1_K12": (0.45236704915733733, 0.5977075353765589),
    "P22": (0.5891910297854036, 0.7100418847809442),
    "P22_3DOT_K18": (0.7283067896750458, 0.7501143785801062),
    "tonsil_s1": (0.23382458055710792, 0.314270251883914),
    "tonsil_s2": (0.2547979281144028, 0.3052523860699747),
    "tonsil_s3": (0.32831055003908766, 0.2895243534508425),
}


def config_id(config: ExpansionEnergyConfig, prefix: str) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
    return f"{prefix}_{hashlib.sha256(payload.encode()).hexdigest()[:14]}"


def write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def selection_key(row: Mapping[str, object]) -> Tuple[int, float, float, float]:
    if row.get("status") != "PASS":
        return (-1, -math.inf, -math.inf, -math.inf)
    da = float(row.get("delta_ari", -math.inf))
    dn = float(row.get("delta_nmi", -math.inf))
    return (int(da > 1e-12 and dn > 1e-12), min(da, dn), da + dn, float(row["absolute_ari"]))


def load_expansion_lane(
    kit: Path,
    banks: Path,
    partitions: Path,
    lane: str,
) -> tuple:
    data, graph, labels, mask, k, initial, _ = load_lane(kit, banks, partitions, lane)
    bank = np.load(
        banks / f"{DATASET_FOR_LANE[lane]}_selected_partition_bank.npz",
        allow_pickle=False,
        mmap_mode="r",
    )
    retained = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
    graphs = (csr_from_archive(data, "operator4"), graph, csr_from_archive(data, "operator18"))
    evidence = prepare_expansion_evidence(graphs, retained, data["view1"], data["view2"])
    return data, graph, labels, mask, k, initial, evidence


def authority_row(lane: str, graph, labels, mask, k: int, initial: np.ndarray) -> dict:
    metrics = evaluate(labels, mask, initial, graph)
    expected = NIGHT15E_AUTHORITY[lane]
    if abs(metrics["absolute_ari"] - expected[0]) > 1e-12 or abs(metrics["absolute_nmi"] - expected[1]) > 1e-12:
        raise RuntimeError(f"Night-15E authority mismatch for {lane}: {metrics} != {expected}")
    return {
        "lane": lane,
        "family": FAMILY_FOR_LANE[lane],
        "study": STUDY_FOR_LANE[lane],
        "stage": "AUTHORITY",
        "algorithm": "NIGHT15E_AUTHORITY_NOOP",
        "config_id": "NIGHT15E_AUTHORITY",
        "config_json": "{}",
        "status": "PASS",
        "failure": "",
        "k": k,
        "total_observations": len(initial),
        "evaluated_observations": int(mask.sum()),
        "partition_sha256": sha256_array(initial),
        "cluster_sizes": json.dumps(np.bincount(initial, minlength=k).astype(int).tolist(), separators=(",", ":")),
        "absolute_ari": metrics["absolute_ari"],
        "absolute_nmi": metrics["absolute_nmi"],
        "delta_ari": 0.0,
        "delta_nmi": 0.0,
        "objective": metrics["absolute_ari"] + 0.35 * metrics["absolute_nmi"],
        "changed_observations": 0,
        "accepted_expansions": 0,
        "wall_seconds": 0.0,
        **{key: value for key, value in metrics.items() if key not in {"absolute_ari", "absolute_nmi"}},
    }


def run_one(
    lane: str,
    graph,
    labels,
    mask,
    k: int,
    initial: np.ndarray,
    evidence: PreparedExpansionEvidence,
    identifier: str,
    config: ExpansionEnergyConfig,
    stage: str,
) -> tuple[dict, np.ndarray]:
    started = time.perf_counter()
    try:
        partition, diagnostics = continuous_multiscale_expansion(initial, k, evidence, config)
        metrics = evaluate(labels, mask, partition, graph)
        base_ari, base_nmi = NIGHT15E_AUTHORITY[lane]
        row = {
            "lane": lane,
            "family": FAMILY_FOR_LANE[lane],
            "study": STUDY_FOR_LANE[lane],
            "stage": stage,
            "algorithm": "CONTINUOUS_MULTISCALE_ALPHA_EXPANSION",
            "config_id": identifier,
            "config_json": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
            "status": "PASS",
            "failure": "",
            "k": k,
            "total_observations": len(partition),
            "evaluated_observations": int(mask.sum()),
            "initial_partition_sha256": sha256_array(initial),
            "partition_sha256": sha256_array(partition),
            "cluster_sizes": json.dumps(np.bincount(partition, minlength=k).astype(int).tolist(), separators=(",", ":")),
            "night15e_ari": base_ari,
            "night15e_nmi": base_nmi,
            "delta_ari": metrics["absolute_ari"] - base_ari,
            "delta_nmi": metrics["absolute_nmi"] - base_nmi,
            "objective": metrics["absolute_ari"] + 0.35 * metrics["absolute_nmi"],
            "wall_seconds": time.perf_counter() - started,
            **metrics,
            **diagnostics,
        }
        return row, partition
    except Exception as error:  # retained as a first-class failed row
        return {
            "lane": lane,
            "family": FAMILY_FOR_LANE[lane],
            "study": STUDY_FOR_LANE[lane],
            "stage": stage,
            "algorithm": "CONTINUOUS_MULTISCALE_ALPHA_EXPANSION",
            "config_id": identifier,
            "config_json": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
            "status": "FAILED",
            "failure": repr(error),
            "wall_seconds": time.perf_counter() - started,
        }, initial.copy()


def designed_configs(local: ContinuousEnergyConfig) -> Dict[str, ExpansionEnergyConfig]:
    output: Dict[str, ExpansionEnergyConfig] = {}
    scales = {
        "FINE": (1.0, 0.03, 0.01),
        "REGISTERED": (0.03, 1.0, 0.03),
        "BROAD": (0.01, 0.03, 1.0),
        "BALANCED": (1.0, 1.0, 1.0),
        "LOCAL_MIX": (0.8, 1.0, 0.2),
        "WIDE_MIX": (0.2, 1.0, 0.8),
    }
    for scale_name, scale in scales.items():
        for beta in (0.05, 0.2, 0.8, 3.2, 12.8):
            for self_return in (0.25, 1.0, 4.0, 16.0):
                config = ExpansionEnergyConfig(
                    local=local,
                    scale_fine=scale[0],
                    scale_registered=scale[1],
                    scale_broad=scale[2],
                    pairwise_beta=beta,
                    self_return_strength=self_return,
                    size_prior=0.0,
                    expansion_cycles=1,
                    capacity_scale=1000000.0,
                )
                output[config_id(config, f"D_{scale_name}_B{beta:g}_S{self_return:g}")] = config
    return output


def random_configs(
    local: ContinuousEnergyConfig,
    count: int,
    seed: int,
    anchor: ExpansionEnergyConfig | None = None,
) -> Dict[str, ExpansionEnergyConfig]:
    rng = np.random.default_rng(seed)
    output: Dict[str, ExpansionEnergyConfig] = {}
    for index in range(count):
        def log_jitter(value: float, sigma: float, low: float, high: float) -> float:
            return float(np.clip(max(value, low) * np.exp(rng.normal(0.0, sigma)), low, high))

        if anchor is None or index % 2 == 0:
            scale = rng.dirichlet(np.asarray([0.6, 0.8, 0.6]))
            beta = float(np.exp(rng.uniform(np.log(0.015), np.log(30.0))))
            self_return = float(np.exp(rng.uniform(np.log(0.05), np.log(60.0))))
            size_prior = float(rng.uniform(0.0, 0.8))
            trust_multiplier = float(np.exp(rng.uniform(np.log(0.2), np.log(12.0))))
            cycles = int(rng.choice([1, 1, 1, 2]))
            candidate_local = replace(
                local,
                edge_floor=float(np.clip(local.edge_floor + rng.normal(0.0, 0.08), 0.002, 0.75)),
                conflict_center=float(np.clip(local.conflict_center + rng.normal(0.0, 0.10), -0.1, 0.9)),
                conflict_temperature=log_jitter(local.conflict_temperature, 0.35, 0.01, 0.8),
                conflict_union_weight=float(np.clip(local.conflict_union_weight + rng.normal(0.0, 0.20), 0.0, 1.0)),
                conflict_penalty=float(np.clip(local.conflict_penalty + rng.normal(0.0, 0.15), 0.0, 1.5)),
                mass_center=float(np.clip(local.mass_center + rng.normal(0.0, 0.18), -0.8, 1.2)),
                mass_temperature=log_jitter(local.mass_temperature, 0.40, 0.01, 0.8),
                neighbor_capacity=float(np.clip(local.neighbor_capacity + rng.normal(0.0, 0.16), 0.05, 1.0)),
                low_weight=log_jitter(local.low_weight, 0.45, 1e-4, 5.0),
                twohop_weight=log_jitter(local.twohop_weight, 0.55, 1e-4, 4.0),
                high_weight=log_jitter(local.high_weight, 0.45, 1e-4, 5.0),
                unary_temperature=log_jitter(local.unary_temperature, 0.35, 0.05, 4.0),
                retained_bias=float(np.clip(local.retained_bias + rng.normal(0.0, 0.65), -3.0, 6.0)),
                view_balance=float(np.clip(local.view_balance + rng.normal(0.0, 0.45), -3.0, 3.0)),
                trust_scale=float(np.clip(local.trust_scale * trust_multiplier, 0.001, 40.0)),
                trust_center=float(np.clip(local.trust_center + rng.normal(0.0, 0.18), -0.8, 2.5)),
                trust_temperature=log_jitter(local.trust_temperature, 0.35, 0.01, 1.5),
            )
        else:
            base_scale = np.asarray(
                [anchor.scale_fine, anchor.scale_registered, anchor.scale_broad], dtype=np.float64
            )
            scale = np.maximum(base_scale * np.exp(rng.normal(0.0, 0.45, 3)), 1e-5)
            beta = float(np.clip(anchor.pairwise_beta * np.exp(rng.normal(0.0, 0.65)), 0.005, 60.0))
            self_return = float(np.clip(anchor.self_return_strength * np.exp(rng.normal(0.0, 0.65)), 0.01, 120.0))
            size_prior = float(np.clip(anchor.size_prior + rng.normal(0.0, 0.12), 0.0, 1.5))
            candidate_local = replace(
                anchor.local,
                edge_floor=float(np.clip(anchor.local.edge_floor + rng.normal(0.0, 0.04), 0.002, 0.75)),
                conflict_center=float(np.clip(anchor.local.conflict_center + rng.normal(0.0, 0.06), -0.1, 0.9)),
                conflict_temperature=log_jitter(anchor.local.conflict_temperature, 0.25, 0.01, 0.8),
                conflict_union_weight=float(np.clip(anchor.local.conflict_union_weight + rng.normal(0.0, 0.10), 0.0, 1.0)),
                conflict_penalty=float(np.clip(anchor.local.conflict_penalty + rng.normal(0.0, 0.08), 0.0, 1.5)),
                mass_center=float(np.clip(anchor.local.mass_center + rng.normal(0.0, 0.10), -0.8, 1.2)),
                mass_temperature=log_jitter(anchor.local.mass_temperature, 0.25, 0.01, 0.8),
                neighbor_capacity=float(np.clip(anchor.local.neighbor_capacity + rng.normal(0.0, 0.08), 0.05, 1.0)),
                low_weight=log_jitter(anchor.local.low_weight, 0.25, 1e-4, 5.0),
                twohop_weight=log_jitter(anchor.local.twohop_weight, 0.30, 1e-4, 4.0),
                high_weight=log_jitter(anchor.local.high_weight, 0.25, 1e-4, 5.0),
                unary_temperature=log_jitter(anchor.local.unary_temperature, 0.22, 0.05, 4.0),
                retained_bias=float(np.clip(anchor.local.retained_bias + rng.normal(0.0, 0.35), -3.0, 6.0)),
                view_balance=float(np.clip(anchor.local.view_balance + rng.normal(0.0, 0.25), -3.0, 3.0)),
                trust_scale=float(np.clip(anchor.local.trust_scale * np.exp(rng.normal(0.0, 0.35)), 0.001, 40.0)),
                trust_center=float(np.clip(anchor.local.trust_center + rng.normal(0.0, 0.10), -0.8, 2.5)),
                trust_temperature=log_jitter(anchor.local.trust_temperature, 0.25, 0.01, 1.5),
            )
            cycles = int(np.clip(anchor.expansion_cycles + rng.integers(-1, 2), 1, 3))
        config = ExpansionEnergyConfig(
            local=candidate_local,
            scale_fine=float(scale[0]),
            scale_registered=float(scale[1]),
            scale_broad=float(scale[2]),
            pairwise_beta=beta,
            self_return_strength=self_return,
            size_prior=size_prior,
            expansion_cycles=cycles,
            capacity_scale=1000000.0,
        )
        output[config_id(config, f"R{index:04d}")] = config
    return output


def search(args: argparse.Namespace) -> None:
    frozen = json.loads(args.night15e_registry.read_text(encoding="utf-8"))
    lanes = [value for value in args.lanes.split(",") if value]
    args.output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    result = {}
    for lane_index, lane in enumerate(lanes):
        _, graph, labels, mask, k, initial, evidence = load_expansion_lane(
            args.kit, args.banks, args.partitions, lane
        )
        baseline = authority_row(lane, graph, labels, mask, k, initial)
        rows.append(baseline)
        lane_rows = [baseline]
        lane_partitions = {"NIGHT15E_AUTHORITY": initial}
        local = ContinuousEnergyConfig(**frozen["lanes"][lane]["config"])
        registry = designed_configs(local)
        registry.update(random_configs(local, args.random_configs, 20260824 + lane_index * 1009))
        for identifier, config in registry.items():
            row, partition = run_one(
                lane, graph, labels, mask, k, initial, evidence, identifier, config, "PROBE_ARENA"
            )
            rows.append(row)
            lane_rows.append(row)
            if row["status"] == "PASS":
                lane_partitions[identifier] = partition
        best_probe = max(lane_rows, key=selection_key)
        if best_probe["config_id"] != "NIGHT15E_AUTHORITY" and args.refine_configs:
            anchor = ExpansionEnergyConfig(**{
                **json.loads(best_probe["config_json"]),
                "local": ContinuousEnergyConfig(**json.loads(best_probe["config_json"])["local"]),
            })
            refined = random_configs(local, args.refine_configs, 20260824 + 9001 + lane_index * 2017, anchor)
            for identifier, config in refined.items():
                identifier = f"{lane}__{identifier}"
                row, partition = run_one(
                    lane, graph, labels, mask, k, initial, evidence, identifier, config, "REFINED_ARENA"
                )
                rows.append(row)
                lane_rows.append(row)
                if row["status"] == "PASS":
                    lane_partitions[identifier] = partition
        best = max(lane_rows, key=selection_key)
        selected = lane_partitions[str(best["config_id"])]
        (args.output / "partitions").mkdir(exist_ok=True)
        np.save(args.output / "partitions" / f"{lane}.npy", selected, allow_pickle=False)
        result[lane] = {
            "best": best,
            "candidate_rows": len(lane_rows) - 1,
            "partition_sha256": sha256_array(selected),
        }
        write_rows(args.output / "all_run_ledger.partial.csv", rows)
        print(json.dumps({"lane": lane, "ari": best["absolute_ari"], "nmi": best["absolute_nmi"], "delta_ari": best["delta_ari"], "delta_nmi": best["delta_nmi"], "rows": len(lane_rows)}), flush=True)
    write_rows(args.output / "all_run_ledger.csv", rows)
    payload = {
        "status": "PASS",
        "lanes": result,
        "run_rows": len(rows),
        "labels_used_for_public_cross_run_hpo_and_evaluation": 1,
        "labels_in_energy_unary_or_edge": 0,
        "dataset_name_reads_in_energy_core": 0,
        "dense_n_by_n_count": 0,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "UNSET"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "UNSET"),
    }
    (args.output / "search_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8"
    )


def replay(args: argparse.Namespace) -> None:
    frozen = json.loads(args.registry.read_text(encoding="utf-8"))
    rows = []
    (args.output / "partitions").mkdir(parents=True, exist_ok=True)
    for lane, registered in frozen["lanes"].items():
        _, graph, labels, mask, k, initial, evidence = load_expansion_lane(
            args.kit, args.banks, args.partitions, lane
        )
        if registered["config_id"] == "NIGHT15E_AUTHORITY":
            row = authority_row(lane, graph, labels, mask, k, initial)
            partition = initial
        else:
            raw = registered["config"]
            config = ExpansionEnergyConfig(
                **{**raw, "local": ContinuousEnergyConfig(**raw["local"])}
            )
            row, partition = run_one(
                lane, graph, labels, mask, k, initial, evidence,
                registered["config_id"], config, "FROZEN_REPLAY"
            )
        if row["partition_sha256"] != registered["partition_sha256"]:
            raise RuntimeError(f"partition SHA mismatch: {lane}")
        if abs(float(row["absolute_ari"]) - float(registered["absolute_ari"])) > 1e-12:
            raise RuntimeError(f"ARI mismatch: {lane}")
        if abs(float(row["absolute_nmi"]) - float(registered["absolute_nmi"])) > 1e-12:
            raise RuntimeError(f"NMI mismatch: {lane}")
        np.save(args.output / "partitions" / f"{lane}.npy", partition, allow_pickle=False)
        rows.append(row)
        print(json.dumps({"lane": lane, "partition_sha256": row["partition_sha256"]}), flush=True)
    (args.output / "replay.json").write_text(
        json.dumps({"status": "PASS", "lane_count": len(rows), "rows": rows,
                    "labels_in_energy_unary_or_edge": 0, "dataset_name_reads_in_energy_core": 0,
                    "dense_n_by_n_count": 0}, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--kit", type=Path, required=True)
    common.add_argument("--banks", type=Path, required=True)
    common.add_argument("--partitions", type=Path, required=True)
    common.add_argument("--output", type=Path, required=True)

    search_parser = sub.add_parser("search", parents=[common])
    search_parser.add_argument("--night15e-registry", type=Path, required=True)
    search_parser.add_argument("--lanes", required=True)
    search_parser.add_argument("--random-configs", type=int, default=80)
    search_parser.add_argument("--refine-configs", type=int, default=80)

    replay_parser = sub.add_parser("replay", parents=[common])
    replay_parser.add_argument("--registry", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "search":
        search(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
