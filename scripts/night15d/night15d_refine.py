#!/usr/bin/env python3
"""Adaptive coarse-to-fine refinement for Night-15D finalists."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Tuple

import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SCRIPT_ROOT))

from SpaLORA.night15c_cluster_energy import sha256_array
from SpaLORA.night15d_reliability_energy import (
    csr_from_archive,
    edge_similarity,
    multiscale_feature_bank,
    reduce_full,
    reliability_energy_icm,
    reliability_transition,
)
from night15d_arena import (
    DATASET_FOR_LANE,
    NIGHT15C_AUTHORITY,
    append_row,
    evaluate,
    lane_semantics,
    objective,
    reproduce_night15c_authority,
    write_rows,
)


TOLERANCE = 1e-10


def dual_positive(row: Mapping[str, object]) -> bool:
    return (
        row.get("status") == "PASS"
        and float(row.get("delta_ari", -1.0)) > TOLERANCE
        and float(row.get("delta_nmi", -1.0)) > TOLERANCE
    )


def choose(rows: List[dict], authority: dict) -> dict:
    eligible = [row for row in rows if dual_positive(row)]
    return max(eligible, key=objective) if eligible else authority


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--banks", type=Path, required=True)
    parser.add_argument("--night15c-registry", type=Path, required=True)
    parser.add_argument("--seed-configs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lanes",
        default="P22,P22_3DOT_K18,MISAR_E15_5_S1,MISAR_E15_5_S1_K12,A1,D1,tonsil_s1,tonsil_s2,tonsil_s3",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    registered = json.loads(args.night15c_registry.read_text(encoding="utf-8"))[
        "stable_configs"
    ]
    seed_file = json.loads(args.seed_configs.read_text(encoding="utf-8"))
    seeds = seed_file["seeds"]
    all_rows: List[dict] = []
    summaries = []
    global_started = time.perf_counter()

    for lane in [value for value in args.lanes.split(",") if value]:
        lane_started = time.perf_counter()
        dataset = DATASET_FOR_LANE[lane]
        data = np.load(args.kit / f"{dataset}.npz", allow_pickle=False, mmap_mode="r")
        bank = np.load(
            args.banks / f"{dataset}_selected_partition_bank.npz",
            allow_pickle=False,
            mmap_mode="r",
        )
        k, labels, mask = lane_semantics(data, lane)
        graph = csr_from_archive(data, "graph")
        retained_raw = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
        features = multiscale_feature_bank(
            retained_raw, data["view1"], data["view2"], graph
        )
        view1 = reduce_full(data["view1"], min(24, data["view1"].shape[1]))
        view2 = reduce_full(data["view2"], min(24, data["view2"].shape[1]))
        authority_partition = reproduce_night15c_authority(
            data, bank, lane, k, registered[lane]
        )
        authority_metrics = evaluate(labels, mask, authority_partition)
        expected = NIGHT15C_AUTHORITY[lane]
        if abs(authority_metrics["absolute_ari"] - expected[0]) > TOLERANCE or abs(
            authority_metrics["absolute_nmi"] - expected[1]
        ) > TOLERANCE:
            raise RuntimeError(f"authority replay mismatch: {lane}")
        lane_rows: List[dict] = []
        authority_row = append_row(
            lane_rows,
            lane,
            k,
            labels,
            mask,
            "AUTHORITY",
            "NIGHT15C_STABLE",
            {"source": "night15c_authority"},
            authority_partition,
            authority_partition,
            {
                "stage": "AUTHORITY",
                "steps_completed": 0,
                "collapse_guard_triggered": False,
            },
            time.perf_counter(),
        )
        transition_cache: Dict[
            Tuple[str, str, float], Tuple[sp.csr_matrix, Dict[str, float]]
        ] = {}
        config_cache = set()

        def transition(edge: str, normalization: str, tau: float):
            key = (edge, normalization, float(tau))
            if key not in transition_cache:
                binary, weighted = edge_similarity(
                    graph,
                    data["view1"],
                    data["view2"],
                    edge,
                    tau=tau,
                    dim=16,
                )
                transition_cache[key] = reliability_transition(
                    binary, weighted, normalization
                )
            return transition_cache[key]

        def execute(config: Mapping[str, object], stage: str) -> None:
            config = dict(config)
            canonical = json.dumps(config, sort_keys=True, separators=(",", ":"))
            if canonical in config_cache:
                return
            config_cache.add(canonical)
            started = time.perf_counter()
            try:
                support, support_diag = transition(
                    str(config["edge_mode"]),
                    str(config["normalization"]),
                    float(config["tau"]),
                )
                partition, completed, collapse, unary_diag = reliability_energy_icm(
                    authority_partition,
                    support,
                    k,
                    float(config["beta"]),
                    int(config["steps"]),
                    str(config["unary_mode"]),
                    features[str(config["feature"])],
                    view1,
                    view2,
                    margin_temperature=float(config["margin_temperature"]),
                    retained_weight=float(config["retained_weight"]),
                    switch_penalty=float(config["switch_penalty"]),
                )
                append_row(
                    lane_rows,
                    lane,
                    k,
                    labels,
                    mask,
                    "RELIABILITY_ENERGY",
                    "ADAPTIVE_RELIABILITY_MASS_PROTOTYPE",
                    config,
                    authority_partition,
                    partition,
                    {
                        **support_diag,
                        **unary_diag,
                        "stage": stage,
                        "steps_completed": int(completed),
                        "collapse_guard_triggered": bool(collapse),
                        "initial_source": "night15c_authority",
                    },
                    started,
                )
            except Exception as error:
                lane_rows.append(
                    {
                        "lane": lane,
                        "k": k,
                        "family": "RELIABILITY_ENERGY",
                        "algorithm": "ADAPTIVE_RELIABILITY_MASS_PROTOTYPE",
                        "config_json": canonical,
                        "stage": stage,
                        "status": "FAILED",
                        "failure": repr(error),
                        "wall_seconds": time.perf_counter() - started,
                    }
                )

        seed = dict(seeds[lane])
        execute(seed, "SEED")

        edge_modes = ("spatial", "either", "both", "geomean", "agreement")
        normalizations = ("row", "mass", "self")
        betas = (0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0)
        steps_grid = (1, 2, 3, 5, 7, 10)
        for edge in edge_modes:
            for normalization in normalizations:
                if edge == "spatial" and normalization != "row":
                    continue
                for beta in betas:
                    for steps in steps_grid:
                        config = dict(seed)
                        config.update(
                            edge_mode=edge,
                            normalization=normalization,
                            beta=beta,
                            steps=steps,
                        )
                        execute(config, "EDGE_MASS_GRID")

        stage1 = choose(lane_rows, authority_row)
        stage1_config = json.loads(stage1["config_json"])
        unary_variants = [
            ("retained", 0.5, 0.35),
            ("dual_equal", 0.5, 0.0),
            ("dual_margin", 0.15, 0.0),
            ("dual_margin", 0.25, 0.0),
            ("dual_margin", 0.5, 0.0),
            ("dual_margin", 0.75, 0.0),
            ("dual_margin", 1.2, 0.0),
        ]
        unary_variants.extend(
            ("retained_dual_margin", temperature, weight)
            for temperature in (0.2, 0.5, 0.8)
            for weight in (0.2, 0.35, 0.5, 0.7)
        )
        for feature in features:
            for unary_mode, temperature, retained_weight in unary_variants:
                if unary_mode in ("dual_equal", "dual_margin") and feature != "retained":
                    continue
                config = dict(stage1_config)
                config.update(
                    feature=feature,
                    unary_mode=unary_mode,
                    margin_temperature=temperature,
                    retained_weight=retained_weight,
                )
                execute(config, "FEATURE_UNARY_GRID")

        stage2 = choose(lane_rows, authority_row)
        stage2_config = json.loads(stage2["config_json"])
        center_beta = max(float(stage2_config["beta"]), 0.025)
        beta_local = sorted(
            {
                round(center_beta * factor, 8)
                for factor in (0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)
            }
            | {0.025, 0.05, 0.1}
        )
        for tau in (0.5, 0.7, 1.0, 1.4, 2.0):
            for beta in beta_local:
                for steps in steps_grid:
                    for switch_penalty in (0.0, 0.02, 0.05, 0.1, 0.2):
                        config = dict(stage2_config)
                        config.update(
                            tau=tau,
                            beta=beta,
                            steps=steps,
                            switch_penalty=switch_penalty,
                        )
                        execute(config, "LOCAL_TEMPERATURE_TRUST_GRID")

        best = choose(lane_rows, authority_row)
        best_config = json.loads(best["config_json"])
        if best["family"] == "AUTHORITY":
            best_partition = authority_partition
        else:
            support, _ = transition(
                str(best_config["edge_mode"]),
                str(best_config["normalization"]),
                float(best_config["tau"]),
            )
            best_partition, _, _, _ = reliability_energy_icm(
                authority_partition,
                support,
                k,
                float(best_config["beta"]),
                int(best_config["steps"]),
                str(best_config["unary_mode"]),
                features[str(best_config["feature"])],
                view1,
                view2,
                margin_temperature=float(best_config["margin_temperature"]),
                retained_weight=float(best_config["retained_weight"]),
                switch_penalty=float(best_config["switch_penalty"]),
            )
        np.save(args.output / f"{lane}__best_partition.npy", best_partition, allow_pickle=False)
        summary = {
            "lane": lane,
            "k": k,
            "authority_partition_sha256": sha256_array(authority_partition),
            "best_partition_sha256": sha256_array(best_partition),
            "best": best,
            "run_rows": len(lane_rows),
            "pass_rows": sum(row.get("status") == "PASS" for row in lane_rows),
            "failed_rows": sum(row.get("status") != "PASS" for row in lane_rows),
            "dual_positive_rows": sum(dual_positive(row) for row in lane_rows),
            "wall_seconds": time.perf_counter() - lane_started,
            "labels_used_for_development_hpo_and_evaluation": 1,
            "labels_in_model_or_energy": 0,
            "dense_n_by_n_count": 0,
        }
        summaries.append(summary)
        all_rows.extend(lane_rows)
        write_rows(args.output / "all_run_ledger.partial.csv", all_rows)
        (args.output / f"{lane}__summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "lane": lane,
                    "rows": len(lane_rows),
                    "dual_positive_rows": summary["dual_positive_rows"],
                    "best_ari": best["absolute_ari"],
                    "best_nmi": best["absolute_nmi"],
                    "delta_ari": best["delta_ari"],
                    "delta_nmi": best["delta_nmi"],
                    "wall_seconds": summary["wall_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_rows(args.output / "all_run_ledger.csv", all_rows)
    result = {
        "status": "PASS",
        "selection_rule": seed_file["selection_rule"],
        "initial_rule": seed_file["initial_rule"],
        "lanes": summaries,
        "run_rows": len(all_rows),
        "wall_seconds": time.perf_counter() - global_started,
        "labels_used_for_development_hpo_and_evaluation": 1,
        "labels_in_model_or_energy": 0,
        "dataset_name_reads_in_model_core": 0,
        "dense_n_by_n_count": 0,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "UNSET"),
        "mkl_num_threads": os.environ.get("MKL_NUM_THREADS", "UNSET"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS", "UNSET"),
    }
    (args.output / "refine_summary.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "run_rows": result["run_rows"],
                "wall_seconds": result["wall_seconds"],
            }
        )
    )


if __name__ == "__main__":
    main()
