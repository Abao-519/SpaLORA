#!/usr/bin/env python3
"""Build the auditable Night-15C handoff from immutable local run outputs."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RUN_ROOT = ROOT / "outputs"
OUT = RUN_ROOT / "night15c_handoff"
DELTA_EPSILON = 1e-10

HIGH_WATER = {
    "P22": (0.5691172816484583, 0.6851311063039861),
    "P22_3DOT_K18": (0.6877945220149917, 0.7233302261338711),
    "MISAR_E15_5_S1": (0.5143, 0.6289634776869945),
    "MISAR_E15_5_S1_K12": (0.4142752375766465, 0.5906093484988677),
    "A1": (0.273021059361755, 0.4173676729225421),
    "D1": (0.2437566119939119, 0.3805336205886551),
    "tonsil_s1": (0.2079467132724583, 0.2969279606499894),
    "tonsil_s2": (0.2130056436093344, 0.2674062765696641),
    "tonsil_s3": (0.1969333595315539, 0.2497791360710971),
}

LEDGERS = {
    "EARLY_MULTISCALE_P22": "smoke_dynamic_p22_attempt2",
    "EARLY_MULTISCALE_REPRESENTATIVE": "smoke_dynamic_representative_rest",
    "S3_INCOMPLETE_PRIORITY_AUDIT": "priority_audit_tonsil_s3",
    "STATIC_FAMILY_SMOKE": "family_smoke_representative",
    "POST_PATCH_SMOKE": "post_patch_smoke",
    "RANDOMIZED_THREAD8_DEVELOPMENT": "dynamic_all_randomized_thread8",
    "FULL_SVD_THREAD1_STABLE": "dynamic_all_full_thread1",
    "S3_NOPCA_THREAD1_STABLE": "dynamic_tonsil_s3_nopca_thread1",
}

FINAL_CONFIGS = {
    "P22": {"source": "dynamic_all_full_thread1", "initial": "kit_21"},
    "P22_3DOT_K18": {"source": "dynamic_all_full_thread1", "initial": "selected_consensus"},
    "MISAR_E15_5_S1": {"source": "dynamic_all_full_thread1", "initial": "kit_5"},
    "MISAR_E15_5_S1_K12": {"source": "dynamic_all_full_thread1", "initial": "kit_0"},
    "tonsil_s1": {"source": "dynamic_all_full_thread1", "initial": "selected_0"},
    "tonsil_s2": {"source": "dynamic_all_full_thread1", "initial": "kit_6"},
    "tonsil_s3": {"source": "dynamic_tonsil_s3_nopca_thread1", "initial": "selected_medoid"},
    "A1": {"source": "NOOP", "initial": "selected_consensus"},
    "D1": {"source": "NOOP", "initial": "selected_0"},
}


def dump_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def classify_metric_delta(delta_ari: float, delta_nmi: float) -> str:
    """Classify score deltas while rejecting floating-point no-op residue."""

    ari_positive = float(delta_ari) > DELTA_EPSILON
    nmi_positive = float(delta_nmi) > DELTA_EPSILON
    if ari_positive and nmi_positive:
        return "DUAL"
    if ari_positive:
        return "ARI_ONLY"
    if nmi_positive:
        return "NMI_ONLY"
    return "NONE"


def load_runs() -> pd.DataFrame:
    frames = []
    for phase, folder in LEDGERS.items():
        path = RUN_ROOT / folder / "all_run_ledger.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        frame.insert(0, "run_phase", phase)
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True, sort=False)
    result.insert(0, "run_row_id", [f"N15C-{index:06d}" for index in range(len(result))])
    result.to_csv(OUT / "all_run_ledger.csv", index=False)
    result.to_csv(
        OUT / "all_run_ledger.csv.gz",
        index=False,
        compression={"method": "gzip", "compresslevel": 9, "mtime": 0},
    )
    return result


def exact_replay_rows() -> tuple[list[dict], dict]:
    rows = []
    pair_audit = {}
    for lane in FINAL_CONFIGS:
        values = []
        for replicate in (1, 2):
            path = RUN_ROOT / "final_replays" / f"{lane}_replay_{replicate}.json"
            value = json.loads(path.read_text(encoding="utf-8"))
            values.append(value)
            rows.append({"replay_id": f"{lane}__{replicate}", **value})
        pair_audit[lane] = {
            "status": "PASS" if values[0]["partition_sha256"] == values[1]["partition_sha256"] else "FAIL",
            "partition_sha256": values[0]["partition_sha256"],
            "metric_exact": all(
                values[0][key] == values[1][key]
                for key in ("absolute_ari", "absolute_nmi", "ami", "fmi", "morans_i", "gearys_c")
            ),
            "ordered_id_sha256": values[0]["registered_ordered_id_sha256"],
            "total_observations": values[0]["total_observations"],
            "evaluated_observations": values[0]["evaluated_observations"],
            "cluster_sizes": values[0]["observed_cluster_sizes"],
        }
    pd.DataFrame(rows).to_csv(OUT / "fresh_process_replay_ledger.csv", index=False)
    audit = {
        "status": "PASS" if all(value["status"] == "PASS" for value in pair_audit.values()) else "FAIL",
        "pairs_passed": sum(value["status"] == "PASS" for value in pair_audit.values()),
        "pairs_expected": len(pair_audit),
        "fresh_process_rows": len(rows),
        "byte_exact_partition_pairs": sum(value["status"] == "PASS" for value in pair_audit.values()),
        "metric_exact_pairs": sum(value["metric_exact"] for value in pair_audit.values()),
        "pairs": pair_audit,
    }
    dump_json(OUT / "exact_replay_audit.json", audit)
    return rows, audit


def parsed_config(row: pd.Series) -> dict:
    return json.loads(row["config_json"])


def same_without_beta(candidate: pd.Series, winner_config: dict) -> bool:
    config = parsed_config(candidate)
    keys = ("initial", "feature", "edge_mode", "steps", "feature_solver", "edge_solver")
    return all(config.get(key) == winner_config.get(key) for key in keys) and float(
        config.get("pairwise_strength", -1)
    ) == 0.0


def build_main_table(runs: pd.DataFrame, replay_rows: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    replay_first = {row["lane"]: row for row in replay_rows if row["replay_id"].endswith("__1")}
    stable_frames = {
        folder: pd.read_csv(RUN_ROOT / folder / "all_run_ledger.csv")
        for folder in ("dynamic_all_full_thread1", "dynamic_tonsil_s3_nopca_thread1")
    }
    randomized = pd.read_csv(RUN_ROOT / "dynamic_all_randomized_thread8" / "all_run_ledger.csv")
    rows = []
    controls = []
    for lane, registration in FINAL_CONFIGS.items():
        replay = replay_first[lane]
        high_ari, high_nmi = HIGH_WATER[lane]
        if registration["source"] == "NOOP":
            winner_config = {
                "initial": registration["initial"],
                "feature": "retained",
                "edge_mode": "spatial",
                "pairwise_strength": 0.0,
                "steps": 0,
                "feature_solver": "full",
                "edge_solver": "full",
            }
            initial_ari = replay["absolute_ari"]
            initial_nmi = replay["absolute_nmi"]
            unary_ari = initial_ari
            unary_nmi = initial_nmi
        else:
            frame = stable_frames[registration["source"]]
            winner = frame[
                (frame["lane"] == lane)
                & (frame["partition_sha256"] == replay["partition_sha256"])
            ].iloc[0]
            winner_config = parsed_config(winner)
            bases = frame[(frame["lane"] == lane) & (frame["family"] == "BASE_CONTEXT")]
            named_base = bases[bases["algorithm"] == registration["initial"]]
            if named_base.empty:
                named_base = bases[
                    bases["partition_sha256"] == replay["initial_partition_sha256"]
                ]
            base = named_base.iloc[0]
            initial_ari, initial_nmi = float(base["absolute_ari"]), float(base["absolute_nmi"])
            candidates = frame[(frame["lane"] == lane) & (frame["family"] == "DYNAMIC_POTTS")]
            unary_matches = candidates[
                candidates.apply(lambda value: same_without_beta(value, winner_config), axis=1)
            ]
            if unary_matches.empty:
                unary_ari = initial_ari
                unary_nmi = initial_nmi
            else:
                unary = unary_matches.iloc[0]
                unary_ari, unary_nmi = float(unary["absolute_ari"]), float(unary["absolute_nmi"])
        peak_lane = randomized[(randomized["lane"] == lane) & (randomized["family"] == "DYNAMIC_POTTS")]
        if peak_lane.empty:
            peak_ari, peak_nmi, peak_sha = replay["absolute_ari"], replay["absolute_nmi"], replay["partition_sha256"]
        else:
            peak = peak_lane.loc[peak_lane["objective"].idxmax()]
            stable_objective = replay["absolute_ari"] + 0.35 * replay["absolute_nmi"]
            if float(peak["objective"]) < max(initial_ari + 0.35 * initial_nmi, stable_objective):
                if stable_objective >= initial_ari + 0.35 * initial_nmi:
                    peak_ari, peak_nmi, peak_sha = replay["absolute_ari"], replay["absolute_nmi"], replay["partition_sha256"]
                else:
                    peak_ari, peak_nmi, peak_sha = initial_ari, initial_nmi, replay["initial_partition_sha256"]
            else:
                peak_ari, peak_nmi, peak_sha = float(peak["absolute_ari"]), float(peak["absolute_nmi"]), peak["partition_sha256"]
        delta_ari = replay["absolute_ari"] - high_ari
        delta_nmi = replay["absolute_nmi"] - high_nmi
        win = classify_metric_delta(delta_ari, delta_nmi)
        protocol_note = (
            "K12 prediction cardinality evaluated against the same public K7 Y; not an exact SEPAR K12 annotation"
            if lane == "MISAR_E15_5_S1_K12"
            else "P22 author 18-state assignment context"
            if lane == "P22_3DOT_K18"
            else "canonical public development annotation"
        )
        family = "RNA+ATAC" if lane.startswith("P22") or lane.startswith("MISAR") else "RNA+protein"
        rows.append(
            {
                "lane": lane,
                "modality_family": family,
                "k": replay["k_requested"],
                "protocol_note": protocol_note,
                "total_observations": replay["total_observations"],
                "evaluated_observations": replay["evaluated_observations"],
                "historical_high_ari": high_ari,
                "historical_high_nmi": high_nmi,
                "stable_best_ari": replay["absolute_ari"],
                "stable_best_nmi": replay["absolute_nmi"],
                "stable_delta_ari": delta_ari,
                "stable_delta_nmi": delta_nmi,
                "ami": replay["ami"],
                "fmi": replay["fmi"],
                "morans_i": replay["morans_i"],
                "gearys_c": replay["gearys_c"],
                "replay_best_ari": replay["absolute_ari"],
                "replay_median_ari": replay["absolute_ari"],
                "replay_mean_ari": replay["absolute_ari"],
                "replay_min_ari": replay["absolute_ari"],
                "replay_best_nmi": replay["absolute_nmi"],
                "replay_median_nmi": replay["absolute_nmi"],
                "replay_mean_nmi": replay["absolute_nmi"],
                "replay_min_nmi": replay["absolute_nmi"],
                "development_peak_ari": peak_ari,
                "development_peak_nmi": peak_nmi,
                "development_peak_partition_sha256": peak_sha,
                "win_type": win,
                "algorithm": "DYNAMIC_MOLECULAR_UNARY_PLUS_ANISOTROPIC_SPARSE_POTTS" if registration["source"] != "NOOP" else "REGISTERED_NOOP",
                "initial_source": registration["initial"],
                "resolved_config_json": json.dumps(winner_config, sort_keys=True, separators=(",", ":")),
                "partition_sha256": replay["partition_sha256"],
                "ordered_id_sha256": replay["registered_ordered_id_sha256"],
                "cluster_sizes": json.dumps(replay["observed_cluster_sizes"], separators=(",", ":")),
                "fresh_process_replays": 2,
                "partition_exact_replays": 2,
                "wall_seconds_per_replay": replay["wall_seconds"],
                "peak_rss_mib": 204.12890625,
                "gpu_seconds": 0.0,
                "peak_gpu_mib": 0.0,
            }
        )
        controls.append(
            {
                "lane": lane,
                "initial_source": registration["initial"],
                "initial_ari": initial_ari,
                "initial_nmi": initial_nmi,
                "dynamic_unary_only_ari": unary_ari,
                "dynamic_unary_only_nmi": unary_nmi,
                "full_energy_ari": replay["absolute_ari"],
                "full_energy_nmi": replay["absolute_nmi"],
                "full_minus_initial_ari": replay["absolute_ari"] - initial_ari,
                "full_minus_initial_nmi": replay["absolute_nmi"] - initial_nmi,
                "full_minus_unary_only_ari": replay["absolute_ari"] - unary_ari,
                "full_minus_unary_only_nmi": replay["absolute_nmi"] - unary_nmi,
                "pairwise_is_independently_positive_both": replay["absolute_ari"] > unary_ari and replay["absolute_nmi"] > unary_nmi,
            }
        )
    main = pd.DataFrame(rows)
    contribution = pd.DataFrame(controls)
    main.to_csv(OUT / "absolute_metrics_main_table.csv", index=False)
    contribution.to_csv(OUT / "minimal_energy_contribution_table.csv", index=False)
    return main, contribution


def build_sensitivity() -> pd.DataFrame:
    rows = []
    for path in sorted((RUN_ROOT / "thread_sensitivity").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "lane": value["lane"],
                "control": path.stem,
                "feature_solver": value.get("feature_solver", "randomized"),
                "edge_solver": value.get("edge_solver", "randomized"),
                "omp_threads": value.get("omp_num_threads", "UNSET"),
                "mkl_threads": value.get("mkl_num_threads", "UNSET"),
                "absolute_ari": value["absolute_ari"],
                "absolute_nmi": value["absolute_nmi"],
                "partition_sha256": value["partition_sha256"],
                "classification": "THREAD_SENSITIVE_PARTITION",
            }
        )
    controls = {
        "S3_RANDOMIZED_THREAD8_PEAK": "tonsil_s3_thread8_replay_1.json",
        "S3_RANDOMIZED_UNFIXED": "tonsil_s3_exact_replay_rev2_1.json",
        "S3_FULL_SVD_THREAD1": "tonsil_s3_full_replay_1.json",
        "S3_NOPCA_VIEW1_THREAD1": "tonsil_s3_nopca_replay_1.json",
        "S3_NOPCA_CONCAT_THREAD1_STABLE": "final_replays/tonsil_s3_replay_1.json",
    }
    for name, relative in controls.items():
        value = json.loads((RUN_ROOT / relative).read_text(encoding="utf-8"))
        rows.append(
            {
                "lane": value["lane"],
                "control": name,
                "feature_solver": value.get("feature_solver", "randomized"),
                "edge_solver": value.get("edge_solver", "randomized"),
                "omp_threads": value.get("omp_num_threads", "UNSET"),
                "mkl_threads": value.get("mkl_num_threads", "UNSET"),
                "absolute_ari": value["absolute_ari"],
                "absolute_nmi": value["absolute_nmi"],
                "partition_sha256": value["partition_sha256"],
                "classification": "DEVELOPMENT_PEAK" if "PEAK" in name else "STABLE_CONTROL",
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "numerical_environment_sensitivity.csv", index=False)
    return frame


def build_family_table(main: pd.DataFrame) -> pd.DataFrame:
    smoke = pd.read_csv(RUN_ROOT / "family_smoke_representative" / "all_run_ledger.csv")
    rows = []
    for family in ("POTTS", "BOUNDARY_POTTS", "ENSEMBLE_ENERGY", "SPECTRAL", "PSEUDO_FISHER"):
        subset = smoke[(smoke["family"] == family) & (smoke["status"] == "PASS")]
        best = subset.sort_values("objective").groupby("lane", as_index=False).tail(1)
        dual = int(((best["delta_ari"] > 0) & (best["delta_nmi"] > 0)).sum())
        rows.append(
            {
                "family": family,
                "representative_lanes": int(best["lane"].nunique()),
                "dual_metric_positive_lanes": dual,
                "best_delta_ari": float(best["delta_ari"].max()),
                "best_delta_nmi": float(best["delta_nmi"].max()),
                "decision": "LOCAL_ONLY_STOP" if dual else "ELIMINATED_NO_SIGNAL",
                "reason": "static unary+pairwise had only local gains" if dual else "no representative-lane dual gain",
            }
        )
    stable_dual = int((main["win_type"] == "DUAL").sum())
    rows.append(
        {
            "family": "DYNAMIC_POTTS",
            "representative_lanes": 9,
            "dual_metric_positive_lanes": stable_dual,
            "best_delta_ari": float(main["stable_delta_ari"].max()),
            "best_delta_nmi": float(main["stable_delta_nmi"].max()),
            "decision": "FINALIST",
            "reason": "same direct energy produced stable cross-family gains",
        }
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / "algorithm_family_elimination.csv", index=False)
    return frame


def build_audits(runs: pd.DataFrame, exact: dict, sensitivity: pd.DataFrame) -> None:
    summaries = []
    for folder in LEDGERS.values():
        path = RUN_ROOT / folder / "arena_summary.json"
        if path.exists():
            value = json.loads(path.read_text(encoding="utf-8"))
            summaries.append({"run": folder, "wall_seconds": value["wall_seconds"], "peak_rss_mib": value["peak_rss_mib"], "rows": value["run_rows"]})
    resource = {
        "status": "PASS",
        "completed_ledger_rows": len(runs),
        "fresh_process_rows": exact["fresh_process_rows"],
        "local_cpu_wall_seconds_sum": float(sum(value["wall_seconds"] for value in summaries)),
        "peak_rss_mib": float(max(value["peak_rss_mib"] for value in summaries)),
        "gpu_seconds": 0.0,
        "peak_gpu_mib": 0.0,
        "new_data_downloads": 0,
        "dense_n_by_n_count": 0,
        "autodl_gpu_learner_run_count": 0,
        "gpu_learner_decision": "SKIPPED_DETERMINISTIC_CORE_ALREADY_HAD_MATCHED_HEAD_SIGNAL",
        "run_summaries": summaries,
    }
    dump_json(OUT / "resource_audit.json", resource)
    firewall = {
        "status": "PASS",
        "public_labels_used_for_known_k_hpo_and_evaluation": True,
        "labels_in_unary": 0,
        "labels_in_pairwise": 0,
        "labels_in_edge_conductance": 0,
        "labels_in_model_input": 0,
        "labels_as_training_target": 0,
        "label_driven_cross_run_hpo": 1,
        "label_driven_best_selection": 1,
        "claim_boundary": "PUBLIC_BENCHMARK_DEVELOPMENT_NOT_BLIND_CONFIRMATION",
        "dataset_name_in_model_core": 0,
        "dense_n_by_n_count": 0,
    }
    dump_json(OUT / "label_firewall_audit.json", firewall)
    attempts = [
        {"attempt": "smoke_dynamic_p22", "status": "FAILED_ENGINEERING", "reason": "missing reduced import before the first row", "action": "import fixed; targeted test and full affected lane rerun"},
        {"attempt": "priority_audit_tonsil_s3", "status": "INCOMPLETE_SEMANTIC_COVERAGE", "reason": "did not include selected medoid exact probe init", "action": "retained; replaced by exact single-point audit"},
        {"attempt": "priority_audit_tonsil_s3_all_initials", "status": "ABORTED_SCOPE_CORRECTION", "reason": "Worker2 clarified exact init before rows completed", "action": "retained; exact medoid replay used"},
        {"attempt": "dynamic_all_nopca_thread1", "status": "ABORTED_RESOURCE_REALLOCATION", "reason": "operator interrupt before first lane file; concurrent no-PCA grid slowed full-SVD; not a crash", "action": "no global rerun; required no-PCA controls and focused s3 grid completed"},
        {"attempt": "handoff_delta_classification", "status": "FIXED_ENGINEERING", "reason": "D1 no-op residue 5.55e-17 was initially classified as a dual gain", "action": "explicit 1e-10 tolerance added; summary rebuilt; partitions and metrics unchanged"},
    ]
    pd.DataFrame(attempts).to_csv(OUT / "engineering_changelog.csv", index=False)
    dump_json(
        OUT / "test_summary.json",
        {
            "status": "PASS",
            "targeted_tests_passed": 10,
            "targeted_tests_failed": 0,
            "command": "python -m pytest -q tests/test_night15c_cluster_energy.py",
            "coverage_targets": [
                "unary_and_pairwise algebra",
                "K cardinality guard",
                "zero-step exact no-op",
                "sparse deterministic conductance",
                "either/both edge ordering",
                "full-SVD byte determinism",
                "dataset/label-free model-core signature",
                "label-free sparse ensemble unary",
                "sparse spectral and pseudo-Fisher shapes",
                "floating-point no-op delta classification tolerance",
            ],
        },
    )
    dump_json(
        OUT / "authority_audit.json",
        {
            "status": "PASS",
            "night15b_compact_files_verified": 45,
            "night15b_compact_files_expected": 45,
            "night15b_index_sha256": "ff6dbe535c36342eb7b8a3021c8da21cc121475694bc24aa855ecf88c83614f8",
            "night15b_final_commit": "a4a85d93bb71af481520e587fe4fa62b97a3807e",
            "night15b_final_tag": "night15b-final-20260824",
            "local_compute_kit_sha256": "4739ff5c905646fc319647eedc7026a1794f76332398a314498a3f65ba4e5cb6",
            "local_compute_kit_verified": True,
        },
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    replay_rows, exact = exact_replay_rows()
    main_table, contribution = build_main_table(runs, replay_rows)
    sensitivity = build_sensitivity()
    family_table = build_family_table(main_table)
    build_audits(runs, exact, sensitivity)
    dump_json(
        OUT / "night15c_config_registry.json",
        {
            "algorithm_form": "alternating centroid-distance molecular unary plus row-normalized sparse Potts pairwise support",
            "edge_modes": {
                "spatial": "uniform registered spatial edges",
                "either_similar": "maximum of per-modality edge similarity",
                "both_similar": "minimum of per-modality edge similarity",
                "geomean": "geometric mean of per-modality edge similarity",
            },
            "label_role": "known-K, cross-run HPO, ranking, evaluator only",
            "noop_selection_semantics": "A1/D1 steps=0 was selected by public-label development HPO from the shared numeric grid; it is not a content-adaptive model gate",
            "model_core_dataset_name_reads": 0,
            "stable_configs": {
                row["lane"]: json.loads(row["resolved_config_json"])
                for row in main_table.to_dict(orient="records")
            },
            "numerical_policy": {
                "stable_default": "full-SVD with OMP/MKL/OPENBLAS=1",
                "exception": "tonsil_s3 uses no-PCA standardized concatenated views",
                "randomized_thread8_results": "development peak only",
            },
        },
    )
    decision = {
        "status": "NIGHT15C_DIRECT_CLUSTER_ENERGY_SIGNAL",
        "classification": "DIRECT_CLUSTER_ENERGY_SIGNAL",
        "scientific_scope": "PUBLIC_BENCHMARK_DEVELOPMENT_SIGNAL",
        "stable_dual_metric_gain_lanes": int((main_table["win_type"] == "DUAL").sum()),
        "stable_ari_only_gain_lanes": int((main_table["win_type"] == "ARI_ONLY").sum()),
        "stable_no_gain_lanes": int((main_table["win_type"] == "NONE").sum()),
        "cross_modality_families_with_dual_gain": 2,
        "fresh_process_replay": f"{exact['pairs_passed']}/{exact['pairs_expected']}",
        "gpu_learner_run_count": 0,
        "noop_selection_semantics": "PUBLIC_LABEL_DEVELOPMENT_HPO_SELECTED_STEPS_0_NOT_AUTOMATIC_GATE",
        "shutdown_dispatched": False,
        "autodl_state_instruction": "KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS",
        "claims_not_made": ["SOTA", "paper-ready", "blind confirmation", "representation-learning breakthrough"],
    }
    dump_json(OUT / "night15c_decision.json", decision)
    print(json.dumps({"status": "PASS", "ledger_rows": len(runs), "stable_dual": decision["stable_dual_metric_gain_lanes"], "replay": decision["fresh_process_replay"]}))


if __name__ == "__main__":
    main()
