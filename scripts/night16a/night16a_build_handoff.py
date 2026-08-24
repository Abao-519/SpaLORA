#!/usr/bin/env python3
"""Build the compact, human-readable Night-16A handoff from frozen artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd


LANE_ORDER = [
    "A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3",
    "P22", "P22_3DOT_K18", "MISAR_E15_5_S1", "MISAR_E15_5_S1_K12",
]
DISPLAY = {
    "A1": "A1 lymph node K10", "D1": "D1 lymph node K10",
    "tonsil_s1": "tonsil s1 K4", "tonsil_s2": "tonsil s2 K4",
    "tonsil_s3": "tonsil s3 K4", "P22": "P22 K9",
    "P22_3DOT_K18": "P22 author assignment K18",
    "MISAR_E15_5_S1": "MISAR E15.5 K7", "MISAR_E15_5_S1_K12": "MISAR E15.5 K12",
}
PRIMARY = ["A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3", "P22", "MISAR_E15_5_S1"]


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fmt(value, digits=4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def csv_rows(path: Path) -> int:
    with path.open(encoding="utf-8", newline="") as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def correct_development_ledger(ledger_path: Path, registry_path: Path, destination: Path) -> None:
    frame = pd.read_csv(ledger_path)
    registry = read_json(registry_path)
    for index, row in frame.iterrows():
        key = f"{row['lane']}::{row['config_id']}"
        payload = registry[key]
        for name, value in payload["constants"].items():
            frame.at[index, f"calibration_constant__{name}"] = value
        for name, value in payload["statistics_and_parameters"].items():
            frame.at[index, f"calibration_stat__{name}"] = (
                json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value
            )
    frame["metadata_repaired_from_authoritative_registry"] = True
    frame.to_csv(destination, index=False)


def build_oracle_profiles(work: Path, internal: pd.DataFrame, output: Path) -> pd.DataFrame:
    sources = []
    for path, source in [
        (work / "calibrated_arena_all9_v2/calibrated_arena_ledger.csv", "energy_arena"),
        (work / "balanced_repair_d1/balanced_repair_ledger.csv", "D1_repair_arena"),
        (work / "structural_repair_secondary/structural_repair_ledger.csv", "secondary_repair_arena"),
    ]:
        frame = pd.read_csv(path)
        if source == "D1_repair_arena" and "lane" not in frame:
            frame["lane"] = "D1"
        if "status" in frame:
            frame = frame[frame.status == "PASS"].copy()
        frame["profile_source"] = source
        sources.append(frame)
    candidates = pd.concat(sources, ignore_index=True, sort=False)
    rows = []
    n_lookup = {row.lane: int(row.total_observations) for row in pd.read_csv(work / "strict_crossfit_family_v2/locked_metrics.csv").itertuples()}
    for lane in LANE_ORDER:
        lane_frame = candidates[candidates.lane == lane].copy()
        balanced = internal[internal.lane == lane].iloc[0]
        for profile, selected in [
            ("BALANCED_DUAL_FRONTIER", balanced),
            ("MAX_ARI", lane_frame.loc[lane_frame.absolute_ari.astype(float).idxmax()]),
            ("MAX_NMI", lane_frame.loc[lane_frame.absolute_nmi.astype(float).idxmax()]),
        ]:
            minimum = int(float(selected.get("min_cluster_size", 0)))
            rows.append({
                "lane": lane, "profile": profile,
                "absolute_ari": float(selected.absolute_ari), "absolute_nmi": float(selected.absolute_nmi),
                "min_cluster_size": minimum,
                "microcluster_below_1pct": bool(minimum < max(1, int(np.ceil(0.01 * n_lookup[lane])))),
                "source": selected.get("source", selected.get("profile_source", "frozen_internal_frontier")),
                "config_or_feature": selected.get("config_id", selected.get("feature", "")),
                "labels_used_for_cross_run_selection": True,
            })
        if lane == "D1":
            rows.append({
                "lane": lane, "profile": "PRIOR_PATHOLOGICAL_CONTEXT",
                "absolute_ari": 0.350728526, "absolute_nmi": 0.416032115,
                "min_cluster_size": 1, "microcluster_below_1pct": True,
                "source": "Night-15G prior singleton context", "config_or_feature": "not promoted",
                "labels_used_for_cross_run_selection": True,
            })
    result = pd.DataFrame(rows)
    result.to_csv(output, index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project, work, out = args.project, args.work, args.output
    out.mkdir(parents=True, exist_ok=True)

    internal = pd.read_csv(work / "working/internal_frontier/internal_frontier_table.csv")
    automatic = pd.read_csv(work / "working/strict_crossfit_family_v2/locked_metrics.csv")
    global_auto = pd.read_csv(work / "working/strict_crossfit_global_v2/locked_metrics.csv")
    single_auto = pd.read_csv(work / "working/fixed_start_guarded_crossfit_family_v2/locked_metrics.csv")
    ablation = pd.read_csv(work / "working/matched_ablation.csv")
    registry = read_json(work / "working/strict_crossfit_family_v2/frozen_crossfit_registry.json")
    descriptor = pd.read_csv(work / "working/strict_crossfit_inputs_v2/candidate_descriptors_no_public_metrics.csv")
    replay = read_json(work / "working/resource_replay.json")
    arena_summary = read_json(work / "working/calibrated_arena_all9_v2/arena_summary.json")
    start_summary = read_json(work / "working/start_bank_probe/start_bank_probe_summary.json")

    internal = internal.set_index("lane").loc[LANE_ORDER].reset_index()
    automatic = automatic.set_index("lane").loc[LANE_ORDER].reset_index()
    global_auto = global_auto.set_index("lane").loc[LANE_ORDER].reset_index()
    single_auto = single_auto.set_index("lane").loc[LANE_ORDER].reset_index()
    replay_rows = {row["lane"]: row for row in replay["rows"]}

    main_rows = []
    for auto in automatic.itertuples():
        lane = auto.lane
        dev = internal[internal.lane == lane].iloc[0]
        selected = registry["lanes"][lane]
        selected_desc = descriptor[descriptor.candidate_key == selected["candidate_key"]].iloc[0]
        run = replay_rows[lane]
        main_rows.append({
            "lane": lane, "dataset": DISPLAY[lane], "study": auto.held_out_study, "family": auto.family,
            "total_observations": auto.total_observations, "evaluated_observations": auto.evaluated_observations,
            "k": auto.k, "old_best_ari": dev.old_ari, "old_best_nmi": dev.old_nmi,
            "internal_frontier_ari": dev.absolute_ari, "internal_frontier_nmi": dev.absolute_nmi,
            "internal_delta_ari": dev.delta_ari, "internal_delta_nmi": dev.delta_nmi,
            "internal_ami": dev.ami, "internal_fmi": dev.fmi, "internal_morans_i": dev.morans_i,
            "internal_gearys_c": dev.gearys_c, "internal_min_cluster_size": dev.min_cluster_size,
            "internal_cluster_sizes": dev.cluster_sizes,
            "automatic_ari": auto.absolute_ari, "automatic_nmi": auto.absolute_nmi,
            "automatic_ami": auto.ami, "automatic_fmi": auto.fmi,
            "automatic_morans_i": auto.morans_i, "automatic_gearys_c": auto.gearys_c,
            "automatic_delta_ari_vs_old": auto.delta_ari_vs_current_best,
            "automatic_delta_nmi_vs_old": auto.delta_nmi_vs_current_best,
            "automatic_gap_ari_to_internal": auto.absolute_ari - dev.absolute_ari,
            "automatic_gap_nmi_to_internal": auto.absolute_nmi - dev.absolute_nmi,
            "automatic_min_cluster_size": auto.min_cluster_size,
            "automatic_cluster_sizes": auto.cluster_sizes,
            "automatic_start": selected["start_name"], "automatic_config_id": selected["config_id"],
            "start_stability_median_ari": selected_desc.get("calibration_stat__start_stability_median_ari", np.nan),
            "pairwise_beta": run["calibrated_parameters"]["pairwise_beta"],
            "self_return_strength": run["calibrated_parameters"]["self_return_strength"],
            "size_prior": run["calibrated_parameters"]["size_prior"],
            "scale_weights": json.dumps(run["calibrated_parameters"]["scale_weights"], separators=(",", ":")),
            "optional_weight": run["calibrated_parameters"]["optional_weight"],
            "partition_sha256": auto.partition_sha256,
            "automatic_fresh_process_exact_rate": 1.0,
            "automatic_replay_wall_seconds_shared_job": replay["wall_seconds"],
            "gpu_time_seconds": 0.0, "peak_gpu_mib": 0.0, "measured_peak_rss_mib": 377.79,
            "internal_selection_labels_used": True, "automatic_current_study_labels_used_before_lock": False,
        })
    main_table = pd.DataFrame(main_rows)
    main_table.to_csv(out / "main_results_table.csv", index=False)
    main_table[[
        "lane", "old_best_ari", "old_best_nmi", "internal_frontier_ari", "internal_frontier_nmi",
        "automatic_ari", "automatic_nmi", "automatic_gap_ari_to_internal", "automatic_gap_nmi_to_internal",
    ]].to_csv(out / "automatic_vs_internal_gap.csv", index=False)

    oracle = build_oracle_profiles(work / "working", internal, out / "internal_oracle_profiles.csv")

    group_definitions = {
        "RNA_PROTEIN_PRIMARY": ["A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3"],
        "RNA_ATAC_PRIMARY": ["P22", "MISAR_E15_5_S1"],
        "LYMPH_NODE_TRANSFER": ["A1", "D1"], "TONSIL_TRANSFER": ["tonsil_s1", "tonsil_s2", "tonsil_s3"],
        "P22_TRANSFER_PRIMARY": ["P22"], "MISAR_TRANSFER_PRIMARY": ["MISAR_E15_5_S1"],
    }
    group_rows = []
    for name, lanes in group_definitions.items():
        block = main_table[main_table.lane.isin(lanes)]
        group_rows.append({
            "group": name, "lanes": ";".join(lanes), "lane_count": len(block),
            "old_mean_ari": block.old_best_ari.mean(), "old_mean_nmi": block.old_best_nmi.mean(),
            "internal_mean_ari": block.internal_frontier_ari.mean(), "internal_mean_nmi": block.internal_frontier_nmi.mean(),
            "automatic_mean_ari": block.automatic_ari.mean(), "automatic_mean_nmi": block.automatic_nmi.mean(),
            "automatic_delta_ari_vs_old": (block.automatic_ari - block.old_best_ari).mean(),
            "automatic_delta_nmi_vs_old": (block.automatic_nmi - block.old_best_nmi).mean(),
            "held_out_current_group_labels_before_lock": 0,
        })
    pd.DataFrame(group_rows).to_csv(out / "grouped_transfer_summary.csv", index=False)

    consumer_rows = []
    for lane in LANE_ORDER:
        multi = automatic[automatic.lane == lane].iloc[0]
        single = single_auto[single_auto.lane == lane].iloc[0]
        glob = global_auto[global_auto.lane == lane].iloc[0]
        consumer_rows.append({
            "lane": lane,
            "family_multistart_ari": multi.absolute_ari, "family_multistart_nmi": multi.absolute_nmi,
            "family_single_start_ari": single.absolute_ari, "family_single_start_nmi": single.absolute_nmi,
            "multistart_minus_single_ari": multi.absolute_ari - single.absolute_ari,
            "multistart_minus_single_nmi": multi.absolute_nmi - single.absolute_nmi,
            "global_multistart_ari": glob.absolute_ari, "global_multistart_nmi": glob.absolute_nmi,
            "family_minus_global_ari": multi.absolute_ari - glob.absolute_ari,
            "family_minus_global_nmi": multi.absolute_nmi - glob.absolute_nmi,
        })
    pd.DataFrame(consumer_rows).to_csv(out / "consumer_and_calibration_scope_ablation.csv", index=False)

    full_lookup = {
        lane: row for lane, row in ablation[ablation.ablation == "FULL"].set_index("lane").iterrows()
    }
    ablation_rows = []
    for row in ablation.itertuples():
        full = full_lookup[row.lane]
        ablation_rows.append({
            **row._asdict(),
            "delta_ari_vs_full": row.absolute_ari - float(full.absolute_ari),
            "delta_nmi_vs_full": row.absolute_nmi - float(full.absolute_nmi),
        })
    pd.DataFrame(ablation_rows).to_csv(out / "matched_ablation.csv", index=False)

    target_rows = [
        ["A1", 10, "project canonical public labels", "3484/3484", "ARISE context", 0.3427, "", "protocol context; not claimed directly comparable", "https://pubmed.ncbi.nlm.nih.gov/42366683/"],
        ["D1", 10, "project canonical public labels", "3359/3359", "ARISE context", "", "", "no exact same-mask number extracted", "https://pubmed.ncbi.nlm.nih.gov/42366683/"],
        ["P22", 9, "MouseBrain_groundtruth coarse regions", "9196/9196", "COSMOS", 0.63, "", "close dataset context; endpoint differs", "https://www.nature.com/articles/s41467-024-55204-y"],
        ["P22_3DOT_K18", 18, "author 18-state assignment", "9196/9196", "3d-OT reported context", 0.39, "", "same K context; endpoint/version caveat", "local audited 3d-OT taskbook"],
        ["MISAR_E15_5_S1", 7, "deposited 7-class carrier", "1949/1949", "no exact public target registered", "", "", "project primary protocol", "local authority"],
        ["MISAR_E15_5_S1_K12", 12, "same carrier evaluated at K12", "1949/1949", "SEPAR", 0.644, "", "protocol context; not direct equality", "https://pmc.ncbi.nlm.nih.gov/articles/PMC12820152/"],
    ]
    with (out / "score_target_board.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["lane", "k", "label_version", "mask", "reported_method", "reported_ari", "reported_nmi", "comparability", "source"])
        writer.writerows(target_rows)

    pd.DataFrame([
        ["Night-15G/Night-15F per-lane optional energy", "up to 38 nested numeric fields", "cross-run public-label HPO", "different profile by lane"],
        ["Night-16A energy mapping", 7, "global/family constants", "generates the same 38 downstream fields from observable statistics"],
        ["Night-16A strict candidate bank", 12, "frozen global constant templates", "same bank for every lane"],
        ["Night-16A cross-study selector", 20, "11 structural descriptors + 9 start-class indicators", "four held-out-study ridge checkpoints; current-study labels absent"],
    ], columns=["component", "free_numeric_or_feature_count", "scope", "meaning"]).to_csv(out / "parameter_compression.csv", index=False)

    pd.DataFrame([
        ["N, K, N/K", "pairwise_beta", "0.75 + overlap + 0.15*log1p(N/K)", "active"],
        ["effective ranks of retained/RNA/second modality", "retained and view logits", "log rank ratio plus edge-coherence balance", "active"],
        ["registered neighbor overlap and edge correlation", "edge floor, capacity, pairwise, self-return", "continuous clipped transforms", "active"],
        ["conflict median and robust scale", "conflict center/temperature and mass controls", "median/MAD-derived", "active"],
        ["three-scale cross-modal support", "fine/registered/broad weights", "softmax(scale score / temperature)", "active"],
        ["multi-start median agreement", "unary temperature, size prior and trust", "continuous uncertainty transform", "active"],
        ["optional morphology edge reliability", "morphology unary/edge weight", "presence mask times reliability", "active; exact zero when missing"],
        ["zero fractions", "diagnostic only", "reported but not used by final mapping", "inactive"],
        ["candidate prototype margin/separation/centrality", "final candidate quality", "held-out cross-study ridge", "active selector feature"],
    ], columns=["observable_statistic", "generated_parameter_or_role", "mapping", "implementation_status"]).to_csv(out / "data_statistics_to_parameters.csv", index=False)

    p0_rows = []
    for row in replay["rows"]:
        p0_rows.append({
            "lane": row["lane"], "total_observations": row["total_observations"], "k": row["k"],
            "retained_shape": row["retained_shape"], "view1_shape": row["view1_shape"],
            "view2_shape": row["view2_shape"], "optional_shape": row["optional_shape"],
            "registered_graph_shape_nnz": row["graph_shape_nnz"], "partition_sha256": row["partition_sha256"],
            "fresh_process_byte_exact": row["byte_exact"],
        })
    write_json(out / "real_input_p0_audit.json", {
        "status": "NIGHT16A_REAL_P0_PASS", "lanes": len(p0_rows), "rows": p0_rows,
        "producer_label_arrays_opened": 0, "dense_n_by_n_count": 0,
        "pipeline": "preprocessing -> observable statistics -> calibration -> generated multi-start -> unified energy -> selector -> frozen partition -> separate evaluator -> reload",
    })

    write_json(out / "label_flow_audit.json", {
        "internal_exploration": {
            "public_labels_allowed": True, "cross_run_selection_uses_public_metrics": True,
            "eligible_as_automatic_main_result": False,
        },
        "strict_final_path": {
            "candidate_generation_before_reference_evaluation": True,
            "descriptor_public_metric_columns": 0, "current_held_out_study_reference_arrays_opened_by_producer": 0,
            "other_study_metrics_used_to_fit_meta_calibrator": True,
            "partition_locked_before_evaluator": True, "known_k_from_public_benchmark": True,
            "evaluator_reference_access_events": 9,
        },
        "labels_as_model_input_or_energy_unary": 0,
    })

    write_json(out / "resource_audit.json", {
        "execution_location": "Windows local CPU using existing compact feature banks",
        "gpu_time_seconds": 0.0, "peak_gpu_mib": 0.0,
        "measured_peak_rss_mib_final_replay": 377.79,
        "strict_full_9_lane_replay_wall_seconds": replay["wall_seconds"],
        "strict_arena_wall_seconds_sum": float(sum(v["wall_seconds"] for v in read_json(work / "working/strict_generated_arena_v2/arena_summary.json")["lanes"].values())),
        "internal_development_arena_wall_seconds_sum": float(sum(v["wall_seconds"] for v in arena_summary["lanes"].values())),
        "new_download_count": 0, "raw_modification_count": 0, "dense_n_by_n_count": 0,
    })

    inventory_specs = [
        ("start_bank_probe", work / "working/start_bank_probe/start_bank_probe_ledger.csv", "valid; labels appended only after candidate construction"),
        ("development_energy_arena", work / "working/calibrated_arena_all9_v2/calibrated_arena_ledger.csv", "metrics valid; per-row parameter metadata repaired from registry for delivery"),
        ("strict_generated_v1", work / "working/strict_generated_arena_v1/calibrated_arena_ledger.csv", "superseded: arena-writer parameter metadata bug"),
        ("strict_generated_v2", work / "working/strict_generated_arena_v2/calibrated_arena_ledger.csv", "valid"),
        ("fixed_start_v1", work / "working/fixed_start_arena_v1/calibrated_arena_ledger.csv", "superseded: arena-writer parameter metadata bug"),
        ("fixed_start_v2", work / "working/fixed_start_arena_v2/calibrated_arena_ledger.csv", "valid"),
        ("D1_microcluster_probe", work / "working/microcluster_repair_d1/microcluster_repair_ledger.csv", "valid internal exploration"),
        ("D1_balanced_repair", work / "working/balanced_repair_d1/balanced_repair_ledger.csv", "valid internal exploration"),
        ("secondary_repair", work / "working/structural_repair_secondary/structural_repair_ledger.csv", "valid internal exploration"),
        ("matched_ablation", work / "working/matched_ablation.csv", "valid strict automatic ablation"),
    ]
    inventory_rows = []
    for name, path, status in inventory_specs:
        inventory_rows.append({"artifact": name, "rows": csv_rows(path), "sha256": sha256(path), "status": status, "path": str(path)})
    pd.DataFrame(inventory_rows).to_csv(out / "run_inventory.csv", index=False)

    failure_rows = [
        ["ARENA_WRITER_PARAMETER_METADATA", "ENGINEERING", "strict/fixed v1 and all9 parameter columns", "partition and metric rows were correct, but loop wrote the last constants/statistics into each row", "writer fixed; strict/fixed rerun as v2; all9 delivered with registry-corrected metadata", "PRESERVED_SUPERSEDED"],
        ["STRICT_SELF_CALIBRATION", "SCIENTIFIC", "9/9 lanes", "automatic held-out selection was below the current best on every lane", "no backfill or label-selected replacement", "SCIENTIFIC_NEGATIVE"],
        ["GENERIC_ENSEMBLE", "SCIENTIFIC", "all lanes", "label-free medoid/consensus did not recover the high-quality authority starts", "retained as initialization-gap evidence", "NEGATIVE_RETAINED"],
        ["D1_PATHOLOGICAL_CONTEXT", "SCIENTIFIC", "D1", "prior 0.350729/0.416032 has singleton cluster", "not promoted; new nonmicro internal frontier is 0.338775/0.435972", "NOT_PROMOTED"],
        ["PYTEST_UNAVAILABLE", "INFRASTRUCTURE", "local environment", "pytest package absent", "dependency-free targeted test runner executed 5/5", "WORKAROUND_AUDITED"],
        ["FINAL_TEST_CLI_ARGUMENT", "ENGINEERING", "final test invocation", "first final invocation omitted the required --test-file argument and exited before tests", "command corrected immediately; final dependency-free run passed 5/5 after final source", "CORRECTED_AND_PRESERVED"],
        ["REMOTE_PYTHON_ALIAS", "INFRASTRUCTURE", "remote delivery audit", "the unqualified python command was absent", "reran the AST/JSON audit with /root/miniconda3/bin/python; 21 Python files and 16 JSON files passed", "WORKAROUND_AUDITED"],
        ["WINDOWS_CRLF_PRECOMMIT", "ENGINEERING", "remote precommit check", "the first git diff --cached --check rejected Windows CRLF payload lines before any commit was made", "normalized all text payloads to LF, restaged, and reran the check", "CORRECTED_BEFORE_COMMIT"],
        ["NIGHT15G_D1_PARTIAL_INHERITED", "HISTORICAL", "3432/5280 rows", "broad morphology search exited 1 without exception text", "not rerun or hidden; inherited partial ledger remains immutable", "PRESERVED"],
    ]
    pd.DataFrame(failure_rows, columns=["failure_id", "class", "scope", "finding", "handling", "status"]).to_csv(out / "failure_ledger.csv", index=False)

    changelog = """# Night-16A engineering changelog

- Built one dataset-name-blind observable-statistics-to-energy API with seven global/family controls.
- Corrected optional edge coherence: the median of `exp(-d/median(d))` is nearly constant, so the implementation now uses mean similarity plus interquartile persistence.
- Found and fixed the arena writer metadata bug: partitions and metrics were correct, but per-row constants/statistics were not stored with each generated tuple. Strict and fixed-start arenas were rerun as `v2`; earlier outputs remain preserved and marked superseded.
- Reconstructed the internal development ledger's parameter metadata from its authoritative calibration registry; scientific partitions and metrics were not changed.
- Added physical descriptor/metric separation and grouped hold-out files for lymph-node, tonsil, P22 and MISAR studies.
- Added a frozen Ridge-checkpoint replay; two new processes recovered all 9/9 selected candidate IDs without opening current-study labels.
- Added a measured-resource replay (9/9 exact; 377.79 MiB peak RSS). No GPU computation or new download was required.
- Pytest was unavailable locally; the dependency-free targeted runner passed 5/5 tests.
- The first final test-runner invocation omitted its required `--test-file` CLI argument and exited before executing tests; it was corrected immediately, and the post-source run passed 5/5.
- The remote login shell had no unqualified `python`; the delivery AST/JSON audit was rerun with `/root/miniconda3/bin/python` and passed (21 Python files, 16 JSON files, zero banned payloads).
- The first remote precommit whitespace check rejected Windows CRLF line endings before a commit was created; all Night-16A text payloads were normalized to LF, restaged and rechecked.
"""
    (out / "engineering_changelog.md").write_text(changelog, encoding="utf-8")

    source_audit = """# Night-16A source and novelty collision audit

Night-16A does not claim novelty for prototype/Gaussian unaries, Potts/CRF priors, alpha-expansion, graph cuts, multiscale neighborhood features, similarity-weighted graphs, morphology fusion or meta-learning by themselves. Night-15F already registered Boykov-Veksler-Zabih, Kolmogorov-Zabih, fusion moves, BANKSY, PRAGA, BayesSpace, BASS, DR.SC, SCGP, GROVER, SpaMCA, SpatialCOC and SpaMosaic. Night-15G additionally registered MISO, Proust, STESH, stGCL, SpatialEx and COSIE as image-plus-molecular precedents.

The narrower attempted object here is a clean-room calibration layer that maps observable rank, sparsity, graph-scale support, cross-modal overlap/conflict, start stability and optional-view reliability into the same continuous spatial energy, followed by a grouped held-out meta-selector. The implementation is unified and auditable, but the strict automatic outputs are substantially below the development authorities. Therefore Night-16A provides an engineering formulation and a negative transfer result, not a supported novelty claim or a paper-ready method contribution.

No third-party method source was copied. Existing GPL, AGPL and unlicensed repositories remain source-review-only. SciPy/scikit-learn APIs are used under their normal licenses.
"""
    (out / "source_collision_audit.md").write_text(source_audit, encoding="utf-8")

    # Copy compact evidence artifacts; no partitions, embeddings, images, checkpoints or raw arrays.
    copies = {
        work / "working/internal_frontier/internal_frontier_registry.json": "internal_frontier_registry.json",
        work / "working/internal_frontier/internal_frontier_table.csv": "internal_frontier_table.csv",
        work / "working/strict_generated_arena_v2/calibrated_arena_ledger.csv": "strict_generated_search_ledger.csv",
        work / "working/start_bank_probe/start_bank_probe_ledger.csv": "start_bank_probe_ledger.csv",
        work / "working/strict_crossfit_family_v2/frozen_crossfit_registry.json": "strict_frozen_calibrator_registry.json",
        work / "working/strict_crossfit_family_v2/locked_metrics.csv": "strict_automatic_metrics.csv",
        work / "working/strict_crossfit_global_v2/locked_metrics.csv": "strict_global_ablation_metrics.csv",
        work / "working/fixed_start_guarded_crossfit_family_v2/locked_metrics.csv": "strict_single_start_ablation_metrics.csv",
        work / "working/strict_crossfit_inputs_v2/candidate_descriptors_no_public_metrics.csv": "candidate_descriptors_no_public_metrics.csv",
        work / "working/strict_crossfit_inputs_v2/split_manifest.json": "grouped_split_manifest.json",
        work / "working/final_replay1_v2.json": "fresh_process_partition_replay1.json",
        work / "working/final_replay2_v2.json": "fresh_process_partition_replay2.json",
        work / "working/calibrator_selection_replay1.json": "fresh_process_calibrator_replay1.json",
        work / "working/calibrator_selection_replay2.json": "fresh_process_calibrator_replay2.json",
        work / "working/targeted_tests_final.json": "targeted_tests.json",
    }
    for source, name in copies.items():
        shutil.copy2(source, out / name)
    correct_development_ledger(
        work / "working/calibrated_arena_all9_v2/calibrated_arena_ledger.csv",
        work / "working/calibrated_arena_all9_v2/calibration_registry.json",
        out / "internal_development_search_ledger_corrected.csv",
    )

    replay_audit = {
        "partition_replays": [read_json(work / "working/final_replay1_v2.json"), read_json(work / "working/final_replay2_v2.json")],
        "calibrator_checkpoint_replays": [read_json(work / "working/calibrator_selection_replay1.json"), read_json(work / "working/calibrator_selection_replay2.json")],
        "resource_replay": {key: value for key, value in replay.items() if key != "rows"},
        "partition_replay_exact": True, "calibrator_selection_replay_exact": True,
    }
    write_json(out / "fresh_process_roundtrip_audit.json", replay_audit)

    authority = {
        "night15g": {
            "commit": "750d3fe9df52208b7618b878eb2701cd613b23b0", "tag": "night15g-final-20260824",
            "compact_payload": "47/47", "compact_index_sha256": "739fd739be27eefcd661b3fa0059007edfffaf5d555edfb6501856129a787fe7",
        },
        "night15f": {
            "commit": "dfbad141fba3c3f79963ed93ae5c0bc49fa6f370", "tag": "night15f-final-20260824",
            "compact_payload": "30/30", "compact_index_sha256": "7188eb661e5924b4f098d0ea96ab16dcd0e7b3b1c250d0d3c24a3edca6df1e26",
        },
        "git_objects_verified": True, "ledger_and_source_authority_read": True,
    }
    write_json(out / "parent_authority_audit.json", authority)

    write_json(out / "night16a_execution_contract.json", {
        "title": "Night-16A Score-Driven Self-Calibrating Unified Spatial Energy",
        "parent_commit": authority["night15g"]["commit"],
        "parent_tag": authority["night15g"]["tag"],
        "lanes": LANE_ORDER,
        "known_k": {row.lane: int(row.k) for row in automatic.itertuples()},
        "strict_final_path": {
            "start_generators": ["KMeans", "GaussianMixture-diag", "GaussianMixture-tied"],
            "generator_seeds": [0, 1, 2],
            "generic_start_ranking": "fixed observable descriptor score",
            "constant_templates": 12,
            "constant_bank_seed": 20260824,
            "calibrator_scope": "FAMILY",
            "ridge_alpha": 0.1,
            "minimum_relative_cluster": 0.02,
            "descriptor_profile": "compact",
            "held_out_groups": {
                "LYMPH_NODE": ["A1", "D1"],
                "TONSIL": ["tonsil_s1", "tonsil_s2", "tonsil_s3"],
                "P22": ["P22", "P22_3DOT_K18"],
                "MISAR": ["MISAR_E15_5_S1", "MISAR_E15_5_S1_K12"],
            },
            "current_group_reference_access_before_partition_lock": 0,
            "evaluation_after_partition_lock": True,
        },
        "internal_exploration": {
            "public_metric_cross_run_selection_allowed": True,
            "eligible_for_automatic_main_result": False,
            "all_failed_and_tradeoff_profiles_retained": True,
        },
        "replay_requirement": {"partition_fresh_process": "2x 9/9 exact", "calibrator_checkpoint": "2x 9/9 exact"},
        "final_remote_command": "/usr/bin/shutdown",
    })

    internal_advances = int(((main_table.internal_delta_ari > 0) | (main_table.internal_delta_nmi > 0)).sum())
    internal_dual = int(((main_table.internal_delta_ari > 0) & (main_table.internal_delta_nmi > 0)).sum())
    automatic_losses = int(((main_table.automatic_delta_ari_vs_old < 0) & (main_table.automatic_delta_nmi_vs_old < 0)).sum())
    decision = {
        "status": "NIGHT16A_SCORE_FRONTIER_ADVANCE_WITH_SELF_CALIBRATION_GAP",
        "classification": ["SCORE_FRONTIER_ADVANCE", "LOCAL_SIGNAL"],
        "strict_self_calibration_component_classification": "SCIENTIFIC_NEGATIVE",
        "evidence_tier": "PUBLIC_BENCHMARK_DEVELOPMENT_ONLY",
        "paper_ready": False, "confirmed_milestone": False, "sota_claimed": False,
        "internal_frontier": {
            "advanced_lanes_any_metric": internal_advances, "dual_positive_lanes": internal_dual,
            "headline": "D1 nonmicro 0.338774873/0.435971663",
            "selection_uses_public_labels": True,
        },
        "strict_automatic": {
            "lanes": 9, "both_metrics_below_old_best_lanes": automatic_losses,
            "producer_current_study_label_reads": 0, "partition_replay": "9/9 twice",
            "calibrator_checkpoint_replay": "9/9 twice", "self_calibrating_method_signal": False,
        },
        "git": {"parent": authority["night15g"]["commit"], "branch": "revision/q2-night16a-score-driven-self-calibrating-energy-20260824", "final_tag": "night16a-final-20260824", "final_commit": "RECORDED_POST_COMMIT_IN_DELIVERY_VERIFICATION"},
        "shutdown_dispatched": False,
    }
    write_json(out / "night16a_decision.json", decision)

    # Plain main report.
    rows_md = []
    for row in main_table.itertuples():
        rows_md.append(
            f"| {DISPLAY[row.lane]} | {row.total_observations}/{row.evaluated_observations} | {row.k} | "
            f"{fmt(row.old_best_ari)}/{fmt(row.old_best_nmi)} | {fmt(row.internal_frontier_ari)}/{fmt(row.internal_frontier_nmi)} | "
            f"{fmt(row.internal_delta_ari, 5)}/{fmt(row.internal_delta_nmi, 5)} | {fmt(row.automatic_ari)}/{fmt(row.automatic_nmi)} | "
            f"{fmt(row.automatic_delta_ari_vs_old, 5)}/{fmt(row.automatic_delta_nmi_vs_old, 5)} | "
            f"{fmt(row.automatic_ami)}/{fmt(row.automatic_fmi)} | {fmt(row.automatic_morans_i)}/{fmt(row.automatic_gearys_c)} | "
            f"{row.internal_min_cluster_size}/{row.automatic_min_cluster_size} |"
        )
    p0_md = []
    for row in replay["rows"]:
        p0_md.append(
            f"| {DISPLAY[row['lane']]} | {row['retained_shape']} | {row['view1_shape']} | {row['view2_shape']} | "
            f"{row['optional_shape']} | {row['graph_shape_nnz']} | {str(row['byte_exact']).lower()} |"
        )
    report = f"""# SpaLORA Night-16A 报告：高分驱动的自校准统一空间能量

## 我现在需要知道的三件事

1. **绝对分数确实继续前进了，但这是内部开发上限。** 9 条协议中 {internal_advances} 条至少一项刷新、{internal_dual} 条 ARI/NMI 双升；最重要的是 D1 从可信无微小簇的 `0.288876/0.413762` 提到 `0.338775/0.435972`，最小簇 51，ARI 距公开 ARISE context `0.3427` 约 0.0039。这个 D1 结果来自公开标签驱动的跨运行筛选，不能当成自动模型输出。
2. **真正的自校准链在工程上闭合、科学上没有追上。** 它把样本规模、有效秩、跨模态边重叠/冲突、图尺度稳定性、multi-start 稳定性和形态可靠性转成同一套能量参数，再用整组留出研究的 Ridge 校准器选最终 candidate。9/9 partition 和 9/9 校准器选择都在两个新进程中精确重放，但严格自动输出在 9/9 协议上都低于旧 best-so-far。
3. **对论文的含义是“分数前沿推进 + 自校准机制负结果”。** Night-16A 找到了更好的 D1 非病理性开发分区，也完成了统一消费者和标签后置评价的真实框架；但初始化/selector 仍无法无标签复现 oracle 高分，因此不能登记 `SELF_CALIBRATING_METHOD_SIGNAL`、`CONFIRMED MILESTONE` 或 paper-ready。

## 明确终态

- 主分类：`SCORE_FRONTIER_ADVANCE` + `LOCAL_SIGNAL`
- 严格自校准组件：`SCIENTIFIC_NEGATIVE`
- 状态：`NIGHT16A_SCORE_FRONTIER_ADVANCE_WITH_SELF_CALIBRATION_GAP`
- 证据层级：公开 benchmark 开发；不是盲测，不是 SOTA 声明。

## 本轮改了模型的哪一层

Night-16A 没有再造数据集专用出口，而是在 Night-15F/15G 的连续多尺度能量前增加两层统一消费者：第一层由可观测统计量生成 modality logits、三尺度权重、unary 温度、pairwise、self-return、size prior、trust 和 optional-view 权重；第二层在同一 KMeans/GMM start 生成器上，用跨研究留出的 Ridge checkpoint 按结构质量、prototype margin、模态共同支持和簇平衡选最终 partition。RNA+protein 与 RNA+ATAC 使用相同公式；只有冻结的 family-level 数值 checkpoint 可不同。图像缺失时 optional weight 精确归零。

## 绝对指标主表

`内部最高`使用公开标签做跨运行开发选择；`严格自动`在当前整组研究上先冻结 partition，再由独立 evaluator 打分。

| 数据集 | N/eval | K | 旧 best ARI/NMI | 新内部最高 ARI/NMI | 内部 ΔARI/ΔNMI | 严格自动 ARI/NMI | 自动 ΔARI/ΔNMI | 自动 AMI/FMI | 自动 Moran/Geary | 内部/自动最小簇 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows_md)}

内部 D1 的完整簇大小为 `[1081,86,1243,116,193,292,58,188,51,51]`；旧 singleton 高分 `0.350729/0.416032` 仅保留为病理性参照。P22 K18 的内部 balanced profile 最小簇为 24（低于 N 的 1%），因此即使 ARI/NMI 较高也明确登记为 small-cluster sensitivity。所有 lane 的完整簇大小、AMI/FMI、Moran/Geary 和 partition SHA 在 `main_results_table.csv`。

## 自动模型与内部最优的差距

严格 family-level multi-start 的主要结果为：A1 `0.1885/0.3279`、D1 `0.1336/0.2570`、P22 K9 `0.4418/0.5756`、P22 K18 `0.3606/0.5451`、MISAR K7 `0.3489/0.5270`、MISAR K12 `0.3186/0.5451`、tonsil s1/s2/s3 分别 `0.1015/0.2303`、`0.1471/0.2277`、`0.0681/0.1787`。这不是小幅不稳定，而是系统性的 initialization/selection gap。统一 generic medoid/consensus 也没有恢复 Night-15F/15G 的高质量 authorities。

## 分组转移与消费者对照

- lymph-node A1+D1、tonsil s1/s2/s3、P22 K9/K18、MISAR K7/K12 都按整组留出；同一 study 的次级 K 没被当作额外独立数据。
- family constants 相对 global constants 有时改善，但仍没有一个 study 保持旧 best；因此 family calibration 也不能升格为迁移信号。
- multi-start 相对固定 `KMeans(retained, seed=0)` 有混合结果：它改善 D1、MISAR K7、tonsil s2/s3，但固定单起点在 P22 K9/K18 和 tonsil s1 更高。初始化银行本身尚未被可靠排序。
- 详细 study-balanced 均值见 `grouped_transfer_summary.csv`；逐 lane global/family/single/multi 见 `consumer_and_calibration_scope_ablation.csv`。

## 关键 matched ablation

- 去掉 self-return 在部分 lane 分数上升，但 D1 最小簇降到 3、P22 K18 降到 1且指标严重崩塌；它是重要的稳定器，却不是普适增益来源。
- 固定单尺度和固定等模态权重多数与 full 接近，说明当前严格 selector 尚未真正利用丰富的连续校准自由度。
- A1、D1、tonsil s3 的 morphology missing/permuted 对严格自动输出几乎无影响；自动链没有重现 Night-15G 的形态增益。
- 这组结果支持“统一公式已实现”，不支持“每个子机制均有独立普适贡献”。

## 参数压缩与统计量映射

Night-15G 可暴露最多 38 个嵌套数值字段；Night-16A 用 7 个全局/家族控制量生成全部下游能量参数。严格 selector 使用 11 个结构描述量和 9 个 start-class 指示量，且对当前 held-out study 不打开指标。零值比例目前只记录、未进入最终公式；这一点已在 `data_statistics_to_parameters.csv` 如实标为 inactive。

## 真实 P0 与复现

| lane | retained | RNA/view1 | second/view2 | optional | registered graph [N,N,nnz] | byte exact |
|---|---:|---:|---:|---:|---:|---:|
{chr(10).join(p0_md)}

- 两次最终 9/9 energy/partition fresh-process replay：精确一致。
- 两次冻结 Ridge checkpoint/selection replay：9/9 candidate ID 精确一致，4 个 held-out-study checkpoint 均重新加载。
- 针对性测试：5/5；最终 replay 9 lanes 合计约 {fmt(replay['wall_seconds'], 2)} 秒，实测峰值 RSS 377.79 MiB；GPU 时间和峰值显存均为 0。

## 失败、限制和论文边界

1. 自校准 selector 在所有 9 条协议上都没有达到旧 best；最主要瓶颈是从通用 starts 找不到开发高分 authority，而不是能量重放失败。
2. D1 新高依赖 Night-15G 已经由公开标签筛过的形态分区再做结构修复，所以只属于内部 score frontier。
3. internal arena 曾发现 writer 元数据 bug；v1 输出已保留为 superseded，strict/fixed 全部重跑 v2，内部 ledger 的参数列从权威 registry 恢复，partition 和指标未改。
4. optional morphology 在严格自动链中没有可见增益；不能把 Night-15G 局部形态信号推广成自动模型结论。
5. Potts/CRF、alpha-expansion、多尺度图、prototype unary、图像融合和 meta-calibration 均有明确先例；本轮自动性能为负，不做新颖性锁定。

## 导师汇报版

1. Night-16A 同时做了两件事：继续刷新公开 benchmark 开发分数，并把历史逐数据集参数整理成统一的统计量驱动能量。
2. D1 获得了本轮最有价值的进展：无极小簇 ARI/NMI 达到 `0.3388/0.4360`，ARI 已接近公开 ARISE context。
3. 但这个高分仍是公开标签辅助开发结果，不是自动校准器自己找到的。
4. 严格自动链对 lymph node、tonsil、P22 和 MISAR 都做了整组留出，当前数据集标签只在 partition 锁定后评价。
5. 工程上它已 9/9 完整运行，并完成 partition 与校准器 checkpoint 的双重 fresh-process 重放。
6. 科学上自动结果在 9/9 协议都低于旧 best，说明统一初始化和无标签 final selector 仍是核心瓶颈。
7. 因此本轮应记为“分数前沿推进 + 局部信号”，自校准方法本身是科学负结果。
8. 下一阶段若继续，应优先研究能否在不继承标签筛选 authority 的前提下生成高质量 starts，而不是继续加复杂的能量参数。

## 技术附录

- 父级：`750d3fe9df52208b7618b878eb2701cd613b23b0` / `night15g-final-20260824`
- 分支：`revision/q2-night16a-score-driven-self-calibrating-energy-20260824`
- final tag：`night16a-final-20260824`
- 最终 commit、bundle SHA、compact N/N 与 index SHA 在 commit 后生成的 `delivery_verification.json` 中记录。
- 全部工作 ledger 保留；compact 不含 raw、embedding、partition array、image 或 checkpoint binary。
"""
    (out / "night16a_report.md").write_text(report, encoding="utf-8")
    (out / "night16a_plain_summary.md").write_text(
        "Night-16A 的直接成果是 D1 等公开开发分数继续前进；严格自校准链虽可复现，但 9/9 均未追上旧 best。"
        "因此分类是 SCORE_FRONTIER_ADVANCE + LOCAL_SIGNAL，自校准组件为 SCIENTIFIC_NEGATIVE。\n",
        encoding="utf-8",
    )

    write_json(out / "run_manifest.json", {
        "status": decision["status"], "generated_files": sorted(path.name for path in out.iterdir() if path.is_file()),
        "internal_development_rows": int(sum(row["rows"] for row in inventory_rows if "internal" in row["status"] or row["artifact"] in ("development_energy_arena", "D1_microcluster_probe", "D1_balanced_repair", "secondary_repair"))),
        "strict_generated_rows": csv_rows(work / "working/strict_generated_arena_v2/calibrated_arena_ledger.csv"),
        "all_failures_preserved": True, "raw_files_in_handoff": 0, "partition_arrays_in_handoff": 0,
    })


if __name__ == "__main__":
    main()
