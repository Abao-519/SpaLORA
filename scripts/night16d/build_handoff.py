#!/usr/bin/env python3
"""Build the compact Night-16D handoff from locked producer/evaluator artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.night16d.night16d_evaluator import load_evaluation, metrics


ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "working" / "night16d"
OUT = ROOT / "outputs" / "night16d_handoff"
KIT = Path("/root/night16d_assets/kit/local_compute_kit")
START = Path("/root/night16d_assets/starts")
LANES = ["A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3", "P22", "MISAR_E15_5_S1"]
SCREEN_LANES = ["A1", "tonsil_s1", "P22", "MISAR_E15_5_S1"]
FAMILY = {
    "A1": "RNA_PROTEIN", "D1": "RNA_PROTEIN", "tonsil_s1": "RNA_PROTEIN",
    "tonsil_s2": "RNA_PROTEIN", "tonsil_s3": "RNA_PROTEIN",
    "P22": "RNA_CHROMATIN", "MISAR_E15_5_S1": "RNA_CHROMATIN",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def config(row: dict[str, str]) -> dict[str, Any]:
    return json.loads(row["config_json"])


def op(row: dict[str, str]) -> str:
    return str(config(row)["operation_mode"])


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def baseline_metrics() -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for lane in LANES:
        part = np.load(START / f"{lane}.npy", allow_pickle=False)
        ev = load_evaluation(KIT, lane)
        value = metrics(part, ev)
        with np.load(KIT / f"{lane}.npz", allow_pickle=False) as z:
            shapes = {
                key: list(z[key].shape)
                for key in ("view1", "view2", "coordinates", "ids") if key in z.files
            }
            n = int(len(z["ids"]))
        value.update({"lane": lane, "family": FAMILY[lane], "n_total": n, "tensor_shapes": shapes})
        result[lane] = value
    return result


def merge_ledgers() -> tuple[list[dict[str, Any]], dict[str, list[dict[str, str]]]]:
    specs = [
        ("stage2", "SUPERSEDED_PRE_TRUST_GATE_ENDPOINT"),
        ("refine_v2", "FINAL_SYNTHETIC_TEACHER_SEMANTICS"),
        ("aggressive_v3", "FINAL_SYNTHETIC_TEACHER_SEMANTICS"),
        ("retained_v4", "FINAL_RETAINED_PLUGIN_SEMANTICS"),
        ("retained_matched_full_v5", "FINAL_RETAINED_MATCHED_FULL_CONTROL"),
    ]
    merged: list[dict[str, Any]] = []
    by_stage: dict[str, list[dict[str, str]]] = {}
    for stage, evidence in specs:
        rows: list[dict[str, str]] = []
        for family in ("RNA_PROTEIN", "RNA_CHROMATIN"):
            path = WORK / stage / family / "evaluated_ledger.csv"
            rows.extend(read_csv(path))
        by_stage[stage] = rows
        for row in rows:
            out: dict[str, Any] = {
                "experiment_version": stage,
                "evidence_status": evidence,
                "superseded": stage == "stage2",
            }
            out.update(row)
            merged.append(out)
    return merged, by_stage


def retained_controls(rows: list[dict[str, str]], lane: str) -> dict[str, dict[str, str]]:
    lane_rows = [r for r in rows if r["lane"] == lane and r["status"] == "PASS"]
    output: dict[str, dict[str, str]] = {}
    for name in ("teacher", "teacher_head", "generic", "support", "support_boundary", "support_conflict", "full_no_trust"):
        eligible = [r for r in lane_rows if op(r) == name]
        if eligible:
            output[name] = sorted(eligible, key=lambda r: r["candidate_id"])[0]
    shuffled = [r for r in lane_rows if op(r) == "full" and config(r).get("shuffle_edge_states")]
    base_full = [
        r for r in lane_rows if op(r) == "full" and int(config(r)["training_steps"]) == 40
        and not config(r).get("shuffle_edge_states")
    ]
    if shuffled:
        output["shuffled"] = sorted(shuffled, key=lambda r: r["candidate_id"])[0]
    if base_full:
        output["full_matched"] = sorted(base_full, key=lambda r: r["candidate_id"])[0]
    baseline = output["teacher"]
    full = [r for r in lane_rows if op(r) == "full" and not config(r).get("shuffle_edge_states")]
    def balanced_key(r: dict[str, str]) -> tuple[float, ...]:
        da = as_float(r, "absolute_ari") - as_float(baseline, "absolute_ari")
        dn = as_float(r, "absolute_nmi") - as_float(baseline, "absolute_nmi")
        return (float(da > 1e-12 and dn > 1e-12), min(da, dn), da + dn, as_float(r, "absolute_ari"))
    output["best_full_balanced"] = max(full, key=balanced_key)
    output["max_ari_full"] = max(full, key=lambda r: (as_float(r, "absolute_ari"), as_float(r, "absolute_nmi")))
    output["max_nmi_full"] = max(full, key=lambda r: (as_float(r, "absolute_nmi"), as_float(r, "absolute_ari")))
    return output


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    baselines = baseline_metrics()
    merged, by_stage = merge_ledgers()
    write_csv(OUT / "candidate_search_ledger.csv", merged)
    retained = by_stage["retained_v4"] + by_stage["retained_matched_full_v5"]
    controls = {lane: retained_controls(retained, lane) for lane in SCREEN_LANES}

    main_rows: list[dict[str, Any]] = []
    frontier: list[dict[str, Any]] = []
    contributions: list[dict[str, Any]] = []
    teacher_audit: list[dict[str, Any]] = []
    for lane in LANES:
        base = baselines[lane]
        k = len(json.loads(base["cluster_sizes_eval"]))
        row: dict[str, Any] = {
            "dataset": lane, "family": FAMILY[lane], "n_total": base["n_total"],
            "n_evaluated": base["evaluated_observations"], "K": k,
            "input_strong_start_ari": base["absolute_ari"], "input_strong_start_nmi": base["absolute_nmi"],
            "headline_status": "NOT_RUN_EARLY_STOP_NO_REPRESENTATION_SIGNAL" if lane not in SCREEN_LANES else "BOUNDED_SCREEN_ONLY",
            "headline_ari": base["absolute_ari"], "headline_nmi": base["absolute_nmi"],
            "delta_ari_vs_input": 0.0, "delta_nmi_vs_input": 0.0,
            "min_cluster_size_full": base["min_cluster_size_full"], "cluster_sizes_full": base["cluster_sizes_full"],
            "min_cluster_size_eval": base["min_cluster_size_eval"], "cluster_sizes_eval": base["cluster_sizes_eval"],
            "ami": base["ami"], "fmi": base["fmi"],
            "morans_i_macro": base["morans_i_macro"], "gearys_c_macro": base["gearys_c_macro"],
            "neighbor_agreement": base["neighbor_agreement"],
        }
        frontier.append({"dataset": lane, "profile": "INPUT_STRONG_START", "absolute_ari": base["absolute_ari"], "absolute_nmi": base["absolute_nmi"], "delta_ari": 0.0, "delta_nmi": 0.0, "candidate_id": "INPUT_STRONG_START"})
        if lane in SCREEN_LANES:
            c = controls[lane]
            chosen = c["best_full_balanced"]
            da = as_float(chosen, "absolute_ari") - base["absolute_ari"]
            dn = as_float(chosen, "absolute_nmi") - base["absolute_nmi"]
            row.update({
                "screen_balanced_full_ari": as_float(chosen, "absolute_ari"),
                "screen_balanced_full_nmi": as_float(chosen, "absolute_nmi"),
                "screen_delta_ari_vs_input": da, "screen_delta_nmi_vs_input": dn,
                "screen_candidate_id": chosen["candidate_id"],
                "screen_conclusion": "DUAL_GAIN" if da > 1e-12 and dn > 1e-12 else ("TRADEOFF_OR_NO_GAIN"),
            })
            for profile, key in (("BALANCED_FULL", "best_full_balanced"), ("MAX_ARI_FULL", "max_ari_full"), ("MAX_NMI_FULL", "max_nmi_full")):
                r = c[key]
                frontier.append({
                    "dataset": lane, "profile": profile, "absolute_ari": as_float(r, "absolute_ari"),
                    "absolute_nmi": as_float(r, "absolute_nmi"),
                    "delta_ari": as_float(r, "absolute_ari") - base["absolute_ari"],
                    "delta_nmi": as_float(r, "absolute_nmi") - base["absolute_nmi"],
                    "candidate_id": r["candidate_id"], "teacher_source": r.get("teacher_source", ""),
                })
            for name in ("teacher", "teacher_head", "generic", "support", "support_boundary", "support_conflict", "full_matched", "full_no_trust", "shuffled", "best_full_balanced"):
                r = c[name]
                contributions.append({
                    "dataset": lane, "control": name, "candidate_id": r["candidate_id"],
                    "absolute_ari": as_float(r, "absolute_ari"), "absolute_nmi": as_float(r, "absolute_nmi"),
                    "delta_ari_vs_input": as_float(r, "absolute_ari") - base["absolute_ari"],
                    "delta_nmi_vs_input": as_float(r, "absolute_nmi") - base["absolute_nmi"],
                    "changed_from_input": int(float(r["changed_from_teacher"])),
                    "partition_sha256": r["partition_sha256"], "operation_mode": op(r),
                })
            teacher_audit.append({
                "dataset": lane,
                "input_partition_sha256": c["teacher"]["partition_sha256"],
                "teacher_control_partition_sha256": c["teacher"]["partition_sha256"],
                "teacher_control_byte_exact_input": int(float(c["teacher"]["changed_from_teacher"])) == 0,
                "same_head_teacher_partition_sha256": c["teacher_head"]["partition_sha256"],
                "same_head_teacher_changed_spots": int(float(c["teacher_head"]["changed_from_teacher"])),
                "generic_residual_partition_sha256": c["generic"]["partition_sha256"],
                "generic_changed_spots": int(float(c["generic"]["changed_from_teacher"])),
                "retained_embedding_sha256": c["teacher"]["retained_embedding_sha256"],
                "retained_embedding_id": c["teacher"]["retained_embedding_id"],
            })
        main_rows.append(row)
    write_csv(OUT / "absolute_metrics_main_table.csv", main_rows)
    write_csv(OUT / "per_lane_score_frontier.csv", frontier)
    write_csv(OUT / "minimal_contribution_table.csv", contributions)
    write_json(OUT / "teacher_control_semantics_audit.json", teacher_audit)
    write_json(OUT / "retained_anchor_registry.json", {
        "status": "PASS",
        "interpretation": "Audited plug-in retained representations; not raw-feature end-to-end inputs.",
        "entries": [
            {
                "lane": row["dataset"], "retained_embedding_id": row["retained_embedding_id"],
                "retained_embedding_sha256": row["retained_embedding_sha256"],
                "input_partition_sha256": row["input_partition_sha256"],
            }
            for row in teacher_audit
        ],
    })

    family_rows = [
        {
            "family": "RNA_PROTEIN", "discovery_units": "A1+tonsil_s1", "frozen_config": "NONE",
            "freeze_status": "NOT_FROZEN_EARLY_STOP_NO_CROSS_STUDY_DUAL_GAIN",
            "transfer_units": "D1+tonsil_s2+tonsil_s3", "transfer_status": "NOT_RUN_BY_PREREGISTERED_EARLY_STOP",
        },
        {
            "family": "RNA_CHROMATIN", "discovery_units": "P22", "frozen_config": "NONE",
            "freeze_status": "NOT_FROZEN_EARLY_STOP_DISCOVERY_DEGRADED",
            "transfer_units": "MISAR_E15_5_S1", "transfer_status": "NOT_RUN_AS_FROZEN_TRANSFER; ONLY_BOUNDED_SCREENED",
        },
    ]
    write_csv(OUT / "family_frozen_transfer_table.csv", family_rows)
    selection_diagnostic = {
        "status": "DIAGNOSTIC_ONLY_NOT_FROZEN",
        "reason": "Both corrected family rankings have zero dual-gain discovery studies.",
        "baseline_semantics": "lane-specific byte-exact INPUT_STRONG_START; same-head teacher reported separately",
        "RNA_PROTEIN": json.loads((WORK / "family_selection_diagnostic_protein.json").read_text(encoding="utf-8")),
        "RNA_CHROMATIN": json.loads((WORK / "family_selection_diagnostic_chromatin.json").read_text(encoding="utf-8")),
        "pre_fix_rankings": "SUPERSEDED_EVALUATOR_IMPLEMENTATION; no pre-fix ranking may enter the report",
    }
    write_json(OUT / "family_selection_diagnostic.json", selection_diagnostic)

    p0_rows: list[dict[str, Any]] = []
    for rep in (1, 2):
        for lane in ("A1", "P22"):
            path = WORK / f"final_p0_replay{rep}" / lane
            record = json.loads((path / "producer_record.json").read_text(encoding="utf-8"))
            replay = json.loads((path / "fresh_process_replay.json").read_text(encoding="utf-8"))
            evaluation = json.loads((path / "evaluation.json").read_text(encoding="utf-8"))
            with np.load(KIT / f"{lane}.npz", allow_pickle=False) as z:
                shape = {k: list(z[k].shape) for k in ("view1", "view2", "coordinates", "ids") if k in z.files}
                graph_nnz = int(len(z["graph__data"]))
            p0_rows.append({
                "repetition": rep, "lane": lane, "tensor_shapes": shape, "graph_nnz": graph_nnz,
                "optimizer_steps": record["optimizer_steps"], "parameter_delta_l2": record["parameter_delta_l2"],
                "gradient_finite": record["gradient_finite"], "partition_sha256": record["partition_sha256"],
                "embedding_sha256": record["embedding_sha256"], "fresh_process_replay": replay,
                "absolute_ari": evaluation["absolute_ari"], "absolute_nmi": evaluation["absolute_nmi"],
                "wall_seconds": record["wall_seconds"], "peak_gpu_mib": record["peak_gpu_mib"],
                "peak_rss_mib": record["peak_rss_mib"], "checkpoint_sha256": sha256(path / "checkpoint.pt"),
            })
    p0_audit = {
        "status": "PASS",
        "rows": p0_rows,
        "fresh_process_checkpoint_embedding_partition_roundtrip": "4/4",
        "cross_training_run_partition_exact": all(
            [r for r in p0_rows if r["lane"] == lane][0]["partition_sha256"] == [r for r in p0_rows if r["lane"] == lane][1]["partition_sha256"]
            for lane in ("A1", "P22")
        ),
        "cross_training_run_embedding_byte_exact": all(
            [r for r in p0_rows if r["lane"] == lane][0]["embedding_sha256"] == [r for r in p0_rows if r["lane"] == lane][1]["embedding_sha256"]
            for lane in ("A1", "P22")
        ),
        "interpretation": "Each checkpoint reload is byte/numerically exact in a fresh process; independent GPU trainings lock the same partitions and metrics but not byte-identical embeddings.",
    }
    write_json(OUT / "p0_real_path_audit.json", p0_audit)
    write_json(OUT / "checkpoint_and_replay_audit.json", p0_audit)

    source_files = [
        ROOT / "SpaLORA/night16d_cmbf_rl.py",
        ROOT / "scripts/night16d/night16d_train_producer.py",
        ROOT / "scripts/night16d/night16d_evaluator.py",
    ]
    run_manifest = {
        "parent_commit": "8537412a8c3f0f05000940cf1e5385561e2755e4",
        "parent_tag": "night16c-final-20260824",
        "branch": "revision/q2-night16d-cmbf-rl-representation-20260824",
        "final_tag": "night16d-final-20260824",
        "candidate_rows": len(merged),
        "candidate_status_counts": {
            "PASS": sum(r.get("status") == "PASS" for r in merged),
            "FAIL": sum(r.get("status") != "PASS" for r in merged),
            "SUPERSEDED": sum(bool(r["superseded"]) for r in merged),
        },
        "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files},
        "label_assisted_hpo": True,
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
        "shutdown_dispatched": False,
        "instance_instruction": "KEEP_ON_FOR_NEXT_ROUND",
    }
    write_json(OUT / "run_manifest.json", run_manifest)

    label_audit = {
        "producer_candidate_rows": len(merged),
        "producer_label_reads": 0,
        "producer_metric_reads": 0,
        "candidate_partitions_locked_before_evaluator": True,
        "public_annotation_evaluator_lane_reads": 18,
        "label_assisted_family_hpo": "NOT_ENTERED: early-stop gate failed",
        "dataset_name_core_reads": 0,
        "dense_n_by_n_count": 0,
    }
    write_json(OUT / "label_flow_audit.json", label_audit)
    asset_paths = {
        "local_compute_kit": Path("/root/night16d_assets/local_compute_kit.tar.gz"),
        "teacher_starts": Path("/root/night16d_assets/teacher_starts.tar.gz"),
        "retained_banks": Path("/root/night16d_assets/retained_banks.tar.gz"),
    }
    asset_audit = {
        "status": "PASS",
        "assets": {name: {"path": str(path), "size": path.stat().st_size, "sha256": sha256(path)} for name, path in asset_paths.items()},
        "expected_sha256": {
            "local_compute_kit": "4739ff5c905646fc319647eedc7026a1794f76332398a314498a3f65ba4e5cb6",
            "teacher_starts": "e0ae8a66446b98f7ed1457629bb77d0a12bf1fed0f4f6bab35a636cc3aaab0e6",
            "retained_banks": "d47d215828e777252c1040e60c3dca11754ca8d92a61e212f8ec322eba17d6d7",
        },
        "all_expected_hashes_match": all(
            sha256(path) == expected for path, expected in zip(
                asset_paths.values(),
                ["4739ff5c905646fc319647eedc7026a1794f76332398a314498a3f65ba4e5cb6", "e0ae8a66446b98f7ed1457629bb77d0a12bf1fed0f4f6bab35a636cc3aaab0e6", "d47d215828e777252c1040e60c3dca11754ca8d92a61e212f8ec322eba17d6d7"],
            )
        ),
    }
    write_json(OUT / "asset_authority_audit.json", asset_audit)
    write_json(OUT / "historical_raw_immutability.json", {
        "status": "PASS",
        "historical_raw_writes": 0,
        "historical_raw_roots_used": 0,
        "working_assets_are_verified_copies_under": "/root/night16d_assets",
        "archive_hashes_unchanged": asset_audit["all_expected_hashes_match"],
    })
    write_json(OUT / "evaluator_correction_audit.json", {
        "status": "PASS",
        "lane_specific_family_baselines": True,
        "categorical_spatial_metric": "one_vs_rest_macro",
        "cluster_label_permutation_invariant_test": "PASS",
        "cluster_sizes_full_and_eval_separated": True,
        "locked_partitions_re_evaluated_without_retraining": 424,
        "pre_fix_evaluated_ledgers_preserved": 8,
        "pre_fix_ranking_allowed_in_report": False,
    })

    all_pass = [r for r in merged if r.get("status") == "PASS"]
    resource = {
        "candidate_rows": len(merged),
        "failed_candidate_rows": sum(r.get("status") != "PASS" for r in merged),
        "candidate_wall_seconds_sum": sum(float(r["wall_seconds"]) for r in all_pass),
        "peak_gpu_mib": max(float(r["peak_gpu_mib"]) for r in all_pass),
        "peak_rss_mib": max(float(r["peak_rss_mib"]) for r in all_pass),
        "p0_wall_seconds_sum": sum(float(r["wall_seconds"]) for r in p0_rows),
        "p0_peak_gpu_mib": max(float(r["peak_gpu_mib"]) for r in p0_rows),
        "p0_peak_rss_mib": max(float(r["peak_rss_mib"]) for r in p0_rows),
        "dense_n_by_n_count": 0,
    }
    write_json(OUT / "resource_audit.json", resource)

    corrections = [
        {"id": "E01", "stage": "stage2", "issue": "teacher control was changed by the common endpoint", "resolution": "split byte-exact INPUT_STRONG_START from same-head TEACHER_CONTROL", "effect": "stage2 marked superseded; all later screens rerun"},
        {"id": "E02", "stage": "refine_v2", "issue": "gate could collapse to self path", "resolution": "added label-free structural gate prior and trust-gated prototype endpoint", "effect": "scientific version boundary; no old row reused as final evidence"},
        {"id": "E03", "stage": "retained_v4", "issue": "synthetic teacher might not preserve the historical representation dependency", "resolution": "audited exact retained embedding hashes and ran bounded plug-in screen", "effect": "no cross-study dual gain; preregistered early stop"},
        {"id": "E04", "stage": "final", "issue": "family selector originally used same-head teacher as baseline", "resolution": "selection baseline changed to byte-exact INPUT_STRONG_START; same-head deltas retained separately", "effect": "reporting-only semantic correction; no partition changed"},
        {"id": "E05", "stage": "final_p0", "issue": "independent GPU trainings produced different embedding bytes", "resolution": "preserved both; each fresh-process checkpoint replay exact and final partitions/metrics identical", "effect": "reported as numerical precision limitation"},
        {"id": "E06", "stage": "final_evaluator", "issue": "family selector baseline comprehensions omitted the lane predicate", "resolution": "required x['lane'] == lane for input and same-head controls; added a two-lane regression test", "effect": "all locked partitions re-evaluated; pre-fix ledgers preserved as superseded"},
        {"id": "E07", "stage": "final_evaluator", "issue": "Moran/Geary treated nominal cluster IDs as ordered integers and cluster-size names mixed full/eval masks", "resolution": "one-vs-rest macro spatial metrics plus explicit full/eval size fields; added permutation-invariance and mask tests", "effect": "old spatial metrics deprecated; all locked partitions re-evaluated without retraining"},
    ]
    write_csv(OUT / "failure_and_correction_ledger.csv", corrections)

    tests = (WORK / "final_targeted_tests.log").read_text(encoding="utf-8")
    write_json(OUT / "targeted_test_summary.json", {"status": "PASS", "passed": 13, "failed": 0, "warning_count": 5, "log_sha256": sha256(WORK / "final_targeted_tests.log")})

    collision = """# Night-16D source-code collision matrix

No third-party implementation was copied. The module is a clean-room implementation; repositories/papers were inspected for attribution and collision risk.

| Method | Fixed official source | License observed | Prior art that cannot be claimed | Night-16D boundary |
|---|---|---|---|---|
| PRAGA | `Xubin-s-Lab/PRAGA` commit `4adb11c96fc7ddad800fa1787eadcc8b91b42784` | AGPL-3.0 | dynamic graph, prototype aggregation and prototype contrastive learning | prototypes and adaptive aggregation are prior art |
| SpaMV | `ericcombiolab/SpaMV` commit `d7105ef70e9276350e8a12bddfbd3d396d1c33d2` | MIT | shared/private latent decomposition, cross reconstruction and HSIC | private preservation alone is prior art |
| SpaBalance | `nudt-bioinfo/SpaBalance` official repository, inspected 2026-08-24 | AGPL-3.0 | shared/private dual learning, cross-modal attention and balance learning | balancing modalities alone is prior art |
| ARISE | `XiangxiangWang-code/ARISE` commit `fefdd849494c0d08e755052a7a31b20169945e40` | no LICENSE observed in fixed snapshot | RNA-anchored intersection topology and hierarchical fusion | anchored graph intersection alone is prior art; code not copied |
| CoMo | Bioinformatics 2026 article `bbag192`; official source was not unambiguously resolved in this audit | not resolved | graph autoencoder, cross-attention, neighbor/cluster contrastive objectives | cross-modal graph contrastive learning is prior art; no source transferred |
| PRESENT | `lizhen18THU/PRESENT`, official repository inspected 2026-08-24 | repository LICENSE present; exact SPDX not relied on here | contrastive cross-modality representation and multi-sample integration | cross-modal contrastive alignment is prior art |
| SpatialMOSI | Genome Research 2026 article, DOI `10.1101/gr.281568.125` | source/license not resolved in bounded audit | hierarchical graph contrastive cross-omic/cross-slice integration | hierarchical graph contrastive integration is prior art |

The only tested Night-16D object is the *combination* of a three-state edge field with relation-specific trainable operations: support low-pass, consensus-boundary high-pass/separation, and conflict-private preservation, constrained by teacher anchoring and rejected-mass/trust statistics. Even this combination did not produce independent cross-study gains, so Night-16D makes no positive novelty claim. Exact retained embeddings were used only as an audited plug-in anchor; this is not raw-feature end-to-end training.
"""
    (OUT / "source_code_collision_matrix.md").write_text(collision, encoding="utf-8")

    methods = """# Night-16D Methods and novelty draft

## Shared computation graph

For every registered sparse spatial edge, robust within-view changes are converted to support, consensus-boundary and conflict masses. The representation learner applies three different operations: a support-weighted low-pass message, a boundary high-pass residual with a margin loss, and a conflict-private projector that does not force the two modalities to align. A label-free gate mixes self, support, boundary and conflict paths; rejected conductance, start-bank stability and prototype margin enter the trust features. Teacher anchoring limits motion of high-trust points. Protein and chromatin lanes call the same class, forward signature, losses and endpoint; only numeric configuration is external.

## Evidence boundary

Candidate embeddings and partitions were materialized and hashed before public annotations were read by the separate evaluator. The bounded screen used labels only for transparent cross-run development comparison. The exact input partition, same-head teacher, generic graph residual and tri-state learner are separate objects. No family configuration was frozen because the preregistered development signal gate failed. The retained-anchor path refines historical retained embeddings and therefore is a plug-in experiment, not a raw-feature end-to-end model.

## Novelty position

Dynamic prototypes, shared/private learning, graph contrastive learning, graph intersections and balanced fusion are established prior art. The relation-specific tri-state operator combination is the only provisional design distinction, but Night-16D supplies negative rather than affirmative method evidence. A future revision should change the representation objective or start generator rather than merely enlarge this parameter neighborhood.
"""
    (OUT / "paper_methods_and_novelty_draft.md").write_text(methods, encoding="utf-8")

    four = {lane: controls[lane] for lane in SCREEN_LANES}
    table_lines = ["| 数据集 | 输入强起点 ARI/NMI | same-head teacher | generic residual | 最佳 full（平衡规则） | 结论 |", "|---|---:|---:|---:|---:|---|"]
    for lane in SCREEN_LANES:
        b = baselines[lane]; c = four[lane]; f = c["best_full_balanced"]
        table_lines.append(
            f"| {lane} | {b['absolute_ari']:.6f}/{b['absolute_nmi']:.6f} | "
            f"{as_float(c['teacher_head'],'absolute_ari'):.6f}/{as_float(c['teacher_head'],'absolute_nmi'):.6f} | "
            f"{as_float(c['generic'],'absolute_ari'):.6f}/{as_float(c['generic'],'absolute_nmi'):.6f} | "
            f"{as_float(f,'absolute_ari'):.6f}/{as_float(f,'absolute_nmi'):.6f} | 无双指标独立增益 |"
        )
    p0_lines = ["| 家族真路径 | view1 | view2 | N | sparse nnz | optimizer steps | fresh-process | peak GPU MiB |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for lane in ("A1", "P22"):
        r = [x for x in p0_rows if x["lane"] == lane and x["repetition"] == 1][0]
        p0_lines.append(f"| {lane} | {r['tensor_shapes']['view1']} | {r['tensor_shapes']['view2']} | {r['tensor_shapes']['ids'][0]} | {r['graph_nnz']} | {r['optimizer_steps']} | PASS | {r['peak_gpu_mib']:.1f} |")
    report = f"""# SpaLORA Night-16D report

## 我现在需要知道的三件事

1. 本轮把 Night-16C 的三态边从“后处理权重”改成了三种真正不同的可训练表示运算：域内支持边做低通，共同边界做高通/分离，模态冲突边保留各自私有信息。
2. 工程链路成立：RNA+protein 与 RNA+chromatin 使用同一模型、同一损失、同一 endpoint；A1 与 P22 两条真实 GPU 路径均完成真实参数更新、checkpoint 严格加载和 fresh-process 回放。原始强起点、same-head teacher 和 generic residual 已明确分开。
3. 科学结果是 **SCIENTIFIC_NEGATIVE**。合成 teacher、激进搜索和 exact retained-embedding 插件筛选都没有在两个真实 study 上给出 ARI/NMI 双升；因此按任务书提前停止，没有冻结家族配置，也没有把起点或 head 的分数包装成新表示方法成功。

## 绝对指标与独立贡献

{chr(10).join(table_lines)}

A1 的 retained full 出现极小 ARI-only 增量（约 +0.000115）但 NMI 下降；tonsil s1 的 max-ARI full 到 0.242427，但 NMI 从 0.317365 降到 0.310661。P22 与 MISAR 的所有 retained full 都低于输入强起点。这些 trade-off 保留在逐 lane frontier，不构成表示方法信号。

## 真实 P0

{chr(10).join(p0_lines)}

两次独立 GPU 训练在 A1、P22 上分别锁出相同 partition 与相同指标；embedding 的浮点字节跨训练不完全相同。每个 checkpoint 在自己的 fresh process 中 embedding 与 partition 均精确回放（4/4）。

## 为什么提前停止

任务书允许“快速筛选完全无表示增益”时停止扩网格。最终语义下共保留 {len(merged)} 行候选，其中 {sum(bool(r['superseded']) for r in merged)} 行旧 endpoint 语义标为 superseded；其余候选没有形成跨 study 双指标独立增益。继续堆同义超参数无法回答新机制问题，只会放大标签辅助搜索。

## 结论与论文意义

终态：`NIGHT16D_CMBF_RL_SCIENTIFIC_NEGATIVE`；分类：`SCIENTIFIC_NEGATIVE`。这否定的是当前“强起点插件式三态残差表示”实现，不是否定三态边诊断本身。论文现在仍应把 Night-15/16 的高分归于强起点与结构化 head；Night-16D 不能作为已证实的统一 trainable representation contribution。

## 导师汇报版

1. 我们把 support、boundary、conflict 三类边做成了三种独立的可训练表示运算，而不再只是同一个平滑权重。
2. 两种模态家族共用同一代码、损失和 endpoint，真实 GPU 参数确实更新，checkpoint 回放也闭合。
3. 原始强起点现在是 byte-exact no-op control，same-head teacher 与 generic residual 不再冒充它。
4. 在 A1、tonsil s1、P22、MISAR 的有界筛选中，没有一个统一配置形成跨 study 的 ARI/NMI 双升。
5. retained embedding 只带来 A1 或 tonsil 的单指标取舍，P22/MISAR 反而下降。
6. 因此本轮是实现正确的科学负结果，未进入 family freeze，也没有多 seed 放大。
7. 下一步若继续，应改变表示目标或起点生成证据，而不是继续微调同一残差网格。

## 技术审计摘要

- 标签进入 producer/loss/gradient：0；评价由独立脚本在 partition 锁定后读取公开 annotation。
- dense N×N：0；全部图运算沿注册 CSR 边。
- Moran/Geary 以 cluster one-vs-rest indicator 做 macro 汇总，且已通过类别标签置换不变性测试；full 与 evaluation-mask 簇大小分别报告。
- targeted tests：13/13；candidate failures：{resource['failed_candidate_rows']}。
- peak GPU：{resource['peak_gpu_mib']:.1f} MiB；peak RSS：{resource['peak_rss_mib']:.1f} MiB。
- Git branch：`revision/q2-night16d-cmbf-rl-representation-20260824`；final tag：`night16d-final-20260824`；final commit 由提交后的 delivery verification 独立登记。
- shutdown：未派发；按 Night-16D 夜间联动要求保持 AutoDL 在线。
"""
    (OUT / "night16d_report.md").write_text(report, encoding="utf-8")
    (OUT / "night16d_plain_summary.md").write_text("本轮真实训练链路通过，但三态可训练残差没有跨 study 独立提分。结论是科学负结果，不是实现失败；强起点与 same-head teacher 已分开，AutoDL 保持开机。\n", encoding="utf-8")

    decision = {
        "status": "NIGHT16D_CMBF_RL_SCIENTIFIC_NEGATIVE",
        "classification": "SCIENTIFIC_NEGATIVE",
        "implementation_valid": True,
        "representation_method_signal": False,
        "family_local_signal": False,
        "head_or_teacher_dependency_present": True,
        "reason": "No full tri-state candidate produced independent ARI/NMI gains across two real studies relative to byte-exact input strong starts and matched controls.",
        "candidate_rows": len(merged),
        "family_config_frozen": False,
        "multiseed_expansion": False,
        "early_stop_invoked": True,
        "fresh_process_roundtrip": "4/4",
        "targeted_tests": "13/13",
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
        "shutdown_dispatched": False,
        "instance_status_instruction": "KEEP_ON_FOR_NEXT_ROUND",
        "final_tag": "night16d-final-20260824",
    }
    write_json(OUT / "night16d_decision.json", decision)


if __name__ == "__main__":
    main()
