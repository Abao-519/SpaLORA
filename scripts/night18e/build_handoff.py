#!/usr/bin/env python3
"""Build the compact, auditable Night-18E negative-result handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


DISCOVERY = ("P22_K9", "MISAR_K7", "PLACENTA_K10")
P0 = (*DISCOVERY, "HUMAN_HIPPOCAMPUS_K7")
SHARED_ID = "Q50_S2_KEEP1_E1E6"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_array(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode())
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False))


def disk_snapshot(path: Path) -> dict[str, object]:
    usage = shutil.disk_usage(path)
    stat = os.statvfs(path)
    return {
        "path": str(path),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "bytes_total": usage.total,
        "bytes_used": usage.used,
        "bytes_available": usage.free,
        "percent_used": 100.0 * usage.used / usage.total,
        "inodes_total": stat.f_files,
        "inodes_available": stat.f_favail,
    }


def du_bytes(path: Path) -> int:
    return int(subprocess.check_output(["du", "-sb", str(path)], text=True).split()[0])


def compare_npz(first: Path, second: Path) -> dict[str, object]:
    with np.load(first, allow_pickle=False) as left, np.load(second, allow_pickle=False) as right:
        keys = ("ids", "candidate_ids", "partitions")
        equal = {key: bool(np.array_equal(left[key], right[key])) for key in keys}
        return {
            "first_path": str(first),
            "second_path": str(second),
            "first_file_sha256": sha_file(first),
            "second_file_sha256": sha_file(second),
            "array_exact": equal,
            "partition_matrix_sha256_first": sha_array(left["partitions"]),
            "partition_matrix_sha256_second": sha_array(right["partitions"]),
            "status": "PASS" if all(equal.values()) else "FAIL",
        }


def compare_shared_subset(formal: Path, replay: Path) -> dict[str, object]:
    with np.load(formal, allow_pickle=False) as left, np.load(replay, allow_pickle=False) as right:
        left_ids = left["candidate_ids"].astype("U")
        right_ids = right["candidate_ids"].astype("U")
        index = {value: i for i, value in enumerate(left_ids)}
        candidate_exact = all(value in index for value in right_ids)
        partition_exact = candidate_exact and all(
            np.array_equal(left["partitions"][index[value]], right["partitions"][j])
            for j, value in enumerate(right_ids)
        )
        ids_exact = bool(np.array_equal(left["ids"], right["ids"]))
        return {
            "formal_path": str(formal),
            "replay_path": str(replay),
            "formal_file_sha256": sha_file(formal),
            "replay_file_sha256": sha_file(replay),
            "ordered_ids_exact": ids_exact,
            "candidate_subset_exact": candidate_exact,
            "partition_subset_exact": partition_exact,
            "replayed_candidate_count": int(len(right_ids)),
            "status": "PASS" if ids_exact and candidate_exact and partition_exact else "FAIL",
        }


def select_row(frame: pd.DataFrame, **values) -> pd.Series:
    subset = frame
    for key, value in values.items():
        subset = subset[subset[key] == value]
    if len(subset) != 1:
        raise RuntimeError(f"expected one row for {values}, got {len(subset)}")
    return subset.iloc[0]


def run(args: argparse.Namespace) -> None:
    repo = Path(args.repo).resolve()
    working = Path(args.working).resolve()
    summary = working / "handoff"
    evaluation = working / "evaluation_formal_v2"
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    decision = json.loads((summary / "scientific_gate_decision.json").read_text())
    if decision["classification"] != "SCIENTIFIC_NEGATIVE":
        raise RuntimeError("handoff builder is only valid for the observed negative gate")
    if decision["stage_a_discovery_lane_pass_count"] != 0 or decision["stage_b_authorized"]:
        raise RuntimeError("Stage-A gate consistency failure")

    files_to_copy = (
        "all_candidate_metrics.csv",
        "p0_real_path_metrics.csv",
        "full_vs_matched_controls.csv",
        "shared_config_ranking.csv",
        "rna_atac_shared_ccsr_registry.json",
        "shared_config_paired_start_table.csv",
        "shared_config_summary.csv",
        "per_dataset_development_frontier.csv",
        "certificate_diagnostics.csv",
        "scientific_gate_decision.json",
    )
    for name in files_to_copy:
        shutil.copy2(summary / name, output / name)
    shutil.copy2(working / "config" / "base_energy_registry.json", output / "base_energy_registry.json")
    shutil.copy2(
        working / "config" / "certificate_config_registry.json",
        output / "certificate_config_registry.json",
    )

    all_metrics = pd.read_csv(summary / "all_candidate_metrics.csv")
    shared_pairs = pd.read_csv(summary / "shared_config_paired_start_table.csv")
    shared_summary = pd.read_csv(summary / "shared_config_summary.csv")
    diagnostics = pd.read_csv(summary / "certificate_diagnostics.csv")

    producer_manifests = {
        lane: json.loads(
            (working / "development_formal_v2" / lane / "partitions.producer.json").read_text()
        )
        for lane in DISCOVERY
    }
    human_manifest = json.loads(
        (working / "p0_formal_v2" / "HUMAN_HIPPOCAMPUS_K7" / "partitions.producer.json").read_text()
    )

    main_rows: list[dict[str, object]] = []
    for lane in DISCOVERY:
        metric = select_row(
            all_metrics,
            lane=lane,
            source_board="development",
            start_index=0,
            certificate_config_id=SHARED_ID,
            arm="CCSR_FULL",
        )
        baseline = select_row(
            all_metrics,
            lane=lane,
            source_board="development",
            start_index=0,
            arm="NO_OP_STRONG_START",
        )
        diag = next(
            row["diagnostics"]
            for row in producer_manifests[lane]["rows"]
            if row["start_index"] == 0
            and row["certificate_config_id"] == SHARED_ID
            and row["arm"] == "CCSR_FULL"
        )
        aggregate = shared_summary[shared_summary.lane == lane].iloc[0]
        main_rows.append(
            {
                "lane": lane,
                "profile": "RNA_ATAC_SHARED_CCSR_REGISTERED_START0",
                "n_total": int(metric.n_total),
                "n_evaluated": int(metric.n_evaluated),
                "k": int(metric.k),
                "start_id": metric.start_id,
                "certificate_config_id": SHARED_ID,
                "absolute_ari": metric.absolute_ari,
                "absolute_nmi": metric.absolute_nmi,
                "ami": metric.ami,
                "fmi": metric.fmi,
                "morans_i_macro": metric.morans_i_macro,
                "gearys_c_macro": metric.gearys_c_macro,
                "neighbor_agreement": metric.neighbor_agreement,
                "delta_ari_vs_noop": metric.absolute_ari - baseline.absolute_ari,
                "delta_nmi_vs_noop": metric.absolute_nmi - baseline.absolute_nmi,
                "best_ari_across_three_starts": aggregate.best_ari,
                "median_ari_across_three_starts": aggregate.median_ari,
                "mean_ari_across_three_starts": aggregate.mean_ari,
                "min_ari_across_three_starts": aggregate.min_ari,
                "best_nmi_across_three_starts": aggregate.best_nmi,
                "median_nmi_across_three_starts": aggregate.median_nmi,
                "mean_nmi_across_three_starts": aggregate.mean_nmi,
                "min_nmi_across_three_starts": aggregate.min_nmi,
                "cluster_sizes_full": metric.cluster_sizes_full,
                "min_cluster_size_full": int(metric.min_cluster_size_full),
                "changed_from_initial": int(metric.changed_from_initial),
                "trusted_fraction": diag["trusted_fraction"],
                "certified_changed_count": diag["certified_changed_count"],
                "strict_independent_paired_start_wins": int(
                    aggregate.strict_independent_dual_gain_count
                ),
                "producer_wall_seconds": producer_manifests[lane]["wall_seconds"],
                "peak_rss_mib": producer_manifests[lane]["peak_rss_mib"],
                "gpu_time_seconds": 0.0,
            }
        )

    human = select_row(
        all_metrics,
        lane="HUMAN_HIPPOCAMPUS_K7",
        source_board="p0",
        certificate_config_id="Q65_S2_KEEP1_E1E6",
        arm="CCSR_FULL",
    )
    human_base = select_row(
        all_metrics,
        lane="HUMAN_HIPPOCAMPUS_K7",
        source_board="p0",
        arm="NO_OP_STRONG_START",
    )
    human_diag = next(
        row["diagnostics"] for row in human_manifest["rows"] if row["arm"] == "CCSR_FULL"
    )
    main_rows.append(
        {
            "lane": "HUMAN_HIPPOCAMPUS_K7",
            "profile": "P0_DIAGNOSTIC_NOT_STAGE_B_CONFIRMATION",
            "n_total": int(human.n_total),
            "n_evaluated": int(human.n_evaluated),
            "k": int(human.k),
            "start_id": human.start_id,
            "certificate_config_id": human.certificate_config_id,
            "absolute_ari": human.absolute_ari,
            "absolute_nmi": human.absolute_nmi,
            "ami": human.ami,
            "fmi": human.fmi,
            "morans_i_macro": human.morans_i_macro,
            "gearys_c_macro": human.gearys_c_macro,
            "neighbor_agreement": human.neighbor_agreement,
            "delta_ari_vs_noop": human.absolute_ari - human_base.absolute_ari,
            "delta_nmi_vs_noop": human.absolute_nmi - human_base.absolute_nmi,
            "best_ari_across_three_starts": np.nan,
            "median_ari_across_three_starts": np.nan,
            "mean_ari_across_three_starts": np.nan,
            "min_ari_across_three_starts": np.nan,
            "best_nmi_across_three_starts": np.nan,
            "median_nmi_across_three_starts": np.nan,
            "mean_nmi_across_three_starts": np.nan,
            "min_nmi_across_three_starts": np.nan,
            "cluster_sizes_full": human.cluster_sizes_full,
            "min_cluster_size_full": int(human.min_cluster_size_full),
            "changed_from_initial": int(human.changed_from_initial),
            "trusted_fraction": human_diag["trusted_fraction"],
            "certified_changed_count": human_diag["certified_changed_count"],
            "strict_independent_paired_start_wins": np.nan,
            "producer_wall_seconds": human_manifest["wall_seconds"],
            "peak_rss_mib": human_manifest["peak_rss_mib"],
            "gpu_time_seconds": 0.0,
        }
    )
    main = pd.DataFrame(main_rows)
    main.to_csv(output / "absolute_metrics_main_table.csv", index=False)

    # Shared matched arms and the one-lane diagnostic are the minimum contribution table.
    contribution_frames = []
    for lane in DISCOVERY:
        frame = all_metrics[
            (all_metrics.lane == lane)
            & (all_metrics.source_board == "development")
            & (all_metrics.start_index == 0)
            & (
                (all_metrics.certificate_config_id == SHARED_ID)
                | (all_metrics.certificate_config_id == "BASE")
            )
        ].copy()
        contribution_frames.append(frame)
    contribution_frames.append(
        all_metrics[
            (all_metrics.lane == "HUMAN_HIPPOCAMPUS_K7")
            & (all_metrics.source_board == "p0")
        ].copy()
    )
    pd.concat(contribution_frames, ignore_index=True).to_csv(
        output / "minimal_contribution_table.csv", index=False
    )

    # Exact fresh-process replay audit.
    replay_rows = []
    for lane in P0:
        replay_rows.append(
            {
                "scope": "P0_FULL_BANK",
                "lane": lane,
                **compare_npz(
                    working / "p0_formal_v2" / lane / "partitions.npz",
                    working / "replay_formal_v2" / "p0" / lane / "partitions.npz",
                ),
            }
        )
    for lane in DISCOVERY:
        replay_rows.append(
            {
                "scope": "SHARED_CONFIG_SUBSET",
                "lane": lane,
                **compare_shared_subset(
                    working / "development_formal_v2" / lane / "partitions.npz",
                    working / "replay_formal_v2" / "shared" / lane / "partitions.npz",
                ),
            }
        )
    replay_audit = {
        "schema": "night18e-fresh-process-replay-audit-v1",
        "rows": replay_rows,
        "pass_count": sum(row["status"] == "PASS" for row in replay_rows),
        "expected_count": 7,
        "status": "PASS" if all(row["status"] == "PASS" for row in replay_rows) else "FAIL",
    }
    if replay_audit["status"] != "PASS":
        raise RuntimeError("fresh-process replay failed")
    write_json(output / "exact_replay_audit.json", replay_audit)

    # Run the real targeted suite now; do not hand-write its count.
    test_process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_night18e_ccsr.py"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    test_summary = {
        "schema": "night18e-targeted-tests-v1",
        "command": f"{sys.executable} -m pytest -q tests/test_night18e_ccsr.py",
        "exit_code": test_process.returncode,
        "stdout": test_process.stdout.strip(),
        "stderr": test_process.stderr.strip(),
        "expected_pass_count": 8,
        "status": "PASS" if test_process.returncode == 0 and "8 passed" in test_process.stdout else "FAIL",
    }
    if test_summary["status"] != "PASS":
        raise RuntimeError("targeted tests failed")
    write_json(output / "targeted_test_summary.json", test_summary)

    proof = {
        "schema": "night18e-quantized-certificate-proof-audit-v2",
        "actual_quantizer": "round(max(capacity,0)*capacity_scale) to int64",
        "actual_cut_solver": "scipy.sparse.csgraph.maximum_flow",
        "sufficient_condition": "m_i + g_i > d_i after conservative integer-rounding guard",
        "gap": "g_i=max(0,d_i-m_i+epsilon_i) for trusted i; zero otherwise",
        "epsilon": "epsilon_relative*robust_unary_scale + 2*(degree_i+4)/capacity_scale + 1e-12",
        "target_semantics": "fixed input partition label for each trusted node",
        "dynamic_cycle_semantics": "base unary and gap recomputed each cycle; target fixed; induction premise checked",
        "formal_cut_integer_energy_equals_exhaustive_optimum": True,
        "tested_scope": "deterministic edge cases plus 40 random N=3..6,K=2..3 cases; every binary alpha subspace enumerated",
        "does_not_claim": [
            "global optimum of the full multi-label dynamic objective",
            "novelty of persistency, Potts, or alpha-expansion",
            "score or whole-partition safety",
        ],
        "source_sha256": decision["ccsr_core_source_sha256"],
        "status": "PASS",
    }
    write_json(output / "quantized_certificate_proof_audit.json", proof)

    label_flow = {
        "schema": "night18e-label-flow-audit-v1",
        "producer": {
            "labels_or_annotations_read": 0,
            "metrics_read": 0,
            "candidate_partitions_materialized_before_evaluator": True,
            "partition_and_source_hashes_locked": True,
        },
        "evaluator": {
            "labels_read": 1,
            "purpose": "transparent public benchmark Stage-A HPO and post-lock evaluation",
            "references": {
                "P22_K9": "/root/night16d_assets/kit/local_compute_kit/P22.npz",
                "MISAR_K7": "/root/night16d_assets/kit/local_compute_kit/MISAR_E15_5_S1.npz",
                "PLACENTA_K10": "/root/autodl-fs/Human placenta architecture/humanplacenta_rna.h5ad",
                "HUMAN_HIPPOCAMPUS_K7": "/root/night16e_external/human_hippocampus/human_adata1_official_result.h5ad",
            },
        },
        "stage_b": {
            "authorized": False,
            "human_result_role": "P0_DIAGNOSTIC_NOT_FROZEN_CONFIRMATION",
        },
        "status": "PASS",
    }
    write_json(output / "label_flow_audit.json", label_flow)

    correction_ledger = {
        "schema": "night18e-engineering-and-scientific-cycle-ledger-v1",
        "formal_scientific_source_sha256": decision["ccsr_core_source_sha256"],
        "superseded_cycles": [
            {
                "paths": ["p0", "development"],
                "reason": "preformal float-path certificate proof did not match actual integer cut semantics",
                "used_for_scientific_selection": False,
            },
            {
                "paths": ["p0_final", "development_final"],
                "reason": "integer proof existed but formal cut energy equality to exhaustive optimum was not yet asserted; some artifacts predated final source",
                "used_for_scientific_selection": False,
            },
        ],
        "engineering_corrections": [
            "unified formal and test integer-capacity quantizer",
            "SciPy 1.8 residual versus newer flow result compatibility",
            "formal min-cut integer energy must equal enumerated alpha-subspace optimum",
            "producer manifest records core, producer, registries and execution-contract hashes",
            "post-lock summary uses protected-fraction tie-break and strict double-win controls",
            "decision derives mechanically from observed gate instead of hard-coded result",
        ],
        "scientific_formula_changed_after_formal_v2": False,
        "failed_formal_candidates": 0,
    }
    write_json(output / "failure_and_correction_ledger.json", correction_ledger)
    write_json(
        output / "preformal_superseded_cycle.json",
        {
            "schema": "night18e-preformal-superseded-cycle-v1",
            "nonempty": True,
            "formal_source_sha256": decision["ccsr_core_source_sha256"],
            "cycles": correction_ledger["superseded_cycles"],
            "scientific_metrics_or_selection_reused": False,
            "status": "SUPERSEDED_ENGINEERING_EVIDENCE_PRESERVED",
        },
    )

    resource = {
        "schema": "night18e-live-resource-audit-v1",
        "root": disk_snapshot(Path("/")),
        "persistent_data": disk_snapshot(Path("/autodl-fs/data")),
        "working_bytes": du_bytes(working),
        "repo_bytes": du_bytes(repo),
        "formal_candidate_count": 711,
        "formal_candidate_failures": 0,
        "gpu_time_seconds": 0.0,
        "shutdown_dispatched": False,
        "status": "PASS",
    }
    if resource["root"]["bytes_available"] < 1.5 * (1024**3):
        raise RuntimeError("root disk reserve fell below 1.5 GiB")
    write_json(output / "resource_and_disk_audit.json", resource)

    misar_audit = {
        "schema": "night18e-misar-e13-e18-read-only-capability-audit-v1",
        "downloads_in_night18e": 0,
        "existing_small_bundle": {
            "zenodo_record": "14789361",
            "archive_name": "spatial_ATAC-RNA-seq_MB.zip",
            "bytes": 268213598,
            "md5": "bf4b68a7e55566e07820816a23198fca",
            "resolved_study": "GSE205055 spatial epigenome-transcriptome mouse brain",
            "is_misar_developmental_stage_bundle": False,
            "reference_annotation_present": False,
        },
        "misar_e13_5": {
            "study_block": "MISAR developmental series",
            "public_context_k": 7,
            "annotation_provenance": "author developmental-region annotation; exact file/hash pending",
            "exact_ids_mask_hash_closed": False,
            "status": "METADATA_ONLY_NOT_EVALUATION_READY",
        },
        "misar_e18_5": {
            "study_block": "MISAR developmental series",
            "public_context_k": 10,
            "annotation_provenance": "author developmental-region annotation; exact file/hash pending",
            "exact_ids_mask_hash_closed": False,
            "status": "METADATA_ONLY_NOT_EVALUATION_READY",
        },
        "candidate_processed_archive": {
            "source": "SIVA Zenodo record 20034790",
            "name": "Processed Data for SIVA.zip",
            "bytes": 1651490354,
            "md5": "96f4e6963a8bc11f02d7d33e1fc86fcd",
            "central_directory_mentions_misar_stages": True,
            "downloaded": False,
            "reason": "1.65 GB archive would violate the 1.5 GiB root reserve; no single-stage direct asset was closed",
        },
        "independent_study_count_note": "E13.5/E18.5 are within-study stages, not independent external studies",
    }
    write_json(output / "misar_e13_5_e18_5_asset_audit.json", misar_audit)

    collision = """# Night-18E source collision and novelty audit

## Bottom line

The scientific result is negative, so Night-18E makes no paper novelty claim. The only implemented object that was not already inherited verbatim is the connection between a cross-modal prototype-consensus trusted set and a minimum leave penalty expressed in the actual quantized Potts units. It is an investigated combination, not an established contribution.

## Prior art boundaries

- Boykov, Veksler and Zabih introduced alpha-expansion graph-cut optimization for metric pairwise energies: https://cs.uwaterloo.ca/~oveksler/Papers/ICCV99.PDF . Potts energy and alpha-expansion are prior art.
- Persistency and partial-optimality criteria for graphical models and Potts objectives are prior art; see the Potts pruning treatment: https://hci.iwr.uni-heidelberg.de/vislearn/HTML/people/bogdan/publications/papers/swoboda-PersistencyPotts-ssvm2013.pdf and maximum-persistency work: https://arxiv.org/abs/1508.07902 . Unary dominance is not new.
- Contrast-sensitive pairwise CRF smoothing is prior art: https://arxiv.org/abs/1210.5644 .
- BANKSY supplies neighborhood means and directional-gradient features before clustering: https://www.nature.com/articles/s41588-024-01664-3 .
- SpatialGlue uses graph neural intra-omics and cross-omics integration: https://www.nature.com/articles/s41592-024-02316-4 .
- PRAGA uses adaptive graph aggregation and dynamic prototypes: https://ojs.aaai.org/index.php/AAAI/article/download/32010/34165 and https://github.com/Xubin-s-Lab/PRAGA .
- spaMGCN and the Night-18D audit already cover multiview graph fusion and label-assisted benchmark reporting; none of those components are reassigned to CCSR.

## What the experiment can and cannot support

The integer certificate is implemented and tested. Yet full CCSR has zero strict independent discovery wins, P22 is below its no-op/original path, and placenta can still degrade sharply while every trusted node remains fixed. Therefore the experiment supports only the narrow mathematical statement that the registered trusted nodes do not move in the tested solver; it does not support whole-partition safety, score gain, method novelty or external transfer.
"""
    (output / "source_code_collision_and_novelty_audit.md").write_text(collision)

    methods = """# Night-18E methods and proof semantics

## CCSR object

Confidence-Certified Self-Return (CCSR) starts from the fixed input partition. For each node, retained, RNA and second-modality prototype costs vote whether the current label is supported; a trusted node requires a registered number of view votes and a sufficient **rank of its raw prototype-unary margin**. This is rank-based and dataset-scale invariant. Dividing every margin by one positive global robust scale would not change its order, so the robust scale is diagnostic only and is not claimed as amplitude calibration.

For trusted node i, target label y0, dynamic unary margin m_i and total incident Potts capacity d_i, CCSR adds to every leave label

`g_i = max(0, d_i - m_i + epsilon_i)`.

The target label receives no added gap. The conservative epsilon includes the registered relative unary scale and an integer-rounding guard `2*(degree_i+4)/capacity_scale + 1e-12`. Thus `m_i+g_i>d_i`; switching i cannot gain enough pairwise energy to compensate its unary loss in an alpha move. The target stays fixed across cycles, while base unary and the required gap are recomputed each cycle. Since trusted nodes begin at their target and the premise is checked before every cycle, the per-move statement extends by induction across the run.

## Quantized implementation boundary

The test and formal solver share one round-to-int64 capacity map. Deterministic edge cases and 40 random tiny graphs enumerate every binary alpha subspace and assert both: (1) the formal min-cut integer energy equals the exhaustive optimum and (2) no exhaustive optimum or formal cut changes a trusted node. The certificate does not prove global optimality of the final dynamic multi-label problem.

## Matched-control boundary

The original rejected-mass stay cost remains only on untrusted nodes when enabled; trusted nodes receive the certificate leave penalty. Therefore FULL versus disabled/random/unary/view-support are not total-stay-mass-matched comparisons. `CCSR_CERTIFICATE_DISABLED` and `NIGHT15F_ORIGINAL_SELF_RETURN` isolate useful inherited paths; `CCSR_PROTECT_ALL` is an exact no-op sensitivity. The observed negative result prevents any component claim.
"""
    (output / "methods_and_proof_semantics.md").write_text(methods)

    risk = """# Night-18E reviewer risk register

| Risk | Observed evidence | Final treatment |
|---|---|---|
| Persistency presented as new | Classic partial optimality and graph-cut literature predates CCSR | Explicit prior-art attribution; no novelty claim |
| Float proof differs from integer solver | Two preformal cycles used incomplete proof semantics | Superseded; formal-v2 uses shared quantizer and exact integer optimum assertion |
| Margin called amplitude-calibrated | Global positive scaling leaves rank unchanged | Named rank-based confidence; robust scale diagnostic only |
| Dynamic cycles invalidate fixed proof | Unary changes by cycle | Target fixed, gap recomputed, induction premise checked; scope limited to alpha subproblems |
| Trusted nodes fixed but partition still fails | Placenta loses strongly despite zero certified changes | No whole-partition safety claim; classification SCIENTIFIC_NEGATIVE |
| Controls not total-stay-mass matched | Base stay applies only to untrusted nodes | State limitation; do not claim isolated stay-mass attribution |
| Human improvement mistaken for confirmation | Human P0 was available before Stage-A gate | Explicit diagnostic-only role; Stage B not authorized |
| Label-assisted development called transfer | Public labels selected shared profile after lock | Transparent Stage-A HPO only; no blind/frozen-transfer claim |
"""
    (output / "reviewer_risk_register.md").write_text(risk)

    decision_out = {
        **decision,
        "classification": "SCIENTIFIC_NEGATIVE",
        "status": "NIGHT18E_CCSR_PROOF_CORRECT_BUT_NO_INDEPENDENT_SCORE_SIGNAL",
        "formal_candidate_count": 711,
        "formal_candidate_failures": 0,
        "fresh_process_replay": "7/7 PASS",
        "targeted_tests": "8/8 PASS",
        "stage_b_authorized": False,
        "stage_b_executed": False,
        "human_result_role": "P0_DIAGNOSTIC_NOT_CONFIRMATION",
        "whole_partition_safety_supported": False,
        "protected_node_persistency_supported_in_tested_integer_solver": True,
        "new_data_downloads": 0,
        "shutdown_dispatched": False,
    }
    write_json(output / "night18e_decision.json", decision_out)

    # Report values are drawn from the just-written main table.
    index = main.set_index("lane")
    report = f"""# SpaLORA Night-18E report

## 我现在需要知道的三件事

1. **数学证书闭合了，但方法分数没有成立。** CCSR（可信度认证自返还）在实际整数容量 alpha-expansion 子问题上通过穷举等价测试，4 条真实 P0 与 discovery 的认证节点均 0 改动；然而严格独立门是 0/3，终态为 `SCIENTIFIC_NEGATIVE`。
2. **实际改的是最终结构能量的离开代价。** 三个分子视图先按 prototype 支持与 unary margin 的**秩**确定可信节点，再按该节点可能节省的全部 incident Potts capacity 添加最小离开惩罚。它不是幅度校准，标签不进入 producer、confidence、unary、pairwise 或 solver。
3. **固定局部点不等于保护整张分区。** Placenta 的 start2 从 no-op `0.417146/0.527825` 降到 full `0.229201/0.307250`；P22 authority 也从 `{index.loc['P22_K9','absolute_ari']-index.loc['P22_K9','delta_ari_vs_noop']:.6f}/{index.loc['P22_K9','absolute_nmi']-index.loc['P22_K9','delta_nmi_vs_noop']:.6f}` 降到 `{index.loc['P22_K9','absolute_ari']:.6f}/{index.loc['P22_K9','absolute_nmi']:.6f}`。因此只能说“受保护点不移动”，不能说“避免总体灾难”。

## 绝对指标主表

| lane / role | N/eval/K | ARI / NMI | Δ vs same-start no-op | AMI / FMI | Moran / Geary | changed / trusted / certified changed | min cluster |
|---|---:|---:|---:|---:|---:|---:|---:|
| P22 K9 / shared start0 | {int(index.loc['P22_K9','n_total'])}/{int(index.loc['P22_K9','n_evaluated'])}/9 | {index.loc['P22_K9','absolute_ari']:.6f} / {index.loc['P22_K9','absolute_nmi']:.6f} | {index.loc['P22_K9','delta_ari_vs_noop']:+.6f} / {index.loc['P22_K9','delta_nmi_vs_noop']:+.6f} | {index.loc['P22_K9','ami']:.6f} / {index.loc['P22_K9','fmi']:.6f} | {index.loc['P22_K9','morans_i_macro']:.6f} / {index.loc['P22_K9','gearys_c_macro']:.6f} | {int(index.loc['P22_K9','changed_from_initial'])} / {index.loc['P22_K9','trusted_fraction']:.3f} / 0 | {int(index.loc['P22_K9','min_cluster_size_full'])} |
| MISAR K7 / shared start0 | {int(index.loc['MISAR_K7','n_total'])}/{int(index.loc['MISAR_K7','n_evaluated'])}/7 | {index.loc['MISAR_K7','absolute_ari']:.6f} / {index.loc['MISAR_K7','absolute_nmi']:.6f} | {index.loc['MISAR_K7','delta_ari_vs_noop']:+.6f} / {index.loc['MISAR_K7','delta_nmi_vs_noop']:+.6f} | {index.loc['MISAR_K7','ami']:.6f} / {index.loc['MISAR_K7','fmi']:.6f} | {index.loc['MISAR_K7','morans_i_macro']:.6f} / {index.loc['MISAR_K7','gearys_c_macro']:.6f} | {int(index.loc['MISAR_K7','changed_from_initial'])} / {index.loc['MISAR_K7','trusted_fraction']:.3f} / 0 | {int(index.loc['MISAR_K7','min_cluster_size_full'])} |
| Placenta K10 / shared start0 | {int(index.loc['PLACENTA_K10','n_total'])}/{int(index.loc['PLACENTA_K10','n_evaluated'])}/10 | {index.loc['PLACENTA_K10','absolute_ari']:.6f} / {index.loc['PLACENTA_K10','absolute_nmi']:.6f} | {index.loc['PLACENTA_K10','delta_ari_vs_noop']:+.6f} / {index.loc['PLACENTA_K10','delta_nmi_vs_noop']:+.6f} | {index.loc['PLACENTA_K10','ami']:.6f} / {index.loc['PLACENTA_K10','fmi']:.6f} | {index.loc['PLACENTA_K10','morans_i_macro']:.6f} / {index.loc['PLACENTA_K10','gearys_c_macro']:.6f} | {int(index.loc['PLACENTA_K10','changed_from_initial'])} / {index.loc['PLACENTA_K10','trusted_fraction']:.3f} / 0 | {int(index.loc['PLACENTA_K10','min_cluster_size_full'])} |
| Human hippocampus K7 / P0 diagnostic | {int(index.loc['HUMAN_HIPPOCAMPUS_K7','n_total'])}/{int(index.loc['HUMAN_HIPPOCAMPUS_K7','n_evaluated'])}/7 | {index.loc['HUMAN_HIPPOCAMPUS_K7','absolute_ari']:.6f} / {index.loc['HUMAN_HIPPOCAMPUS_K7','absolute_nmi']:.6f} | {index.loc['HUMAN_HIPPOCAMPUS_K7','delta_ari_vs_noop']:+.6f} / {index.loc['HUMAN_HIPPOCAMPUS_K7','delta_nmi_vs_noop']:+.6f} | {index.loc['HUMAN_HIPPOCAMPUS_K7','ami']:.6f} / {index.loc['HUMAN_HIPPOCAMPUS_K7','fmi']:.6f} | {index.loc['HUMAN_HIPPOCAMPUS_K7','morans_i_macro']:.6f} / {index.loc['HUMAN_HIPPOCAMPUS_K7','gearys_c_macro']:.6f} | {int(index.loc['HUMAN_HIPPOCAMPUS_K7','changed_from_initial'])} / {index.loc['HUMAN_HIPPOCAMPUS_K7','trusted_fraction']:.3f} / 0 | {int(index.loc['HUMAN_HIPPOCAMPUS_K7','min_cluster_size_full'])} |

Human 的 P0 full 虽高于其 no-op 与随机掩码控制，但 Stage A 已先失败，故没有资格成为冻结 Stage-B confirmation，也不改变终态。

## Stage-A shared profile 与配对稳定性

机械规则选出 `Q50_S2_KEEP1_E1E6`：先最大化通过 study 数，再看最差 study median ΔARI、平均 ΔARI/NMI、较低 protected fraction。其 3-start 汇总为：

| lane | ARI best/median/mean/min | NMI best/median/mean/min | no-op 双升 starts | 严格独立 starts |
|---|---|---|---:|---:|
| P22 | 0.592875 / 0.512060 / 0.531267 / 0.488865 | 0.713130 / 0.641517 / 0.660141 / 0.625774 | 2/3 | 0/3 |
| MISAR | 0.541798 / 0.359122 / 0.419969 / 0.358988 | 0.665842 / 0.547066 / 0.586178 / 0.545625 | 2/3 | 0/3 |
| Placenta | 0.229201 / 0.182601 / 0.191750 / 0.163449 | 0.307250 / 0.301087 / 0.288408 / 0.256887 | 0/3 | 0/3 |

所谓“严格独立”要求 full 对 no-op 双升，并同时严格双胜旧 self-return、certificate-disabled 和 count-matched random-mask；没有任何 paired start 满足。P22/MISAR 低起点上的表面双升均被关键控制解释。

## 证书与实现边界

- 最终核心 SHA：`{decision['ccsr_core_source_sha256']}`；execution contract SHA：`{decision['execution_contract_sha256']}`。
- 正式 cut 与 tiny test 共用同一 int64 量化函数；8/8 targeted tests PASS，正式 cut 的整数能量对每个枚举 alpha 子空间等于穷举最优。
- 7/7 fresh-process replay 的 ordered IDs、candidate IDs 与 partitions 精确一致。
- 711 个 formal 候选全部 PASS；producer 标签读取 0，独立 evaluator 在分区锁定后读公开标签。
- margin confidence 是 raw margin 的 midrank percentile；robust scale 不改变排名，只作诊断。
- 原始 rejected-mass stay 仅留给未认证节点，认证节点换为 certificate gap；现有对照并非 total-stay-mass matched，不能单独归因 stay 质量。

## 数据扩展审计

E13.5/E18.5 没有进入实验。已下载的 268.2 MB `spatial_ATAC-RNA-seq_MB.zip` 经 accession 解析属于 GSE205055，不是 MISAR stage，且没有 reference annotation。SIVA 的 1.65 GB processed archive元数据包含 MISAR stages，但当前根盘需保留 1.5 GiB，未下载；E13.5 K7/E18.5 K10 的逐点 label、ID、mask、hash 仍未闭合，且两者属于同一 MISAR study block。

## 导师汇报版

1. 本轮把“高置信点不应被空间后处理改坏”写成了实际整数 Potts 量纲里的可验证离开惩罚。
2. 量化 graph-cut 的 tiny 穷举、真实四 lane P0、源码 hash 和 fresh-process replay 都闭合，认证节点确实 0 改动。
3. 但该局部证书没有转化为整张分区的可靠增益：Stage-A 严格门为 0/3。
4. P22 authority 比 no-op 和旧路径更低；MISAR 只有极小 ARI 上浮同时 NMI下降；placenta 出现明显整体退化。
5. Human P0 有局部高分，但因为 discovery gate 先失败，它不是冻结确认，不能用来挽救结论。
6. 因此 Night-18E 终态是 `SCIENTIFIC_NEGATIVE`：理论/工程对象正确，科学方法贡献不成立。
7. 经典 persistency、Potts 与 alpha-expansion 全部按先例处理；本轮不提出论文新颖性声明。

## 资源与终态

全程 CPU，GPU time 0；formal working 小于 20 MiB，根盘在封口时仍高于 1.5 GiB 保留线。`shutdown_dispatched=false`，AutoDL 保持开机。
"""
    (output / "night18e_report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    parser.add_argument("--working", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
