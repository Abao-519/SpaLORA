#!/usr/bin/env python3
"""Fail-closed Night-8A delivery after the DEV_WINDOW_1 semantic mismatch."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/night8a_handoff"
REGISTRY = REPO / "protocols/night8a/SpaLORA_Night8A_MFSPC_Registry_2026-08-20.json"
sys.path.insert(0, str(REPO))
from scripts import night8a_finalize as common  # noqa: E402
from SpaLORA.night8a_mfspc import file_sha  # noqa: E402


STATUS = "IMPLEMENTATION_SEMANTICS_INVALID"
REASON = "RNA_PROTEIN B00 rebuilt an approximate affinity instead of reusing the SHA-locked C00 G04/H05 affinity and partition"


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def required_empty_tables() -> None:
    pd.DataFrame(columns=["stage", "config_id", "dataset", "unit_id", "seed", "scientific_valid", "reason"]).to_csv(
        OUT / "per_seed_metrics.csv", index=False)
    pd.DataFrame(columns=["stage", "config_id", "dataset", "unit_id", "seed", "scientific_valid", "reason"]).to_csv(
        OUT / "paired_deltas.csv", index=False)
    pd.DataFrame(columns=["config_id", "scientific_valid", "reason"]).to_csv(OUT / "pareto_frontier.csv", index=False)
    pd.DataFrame(columns=["config_id", "dataset", "scientific_valid", "reason"]).to_csv(
        OUT / "module_ablation_summary.csv", index=False)


def main() -> None:
    invalid = json.loads((OUT / "dev_window1_semantic_invalidation.json").read_text(encoding="utf-8"))
    if invalid["status"] != STATUS or not invalid["labels_opened_before_detection"]:
        raise RuntimeError("semantic invalidation authority mismatch")
    if (OUT / "r3_stage_manifest.json").exists():
        raise RuntimeError("R3 unexpectedly started after semantic invalidation")
    common.STAGES = ("R1", "R2")
    cells, checkpoints, runtime = common.verify_and_collect_stages()
    pd.DataFrame(cells).to_csv(OUT / "training_cells.csv", index=False)
    pd.DataFrame(runtime).to_csv(OUT / "runtime_and_gpu_timeline.csv", index=False)
    aliases = sum(x["status"] == "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL" for x in cells)
    transform_failures = [x for x in cells if x["transform_status"] not in
                          {"SUCCESS_PRE_LABEL", "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL"}]
    if len(transform_failures) != 1 or transform_failures[0]["config_id"] != "B03_RR10_PROTO":
        raise RuntimeError("unexpected transform failure inventory")
    atomic_json(OUT / "training_manifest.json", {
        "status": "LOCKED_FAIL_CLOSED", "registered_cells": len(cells),
        "actual_cuda_training_attempts": len(checkpoints), "content_addressed_aliases": aliases,
        "scientific_retry_count": 0, "fallback_count": 0, "R3_training_attempts": 0,
        "raw_root": "/root/autodl-fs/night8a_raw_runs_20260820", "cells": cells,
    })
    atomic_json(OUT / "checkpoint_and_roundtrip_audit.json", {
        "status": "PASS_FOR_COMPLETED_R1_R2", "audited_checkpoints": len(checkpoints),
        "all_real_files": True, "all_sha_verified": True, "all_fresh_process_roundtrip": True,
        "all_training_used_cuda": True, "rows": checkpoints,
    })
    required_empty_tables()

    authority_names = ["p0_authority.json", "p0_semantic_contract.json", "p0_config_lock.json",
                       "p0_full_size_p22_smoke.json", "historical_metric_recompute.json"]
    atomic_json(OUT / "p0_authority_and_semantic_contract.json", {
        "status_at_gate": "PASS", "post_label_audit_status": STATUS,
        "p0_gap": "P0 checked P22 full-size replay but did not assert exact locked C00 protein affinity/partition reuse",
        "files": {name: {"sha256": file_sha(OUT / name)} for name in authority_names},
        "formal_scientific_training_at_gate": 0, "formal_transforms_at_gate": 0,
        "label_access_at_gate": False,
    })
    family = json.loads((OUT / "family_reference_policy.json").read_text(encoding="utf-8"))
    family.update({"status": STATUS, "scientific_use_in_night8a": False, "root_cause": REASON,
                   "corrected_code_committed_for_future_run_only": True})
    atomic_json(OUT / "resolved_family_policy.json", family)
    shutil.copy2(REGISTRY, OUT / "config_registry_frozen.json")

    b00 = pd.read_csv(OUT / "b00_parity_diagnostic.csv")
    p22_error = float(b00[b00.dataset == "p22"][["delta_ari", "delta_nmi", "delta_q"]].abs().to_numpy().max())
    protein_error = float(b00[b00.dataset != "p22"][["delta_ari", "delta_nmi", "delta_q"]].abs().to_numpy().max())
    decision = {
        "status": STATUS, "scientific_conclusion_allowed": False, "unified_winner": None,
        "balanced_material_winners": [], "frontier_winners": [], "R3_started": False,
        "reason": REASON, "detected_at": "DEV_WINDOW_1 historical parity hard gate",
        "protein_B00_max_abs_metric_error": protein_error, "P22_B00_max_abs_metric_error": p22_error,
        "post_detection_scientific_training": 0, "post_detection_transforms": 0,
        "repair_used_for_scientific_results": False, "external_dataset_locked": "MISAR_E15_5_S1",
        "external_benchmark_run": False,
    }
    atomic_json(OUT / "night8a_decision.json", decision)
    atomic_json(OUT / "dev_window_1_audit.json", {
        "status": "FAILED_CLOSED_SEMANTIC", "authorized_role": "single_evaluator",
        "labels_opened_after_R1_R2_total_lock": True, "failure": REASON,
        "diagnostic_only_outputs": ["b00_parity_diagnostic.csv", "dev_window1_semantic_invalidation.json"],
        "candidate_metrics_released_for_selection": False, "shortlist_created": False,
        "window_closed_without_R3": True, "post_window_training_or_transform": False,
    })
    atomic_json(OUT / "label_firewall_audit.json", {
        "status": "PASS_FAIL_CLOSED", "trainer_or_transformer_label_access": False,
        "R1_R2_locked_before_label_access": True, "authorized_DEV_WINDOW_1_opened": True,
        "DEV_WINDOW_2_opened": False, "labels_used_for_training_checkpoint_family_selector_or_cluster_selection": False,
        "post_semantic_failure_training": 0, "post_semantic_failure_transform": 0,
        "external_label_values_opened": False,
    })
    atomic_json(OUT / "failure_and_retry_audit.json", {
        "status": "COMPLETE", "scientific_retry_count": 0, "fallback_count": 0,
        "formal_transform_failures": transform_failures,
        "preserved_infrastructure_attempts": [
            "p0_attempt1_historical_subset_misinterpretation.log", "p0_attempt2_formula_namespace_mismatch.log",
            "p0_attempt3_rscript_path.log", "p0_attempt4_reference_dimension_contract.log",
            "r1_attempt1_registry_key_case.log",
        ],
        "semantic_failure": "dev_window1_semantic_invalidation.json",
    })
    atomic_json(OUT / "budget_audit.json", {
        "status": "PASS", "R1_R2_registered_cells": len(cells), "actual_scientific_training": len(checkpoints),
        "R3_scientific_training": 0, "science_training_hard_cap": 180, "scientific_retry": 0,
        "infrastructure_attempts": 5, "infrastructure_retry_cap": 12, "fallback": 0,
    })

    external = json.loads((OUT / "external_dataset_lock.json").read_text(encoding="utf-8"))
    report = f"""# SpaLORA Night-8A report

Final status: `{STATUS}`

## Plain result

Night-8A implemented the planned MF-SPC family router and all six registered mechanism classes, but this run cannot support a scientific gain claim. At the first authorized label window, the RNA-protein B00 reference differed from the locked C00 result by as much as {protein_error:.12f}. The P22 R02 reference remained within {p22_error:.3e}, which localized the defect to the protein-family C00 transform path.

The defect was semantic rather than a historical-table problem: nine A1/tonsil/D1 B00 cells rebuilt an approximate equal-view affinity instead of reusing the SHA-locked C00 G04/H05 affinity and partition. The minimum partition agreement with the locked artifact was ARI {min(x.get('partition_ari_current_vs_locked', 1.0) for x in invalid['rows']):.6f}. Therefore the label window failed closed, no shortlist was created, and R3 stayed at zero.

## Dataset interpretation

- Human lymph (A1 and D1): no valid Night-8A delta is reportable because their reference endpoint was implemented incorrectly.
- Tonsil: no valid Night-8A delta is reportable for the same reason.
- P22: the R02 reference replay itself matched historical metrics, but no P22 candidate is promoted because the preregistered joint selection window was invalidated globally.

## Engineering and audit

- R1/R2 registered cells: {len(cells)}; real CUDA training/checkpoint round-trips: {len(checkpoints)}; content-addressed no-op aliases: {aliases}.
- One fixed numerical transform failure was preserved: B03_RR10_PROTO / P22 / seed 1 (`cluster K mismatch`); scientific retry 0, fallback 0.
- The corrected C00 artifact-reuse code and a regression test are delivered for a future clean rerun only. They were not used to backfill, reevaluate, or continue this run.
- External provenance preflight independently locked {external['dataset_id']} ({external['spots']} spots, RNA+ATAC) with labels sealed; no external benchmark ran.

## Scientific boundary

There is no unified winner, specialist winner, Pareto claim, or per-dataset improvement claim from this invalid run. Existing C00 and P22 R02 historical conclusions remain unchanged.
"""
    (OUT / "night8a_report.md").write_text(report, encoding="utf-8")
    plain = f"""# Night-8A 通俗总结

这轮把 MF-SPC 的家族路由、shared/private、同位点对齐、软原型、RNA 锚点、DGI 和 MNN triplet 都实现并完成了 R1/R2，但第一次开标签核验时发现基线语义错误，所以本轮科研结论作废，不能报“涨了多少”。

错误只定位在 RNA+蛋白的 C00 基线路径：程序重新构造了近似 affinity，而不是复用已经锁定的 G04/H05 affinity 和 partition。A1、D1、tonsil 因而都没有可报告的新方法增减；P22 的 R02 基线本身精确复现，但因为联合选择窗口整体失效，也没有候选被晋级。R3 没有启动，失败后新增训练和 transform 都是 0。

修复代码和回归测试已经准备好，但没有拿它回填本轮结果。外部 MISAR E15.5 S1 已完成无标签锁定（{external['spots']} spots，RNA+ATAC），本轮没有运行外部 benchmark。
"""
    (OUT / "plain_language_summary.md").write_text(plain, encoding="utf-8")
    print(json.dumps({"status": STATUS, "registered_cells": len(cells), "cuda_checkpoints": len(checkpoints),
                      "R3_training": 0, "transform_failures": len(transform_failures)}, sort_keys=True))


if __name__ == "__main__":
    main()
