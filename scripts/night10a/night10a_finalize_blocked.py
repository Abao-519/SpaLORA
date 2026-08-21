#!/usr/bin/env python3
"""Create the fail-closed Night-10A compact evidence after semantic re-audit."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
from pathlib import Path

REPO = Path("/root/autodl-fs/SpaLORA-night10a")
RAW = Path("/root/autodl-fs/night10a_qcrd_20260821")
OUT = REPO / "outputs/night10a/handoff"
P0 = RAW / "p0_audit"
STATUS = "IMPLEMENTATION_SEMANTICS_INVALID"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    os.replace(tmp, path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def file_row(path: Path, root: Path) -> dict:
    return {"path": path.relative_to(root).as_posix(), "size_bytes": path.stat().st_size, "sha256": sha(path)}


def csv_header(path: Path, fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()
    os.replace(tmp, path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    atomic_json(OUT / "finalization_attempt1_python38_pathlib_error.json", {
        "category": "DELIVERY_INFRASTRUCTURE_ONLY",
        "error": "Python 3.8 pathlib.Path.write_text does not accept newline",
        "scientific_work_affected": False,
        "formal_training_units_started": 0,
        "formal_transforms_started": 0,
        "resolution": "use Path.open with explicit newline",
        "preserved": True,
    })
    reaudit = {
        "status": "FAIL",
        "supersedes_for_gate_decision": "outputs/night10a/p0/p0_final_gate.json",
        "prior_partial_probe_preserved": True,
        "reason": "The first probe tested implemented behavior but did not establish full registry coverage.",
        "missing_required_global_quality_features": [
            "spatial local consistency on the frozen partition",
            "graph local residual as a global feature",
        ],
        "missing_required_spot_quality_features": ["local neighbor entropy"],
        "authority_ambiguities_that_must_not_be_guessed": [
            "masked-denoising fraction and deterministic masking recipe",
            "weights for correction magnitude, original-representation anchor, and local-boundary preservation",
            "Q07 confidence-weighted MNN loss weight",
        ],
        "implemented_and_verified_but_insufficient": [
            "frozen silhouette/DB/CH evidence",
            "local residual-based spot weighting",
            "cross-modal cosine disagreement",
            "sparse reciprocal MNN support",
            "stop-gradient teacher",
            "bounded correction and exact checkpoint reload",
        ],
        "formal_training_units_started": 0,
        "formal_transforms_started": 0,
        "stage_m_started": False,
        "r1_started": False,
        "r2_started": False,
        "scientific_retry": 0,
        "fallback": 0,
        "labels_used_for_training_or_selection": False,
        "terminal_status": STATUS,
    }
    atomic_json(OUT / "p0_semantic_coverage_reaudit.json", reaudit)

    decision = {
        "protocol": "SpaLORA Night-10A QCRD score RnD",
        "terminal_status": STATUS,
        "decision": "Fail closed before Stage M and all formal candidate training.",
        "qcrd_candidates_locked": [],
        "frontiers_locked": [],
        "score_improvement_claimed": False,
        "stage_m": {"status": "NOT_STARTED_P0_HARD_GATE"},
        "r1": {"status": "NOT_STARTED_P0_HARD_GATE", "real_training_units": 0, "transforms": 0},
        "r2": {"status": "NOT_ELIGIBLE", "real_training_units": 0, "transforms": 0},
        "scientific_retry": 0,
        "fallback": 0,
    }
    atomic_json(OUT / "night10a_decision.json", decision)
    atomic_json(OUT / "frontier_registry.json", {
        "status": "NOT_CREATED_P0_HARD_GATE", "accuracy": [], "balanced": [], "spatial": [],
        "reason": "No valid formal QCRD output exists.",
    })
    atomic_json(OUT / "metric_backfill_coverage.json", {
        "status": "NOT_STARTED_P0_HARD_GATE", "rows": 0,
        "missing_reason": "Stage M is downstream of the P0 semantic hard gate; no metric was fabricated or backfilled from a neighboring candidate.",
    })
    csv_header(OUT / "metric_backfill_long.csv", ["candidate", "dataset", "seed", "metric", "value", "artifact_sha256", "protocol_version", "status"])
    csv_header(OUT / "per_seed_metrics_long.csv", ["stage", "candidate", "dataset", "seed", "metric", "value", "reference", "delta", "status"])

    p0_contract = json.loads((P0 / "p0_authority_and_resource_contract.json").read_text())
    atomic_json(OUT / "resource_audit.json", {
        "status": "PASS",
        "resources": p0_contract["resources"],
        "overall_wall_clock_limit_hours": 12,
        "single_transform_limit_minutes": 30,
        "formal_gpu_training_observed": False,
        "reason_no_formal_gpu_training": "P0 semantic coverage hard gate",
    })
    atomic_json(OUT / "label_read_audit.json", {
        "a1": {"authorized_isolated_p0_evaluator_reads": 4, "used_for_training_or_selection": False},
        "tonsil": {"reads": 0}, "d1": {"reads": 0}, "p22": {"reads": 0},
        "misar_y": {"reads": 0, "policy": "permanently forbidden"},
        "e18_5": {"reads": 0},
        "new_external_data": {"reads": 0},
    })
    atomic_json(OUT / "tests_and_semantics_audit.json", {
        "reference_tests": {"passed": 4, "skipped": 0},
        "qcrd_partial_semantics_tests": {"passed": 6, "skipped": 0},
        "historical_ari_nmi_parity_max_abs_error": 0.0,
        "checkpoint_roundtrip_max_abs_error": 0.0,
        "partial_probe_passed": True,
        "full_registry_coverage_passed": False,
        "full_registry_coverage_report": "p0_semantic_coverage_reaudit.json",
        "terminal_status": STATUS,
    })
    atomic_json(OUT / "independent_recomputation.json", {
        "status": "PASS_FOR_AVAILABLE_P0_NUMERICS_ONLY",
        "historical_ari_nmi_max_abs_error": 0.0,
        "formal_summary_tables_recomputed": 0,
        "reason": "No formal per-seed output or summary was created after the hard gate.",
    })
    atomic_json(OUT / "budget_and_failure_audit.json", {
        "formal_training_budget_used": 0,
        "formal_transform_budget_used": 0,
        "scientific_retries": 0,
        "fallbacks": 0,
        "p0_non_scientific_attempts": 5,
        "preserved_attempt_records": sorted(p.name for p in P0.glob("p0_attempt*.json")),
        "hard_gate": "full registered candidate semantics not uniquely executable",
    })

    summary = """# Night-10A 通俗结论

Night-10A 没有产生新的分数，也没有训练任何 QCRD 候选。原因不是候选效果差，而是正式训练前的逐字段复核发现：现有实现漏掉了注册表明确要求的部分质量特征，同时权威文件没有给出几项训练损失和 masked denoising 的唯一数值定义。执行者如果自行补默认值，就会把预注册实验变成事后自创实验。

因此本轮严格停在 P0，终态为 `IMPLEMENTATION_SEMANTICS_INVALID`。Stage M、R1 和 R2 均未启动；没有候选、seed 或 checkpoint 被挑选，没有 fallback，也没有访问 MISAR Y、E18.5 或新外部数据。A1 标签只在隔离的 P0 指标校验中读取，用于确认 ARI/NMI 实现一致，未进入训练或选择。

后续若要继续，应由规划方签发一个机器可读 REV1：明确 global/spot quality 的完整公式、mask 比例与确定性规则、三项 mandatory regularizer 权重、Q07 MNN 权重，并同步更新真实构造测试。不能仅口头补一句“使用合理默认值”。
"""
    atomic_text(OUT / "plain_language_summary.md", summary)
    report = """# SpaLORA Night-10A report

## Terminal decision

`IMPLEMENTATION_SEMANTICS_INVALID`

The authority and resource checks passed, the server reference suite passed 10/10 tests, historical A1 ARI/NMI parity was exact, and the implemented subset passed real-view probes. A subsequent field-by-field coverage audit found that this subset did not implement the complete registered QCRD semantics. The earlier partial P0 PASS is preserved verbatim and is explicitly superseded for the go/no-go decision by `p0_semantic_coverage_reaudit.json`.

## Blocking differences

- Global quality omits registered spatial local consistency and a global graph-local residual feature.
- Per-spot quality omits registered local-neighbor entropy.
- The authority does not uniquely specify the masking fraction/recipe or the numerical loss weights for correction magnitude, original anchor, boundary preservation, and Q07 MNN supervision.

These are definition-level differences, not ordinary implementation details. No values were guessed.

## Execution counts

- Stage M: not started.
- Formal training: 0.
- Formal transforms: 0.
- R1/R2 evaluation: 0.
- Scientific retry: 0; fallback: 0.
- New third-party benchmark: 0; external-data access: 0.

## Label firewall

A1 was read four times by isolated P0 evaluator/diagnostic processes while fixing evaluator-only serialization/alignment issues. The computed values were used only to establish metric parity and never entered training, loss, candidate selection, or hyperparameter definition. tonsil/D1/P22 label reads were 0; MISAR Y and E18.5 reads were 0.

## Required recovery authority

A Night-10A REV1 must define the omitted feature formulas and all fixed numerical training weights/masking semantics, and must add negative/real-construction tests that cover those definitions. The existing partial implementation must not be promoted as a scientific negative result.
"""
    atomic_text(OUT / "night10a_report.md", report)

    # Copy immutable P0 evidence into the handoff without changing the original files.
    evidence = OUT / "p0_evidence"
    evidence.mkdir(exist_ok=True)
    for src in sorted(P0.glob("*.json")):
        shutil.copy2(src, evidence / src.name)

    # Tracked root-relative index intentionally excludes itself.
    paths = sorted(p for p in OUT.rglob("*") if p.is_file() and p.name != "delivery_index.json")
    atomic_json(OUT / "delivery_index.json", {
        "schema": "spalora.night10a.tracked_delivery_index.v1",
        "root": "outputs/night10a/handoff",
        "root_rule": "all regular files below root except delivery_index.json",
        "entry_count": len(paths),
        "entries": [file_row(p, OUT) for p in paths],
        "terminal_status": STATUS,
    })
    print(json.dumps({"status": STATUS, "files": len(paths) + 1}, sort_keys=True))


if __name__ == "__main__":
    main()
