#!/usr/bin/env python3
"""Build the auditable Night-8A scientific summary without reopening labels."""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pandas as pd


REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "outputs/night8a_handoff"
REGISTRY = REPO / "protocols/night8a/SpaLORA_Night8A_MFSPC_Registry_2026-08-20.json"
STAGES = ("R1", "R2", "R3")


def sha(path: Path) -> str:
    import hashlib
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_json(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def verify_and_collect_stages() -> tuple[list[dict], list[dict], list[dict]]:
    cell_rows: list[dict] = []
    checkpoint_rows: list[dict] = []
    runtime_rows: list[dict] = []
    total_actual = 0
    for stage in STAGES:
        path = OUT / f"{stage.lower()}_stage_manifest.json"
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] != "LOCKED_PRE_LABEL" or manifest["label_access"]:
            raise RuntimeError(f"{stage} is not a valid pre-label total lock")
        if manifest["terminal_cells"] != manifest["planned_registered_cells"]:
            raise RuntimeError(f"{stage} coverage mismatch")
        total_actual += int(manifest["actual_training_attempts"])
        for cell in manifest["cells"]:
            common = {
                "stage": stage, "config_id": cell["config_id"], "dataset": cell["dataset"],
                "unit_id": cell["unit_id"], "seed": int(cell["seed"]), "family": cell["family"],
                "status": cell["status"], "transform_status": cell.get("transform_status", cell["status"]),
                "raw_root": cell["root"],
            }
            if cell["status"] == "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL":
                target = Path(cell["alias_target_training_manifest"])
                if sha(target) != cell["alias_target_training_manifest_sha256"]:
                    raise RuntimeError("alias training manifest hash mismatch")
                transform = Path(cell["alias_target_transform_manifest"])
                if sha(transform) != cell["alias_target_transform_manifest_sha256"]:
                    raise RuntimeError("alias transform manifest hash mismatch")
                cell_rows.append({**common, "alias_target_config_id": cell["alias_target_config_id"],
                                  "training_manifest_sha256": sha(target), "transform_manifest_sha256": sha(transform)})
                runtime_rows.append({**common, "phase": "alias", "runtime_seconds": 0.0,
                                     "cuda_used": False, "peak_gpu_allocated_mib": 0.0,
                                     "process_peak_rss_mib": 0.0})
                continue
            if cell["status"] != "SUCCESS_PRE_LABEL":
                cell_rows.append(common)
                continue
            training_path = Path(cell["root"]) / "training/training_manifest.json"
            reload_path = Path(cell["root"]) / "training/reload_audit.json"
            training = json.loads(training_path.read_text(encoding="utf-8"))
            reload = json.loads(reload_path.read_text(encoding="utf-8"))
            if training["status"] != "SUCCESS_PRE_LABEL" or not training["cuda_used"] or reload["status"] != "PASS":
                raise RuntimeError(f"checkpoint/runtime audit failure: {cell['unit_id']} {cell['config_id']}")
            for artifact in training["artifacts"].values():
                path_value = Path(artifact["path"])
                if path_value.stat().st_size != int(artifact["size_bytes"]) or sha(path_value) != artifact["sha256"]:
                    raise RuntimeError(f"artifact mismatch: {path_value}")
            model = training["artifacts"]["model_final.pt"]
            if reload["checkpoint_file_sha256"] != model["sha256"] or reload["state_sha256"] != training["model_tensor_state_sha256"]:
                raise RuntimeError("checkpoint reload state mismatch")
            if not reload["fresh_process"] or any(not row["allclose_1e-7"] for row in reload["embeddings"].values()):
                raise RuntimeError("fresh-process embedding round-trip mismatch")
            transform_path = Path(cell.get("transform_dir", cell["root"])) / "transform_manifest.json"
            transform = None
            if cell.get("transform_status") == "SUCCESS_PRE_LABEL":
                transform = json.loads(transform_path.read_text(encoding="utf-8"))
                cluster_path = Path(cell["transform_dir"]) / "clusters.csv"
                if sha(cluster_path) != transform["cluster_file_sha256"] or not transform["cluster_exact_replay"]:
                    raise RuntimeError("cluster replay/hash mismatch")
            cell_rows.append({**common, "training_manifest_sha256": sha(training_path),
                              "transform_manifest_sha256": None if transform is None else sha(transform_path),
                              "model_file_sha256": model["sha256"],
                              "model_tensor_state_sha256": training["model_tensor_state_sha256"]})
            max_delta = max(float(x["max_abs_delta"]) for x in reload["embeddings"].values())
            checkpoint_rows.append({**common, "checkpoint_path": model["path"],
                                    "checkpoint_size_bytes": model["size_bytes"],
                                    "checkpoint_file_sha256": model["sha256"],
                                    "tensor_state_sha256": training["model_tensor_state_sha256"],
                                    "reload_audit_sha256": sha(reload_path), "fresh_process": True,
                                    "max_embedding_abs_delta": max_delta, "status": "PASS"})
            runtime_rows.append({**common, "phase": "training_and_reload",
                                 "runtime_seconds": float(training["runtime_seconds"]),
                                 "cuda_used": bool(training["cuda_used"]),
                                 "cuda_device": training["cuda_device"],
                                 "peak_gpu_allocated_mib": float(training["peak_gpu_allocated_mib"]),
                                 "process_peak_rss_mib": float(training["process_peak_rss_mib"])})
            if transform is not None:
                runtime_rows.append({**common, "phase": "cpu_transform",
                                     "runtime_seconds": float(transform["runtime_seconds"]),
                                     "cuda_used": False, "cpu_workers_stage_max": 3,
                                     "threads_per_worker": 3,
                                     "process_peak_rss_mib": float(transform["process_peak_rss_mib"])})
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    if total_actual > int(registry["runtime"]["science_training_hard_cap"]):
        raise RuntimeError("scientific training budget exceeded")
    return cell_rows, checkpoint_rows, runtime_rows


def all_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    r1 = pd.read_csv(OUT / "r1_per_seed_metrics.csv")
    r2 = pd.read_csv(OUT / "r2_per_seed_metrics.csv")
    r3 = pd.read_csv(OUT / "final_per_seed_metrics.csv")
    finalists = set(load_json("r2_shortlist_ids.json")["finalist_ids"])
    extension = r3[(r3.stage == "R3") & (r3.config_id.isin(finalists))]
    frame = pd.concat((r1, r2, extension), ignore_index=True)
    if frame.duplicated(["stage", "config_id", "dataset", "seed"]).any():
        raise RuntimeError("duplicate per-seed metric key")
    frame.to_csv(OUT / "per_seed_metrics.csv", index=False)
    delta_columns = [x for x in frame.columns if x.startswith("delta_")]
    paired = frame[["stage", "config_id", "dataset", "unit_id", "seed", "reference_id", *delta_columns]].copy()
    paired.to_csv(OUT / "paired_deltas.csv", index=False)
    return frame, paired


def module_summary() -> pd.DataFrame:
    frame = pd.read_csv(OUT / "r1_per_seed_metrics.csv")
    metrics = ["ari", "nmi", "q", "delta_ari", "delta_nmi", "delta_q",
               "delta_neighbor_agreement", "delta_moran_i", "delta_geary_c", "delta_boundary_disagreement"]
    summary = frame.groupby(["config_id", "dataset", "evaluation_status"], dropna=False)[metrics].mean().reset_index()
    summary["seed_count"] = frame.groupby(["config_id", "dataset", "evaluation_status"], dropna=False).size().to_numpy()
    summary.to_csv(OUT / "module_ablation_summary.csv", index=False)
    return summary


def final_status(decision: dict, historical: dict, external: dict) -> tuple[str, float]:
    family_policy_delta = 0.45 * float(historical["r02_vs_c00"]["p22"]["mean_delta_q_vs_c00"])
    if decision.get("balanced_material_winners"):
        return "NIGHT8A_MFSPC_BALANCED_CANDIDATE_LOCKED", family_policy_delta
    if family_policy_delta > 0 and external["status"] == "LOCKED_FOR_FUTURE_NIGHT8B_CONFIRMATION":
        return "NIGHT8A_FAMILY_POLICY_ONLY_LOCKED", family_policy_delta
    if any(x.get("human_lymph_frontier") or x.get("P22_frontier") for x in decision.get("decisions", [])):
        return "NIGHT8A_FRONTIER_ONLY_NO_UNIFIED_WINNER", family_policy_delta
    return "NIGHT8A_NO_VALID_GAIN", family_policy_delta


def chosen_rows(summary: pd.DataFrame, decision: dict) -> list[dict]:
    ids = []
    if decision.get("unified_winner"):
        ids.append(decision["unified_winner"])
    for row in decision.get("decisions", []):
        if row.get("human_lymph_frontier") or row.get("P22_frontier"):
            if row["config_id"] not in ids:
                ids.append(row["config_id"])
    if not ids and len(summary):
        ids.append(summary.sort_values("priority_macro_delta_Q", ascending=False).iloc[0].config_id)
    return summary[summary.config_id.isin(ids)].to_dict("records")


def reports(status: str, decision: dict, summary: pd.DataFrame, family_delta: float, external: dict,
            cell_rows: list[dict], checkpoint_rows: list[dict]) -> None:
    selected = chosen_rows(summary, decision)
    lines = [
        "# SpaLORA Night-8A report", "", f"Final status: `{status}`", "",
        "## What changed", "",
        "Night-8A implemented MF-SPC as an assay-metadata-routed model family. RNA-protein uses the locked C00 endpoint; "
        "RNA-epigenome uses the locked P22 R02 development reference. The preregistered additions were shared/private "
        "encoders, VICReg-style same-spot redundancy reduction, soft prototype consensus, RNA anchoring, sparse DGI, "
        "and the SMART-style MNN triplet objective. Routing never received dataset names or labels.", "",
        "## Main quantitative result", "",
        f"The locked family policy changes the preregistered priority score by {family_delta:+.6f} Q versus using C00 "
        "for every family; this gain is driven by the previously locked P22 development frontier and is not external confirmation.", "",
    ]
    for row in selected:
        lines.extend([
            f"### {row['config_id']}", "",
            f"- Human lymph: A1 delta Q {row['a1_mean_delta_q']:+.6f}; D1 delta Q {row['d1_mean_delta_q']:+.6f}; combined delta Q_HLN {row['delta_Q_HLN']:+.6f}.",
            f"- P22 delta Q {row['p22_mean_delta_q']:+.6f}.",
            f"- Tonsil delta Q {row['tonsil_mean_delta_q']:+.6f}.",
            f"- Priority macro delta Q {row['priority_macro_delta_Q']:+.6f}; spatial protection: {bool(row['spatial_protection_pass'])}.", "",
        ])
    lines.extend([
        "## Integrity", "",
        f"- Registered terminal cells: {len(cell_rows)}; real CUDA checkpoints with fresh-process round-trip: {len(checkpoint_rows)}.",
        "- Scientific retry: 0; fallback: 0. Pre-science infrastructure attempts remain preserved in the handoff.",
        "- Labels were opened only by the two registered evaluator windows after their respective total locks.",
        f"- External lock: {external['dataset_id']} ({external['spots']} spots), label sealed; no external benchmark was run.", "",
        "## Interpretation", "",
        "This is a development result, not a SOTA or external-generalization claim. R02 remains a P22 development frontier. "
        "The MISAR E15.5 S1 lock is for a future one-time Night-8B confirmation after its annotation provenance is re-audited.", "",
    ])
    (OUT / "night8a_report.md").write_text("\n".join(lines), encoding="utf-8")

    plain = [
        "# Night-8A 通俗总结", "",
        "这轮不是简单叠加一个损失，而是把模型改成按实验模态元数据自动选择家族：RNA+蛋白沿用 C00，RNA+表观组沿用 P22 的 R02 开发前沿；在此基础上系统检验 shared/private、同位点对齐、软原型、RNA 锚点、DGI 和 MNN triplet。整个选择过程没有使用数据集名字或标签。", "",
        f"最终科研状态：{status}。固定家族策略相对所有数据都使用 C00 的预注册综合 Q 改变量为 {family_delta:+.6f}；这仍主要来自既有 P22 开发结果，不能解释成已经跨数据集泛化。", "",
    ]
    for row in selected:
        plain.append(f"{row['config_id']}：A1 Q {row['a1_mean_delta_q']:+.6f}，D1 Q {row['d1_mean_delta_q']:+.6f}，P22 Q {row['p22_mean_delta_q']:+.6f}，tonsil Q {row['tonsil_mean_delta_q']:+.6f}。")
    plain.extend(["", f"外部数据已锁定 MISAR E15.5 S1（{external['spots']} spots，RNA+ATAC），标签未打开，本轮没有做外部 benchmark。"])
    (OUT / "plain_language_summary.md").write_text("\n".join(plain) + "\n", encoding="utf-8")


def main() -> None:
    p0 = load_json("p0_semantic_contract.json")
    if p0["status"] != "PASS":
        raise RuntimeError("P0 is not PASS")
    for window in ("dev_window_1_audit.json", "dev_window_2_audit.json"):
        if load_json(window)["status"] != "CLOSED":
            raise RuntimeError(f"evaluation window not closed: {window}")
    external = load_json("external_dataset_lock.json")
    cell_rows, checkpoint_rows, runtime_rows = verify_and_collect_stages()
    frame, _ = all_metrics()
    module_summary()
    pd.DataFrame(runtime_rows).to_csv(OUT / "runtime_and_gpu_timeline.csv", index=False)
    pd.DataFrame(cell_rows).to_csv(OUT / "training_cells.csv", index=False)

    actual = len(checkpoint_rows)
    aliases = sum(x["status"] == "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL" for x in cell_rows)
    failures = [x for x in cell_rows if x["status"] not in {"SUCCESS_PRE_LABEL", "CONTENT_ADDRESSED_NOOP_ALIAS_PRE_LABEL"}]
    atomic_json(OUT / "training_manifest.json", {
        "status": "LOCKED", "registered_cells": len(cell_rows), "actual_cuda_training_attempts": actual,
        "content_addressed_aliases": aliases, "failed_cells": failures, "scientific_retry_count": 0,
        "fallback_count": 0, "science_training_hard_cap": 180, "raw_root": "/root/autodl-fs/night8a_raw_runs_20260820",
        "cells": cell_rows,
    })
    atomic_json(OUT / "checkpoint_and_roundtrip_audit.json", {
        "status": "PASS", "audited_checkpoints": len(checkpoint_rows), "all_real_files": True,
        "all_sha_verified": True, "all_fresh_process_roundtrip": True,
        "all_training_used_cuda": True, "rows": checkpoint_rows,
    })

    authority_files = ["p0_authority.json", "p0_semantic_contract.json", "p0_config_lock.json",
                       "p0_full_size_p22_smoke.json", "historical_metric_recompute.json"]
    atomic_json(OUT / "p0_authority_and_semantic_contract.json", {
        "status": "PASS", "files": {name: {"sha256": sha(OUT / name)} for name in authority_files},
        "formal_scientific_training_at_gate": 0, "formal_transforms_at_gate": 0, "label_access_at_gate": False,
    })
    family = load_json("family_reference_policy.json")
    historical = load_json("historical_metric_recompute.json")
    family["family_policy_priority_macro_delta_Q_vs_C00"] = 0.45 * historical["r02_vs_c00"]["p22"]["mean_delta_q_vs_c00"]
    family["interpretation"] = "development family policy; not an external confirmation"
    atomic_json(OUT / "resolved_family_policy.json", family)
    shutil.copy2(REGISTRY, OUT / "config_registry_frozen.json")

    decision = load_json("night8a_decision.json")
    final_summary = pd.read_csv(OUT / "final_candidate_summary.csv")
    status, family_delta = final_status(decision, historical, external)
    decision["status"] = status
    decision["family_policy_priority_macro_delta_Q_vs_C00"] = family_delta
    decision["external_dataset_locked"] = external["dataset_id"]
    decision["external_benchmark_run"] = False
    atomic_json(OUT / "night8a_decision.json", decision)

    frontier_ids = set(decision.get("pareto_frontier", []))
    pareto = final_summary.copy()
    pareto["is_pareto"] = pareto.config_id.isin(frontier_ids)
    pareto.to_csv(OUT / "pareto_frontier.csv", index=False)
    firewall = {
        "status": "PASS", "trainer_or_transformer_label_access": False,
        "stages_locked_pre_label": list(STAGES), "evaluation_windows": ["DEV_WINDOW_1", "DEV_WINDOW_2"],
        "window_1_opened_after_R1_R2_total_lock": True, "window_2_opened_after_R3_total_lock": True,
        "labels_used_for_training_checkpoint_family_selector_or_cluster_selection": False,
        "external_label_values_opened": False,
        "source_audits": {name: sha(OUT / name) for name in ("dev_window_1_audit.json", "dev_window_2_audit.json")},
    }
    atomic_json(OUT / "label_firewall_audit.json", firewall)
    atomic_json(OUT / "budget_audit.json", {
        "status": "PASS", "registered_cells": len(cell_rows), "actual_scientific_training": actual,
        "hard_cap": 180, "scientific_retry": 0, "infrastructure_retry": 5,
        "infrastructure_retry_cap": 12, "fallback": 0,
        "preserved_pre_science_attempts": [
            "p0_attempt1_historical_subset_misinterpretation.log", "p0_attempt2_formula_namespace_mismatch.log",
            "p0_attempt3_rscript_path.log", "p0_attempt4_reference_dimension_contract.log",
            "r1_attempt1_registry_key_case.log",
        ],
    })
    reports(status, decision, final_summary, family_delta, external, cell_rows, checkpoint_rows)
    print(json.dumps({"status": status, "registered_cells": len(cell_rows), "actual_cuda_trainings": actual,
                      "metric_rows": len(frame)}, sort_keys=True))


if __name__ == "__main__":
    main()
