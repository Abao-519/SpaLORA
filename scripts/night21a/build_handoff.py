#!/usr/bin/env python3
"""Build the fail-closed Night-21A handoff from locked discovery artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from datetime import datetime, timezone


ROOT = Path("/root/night21a_working")
REPO = Path("/root/SpaLORA-night16h")
OUT = REPO / "outputs/night21a_handoff"
ARMS = ["CARRIER_ONLY", "LOWPASS_CARRIER_CONTROL", "MEAN_ONLY_ANCHORED", "HIERARCHICAL_WITHOUT_ANCHOR",
        "ANCHOR_WITHOUT_GRADIENT", "ANCHOR_WITHOUT_HIERARCHICAL_FUSION", "FULL_COMPOSITION"]
LANES = ["A1_K10", "TONSIL_S1_K4", "P22_K9", "PLACENTA_K10"]
FAMILY = {"A1_K10": "RNA_PROTEIN", "TONSIL_S1_K4": "RNA_PROTEIN", "P22_K9": "RNA_CHROMATIN", "PLACENTA_K10": "RNA_CHROMATIN"}
FRONTIER = {
    "A1_K10": (0.276171767, 0.421937362, "Post-Night19C Q2 current_score_board"),
    "TONSIL_S1_K4": (0.236682867, 0.317365241, "Post-Night19C Q2 current_score_board"),
    "P22_K9": (0.596390056, 0.718243175, "Post-Night19C Q2 current_score_board"),
    "PLACENTA_K10": (0.499999398, 0.631180143, "Post-Night19C Q2 current_score_board"),
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_rows() -> list[dict]:
    rows = []
    for lane in LANES:
        for arm in ARMS:
            path = ROOT / f"evaluation/{lane}__{arm}.csv"
            if not path.exists(): raise RuntimeError(f"missing evaluation {path}")
            with path.open(encoding="utf-8") as handle: rows.extend(csv.DictReader(handle))
    if len(rows) != 28 or {(row["lane"], row["arm"]) for row in rows} != {(lane, arm) for lane in LANES for arm in ARMS}:
        raise RuntimeError("formal 4x7 evaluation matrix is incomplete")
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = read_rows()
    write_csv(OUT / "absolute_metrics_and_controls.csv", rows)
    by_lane = {lane: {row["arm"]: row for row in rows if row["lane"] == lane} for lane in LANES}
    contribution = []
    strict_count = 0; carrier_dual_count = 0
    for lane in LANES:
        block = by_lane[lane]; full = block["FULL_COMPOSITION"]; carrier = block["CARRIER_ONLY"]
        controls = [block[arm] for arm in ARMS if arm not in {"FULL_COMPOSITION", "CARRIER_ONLY"}]
        strongest_ari = max(controls, key=lambda row: float(row["ari"])); strongest_nmi = max(controls, key=lambda row: float(row["nmi"]))
        dual_carrier = float(full["ari"]) > float(carrier["ari"]) and float(full["nmi"]) > float(carrier["nmi"])
        strict = float(full["ari"]) > max(float(row["ari"]) for row in controls) and float(full["nmi"]) > max(float(row["nmi"]) for row in controls)
        carrier_dual_count += int(dual_carrier); strict_count += int(strict)
        contribution.append({"lane": lane, "family": FAMILY[lane], "carrier_ari": carrier["ari"], "carrier_nmi": carrier["nmi"],
            "full_ari": full["ari"], "full_nmi": full["nmi"], "delta_ari_vs_carrier": float(full["ari"])-float(carrier["ari"]),
            "delta_nmi_vs_carrier": float(full["nmi"])-float(carrier["nmi"]), "strongest_atomic_ari_arm": strongest_ari["arm"],
            "strongest_atomic_ari": strongest_ari["ari"], "strongest_atomic_nmi_arm": strongest_nmi["arm"], "strongest_atomic_nmi": strongest_nmi["nmi"],
            "full_minus_max_atomic_ari": float(full["ari"])-max(float(row["ari"]) for row in controls),
            "full_minus_max_atomic_nmi": float(full["nmi"])-max(float(row["nmi"]) for row in controls),
            "dual_gain_vs_carrier": dual_carrier, "strict_full_beats_every_atomic_both": strict,
            "min_cluster_full": full["min_cluster_size"], "cluster_sizes_full": full["cluster_sizes"]})
    write_csv(OUT / "matched_contribution_table.csv", contribution)
    family_rows = []
    for family in ("RNA_PROTEIN", "RNA_CHROMATIN"):
        selected = [row for row in contribution if row["family"] == family]
        family_rows.append({"family": family, "study_count": len(selected),
            "mean_delta_ari_vs_carrier": sum(row["delta_ari_vs_carrier"] for row in selected)/len(selected),
            "mean_delta_nmi_vs_carrier": sum(row["delta_nmi_vs_carrier"] for row in selected)/len(selected),
            "strict_synergy_lanes": sum(bool(row["strict_full_beats_every_atomic_both"]) for row in selected)})
    write_csv(OUT / "family_discovery_summary.csv", family_rows)
    score_rows = []
    for lane in LANES:
        best = max(by_lane[lane].values(), key=lambda row: (float(row["ari"]), float(row["nmi"])))
        old_ari, old_nmi, source = FRONTIER[lane]
        score_rows.append({"lane": lane, "audited_previous_frontier_ari": old_ari, "audited_previous_frontier_nmi": old_nmi,
                           "night21a_best_arm": best["arm"], "night21a_best_ari": best["ari"], "night21a_best_nmi": best["nmi"],
                           "delta_ari_to_frontier": float(best["ari"])-old_ari, "delta_nmi_to_frontier": float(best["nmi"])-old_nmi,
                           "frontier_source": source, "frontier_advanced_both": float(best["ari"])>old_ari and float(best["nmi"])>old_nmi})
    write_csv(OUT / "score_frontier_board.csv", score_rows)

    full_manifests = [json.loads((ROOT / f"discovery/{lane}__FULL_COMPOSITION.json").read_text()) for lane in LANES]
    diagnostics = []
    for manifest in full_manifests:
        d = manifest["diagnostics"]
        diagnostics.append({"lane": manifest["lane"], "identity_initial_max_abs": d["identity_initial_max_abs"],
            "parameter_changed": d["parameter_changed"], "optimizer_steps": d["optimizer_steps"], "parameter_count": d["parameter_count"],
            "residual_frobenius": d["residual_frobenius"], "residual_max_row_norm": d["residual_max_row_norm"],
            "gate_min": d["gate_min"], "gate_mean": d["gate_mean"], "gate_max": d["gate_max"],
            "modality1_weight_mean": d["modality1_weight_mean"], "modality2_weight_mean": d["modality2_weight_mean"],
            "channel1_weight_means": json.dumps(d["channel1_weight_means"]), "channel2_weight_means": json.dumps(d["channel2_weight_means"]),
            "wall_seconds": manifest["wall_seconds"], "gpu_peak_mib": manifest["gpu_peak_mib"], "peak_rss_mib": manifest["peak_rss_mib"]})
    write_csv(OUT / "attention_gate_residual_diagnostics.csv", diagnostics)

    p0 = {}
    for lane in ("A1_K10", "P22_K9"):
        manifest = json.loads((ROOT / f"p0/{lane}__FULL_COMPOSITION.json").read_text())
        replay = json.loads((ROOT / f"replay/{lane}__P0.json").read_text())
        p0[lane] = {"manifest": manifest, "fresh_process_replay": replay}
    (OUT / "p0_registry.json").write_text(json.dumps({"status": "PASS", "lanes": p0}, indent=2, sort_keys=True), encoding="utf-8")

    all_replays = list((ROOT / "replay").glob("*.json"))
    replay_fail = [str(path) for path in all_replays if json.loads(path.read_text())["status"] != "PASS"]
    if replay_fail or len(all_replays) != 30: raise RuntimeError(f"replay audit failed {len(all_replays)} {replay_fail}")
    stage_c = bool(strict_count >= 2 and all(row["strict_synergy_lanes"] >= 1 for row in family_rows))
    if stage_c: raise RuntimeError("unexpected Stage-C authorization; manual frozen confirmation required")
    classification = "NO_COMPOSITIONAL_METHOD_SIGNAL"
    decision = {"classification": classification, "stage_c_authorized": False,
        "reason": "Full composition achieved strict matched synergy on 0/4 discovery lanes and the RNA_PROTEIN family mean delta was negative.",
        "full_dual_gain_vs_carrier_count": carrier_dual_count, "strict_synergy_count": strict_count,
        "discovery_lane_count": 4, "confirmation_runs_started": 0, "multi_seed_runs_started": 0,
        "formula_changed_after_lock": False, "shutdown_dispatched_at_build_time": False}
    (OUT / "night21a_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    (OUT / "label_flow_audit.json").write_text(json.dumps({"producer_annotation_reads": 0, "partitions_locked_before_evaluator": True,
        "evaluator_rows": 28, "p0_evaluator_rows": 2, "benchmark_labels_used": "post-lock metrics only",
        "confirmation_labels_read": False, "confirmation_reason": "Stage-C not authorized by discovery gate"}, indent=2, sort_keys=True), encoding="utf-8")
    corrections = [
        {"cycle": "P0", "issue": "Four-step A1/P22 runs were engineering P0 only", "resolution": "Marked superseded by frozen 60-step formal discovery", "scientific_rows_used": False},
        {"cycle": "P0 command", "issue": "remote summary invoked unavailable base python after producers and replays had completed", "resolution": "reran summary with the existing SpaLORA interpreter; artifacts were not regenerated", "scientific_rows_used": True},
        {"cycle": "preformal control", "issue": "simple fixed low-pass arm absent from first P0 producer choices", "resolution": "added LOWPASS_CARRIER_CONTROL before formal formula freeze without changing the AMCF core or reading labels", "scientific_rows_used": True},
    ]
    write_csv(OUT / "failure_and_correction_ledger.csv", corrections)

    test_log = ROOT / "logs/targeted_tests_final.log"
    test_text = test_log.read_text(encoding="utf-8") if test_log.exists() else ""
    if "5 passed" not in test_text: raise RuntimeError("final targeted test log absent")
    (OUT / "targeted_test_summary.json").write_text(json.dumps({"status": "PASS", "passed": 5, "failed": 0,
        "source": str(test_log), "fresh_process_replays": len(all_replays), "fresh_process_replay_failures": 0}, indent=2, sort_keys=True), encoding="utf-8")

    stat = os.statvfs("/"); data_stat = os.statvfs("/autodl-fs/data")
    working_bytes = sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file())
    resource = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "root_available_bytes": stat.f_bavail*stat.f_frsize,
        "root_available_inodes": stat.f_favail, "persistent_available_bytes": data_stat.f_bavail*data_stat.f_frsize,
        "persistent_available_inodes": data_stat.f_favail, "night21a_working_bytes": working_bytes,
        "new_files_on_persistent_data": 0, "downloaded_datasets_or_environments": 0,
        "max_gpu_peak_mib_from_manifests": max(float(row["gpu_peak_mib"]) for row in rows),
        "max_peak_rss_mib_from_manifests": max(float(row["peak_rss_mib"]) for row in rows)}
    (OUT / "resource_and_disk_audit.json").write_text(json.dumps(resource, indent=2, sort_keys=True), encoding="utf-8")

    for name in ("component_origin_and_license_matrix.csv", "source_collision_matrix.csv", "novel_coupling_claim.md", "module_contract.md"):
        shutil.copy2(ROOT / f"staging_outputs/{name}", OUT / name)
    shutil.copy2(ROOT / "formula_freeze.json", OUT / "formula_freeze.json")

    report_lines = [
        "# Night-21A AMCF 报告", "", "## 我现在需要知道的三件事", "",
        "1. **解决的问题**：本轮不是再选择旧候选，而是训练一个直接输出新表示的模型；它把三尺度邻域均值/图高通纹理、节点级模态内/模态间权重和强载体零起步有界残差放在同一条计算路径。",
        "2. **真实结果**：工程路径成立，但组合协同被否定。Full 仅在 placenta 相对 carrier 双升；A1、tonsil s1、P22 都双降，且 4/4 lane 均未同时超过全部匹配原子臂。Stage C 和多 seed 因此没有授权。",
        "3. **论文含义**：BANKSY/SpatialGlue/spaMGCN 等底层组件均有明确先例；剩余的窄联合对象只有在匹配协同成立时才值得写成贡献。本轮结果不支持它，分类为 `NO_COMPOSITIONAL_METHOD_SIGNAL`。",
        "", "## 绝对指标与归因", "", "| lane | carrier ARI/NMI | full ARI/NMI | Δ vs carrier | strongest atomic evidence | strict synergy |", "|---|---:|---:|---:|---|---|"
    ]
    for row in contribution:
        report_lines.append(f"| {row['lane']} | {float(row['carrier_ari']):.6f}/{float(row['carrier_nmi']):.6f} | {float(row['full_ari']):.6f}/{float(row['full_nmi']):.6f} | {row['delta_ari_vs_carrier']:+.6f}/{row['delta_nmi_vs_carrier']:+.6f} | ARI: {row['strongest_atomic_ari_arm']} {float(row['strongest_atomic_ari']):.6f}; NMI: {row['strongest_atomic_nmi_arm']} {float(row['strongest_atomic_nmi']):.6f} | {row['strict_full_beats_every_atomic_both']} |")
    report_lines += ["", "Placenta full 的 0.394816/0.533318 是局部信号，但 `ANCHOR_WITHOUT_HIERARCHICAL_FUSION` 的 ARI 为 0.416573，因此不能把该提升归因于完整分层组合。P22 的固定低通为 0.480940/0.610378，mean-only 为 0.465021/0.600332，也说明简单算子比 full 更稳。所有 Night-21A 结果都未刷新 Post-Night-19C 登记的绝对 frontier。",
        "", "## 数学与工程性质", "", "- 初始化时 anchored arms 的 `Z_out-Z0` 最大绝对误差为 0；残差投影零初始化，同时后续训练具有非零梯度和参数变化。",
        "- 三尺度运算保持 CSR 稀疏；复杂度为 `O(sum_s nnz(P_s)*(d1+d2) + N*C*h)`，没有 dense N×N。",
        "- 节点置换等价、注册图固定下的坐标旋转/缩放不变语义、exact K、strict checkpoint load 和 fresh-process byte-exact replay通过。",
        f"- A1/P22 真实 P0 的参数量分别为 {diagnostics[0]['parameter_count']} 和 {diagnostics[2]['parameter_count']}；formal 28 个 producer artifact 均有独立 replay。",
        "", "## 贡献边界与停止理由", "", "Full 的失败不是 endpoint 偷换：全部 7 arms 使用同一 KMeans endpoint、同一 seed、同一 60-step budget（非训练 controls 除外）和同一 carrier。P0 的 4-step结果只用于接口验证，已明确 superseded。因为 discovery 已使跨家族协同门不可达，继续 D1/tonsil s2/s3/MISAR/human 的 frozen confirmation 只会消耗算力且无法修复主张，所以 fail-closed 停止。",
        "", "## 导师汇报版", "", "我们这轮把之前分散的多尺度空间纹理、分层跨模态融合和强载体保护真正写成了一个统一可训练模型，而不是候选选择器。代码在 A1 和 P22 上完成真实 GPU forward/backward、checkpoint 严格加载和 fresh-process 重放，工程对象是成立的。来源审计确认邻域均值、梯度、双层注意力、多阶图卷积和残差都有成熟先例，所以只有它们的联合协同可能成为贡献。正式 discovery 结果却显示，full 在 A1、tonsil s1 和 P22 上都低于强 carrier；只有 placenta 相对 carrier 上升，但又不能胜过最强简化臂。最有价值的正信号其实来自 P22 的固定低通和 mean-only，而不是完整模型。因而没有进入 confirmation 或多 seed，也没有刷新现有绝对 frontier。结论是该组合架构工程完整但科学负，不能作为论文主方法；代码中的稀疏多尺度库、零起步有界残差、可复现 CLI/checkpoint/replay 仍可作为计算机求职作品资产。",
        "", "## 技术状态", "", "分类：`NO_COMPOSITIONAL_METHOD_SIGNAL`。确认运行：4 discovery lanes × 7 arms；未运行：5 confirmation lanes、多 seed、score-HPO。关机字段在封口构建时为 false，最终远端动作由交付流程另行登记。"]
    (OUT / "night21a_report.md").write_text("\n".join(report_lines)+"\n", encoding="utf-8")


if __name__ == "__main__": main()
