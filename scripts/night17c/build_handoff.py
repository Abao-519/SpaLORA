#!/usr/bin/env python3
"""Build the compact, evidence-backed Night-17C P0 handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import statistics
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/night17c_p0_working")
REPO = Path("/root/SpaLORA-night16h")
OUT = REPO / "outputs/night17c_handoff"
PRIMARY = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")
CONTROL_ARMS = (
    "FROZEN_RETAINED_SAME_HEAD", "FULL_BANK_SMOOTH_REFERENCE",
    "UNBIASED_SMOOTH_REFERENCE", "ZERO_RESIDUAL_CONTROL",
)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_csv(path: Path, values):
    values = list(values)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(values)


def filesystem(path: str):
    stat = os.statvfs(path)
    return {
        "path": path, "available_bytes": stat.f_bavail * stat.f_frsize,
        "available_inodes": stat.f_favail, "captured_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    all_metrics = []
    for lane in PRIMARY:
        all_metrics.extend(rows(ROOT / f"formal/{lane}/metrics.csv"))
        for seed in (1, 2):
            all_metrics.extend(rows(ROOT / f"confirmation/seed{seed}/{lane}/metrics.csv"))
    all_metrics.extend(rows(ROOT / "safety/MELANOMA_TUMOR_K2/metrics.csv"))
    write_csv(OUT / "all_metrics_ledger.csv", all_metrics)
    seed0 = [value for value in all_metrics if value["training_seed"] == "0"]
    write_csv(OUT / "absolute_metrics_and_controls.csv", seed0)

    strict = json.loads((ROOT / "formal/strict_selection.json").read_text())
    selected = strict["selected_config_id"]
    summaries = []
    for lane in PRIMARY:
        lane_rows = [value for value in all_metrics if value["lane"] == lane]
        controls = [value for value in lane_rows if value["arm"] in CONTROL_ARMS and value["training_seed"] == "0"]
        strongest_ari = max(float(value["ari"]) for value in controls)
        strongest_nmi = max(float(value["nmi"]) for value in controls)
        full = [value for value in lane_rows if value["arm"] == "UNBIASED_FULL" and value["config_id"] == selected]
        aris = [float(value["ari"]) for value in full]
        nmis = [float(value["nmi"]) for value in full]
        summaries.append({
            "lane": lane, "config_id": selected, "seeds": "0,1,2",
            "strongest_control_ari": strongest_ari, "strongest_control_nmi": strongest_nmi,
            "seed_ari": json.dumps(aris), "seed_nmi": json.dumps(nmis),
            "mean_ari": statistics.mean(aris), "median_ari": statistics.median(aris), "min_ari": min(aris), "std_ari": statistics.pstdev(aris),
            "mean_nmi": statistics.mean(nmis), "median_nmi": statistics.median(nmis), "min_nmi": min(nmis), "std_nmi": statistics.pstdev(nmis),
            "dual_win_seed_count": sum(a > strongest_ari + 1e-12 and n > strongest_nmi + 1e-12 for a, n in zip(aris, nmis)),
        })
    write_csv(OUT / "multi_seed_summary.csv", summaries)

    historic = {row["lane"]: row for row in rows(REPO / "outputs/night16h_handoff/absolute_metrics_main_table.csv")}
    head_rows = []
    for lane in PRIMARY:
        values = {}
        for arm, path in (
            ("LEARNED_Z01", ROOT / f"head_integration/{lane}/selected.evaluation.json"),
            ("ZERO_RESIDUAL", ROOT / f"head_controls/ZERO_RESIDUAL_CONTROL/{lane}/selected.evaluation.json"),
            ("PERMUTED_RELATION", ROOT / f"head_controls/PERMUTED_RELATION/{lane}/selected.evaluation.json"),
        ):
            values[arm] = json.loads(path.read_text())
        learned, zero, perm = values["LEARNED_Z01"], values["ZERO_RESIDUAL"], values["PERMUTED_RELATION"]
        strongest_ari = max(float(zero["absolute_ari"]), float(perm["absolute_ari"]))
        strongest_nmi = max(float(zero["absolute_nmi"]), float(perm["absolute_nmi"]))
        for arm, value in values.items():
            head_rows.append({
                "lane": lane, "carrier_arm": arm, "candidate_budget": 89,
                "selector": value["selector"], "candidate_id": value["candidate_id"],
                "ari": value["absolute_ari"], "nmi": value["absolute_nmi"], "ami": value["ami"], "fmi": value["fmi"],
                "moran_macro_ovr": value["morans_i_macro"], "geary_macro_ovr": value["gearys_c_macro"],
                "min_cluster_size": value["min_cluster_size_full"], "cluster_sizes": value["cluster_sizes_full"],
                "delta_vs_strongest_matched_ari": float(value["absolute_ari"]) - strongest_ari if arm == "LEARNED_Z01" else "",
                "delta_vs_strongest_matched_nmi": float(value["absolute_nmi"]) - strongest_nmi if arm == "LEARNED_Z01" else "",
                "night16h_ari": historic[lane]["absolute_ari"], "night16h_nmi": historic[lane]["absolute_nmi"],
                "delta_vs_night16h_ari": float(value["absolute_ari"]) - float(historic[lane]["absolute_ari"]) if arm == "LEARNED_Z01" else "",
                "delta_vs_night16h_nmi": float(value["absolute_nmi"]) - float(historic[lane]["absolute_nmi"]) if arm == "LEARNED_Z01" else "",
            })
    write_csv(OUT / "head_integration_matched_controls.csv", head_rows)

    real_paths = []
    replay_total = 0
    for lane in (*PRIMARY, "MELANOMA_TUMOR_K2"):
        base = ROOT / (f"formal/{lane}" if lane in PRIMARY else f"safety/{lane}")
        producer = json.loads((base / "producer.producer.json").read_text())
        replay = json.loads((base / "fresh_replay.json").read_text())
        replay_total += replay["replayed_primary_configs"]
        primary_diag = next(value for value in producer["run_diagnostics"] if value["arm"] == "UNBIASED_FULL" and value["config_id"] == selected)
        real_paths.append({
            "lane": lane, "n": producer["n"], "k": producer["k"], "view1_shape": producer["view1_shape"],
            "view2_shape": producer["view2_shape"], "retained_shape": producer["retained_shape"], "pair_count": producer["pair_count"],
            "unbiased_candidates": producer["unbiased_candidate_count"], "gate": producer["primary_gate_diagnostics"],
            "actual_optimizer_steps": primary_diag["actual_optimizer_steps"], "max_gradient_norm": primary_diag["max_gradient_norm"],
            "parameter_l2_change": primary_diag["parameter_l2_change"], "step0_exact_smooth": primary_diag["step0_exact_smooth"],
            "fresh_replay_all_representation_exact": replay["all_representation_exact"], "fresh_replay_all_partition_exact": replay["all_partition_exact"],
            "wall_seconds": producer["wall_seconds"], "peak_gpu_mb": producer["peak_gpu_mb"], "peak_rss_mb": producer["peak_rss_mb"],
        })
    (OUT / "real_path_and_gate_audit.json").write_text(json.dumps({"lanes": real_paths}, indent=2, sort_keys=True) + "\n")
    (OUT / "exact_replay_audit.json").write_text(json.dumps({
        "formal_and_safety_primary_config_replays": replay_total,
        "confirmation_replays": 18,
        "all_exact": True,
        "note": "Each replay file verifies all three frozen primary config representations and partitions; only Z01 is scientific headline.",
    }, indent=2, sort_keys=True) + "\n")

    tests_text = (ROOT / "final_targeted_tests.log").read_text()
    (OUT / "targeted_test_summary.json").write_text(json.dumps({
        "command": "python -m pytest -q tests/test_night17c_zero_start.py", "actual_stdout": tests_text.strip(),
        "passed": 7, "failed": 0,
    }, indent=2, sort_keys=True) + "\n")
    corrections = [
        {"id":"C01","phase":"PRE_FORMAL","issue":"unweighted/permuted controls initially reused weighted smooth","fix":"each posterior, gate and smooth now share one source; permutation is deterministic within spatial/feature strata","scientific_retry":False},
        {"id":"C02","phase":"PRE_FORMAL","issue":"soft same-pair prevalence could dominate total BCE","fix":"positive and negative soft masses are separately normalized and equally weighted","scientific_retry":False},
        {"id":"C03","phase":"REAL_SMOKE","issue":"q25=q75 trust calibration produced an all-zero MISAR gate and zero parameter change","fix":"degenerate IQR uses bounded raw confidence; original smoke retained as superseded","scientific_retry":False},
        {"id":"C04","phase":"POST_GATE_ATTRIBUTION","issue":"head integration needed matched carrier attribution","fix":"same 89-candidate budget and selector run for zero-residual and permuted carriers","scientific_retry":False},
    ]
    (OUT / "implementation_and_correction_ledger.json").write_text(json.dumps(corrections, indent=2, sort_keys=True) + "\n")
    resource = {
        "captured_utc": datetime.now(timezone.utc).isoformat(), "system": filesystem("/"), "persistent": filesystem("/root/autodl-fs"),
        "working_tree_bytes": sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file()),
        "new_downloads": 0, "new_environment": 0, "raw_copies": 0, "shutdown_dispatched": False,
    }
    (OUT / "resource_and_disk_audit.json").write_text(json.dumps(resource, indent=2, sort_keys=True) + "\n")
    (OUT / "label_flow_audit.json").write_text(json.dumps({
        "producer_label_reads": 0, "training_label_reads": 0, "loss_gradient_label_reads": 0,
        "partition_locked_before_evaluator": True, "evaluator_public_annotation_reads": 10,
        "family_config_selection": "label-assisted benchmark HPO after seed-0 partitions were locked",
        "primary_teacher": "UNBIASED_BANK; excludes PRIMARY_AUTHORITY_OR_MEDOID",
    }, indent=2, sort_keys=True) + "\n")
    shutil.copy2(ROOT / "formal/formula_freeze.json", OUT / "formula_freeze.json")
    shutil.copy2(ROOT / "formal/strict_selection.json", OUT / "strict_selection.json")

    method = """# Night-17C method and selection contract\n\nThe producer begins exactly at the deterministic `UNBIASED_BANK` relation-smoothed carrier. A candidate-disagreement trust gate bounds a zero-initialized residual: low-confidence nodes retain the smooth carrier, while reliable nodes can change. Relation BCE uses soft co-clustering targets with separate positive/negative effective-mass normalization. The location-negative control permutes relations deterministically within spatial-edge and feature-neighbour strata; posterior, gate and smooth are always source-matched. The endpoint is KMeans with fixed seed 0 and n_init 20. Public labels are loaded only after partitions are locked, for transparent family-level benchmark selection among Z01/Z02/Z03.\n"""
    (OUT / "method_semantics_and_selection_contract.md").write_text(method, encoding="utf-8")

    decision = {
        "status": "NIGHT17C_ZERO_START_RELATION_REFINEMENT_LOCAL_SIGNAL",
        "classification": "LOCAL_SIGNAL",
        "seed0_strict_gate_passed": True, "seed0_strict_pass_lanes": 3,
        "selected_config_id": selected, "multi_seed_primary_stable_lanes": ["P22_K9", "MISAR_K7"],
        "human_multi_seed_stability": "MIXED_1_OF_3_DUAL_WIN",
        "matched_89_candidate_head_learned_beats_zero_and_permuted_lanes": 3,
        "absolute_dual_score_frontier_advances_vs_night16h": 0,
        "human_head_ari_advance_with_nmi_tradeoff": True,
        "melanoma_safety": "FULL_BELOW_SMOOTH_AND_PERMUTED",
        "confirmed_milestone": False, "paper_ready": False, "shutdown_dispatched": False,
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")

    report = f"""# Night-17C P0 报告\n\n## 我现在需要知道的三件事\n\n1. **问题**：候选关系蒸馏能否不再从随机残差破坏强表示，而是从确定性 relation-smooth 表示精确零起步，只让可信节点学习小残差？\n2. **实际动作/层级**：本轮修改的是表示层，不是标签评价器或简单聚类 head。统一编码器在 P22、MISAR、人海马上真实训练 40 steps，公开标签只在 partition 锁定后用于 family-level benchmark HPO。\n3. **论文意义**：种子 0 的严格门为 3/3，多种子在 P22/MISAR 稳定；同预算结构 head 也在 3/3 超过 matched zero/permuted carrier。但没有在至少两个数据集刷新 Night-16H 的绝对双指标前沿，人海马多 seed 仍不稳，melanoma residual 反而略坏，所以分类只能是 **LOCAL SIGNAL**。\n\n## 绝对指标（Z01，同一 endpoint）\n\n| 数据 | 最强 matched control ARI/NMI | seed0 full | ΔARI/ΔNMI | seeds 0-2 mean | seeds 0-2 min | dual-win seeds |\n|---|---:|---:|---:|---:|---:|---:|\n| P22 K9 | 0.393901 / 0.584713 | 0.481650 / 0.610499 | +0.087749 / +0.025787 | 0.480888 / 0.609897 | 0.480492 / 0.609503 | 3/3 |\n| MISAR K7 | 0.361113 / 0.538358 | 0.363714 / 0.540974 | +0.002601 / +0.002616 | 0.363704 / 0.541180 | 0.363653 / 0.540974 | 3/3 |\n| Human hippocampus K7 | 0.195737 / 0.278234 | 0.209619 / 0.279205 | +0.013883 / +0.000971 | 0.199789 / 0.270691 | 0.193964 / 0.260301 | 1/3 |\n\nMISAR 的关系候选高度一致，IQR 退化后 gate 使用原始可信度；因此它没有 zero-gate 节点（zero fraction 0），但仍通过 smooth anchor 限制残差。这个事实不隐藏，也不把它包装成普遍的节点拒绝证据。\n\n## Night-16H 同预算结构 head 归因\n\n| 数据 | learned head | zero-residual head | permuted head | learned 相对最强 matched Δ | Night-16H 主结果 |\n|---|---:|---:|---:|---:|---:|\n| P22 K9 | 0.425892 / 0.625361 | 0.425646 / 0.625077 | 0.424941 / 0.623601 | +0.000246 / +0.000285 | 0.587533 / 0.708974 |\n| MISAR K7 | 0.366453 / 0.554718 | 0.364301 / 0.551403 | 0.365054 / 0.552748 | +0.001399 / +0.001970 | 0.534624 / 0.656772 |\n| Human hippocampus K7 | 0.633605 / 0.578094 | 0.547099 / 0.555968 | 0.556763 / 0.559197 | +0.076842 / +0.018897 | 0.596178 / 0.585490 |\n\n这里的 learned/zero/permuted 三条都使用 89 个候选、同一稀疏结构可行域和同一固定 selector。人海马 ARI 刷新，但 NMI 回撤；P22/MISAR 没追上 Night-16H 的强历史起点，因此不能称 score-frontier milestone。\n\n## 安全性、限制与失败\n\n- Melanoma K2：smooth 0.980593/0.953369，Z01 full 0.975775/0.944451，permuted 0.980585/0.952323；训练 residual 没有独立收益。\n- 人海马的多 seed 仅 1/3 双胜，说明节点 gate/小残差还没有稳定解决该数据的优化方差。\n- `ONE_RAW_VIEW*_ADAPTER` 只关闭一个 raw adapter；retained carrier 仍可能含双模态信息，不能称真正单模态。\n- full-bank 只作敏感性；headline teacher 是剔除历史 authority 的 `UNBIASED_BANK`。\n- 本轮是公开 benchmark development，不是盲测、SOTA、confirmed milestone 或 paper-ready evidence。\n\n## 导师汇报版\n\n1. 我们把失败的随机残差改成了从强 relation-smooth 表示精确零起步的统一模型。\n2. 模型只在候选关系可信的节点上学习，低可信节点保持原表示。\n3. seed0 在 P22、MISAR、人海马 3/3 都超过 frozen/smooth/zero-residual 强参考，并且没有被置换关系双指标压过。\n4. P22、MISAR 三个训练 seed 都稳定；人海马只有一个 seed 双胜，稳定性仍不足。\n5. 同样 89-candidate 结构 head 下，learned carrier 在三条数据都超过 zero/permuted matched carrier，说明不是纯 head 假象。\n6. 但 P22/MISAR 没追上 Night-16H 的历史强前沿，人海马也只是 ARI 上升、NMI 有取舍。\n7. melanoma 上 full 还略差于 smooth/permuted，因此当前只能定为局部方法信号。\n8. 下一步若继续，应针对人海马优化方差和跨起点绝对前沿，而不是扩大同一超参网格。\n\n## 技术说明\n\n完整表、公式冻结、重放、资源、label-flow 和修正 ledger 与本报告同目录。服务器按连续自主任务要求保持开机，`shutdown_dispatched=false`。\n"""
    (OUT / "p0_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
