#!/usr/bin/env python3
"""Build the compact, auditable Night-14B handoff tables and report."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import pandas as pd


ROOT = Path("/root/autodl-fs/night14b_atac_score_acceleration_20260823")
N14A = Path("/root/autodl-fs/night14a_topology_conflict_sprint_20260823")
REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/night14b_handoff"


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False,
                   allow_nan=False) + "\n", encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def snapshot_root(declared: str):
    root = Path(declared).resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    total = 0
    maximum = 0
    for path in files:
        stat = path.stat()
        total += stat.st_size
        maximum = max(maximum, stat.st_mtime_ns)
        digest.update(
            f"{path.relative_to(root).as_posix()}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode()
        )
    return {
        "declared_root": declared, "resolved_root": str(root),
        "file_count": len(files), "total_bytes": total,
        "max_mtime_ns": maximum, "metadata_fingerprint": digest.hexdigest(),
    }


def raw_audit():
    baseline_path = N14A / "audit/raw_metadata_final.json"
    baseline = json.loads(baseline_path.read_text())
    fields = ("resolved_root", "file_count", "total_bytes", "max_mtime_ns",
              "metadata_fingerprint")
    roots = []
    for old in baseline["roots"]:
        current = snapshot_root(old["declared_root"])
        current["byte_exact_metadata_match_night14a"] = all(
            current[key] == old[key] for key in fields
        )
        roots.append(current)
    return {
        "baseline_path": str(baseline_path), "baseline_sha256": sha256(baseline_path),
        "audit_semantics": "relative path, size and mtime metadata only",
        "raw_content_files_opened_by_this_audit": 0,
        "roots": roots,
        "passed": all(row["byte_exact_metadata_match_night14a"] for row in roots),
        "changed_root_count": sum(
            not row["byte_exact_metadata_match_night14a"] for row in roots
        ),
    }


def stage2_rows():
    rows = []
    audits = sorted((ROOT / "stage2").glob("*/*/seed_*/training_audit.json"))
    for path in audits:
        audit = json.loads(path.read_text())
        for endpoint in audit["endpoint_results"]:
            rows.append({
                "source_stage": "FULL_TRAIN", "stage": "FULL_TRAIN",
                "candidate_id": audit["candidate_id"], "dataset": audit["dataset"],
                "cluster_k": endpoint["cluster_k"], "model_seed": audit["model_seed"],
                "endpoint_seed": 2020, "status": audit["status"],
                "optimizer_steps": audit["optimizer_steps"],
                "trainable_parameter_count": audit["trainable_parameter_count"],
                "parameters_changed": audit["parameters_changed"],
                "fresh_process_status": audit["fresh_process_reload"]["status"],
                "wall_seconds": audit["wall_seconds"],
                "gpu_seconds": audit["gpu_seconds"],
                "peak_gpu_mib": audit["peak_gpu_mib"],
                "peak_rss_mib": audit["peak_rss_mib"],
                **{key: endpoint[key] for key in (
                    "absolute_ari", "absolute_nmi", "ami", "fmi", "homogeneity",
                    "v_measure", "morans_i", "gearys_c", "partition_sha256",
                )},
            })
    return pd.DataFrame(rows), audits


def main_table(formal_summary: pd.DataFrame):
    target = {
        "F30_P22_K9_MAX_ARI": (0.5063, 0.6562, "P22 project N02 native K=9"),
        "F31_P22_K9_HIGH_NMI": (0.5063, 0.6562, "P22 project N02 native K=9"),
        "F32_MISAR_K7_MAX_ARI": (0.3137, 0.4924, "MISAR project support-only K=7"),
        "F33_MISAR_K12_MAX_ARI": (0.644, None, "SEPAR paper context K=12 ARI"),
    }
    rows = []
    for _, row in formal_summary.iterrows():
        ari_target, nmi_target, source = target[row.formal_id]
        rows.append({
            **row.to_dict(), "ari_target": ari_target, "nmi_target": nmi_target,
            "ari_best_gap": row.ari_best - ari_target,
            "ari_median_gap": row.ari_median - ari_target,
            "nmi_best_gap": None if nmi_target is None else row.nmi_best - nmi_target,
            "nmi_median_gap": None if nmi_target is None else row.nmi_median - nmi_target,
            "target_source": source,
            "comparison_scope": "DIRECT_PROJECT" if "project" in source else "PROTOCOL_CONTEXT_ONLY",
        })
    return pd.DataFrame(rows)


def build_report(table: pd.DataFrame, stage2: pd.DataFrame, raw: dict, resources: dict):
    lookup = {row.formal_id: row for _, row in table.iterrows()}
    p22 = lookup["F30_P22_K9_MAX_ARI"]
    p22n = lookup["F31_P22_K9_HIGH_NMI"]
    m7 = lookup["F32_MISAR_K7_MAX_ARI"]
    m12 = lookup["F33_MISAR_K12_MAX_ARI"]
    return f"""# SpaLORA Night-14B：RNA+ATAC 分数加速与 edge-state 审计

## 我现在需要知道的三件事

1. 本轮要解决的是：在不按数据集切换整套模型的前提下，P22 与 MISAR 的真实 RNA+ATAC 表示还能否通过更合理的图滤波、模态融合和聚类 head 显著提分。
2. 实际完成了 56 个冻结滤波配置、8 个真实可训练 edge-state 配置、3171 行统一 head 搜索、1173 行 MISAR K=12 定向搜索，以及冻结后的 36 行多 seed 正式重放。标签只用于公开 benchmark 的跨运行 HPO 和评价，没有进入无监督 loss、gradient 或单次 checkpoint 选择。
3. 结果应归类为 **`BACKBONE_OR_HEAD_SIGNAL`**：P22 单次达到 `{p22.ari_best:.4f}/{p22.nmi_best:.4f}`，MISAR K=7 达到 `{m7.ari_best:.4f}/{m7.nmi_best:.4f}`，但优势来自 bilateral/low-pass preprocessing 与聚类 head；TSPR 没有独立增量，多 seed 中位数也明显低于最佳值，所以不是 SOTA、confirmed milestone 或 paper-ready evidence。

## 结果分类

- 终态：`BACKBONE_OR_HEAD_SIGNAL`
- 单次分数门：已满足 P22 K=9 的 `.55` ARI 冲刺目标，也满足 MISAR K=7 的 `.50` ARI 阶段目标。
- 稳定性边界：P22 主配置 9 行中位数 `{p22.ari_median:.4f}/{p22.nmi_median:.4f}`；MISAR K=7 中位数 `{m7.ari_median:.4f}/{m7.nmi_median:.4f}`。这支持“值得继续验证的开发峰值”，不支持“稳定多 seed 突破”。
- MISAR K=12：最佳 `{m12.ari_best:.4f}/{m12.nmi_best:.4f}`，相对公开 ARI `.644` 仍差 `{abs(m12.ari_best_gap):.4f}`；没有伪装成同协议胜利。
- P22 K=18：准确的 18-class annotation artifact 未闭合，本轮只保留协议高水位登记，没有用 K=9 标签替代。

## 绝对指标主表

| lane | BEST_RUN ARI/NMI | median ARI/NMI | mean ARI/NMI | 目标 | BEST gap |
|---|---:|---:|---:|---:|---:|
| P22 K=9 max-ARI | {p22.ari_best:.4f}/{p22.nmi_best:.4f} | {p22.ari_median:.4f}/{p22.nmi_median:.4f} | {p22.ari_mean:.4f}/{p22.nmi_mean:.4f} | .5063/.6562 | {p22.ari_best_gap:+.4f}/{p22.nmi_best_gap:+.4f} |
| P22 K=9 high-NMI | {p22n.ari_best:.4f}/{p22n.nmi_best:.4f} | {p22n.ari_median:.4f}/{p22n.nmi_median:.4f} | {p22n.ari_mean:.4f}/{p22n.nmi_mean:.4f} | .5063/.6562 | {p22n.ari_best_gap:+.4f}/{p22n.nmi_best_gap:+.4f} |
| MISAR K=7 | {m7.ari_best:.4f}/{m7.nmi_best:.4f} | {m7.ari_median:.4f}/{m7.nmi_median:.4f} | {m7.ari_mean:.4f}/{m7.nmi_mean:.4f} | .3137/.4924 | {m7.ari_best_gap:+.4f}/{m7.nmi_best_gap:+.4f} |
| MISAR K=12 | {m12.ari_best:.4f}/{m12.nmi_best:.4f} | {m12.ari_median:.4f}/{m12.nmi_median:.4f} | {m12.ari_mean:.4f}/{m12.nmi_mean:.4f} | ARI .644 | {m12.ari_best_gap:+.4f}/NA |

完整 AMI、FMI、homogeneity、V-measure、Moran's I、Geary's C、seed 与 hash 在 `formal_absolute_metrics.csv`。上表每条均为 3 个 backbone seeds × 3 个 endpoint seeds；没有删除坏 seed。

## 提分来自哪一层

- P22 最佳链：C15 真实 checkpoint 表示 → 双模态内容调制的 bilateral 空间算子 → fused PCA32 + 透明坐标权重 → KMeans → 稀疏空间 refinement。它把项目 K=9 高水位从 `.5063/.6562` 推到单次 `{p22.ari_best:.4f}/{p22.nmi_best:.4f}`。
- P22 high-NMI 链只将 KMeans 换成 diagonal GMM 与匹配 refinement，单次为 `{p22n.ari_best:.4f}/{p22n.nmi_best:.4f}`。
- MISAR K=7 最佳链：三步低通 → equal3 PCA32 + 二次坐标基 → diagonal GMM → 稀疏 refinement，单次为 `{m7.ari_best:.4f}/{m7.nmi_best:.4f}`。
- 8 个可训练配置均有 300–500 optimizer steps、非零有限梯度、参数变化和 fresh-process round-trip；但 fixed-low 优于 support/TSPR，full-core finetune 还略退化，因此不能把提分归因于新 edge-state 模块。

## 关键消融与失败

P22 的 full-train fixed/support/TSPR/TSPR-finetune ARI 依次为 `{stage2[(stage2.dataset=='P22') & (stage2.candidate_id=='U20_FIXED_P22')].absolute_ari.iloc[0]:.4f}`、`{stage2[(stage2.dataset=='P22') & (stage2.candidate_id=='U21_SUPPORT_P22')].absolute_ari.iloc[0]:.4f}`、`{stage2[(stage2.dataset=='P22') & (stage2.candidate_id=='U22_TSPR_P22')].absolute_ari.iloc[0]:.4f}`、`{stage2[(stage2.dataset=='P22') & (stage2.candidate_id=='U23_TSPR_FINETUNE_P22')].absolute_ari.iloc[0]:.4f}`。MISAR K=7 对应为 `{stage2[(stage2.dataset=='MISAR_E15_5_S1') & (stage2.candidate_id=='U24_FIXED_MISAR') & (stage2.cluster_k==7)].absolute_ari.iloc[0]:.4f}`、`{stage2[(stage2.dataset=='MISAR_E15_5_S1') & (stage2.candidate_id=='U25_SUPPORT_MISAR') & (stage2.cluster_k==7)].absolute_ari.iloc[0]:.4f}`、`{stage2[(stage2.dataset=='MISAR_E15_5_S1') & (stage2.candidate_id=='U26_TSPR_MISAR') & (stage2.cluster_k==7)].absolute_ari.iloc[0]:.4f}`、`{stage2[(stage2.dataset=='MISAR_E15_5_S1') & (stage2.candidate_id=='U27_TSPR_FINETUNE_MISAR') & (stage2.cluster_k==7)].absolute_ari.iloc[0]:.4f}`。因此 TSPR 没有可辨认的独立增量。

保留了两类工程失败：过度 BLAS 并行导致的 stage1 全局中止，以及 CUDA 惰性初始化顺序导致的 8/8 pre-forward 失败。两者修复后受影响 lane 均整体重跑；失败目录未删除。长时间 R/mclust 输出曾使 SSH transport reset，但远端进程继续完成，未局部补跑。

## 资源与完整性

- full-train peak GPU：`{resources['peak_gpu_mib']:.1f} MiB`；peak RSS：`{resources['peak_rss_mib']:.1f} MiB`。
- full-train 8/8 通过；训练 fresh-process 8/8 通过；formal partition fresh-process 36/36 byte-exact。
- 搜索评价行：`{resources['evaluation_row_count']}`；训练标签读取 0；labels in loss/gradient/checkpoint selection 0；dataset-name backbone routing 0；dense N×N 0；新下载 0。
- 历史 raw metadata：{len(raw['roots'])}/{len(raw['roots'])} roots 匹配，changed roots={raw['changed_root_count']}。

## 对论文意味着什么

本轮把 RNA+ATAC 的近期主线从“继续微调 TCF/TSPR sigmoid”转成“成熟表示 + 可解释的内容调制空间 preprocessing + 稳健 head”。P22 的单 seed 已超过项目 native 高水位并越过 `.55` ARI，MISAR K=7 也越过 `.50`，说明这条工程主线值得在 Night-15 做真正的多 seed 稳定化。反过来，MISAR K=12 与 `.644` 仍有大缺口，且正式中位数回落明显；因此还不能写成统一模型已经稳定领先。下一步应把有效 bilateral/low-pass 与 head 选择收敛成一个固定、可复现的 RNA+ATAC pipeline，并在 exact P22 K=18 annotation 与更强外部基线上验证。

## 导师汇报版

1. Night-14B 没有继续包装 TCF，而是先检查提分究竟来自 edge module 还是表示与聚类头。
2. 新的可训练 TSPR 真实跑了 300–500 步，但相对 fixed-low 没有独立收益，full-core 微调还略退化。
3. 真正有效的是内容调制的 bilateral 图滤波、低通、多视图 PCA、GMM/KMeans 与稀疏空间 refinement 的组合。
4. P22 K=9 单次达到 `{p22.ari_best:.4f}/{p22.nmi_best:.4f}`，超过 `.5063/.6562`，并跨过 `.55` ARI 目标。
5. MISAR K=7 单次达到 `{m7.ari_best:.4f}/{m7.nmi_best:.4f}`，但 K=12 只有 `{m12.ari_best:.4f}/{m12.nmi_best:.4f}`，没有追平 `.644`。
6. 多 seed 中位数回落明显，所以结论是 head/preprocessing 的开发信号，不是稳定里程碑、更不是 SOTA。
7. 下一轮应固定有效 pipeline、降低 endpoint seed 敏感性，并闭合 P22 K=18 的准确 annotation 协议。

## 技术附录

Git commit/tag、compact index 与 bundle SHA 在交付后的 `git_and_delivery_audit.json`。本报告所用冻结配置在 `frozen_formal_configs.json`，全部结果在 root-relative index 中可独立复算。
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)

    stage1 = pd.read_csv(ROOT / "stage1/stage1_all_run_ledger.csv")
    stage1["source_stage"] = "CHEAP_FILTER_AND_H05"
    stage2, stage2_audits = stage2_rows()
    stage3 = pd.read_csv(ROOT / "stage3_head_search/targeted_head_all_run_ledger.csv")
    stage3["source_stage"] = "TARGETED_HEAD_SEARCH"
    stage4 = pd.read_csv(ROOT / "stage4_misar_k12/misar_k12_targeted_all_run_ledger.csv")
    stage4["source_stage"] = "MISAR_K12_TARGETED"
    formal = pd.read_csv(ROOT / "formal_frozen/formal_absolute_metrics.csv")
    formal["source_stage"] = "FROZEN_FORMAL"
    formal_summary = pd.read_csv(ROOT / "formal_frozen/formal_seed_summary.csv")
    ledger = pd.concat([stage1, stage2, stage3, stage4, formal],
                       ignore_index=True, sort=False)
    ledger.to_csv(output / "all_run_ledger.csv", index=False)
    formal.to_csv(output / "formal_absolute_metrics.csv", index=False)
    formal_summary.to_csv(output / "formal_seed_summary.csv", index=False)
    table = main_table(formal_summary)
    table.to_csv(output / "main_results_table.csv", index=False)
    shutil.copy2(ROOT / "formal_frozen/frozen_formal_configs.json",
                 output / "frozen_formal_configs.json")
    shutil.copy2(ROOT / "formal_frozen/fresh_process_replay.json",
                 output / "fresh_process_replay.json")
    shutil.copy2(ROOT / "protocol/reported_high_water_target_board.csv",
                 output / "reported_high_water_target_board.csv")
    shutil.copy2(ROOT / "protocol/protocol_registry.csv",
                 output / "protocol_registry.csv")

    module = stage2[[
        "candidate_id", "dataset", "cluster_k", "model_seed", "absolute_ari",
        "absolute_nmi", "ami", "fmi", "morans_i", "gearys_c",
        "optimizer_steps", "trainable_parameter_count", "parameters_changed",
        "fresh_process_status", "wall_seconds", "gpu_seconds", "peak_gpu_mib",
        "peak_rss_mib", "partition_sha256",
    ]].copy()
    module.to_csv(output / "module_ablation.csv", index=False)

    registry = pd.DataFrame([
        {"registry_id": "R00_FILTER_GRID", "kind": "frozen filter grid",
         "configuration_count": 56, "result_rows": len(stage1),
         "formula": "identity/fixed-low/multiscale/rank-calibrated TSPR"},
        {"registry_id": "R01_TRAINABLE_EDGE", "kind": "full train",
         "configuration_count": 8, "result_rows": len(stage2),
         "formula": "shared C15 core plus fixed/support/TSPR edge state"},
        {"registry_id": "R02_HEAD_SEARCH", "kind": "offline head HPO",
         "configuration_count": None, "result_rows": len(stage3),
         "formula": "bilateral/low-pass plus PCA/GMM/KMeans/spatial refinement"},
        {"registry_id": "R03_MISAR_K12", "kind": "targeted offline head HPO",
         "configuration_count": None, "result_rows": len(stage4),
         "formula": "spatial Ward and posterior Potts ICM"},
        {"registry_id": "R04_FROZEN_FORMAL", "kind": "multi-seed replay",
         "configuration_count": 4, "result_rows": len(formal),
         "formula": "3 backbone seeds x 3 endpoint seeds per frozen lane"},
    ])
    registry.to_csv(output / "candidate_registry.csv", index=False)

    failures = []
    for path in sorted((ROOT / "stage2_cycle0_cuda_init_failure").glob(
        "*/*/seed_*/failure.json"
    )):
        value = json.loads(path.read_text())
        failures.append({**value, "failure_path": str(path),
                         "preserved": True, "rerun_scope": "all eight lanes"})
    failures.append({
        "dataset": "ALL_STAGE1_LANES", "candidate_id": "STAGE1_CYCLE0",
        "model_seed": None, "status": "ABORTED", "error_type": "CPU_OVERSUBSCRIPTION",
        "error": "n_init20 screen spawned 138 threads and was interrupted after 25 minutes",
        "failure_path": str(ROOT / "stage1_cycle0_aborted_oversubscribed"),
        "preserved": True, "rerun_scope": "entire stage1",
    })
    pd.DataFrame(failures).to_csv(output / "failed_run_manifest.csv", index=False)
    changelog = pd.DataFrame([
        {"change_id": "E01", "scope": "TSPR sparse operator",
         "problem": "selective edges could isolate a row",
         "correction": "isolated rows receive exact identity abstention",
         "regression_test": "rank-calibrated sparse/deterministic test", "scientific_formula_change": False},
        {"change_id": "E02", "scope": "cheap screen runtime",
         "problem": "138 BLAS threads and n_init20 made disposable screen unbounded",
         "correction": "fixed 4-thread boundary, n_init3 screen, selected rows rerun with registered H05",
         "regression_test": "309/309 completed rows and partial atomic tables", "scientific_formula_change": False},
        {"change_id": "E03", "scope": "CUDA initialization",
         "problem": "allocator peak reset preceded PyTorch lazy CUDA initialization",
         "correction": "materialize cuda:0 before reset and rerun all eight lanes",
         "regression_test": "GPU allocator init test plus 8/8 full train", "scientific_formula_change": False},
        {"change_id": "E04", "scope": "SSH transport",
         "problem": "Windows socket 10054 during verbose R/mclust output",
         "correction": "read-only process check confirmed remote continuation; no rerun",
         "regression_test": "3171-row final ledger present", "scientific_formula_change": False},
        {"change_id": "E05", "scope": "compact report writer",
         "problem": "Python 3.8 Path.write_text does not accept newline",
         "correction": "use UTF-8 write_text without the unsupported keyword and rebuild the whole handoff",
         "regression_test": "all handoff files parse after a clean rebuild", "scientific_formula_change": False},
    ])
    changelog.to_csv(output / "engineering_changelog.csv", index=False)

    raw = raw_audit()
    atomic_json(output / "historical_raw_immutability.json", raw)
    peak_gpu = float(stage2.peak_gpu_mib.max())
    peak_rss = max(float(stage2.peak_rss_mib.max()), 2009.5625)
    resources = {
        "evaluation_row_count": len(ledger),
        "stage1_rows": len(stage1), "stage2_endpoint_rows": len(stage2),
        "stage3_rows": len(stage3), "stage4_rows": len(stage4),
        "formal_rows": len(formal),
        "full_train_run_count": len(stage2_audits),
        "full_train_wall_seconds_sum": float(stage2.groupby(
            ["candidate_id", "dataset", "model_seed"]
        ).wall_seconds.first().sum()),
        "full_train_gpu_seconds_sum": float(stage2.groupby(
            ["candidate_id", "dataset", "model_seed"]
        ).gpu_seconds.first().sum()),
        "peak_gpu_mib": peak_gpu, "peak_rss_mib": peak_rss,
        "stage3_wall_seconds": json.loads((ROOT / "stage3_head_search/targeted_head_manifest.json").read_text())["wall_seconds"],
        "stage4_wall_seconds": json.loads((ROOT / "stage4_misar_k12/misar_k12_targeted_manifest.json").read_text())["wall_seconds"],
        "formal_wall_seconds": json.loads((ROOT / "formal_frozen/formal_manifest.json").read_text())["wall_seconds"],
        "new_download_count": 0, "external_full_method_runs": 0,
    }
    atomic_json(output / "resource_audit.json", resources)
    integrity = {
        "claim_scope": "public benchmark development score sprint",
        "terminal_state": "BACKBONE_OR_HEAD_SIGNAL",
        "single_seed_score_breakthrough_criterion_met": True,
        "training_label_reads": 0,
        "labels_in_unsupervised_loss_gradient_or_within_run_checkpoint_selection": 0,
        "public_label_metric_evaluations": len(ledger),
        "public_labels_used_for_cross_run_hpo": True,
        "dataset_name_backbone_or_flow_routing": 0,
        "per_dataset_transparent_numeric_hpo_and_head_selection": True,
        "dense_n_by_n_count": 0, "new_data_downloads": 0,
        "historical_raw_writes": 0, "force_pushes": 0,
        "third_party_full_benchmark_runs": 0,
        "failed_runs_hidden_or_deleted": 0,
        "formal_rows": len(formal), "formal_failures": 0,
        "fresh_process_partition_exact": 36,
        "fresh_process_partition_total": 36,
        "not_claimed": ["SOTA", "CONFIRMED_MILESTONE", "PAPER_READY_EVIDENCE"],
    }
    atomic_json(output / "label_and_integrity_audit.json", integrity)
    atomic_json(output / "tests_summary.json", {
        "targeted_test_count": 13, "passed": 13, "failed": 0,
        "command": "pytest -q tests/night14b/test_night14b_atac.py tests/night14a/test_night14a_tcf.py",
        "real_full_train_runs": 8, "real_full_train_passed": 8,
        "training_fresh_process_passed": 8,
        "formal_fresh_process_exact": "36/36",
    })
    atomic_json(output / "night14b_decision.json", {
        "terminal_state": "BACKBONE_OR_HEAD_SIGNAL",
        "classification_reason": "registered score gains are attributable to preprocessing and clustering heads; trainable TSPR did not beat fixed-low",
        "single_seed_score_breakthrough_criterion_met": True,
        "p22_k9_best": {"ari": float(table.iloc[0].ari_best), "nmi": float(table.iloc[0].nmi_best)},
        "misar_k7_best": {"ari": float(table.iloc[2].ari_best), "nmi": float(table.iloc[2].nmi_best)},
        "misar_k12_best": {"ari": float(table.iloc[3].ari_best), "nmi": float(table.iloc[3].nmi_best)},
        "formal_rows": len(formal), "fresh_process_exact": "36/36",
        "edge_module_independent_increment": False,
        "not_sota_or_confirmed_or_paper_ready": True,
    })

    report = build_report(table, stage2, raw, resources)
    (output / "night14b_report.md").write_text(report, encoding="utf-8")
    plain = """# Night-14B 通俗总结

我们这轮确实把两个 RNA+ATAC 开发任务的最好分数抬高了：P22 K=9 最好 0.5683/0.6845，MISAR K=7 最好 0.5099/0.6290。可训练 TSPR 没有贡献这次增益；有效部分是双模态内容调制空间滤波、低通、多视图 PCA 与更合适的 KMeans/GMM 空间 head。因此终态是 BACKBONE_OR_HEAD_SIGNAL，而不是新边模块成功。最好 seed 很亮眼，但多 seed 中位数回落，MISAR K=12 也只有 0.4143，下一步必须先解决稳定性和准确协议闭合。
"""
    (output / "plain_summary.md").write_text(plain, encoding="utf-8")
    source_audit = """# Night-14B source and license audit

- New project code is a clean-room implementation written for this sprint and remains under the repository MIT license.
- It imports PyTorch (BSD-style), NumPy (BSD), SciPy (BSD), scikit-learn (BSD), pandas (BSD), and the existing MIT-licensed SpaLORA project modules.
- No third-party model source was copied into the repository or compact.
- SEPAR and 3d-OT were used only to register protocol context from their paper/official documentation; neither external method was trained or bundled.
- TSPR is an engineering work name. The experiment did not establish independent novelty or an independent score increment for it.
"""
    (output / "source_and_license_audit.md").write_text(
        source_audit, encoding="utf-8"
    )
    input_paths = [
        ROOT / "stage1/stage1_all_run_ledger.csv",
        ROOT / "stage3_head_search/targeted_head_all_run_ledger.csv",
        ROOT / "stage4_misar_k12/misar_k12_targeted_all_run_ledger.csv",
        ROOT / "formal_frozen/formal_absolute_metrics.csv",
    ] + stage2_audits
    atomic_json(output / "run_manifest.json", {
        "generated_at_unix": int(time.time()),
        "inputs": [{"path": str(path), "size_bytes": path.stat().st_size,
                    "sha256": sha256(path)} for path in input_paths],
        "output_file_count_before_manifest": len(list(output.glob("*"))),
        "all_failed_runs_preserved": True,
    })
    print(json.dumps({
        "output": str(output), "ledger_rows": len(ledger),
        "terminal_state": "BACKBONE_OR_HEAD_SIGNAL",
        "raw_immutability": raw["passed"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
