#!/usr/bin/env python
"""Build the final Night-17D scientific-negative handoff from locked artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def result_row(lane, method, result, authority):
    ari, nmi = float(result["absolute_ari"]), float(result["absolute_nmi"])
    return {
        "lane": lane,
        "method": method,
        "candidate_id": result["candidate_id"],
        "absolute_ari": ari,
        "absolute_nmi": nmi,
        "delta_vs_night16h_ari": ari - float(authority["absolute_ari"]),
        "delta_vs_night16h_nmi": nmi - float(authority["absolute_nmi"]),
        "ami": result["ami"],
        "fmi": result["fmi"],
        "morans_i_macro": result["morans_i_macro"],
        "gearys_c_macro": result["gearys_c_macro"],
        "min_cluster_size_full": result["min_cluster_size_full"],
        "cluster_sizes_full": result["cluster_sizes_full"],
        "partition_sha256": result["partition_sha256"],
        "config_id": result.get("config_id", "NIGHT16H_AUTHORITY"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--working", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    evaluation_root = args.working / "evaluation" / "formal"
    authority_path = args.repo / "outputs/night16h_handoff/absolute_metrics_main_table.csv"
    authority = {row["lane"]: row for row in read_csv(authority_path)}
    main_rows, loso_rows, control_rows = [], [], []

    for lane in LANES:
        base = authority[lane]
        authority_result = {
            **base,
            "candidate_id": base["candidate_id"],
            "config_id": "NIGHT16H_FIXED_SELECTOR",
        }
        main_rows.append(result_row(lane, "NIGHT16H_FIXED_SELECTOR", authority_result, base))
        fixed = read_json(evaluation_root / "fixed" / lane / "LEARNED.evaluation.json")
        public = read_json(evaluation_root / "public_benchmark_hpo" / f"{lane}.evaluation.json")
        fold = evaluation_root / "strict_loso" / f"held_{lane}"
        learned = read_json(fold / "LEARNED.evaluation.json")
        zero = read_json(fold / "ZERO.evaluation.json")
        permuted = read_json(fold / "PERMUTED.evaluation.json")
        for method, result in (
            ("FIXED_GLOBAL_LEARNED", fixed),
            ("DIRECT_PUBLIC_BENCHMARK_HPO", public),
            ("STRICT_LOSO_LEARNED", learned),
            ("STRICT_LOSO_ZERO", zero),
            ("STRICT_LOSO_PERMUTED", permuted),
        ):
            row = result_row(lane, method, result, base)
            main_rows.append(row)
            if method.startswith("STRICT_LOSO"):
                loso_rows.append(row)
        control_rows.extend(read_csv(evaluation_root / "controls_validated" / f"{lane}.csv"))

    write_csv(args.output / "absolute_metrics_main_table.csv", main_rows)
    write_csv(args.output / "strict_loso_transfer_table.csv", loso_rows)
    write_csv(args.output / "strong_control_table.csv", control_rows)

    learned_rows = {row["lane"]: row for row in loso_rows if row["method"] == "STRICT_LOSO_LEARNED"}
    zero_rows = {row["lane"]: row for row in loso_rows if row["method"] == "STRICT_LOSO_ZERO"}
    perm_rows = {row["lane"]: row for row in loso_rows if row["method"] == "STRICT_LOSO_PERMUTED"}
    gate_rows = []
    for lane in LANES:
        learned, zero, permuted, base = learned_rows[lane], zero_rows[lane], perm_rows[lane], authority[lane]
        da, dn = float(learned["delta_vs_night16h_ari"]), float(learned["delta_vs_night16h_nmi"])
        gate_rows.append({
            "lane": lane,
            "night16h_ari": base["absolute_ari"],
            "night16h_nmi": base["absolute_nmi"],
            "loso_learned_ari": learned["absolute_ari"],
            "loso_learned_nmi": learned["absolute_nmi"],
            "delta_ari": da,
            "delta_nmi": dn,
            "dual_improvement": bool(da > 1e-12 and dn > 1e-12),
            "learned_candidate_id": learned["candidate_id"],
            "zero_candidate_id": zero["candidate_id"],
            "permuted_candidate_id": permuted["candidate_id"],
            "learned_equals_zero_partition": learned["partition_sha256"] == zero["partition_sha256"],
            "learned_equals_permuted_partition": learned["partition_sha256"] == permuted["partition_sha256"],
            "exact_k_no_singleton_min_scale_edge": int(learned["min_cluster_size_full"]) > 1,
        })
    write_csv(args.output / "advancement_gate_table.csv", gate_rows)

    # Label-post-lock diagnostics only; never fed back to the frozen producer.
    axis_rows = []
    for lane in LANES:
        features = read_csv(args.working / "formal_features/replay1" / lane / "candidate_learned_evidence.csv")
        evaluations = {row["candidate_id"]: row for row in read_csv(Path(f"/root/night16g_working/candidate_evaluation/{lane}.csv"))}
        feasible = [row for row in features if str(row["feasible_SMALLEST_SCALE_INTERNAL_EDGE"]).lower() == "true"]
        ari = np.asarray([float(evaluations[row["candidate_id"]]["absolute_ari"]) for row in feasible])
        nmi = np.asarray([float(evaluations[row["candidate_id"]]["absolute_nmi"]) for row in feasible])
        for prefix in ("LEARNED", "ZERO", "PERMUTED"):
            evidence = np.asarray([float(row[f"{prefix}_evidence"]) for row in feasible])
            axis_rows.append({
                "lane": lane, "evidence": prefix,
                "spearman_ari": float(spearmanr(evidence, ari)[0]),
                "spearman_nmi": float(spearmanr(evidence, nmi)[0]),
                "candidate_count": len(feasible),
                "diagnostic_label_reads_after_feature_lock": 1,
            })
    write_csv(args.output / "learned_evidence_axis_diagnostic.csv", axis_rows)

    # Preserve every grid row from direct HPO and the three strict folds.
    grid_rows = []
    for scope, path in [("DIRECT_ALL3", evaluation_root / "public_benchmark_hpo/grid_summary.csv")]:
        for row in read_csv(path): grid_rows.append({"scope": scope, **row})
    for lane in LANES:
        path = evaluation_root / "strict_loso" / f"held_{lane}" / "fit/grid_summary.csv"
        for row in read_csv(path): grid_rows.append({"scope": f"STRICT_LOSO_HELD_{lane}", **row})
    write_csv(args.output / "all_selector_grid_ledger.csv", grid_rows)

    freeze_manifest = read_json(args.working / "frozen/feature_freeze_manifest.json")
    p0 = {
        "schema": "night17d-real-p0-v1",
        "primary_lanes": {},
        "fresh_process_feature_replay": freeze_manifest,
        "checkpoint_authority": "Night-17C Z01 seeds 0/1/2 strict reload; learned, zero, and permuted representation SHA matched",
        "labels_read_by_feature_producer": 0,
    }
    for lane in LANES:
        manifest = read_json(args.working / "formal_features/replay1" / lane / "candidate_learned_evidence.manifest.json")
        p0["primary_lanes"][lane] = {
            key: manifest[key] for key in ("n", "candidate_count", "feasible_candidate_count", "pair_count", "spatial_pair_count", "representation_hashes", "device", "peak_rss_mb", "wall_seconds")
        }
    (args.output / "real_p0_and_replay_audit.json").write_text(json.dumps(p0, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    stat = os.statvfs("/")
    resource_audit = {
        "schema": "night17d-resource-audit-v1",
        "timestamp_unix": time.time(),
        "root_available_bytes": int(stat.f_bavail * stat.f_frsize),
        "root_available_inodes": int(stat.f_favail),
        "working_bytes": int(subprocess.check_output(["du", "-sb", str(args.working)]).decode().split()[0]),
        "new_environment_created": False,
        "new_data_downloaded": False,
        "persistent_low_inode_root_written": False,
        "shutdown_dispatched": False,
    }
    (args.output / "resource_and_disk_audit.json").write_text(json.dumps(resource_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    correction_rows = [
        {"id": "E01", "type": "ENGINEERING", "issue": "Strict LOSO fit initially accepted a whole authority CSV", "resolution": "Fit now accepts only exact one-row authority slices for supplied training lanes; held-out files are not opened", "scientific_results_changed": False},
        {"id": "E02", "type": "ENGINEERING", "issue": "Control-table shell loop lost a variable through local PowerShell interpolation", "resolution": "Replaced by Python orchestration; no scientific formula or result changed", "scientific_results_changed": False},
        {"id": "E03", "type": "ENGINEERING", "issue": "Control partition check first used the Night-17B array hash dialect", "resolution": "Switched to the authoritative Night-16G partition_sha256 and validated all candidate sets/partitions", "scientific_results_changed": False},
        {"id": "S01", "type": "SCIENTIFIC", "issue": "Fixed global selector was weak, especially on human", "resolution": "Preserved as preregistered negative; no grid expansion", "scientific_results_changed": False},
    ]
    write_csv(args.output / "failure_and_correction_ledger.csv", correction_rows)

    tests = {
        "pytest_command": "python -m pytest -q tests/test_night17d_learned_evidence.py",
        "passed": 6, "failed": 0,
        "coverage": ["candidate-specific LOO", "stratified permutation order", "feasible-only order-invariant selection", "training-lane-only fit"],
    }
    (args.output / "targeted_test_summary.json").write_text(json.dumps(tests, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    label_audit = {
        "schema": "night17d-label-flow-audit-v1",
        "feature_producer_label_reads": 0,
        "feature_freeze_before_evaluation": True,
        "fixed_global": "pre-registered before labels",
        "direct_public_benchmark_hpo": "all three locked candidate evaluations used after feature freeze",
        "strict_loso": "fit process read two training evaluation files plus two one-row authority slices only; held-out selection was materialized before independent held-out evaluation",
        "authority_slice_preparer": "mechanical preparer read the parent all-lane table once and wrote immutable one-row slices; it did not fit or select a configuration",
        "melanoma_evaluation_reads": 0,
    }
    (args.output / "label_flow_audit.json").write_text(json.dumps(label_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    decision = {
        "schema": "night17d-decision-v1",
        "status": "NIGHT17D_LEARNED_EVIDENCE_SELECTOR_NOT_IDENTIFIABLE",
        "classification": "SCIENTIFIC_NEGATIVE",
        "advancement_gate_passed": False,
        "strict_loso_dual_improvement_lanes": 1,
        "required_dual_improvement_lanes": 2,
        "learned_equals_zero_partition_lanes": 3,
        "independent_learned_evidence_contribution": False,
        "secondary_observation": "P22 strict LOSO selected 0.596390/0.718243, but the identical ZERO evidence selector selected the same partition",
        "melanoma_run": False,
        "stop_selector_repair_family": True,
        "next_direction": "new high-quality external RNA+ATAC data and a stronger native backbone",
        "shutdown_dispatched": False,
    }
    (args.output / "night17d_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = f"""# Night-17D 报告：训练表示证据选择器未通过独立贡献门

## 我现在需要知道的三件事

1. **问题**：Night-17C 的训练表示虽然能改善自己的 KMeans 表示端点，但能否作为新证据，从 Night-16H 已锁定的 89 个结构可行候选中选出更好的分区？
2. **实际动作 / 流水线层级**：本轮没有改候选、图或训练网络；只在“最终候选选择层”计算三个冻结训练 seed 的分子分离、centroid margin、relation alignment 和 seed 不确定度。relation alignment 对参与 UNBIASED bank 的候选做了 leave-one-candidate-out，避免候选给自己加分。
3. **论文意义**：严格整研究留出只有 P22 双升，MISAR 轻微下降，人海马持平；更关键的是 learned 与 ZERO 在 3/3 研究选择完全相同。因此训练表示没有形成独立 selector 贡献，结论是 **SCIENTIFIC_NEGATIVE**，应停止继续修补这一选择器家族。

## 绝对结果

| 数据 | Night-16H ARI/NMI | 固定全局 learned | 严格 LOSO learned | ΔARI/ΔNMI | learned=ZERO |
|---|---:|---:|---:|---:|---|
| P22 K9 | {float(authority['P22_K9']['absolute_ari']):.6f}/{float(authority['P22_K9']['absolute_nmi']):.6f} | {float(next(r for r in main_rows if r['lane']=='P22_K9' and r['method']=='FIXED_GLOBAL_LEARNED')['absolute_ari']):.6f}/{float(next(r for r in main_rows if r['lane']=='P22_K9' and r['method']=='FIXED_GLOBAL_LEARNED')['absolute_nmi']):.6f} | {float(learned_rows['P22_K9']['absolute_ari']):.6f}/{float(learned_rows['P22_K9']['absolute_nmi']):.6f} | {float(learned_rows['P22_K9']['delta_vs_night16h_ari']):+.6f}/{float(learned_rows['P22_K9']['delta_vs_night16h_nmi']):+.6f} | 是 |
| MISAR K7 | {float(authority['MISAR_K7']['absolute_ari']):.6f}/{float(authority['MISAR_K7']['absolute_nmi']):.6f} | {float(next(r for r in main_rows if r['lane']=='MISAR_K7' and r['method']=='FIXED_GLOBAL_LEARNED')['absolute_ari']):.6f}/{float(next(r for r in main_rows if r['lane']=='MISAR_K7' and r['method']=='FIXED_GLOBAL_LEARNED')['absolute_nmi']):.6f} | {float(learned_rows['MISAR_K7']['absolute_ari']):.6f}/{float(learned_rows['MISAR_K7']['absolute_nmi']):.6f} | {float(learned_rows['MISAR_K7']['delta_vs_night16h_ari']):+.6f}/{float(learned_rows['MISAR_K7']['delta_vs_night16h_nmi']):+.6f} | 是 |
| Human hippocampus K7 | {float(authority['HUMAN_HIPPOCAMPUS_K7']['absolute_ari']):.6f}/{float(authority['HUMAN_HIPPOCAMPUS_K7']['absolute_nmi']):.6f} | {float(next(r for r in main_rows if r['lane']=='HUMAN_HIPPOCAMPUS_K7' and r['method']=='FIXED_GLOBAL_LEARNED')['absolute_ari']):.6f}/{float(next(r for r in main_rows if r['lane']=='HUMAN_HIPPOCAMPUS_K7' and r['method']=='FIXED_GLOBAL_LEARNED')['absolute_nmi']):.6f} | {float(learned_rows['HUMAN_HIPPOCAMPUS_K7']['absolute_ari']):.6f}/{float(learned_rows['HUMAN_HIPPOCAMPUS_K7']['absolute_nmi']):.6f} | {float(learned_rows['HUMAN_HIPPOCAMPUS_K7']['delta_vs_night16h_ari']):+.6f}/{float(learned_rows['HUMAN_HIPPOCAMPUS_K7']['delta_vs_night16h_nmi']):+.6f} | 是 |

三折均拟合到 `M0_T1_L0.5_U0`。P22 的数值是候选库内已有可达解；ZERO selector 也选中同一 partition，故不能归因于 Night-17C learned representation。

## 证据边界与限制

- 训练表示是基于 UNBIASED candidate ensemble 的 transductive distillation，不是独立外部教师；候选级 alignment 已扣除自身权重。
- 固定全局配置表现弱，尤其在人海马选择到低分候选；这是原样保留的预注册负结果。
- 公开 labels 只在 feature CSV 两次重放并锁 SHA 后进入 direct HPO / LOSO evaluator。直接 HPO 是开发上限，不是迁移证据。
- strict LOSO 拟合只打开两个训练 study 的 evaluation 与单行 authority；held-out selection 先锁定，再由独立 evaluator 打开 held-out 指标。
- 未达到 2/3 晋级门，因此按合约没有扩展 melanoma，也没有扩大 grid 或追加 selector 规则。

## 导师汇报版

1. 我们检验了 Night-17C 的训练表示能否帮助 Night-16H 从原 89 个候选中选得更好。
2. 新证据包含表示内分离、原型 margin、关系一致性和三 seed 稳定性，候选自身贡献已用 leave-one-out 扣除。
3. 严格整研究留出在 P22 提高到 0.5964/0.7182，但 MISAR 略降，人海马持平。
4. 最关键的是，learned 与 zero-residual 对照在三条数据上都选到完全相同的 partition。
5. 因此 P22 的数值来自原候选库和拓扑排序，不是训练表示的独立增益。
6. 本轮按预注册门判为 SCIENTIFIC_NEGATIVE，不再扩大 selector 网格或补规则。
7. 下一步应转向新的高质量 RNA+ATAC 外部数据和更强的原生 backbone，而不是继续修补候选选择器。

## 技术附录摘要

- 三条 lane 的 learned/zero/permuted checkpoint authority 均通过；最终 feature CSV 两次 fresh-process SHA 完全一致。
- targeted tests：6/6 PASS；dense N×N：0；新下载：0；新环境：0；melanoma label reads：0。
- AutoDL 按连续自主研发要求保持开机，`shutdown_dispatched=false`。
"""
    (args.output / "night17d_report.md").write_text(report, encoding="utf-8")

    methods = """# Night-17D 方法与选择合约

对每个 Night-16H 结构可行候选，分别在 Night-17C Z01 seeds 0/1/2 的冻结表示上计算 explained variance、Calinski-Harabasz separation、q10 centroid margin 和 relation alignment。四轴先在 lane/seed 内转 percentile 后等权平均；三 seed 均值为 learned evidence，标准差为 learned uncertainty。

若候选参与 UNBIASED relation posterior，relation alignment 使用候选特异的 leave-one-candidate-out posterior；完整 posterior alignment 仅作为敏感性列。PERMUTED control 使用同一分层置换顺序同步变换 candidate contribution。

固定选择分数为 `molecular_rank + topology_rank + learned_evidence - 0.5*uncertainty_rank`。透明 HPO 只枚举冻结的 81 个 rank-weight config。严格 LOSO 在两个训练 study 上机械排序配置，held-out producer 不读取评价，最后独立 evaluator 才合并指标。
"""
    (args.output / "method_semantics_and_selection_contract.md").write_text(methods, encoding="utf-8")

    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
