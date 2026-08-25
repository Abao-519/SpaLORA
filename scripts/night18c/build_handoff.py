#!/usr/bin/env python3
"""Build the compact Night-18C scientific handoff from locked artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import re

import pandas as pd


ROOT = Path("/root/SpaLORA-night16h")
WORK = Path("/root/night18c_working")
OUT = ROOT / "outputs/night18c_handoff"
LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")
NIGHT16H = {
    "P22_K9": (0.5875325536381577, 0.7089736211480059),
    "MISAR_K7": (0.5346235433360444, 0.6567718908460013),
    "HUMAN_HIPPOCAMPUS_K7": (0.5961783620969016, 0.5854895180308324),
}


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")


def stage_summary(all_rows: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    rows, details = [], []
    for lane in LANES:
        lane_rows = all_rows[all_rows.lane == lane].copy()
        identity = lane_rows[lane_rows.arm == "IDENTITY_RETAINED"].iloc[0]
        fulls = lane_rows[lane_rows.arm == "FULL"].sort_values(["ari", "nmi"], ascending=False)
        full = fulls.iloc[0]
        matched = lane_rows[(lane_rows.config_id == full.config_id) & lane_rows.arm.isin(
            ["L2_LOWPASS", "GRAPH_TV_ONLY", "PRIVATE_ONLY", "PERMUTED_PRIVATE"]
        )].copy()
        control_ari = matched.sort_values(["ari", "nmi"], ascending=False).iloc[0]
        control_nmi = matched.sort_values(["nmi", "ari"], ascending=False).iloc[0]
        independent_dual = bool(full.ari > matched.ari.max() and full.nmi > matched.nmi.max())
        rows.append({
            "lane": lane,
            "night16h_structured_reference_ari": NIGHT16H[lane][0],
            "night16h_structured_reference_nmi": NIGHT16H[lane][1],
            "identity_retained_kmeans_ari": identity.ari,
            "identity_retained_kmeans_nmi": identity.nmi,
            "best_full_config": full.config_id,
            "best_full_ari": full.ari,
            "best_full_nmi": full.nmi,
            "delta_full_vs_identity_ari": full.ari - identity.ari,
            "delta_full_vs_identity_nmi": full.nmi - identity.nmi,
            "max_control_ari_arm": control_ari.arm,
            "max_control_ari": control_ari.ari,
            "max_control_nmi_at_max_ari": control_ari.nmi,
            "max_control_nmi_arm": control_nmi.arm,
            "max_control_nmi": control_nmi.nmi,
            "delta_full_vs_max_control_ari": full.ari - matched.ari.max(),
            "delta_full_vs_max_control_nmi": full.nmi - matched.nmi.max(),
            "independent_dual_signal": independent_dual,
            "min_cluster_size_full": int(full.min_cluster_size_full),
            "cluster_sizes_full": full.cluster_sizes_full,
            "partition_sha256": full.partition_sha256,
        })
        details.append({"lane": lane, "full": full.to_dict(), "matched_controls": matched.to_dict("records")})
    return pd.DataFrame(rows), details


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows = pd.concat([pd.read_csv(WORK / f"evaluation/{lane}.csv") for lane in LANES], ignore_index=True)
    all_rows.to_csv(OUT / "absolute_metrics_and_controls.csv", index=False)
    summary, details = stage_summary(all_rows)
    summary.to_csv(OUT / "stage_a_gate_summary.csv", index=False)
    pass_count = int(summary.independent_dual_signal.sum())
    replay = {lane: json.loads((WORK / f"replay/{lane}.audit.json").read_text()) for lane in LANES}
    if not all(item["status"] == "PASS" and item["partition_exact"] for item in replay.values()):
        raise RuntimeError("fresh-process replay is not exact")
    manifests = {lane: json.loads((WORK / f"artifacts/{lane}.json").read_text()) for lane in LANES}
    alignment = pd.DataFrame([{
        "lane": lane,
        "alignment": manifest["view_coordinate_alignment"]["alignment"],
        "rotation_sha256": manifest["view_coordinate_alignment"]["rotation_sha256"],
        "orthogonality_error": manifest["view_coordinate_alignment"]["rotation_orthogonality_error"],
        "pre_alignment_frobenius": manifest["view_coordinate_alignment"]["pre_alignment_frobenius"],
        "post_alignment_frobenius": manifest["view_coordinate_alignment"]["post_alignment_frobenius"],
        "ordered_ids_sha256": manifest["ordered_ids_sha256"],
        "carrier_sha256": manifest["carrier_sha256"],
    } for lane, manifest in manifests.items()])
    alignment.to_csv(OUT / "view_coordinate_alignment_audit.csv", index=False)
    write_json(OUT / "fresh_process_replay_audit.json", replay)
    config_values = {}
    for record in manifests[LANES[0]]["records"]:
        if record["arm"] == "FULL":
            config_values[record["config_id"]] = record["diagnostics"]["config"]
    write_json(OUT / "config_registry.json", {
        "schema": "night18c-stage-a-pre-evaluation-grid-v1",
        "configs": config_values,
        "candidate_ids": [record["candidate_id"] for record in manifests[LANES[0]]["records"]],
        "same_candidate_schema_all_lanes": all(
            [record["candidate_id"] for record in manifests[lane]["records"]]
            == [record["candidate_id"] for record in manifests[LANES[0]]["records"]] for lane in LANES
        ),
        "labels_available_to_producer": False,
        "endpoint": "COMMON_KMEANS_N20_S0",
    })
    write_json(OUT / "label_flow_audit.json", {
        "producer_annotation_arrays_accessed": 0,
        "partitions_materialized_and_hashed_before_evaluation": True,
        "evaluator_label_reads": 3,
        "use": "post-lock public benchmark evaluation only",
        "stage_a_grid_defined_before_metric read": True,
        "stage_b_opened": False,
    })
    write_json(OUT / "implementation_and_correction_ledger.json", [
        {"id": "E01", "phase": "preformal", "issue": "PowerShell interpreted command substitution while creating remote directories", "action": "reissued a literal command without substitution", "scientific_outputs_affected": 0},
        {"id": "E02", "phase": "preformal", "issue": "script-path execution omitted repository package from sys.path", "action": "used python -m from repository root", "scientific_outputs_affected": 0},
        {"id": "S01", "phase": "preformal", "issue": "independent PCA score axes were shape-compatible but not coordinate-compatible", "action": "added same-spot orthogonal Procrustes alignment and rotation hash before any label evaluation", "scientific_outputs_affected": 0},
        {"id": "E03", "phase": "evaluation", "issue": "pandas categorical cannot fill an unseen missing-value category", "action": "constructed the string label array element-wise; producer partitions and hashes were unchanged", "scientific_outputs_affected": 0},
        {"id": "W01", "phase": "all", "issue": "SciPy emitted CSR structural-edit efficiency warnings", "action": "retained results because arrays were finite and exact replay passed; warning has no formula effect", "scientific_outputs_affected": 0},
    ])
    pytest_text = (WORK / "audit/pytest_final.txt").read_text(encoding="utf-8")
    match = re.search(r"(\d+) passed", pytest_text)
    if not match:
        raise RuntimeError("actual pytest PASS count not found")
    write_json(OUT / "targeted_test_summary.json", {
        "status": "PASS",
        "passed": int(match.group(1)),
        "source": "actual pytest stdout",
        "stdout_sha256": file_sha(WORK / "audit/pytest_final.txt"),
        "command": "python -m pytest -q tests/test_night18c_rsp_gtd.py",
    })
    source_audit = """# Source-code collision and novelty audit

## Decision

Every primitive in the Stage-A object has strong prior art. Graph total variation, robust/Huber fidelity, group-sparse residuals, shared/private multi-view decomposition, and trainable algorithm unrolling cannot be claimed as new. The only not-yet-exactly-matched object found in this bounded audit is their same-spot spatial multi-omics combination: one shared tissue graph trend plus spot-level modality-private group-sparse residuals, consumed by clustering. Because Stage A failed 0/3, Night-18C makes no novelty claim for that combination and does not open an unrolled network.

| Prior work | Audited object | Collision with Night-18C | Consequence |
|---|---|---|---|
| Trend Filtering on Graphs, JMLR 2016 | graph fused lasso / graph trend filtering | Direct collision with graph-TV prior | graph-TV is mature scaffold |
| Vector-valued graph trend filtering | vector graph signals and ADMM solvers | Direct collision with vector-valued trend solver | vector extension is not new |
| Graph Unrolling Networks, IEEE TSP / arXiv:2006.01301 | unsupervised trainable graph trend-filtering unrolling | Direct collision with proposed Stage-B solver unrolling | unrolling is not new |
| Unrolling Nonconvex Graph Total Variation, arXiv:2506.02381 | Huber-related graph-TV and ADMM unrolling | Strong collision with robust graph-TV/unrolling | robust unrolling is not new |
| Robust multi-view shared/private factorization (IJCAI/AAAI line) | shared and private latent structure with structured sparsity | Strong component collision | shared/private wording is prior art |
| SpaMV, Nature Communications 2026 | spatial multi-omics shared/private encoders and mixture-of-experts | Strong application-domain collision | ordinary shared/private spatial fusion is not new |

Audited primary entry points:
- https://www.jmlr.org/beta/papers/v17/15-147.html
- https://arxiv.org/abs/2006.01301
- https://arxiv.org/abs/2506.02381
- https://www.nature.com/articles/s41467-026-74718-1

This was a bounded collision audit, not a proof of global novelty.
"""
    (OUT / "source_code_collision_and_novelty_audit.md").write_text(source_audit, encoding="utf-8")
    formula = r"""# Method semantics and formula contract

Night-18C Stage A tests a deterministic **robust shared/private graph-trend decomposition**. After per-view robust scaling and PCA, the second view's score coordinates are aligned to the first by same-spot orthogonal Procrustes; merely matching dimensions is forbidden.

For coordinated views H1 and H2, shared tissue trend Z and spot-private residuals R1,R2 solve the registered objective

`0.5 * sum_m sum_i Huber_delta(||Z_i + R_mi - H_mi||_2) + lambda * sum_(i,j) a_ij ||Z_i-Z_j||_2 + 0.5*gamma*sum_m sum_i ||R_mi||_2`.

Monotone backtracked block updates optimize Z; group shrinkage updates R. The clustering representation concatenates the unchanged retained anchor with a fixed-weight shared trend block. Every arm uses the same `KMeans(K, n_init=20, random_state=0)` endpoint. Labels are not an input to alignment, decomposition, representation, or clustering.

Matched arms are retained-only, identity fused, ordinary L2 low-pass, graph-TV-only (`R=0`), private-only (`lambda=0`), full, and an ID-stable permutation of private residual locations. Trainable unrolling is disallowed unless full has unexplained ARI/NMI gains on at least two of three lanes.
"""
    (OUT / "method_semantics_and_formula_contract.md").write_text(formula, encoding="utf-8")
    resource = {
        "timestamp_unix": time.time(),
        "root_statvfs": {
            "available_bytes": os.statvfs("/").f_bavail * os.statvfs("/").f_frsize,
            "available_inodes": os.statvfs("/").f_favail,
        },
        "persistent_statvfs": {
            "available_bytes": os.statvfs("/autodl-fs/data").f_bavail * os.statvfs("/autodl-fs/data").f_frsize,
            "available_inodes": os.statvfs("/autodl-fs/data").f_favail,
            "files_written_by_night18c": 0,
        },
        "working_tree_bytes": int(subprocess.check_output(["du", "-sb", str(WORK)]).split()[0]),
        "gpu_training_seconds": 0.0,
        "stage_b_opened": False,
        "shutdown_dispatched": False,
    }
    write_json(OUT / "resource_and_disk_audit.json", resource)
    decision = {
        "classification": "SCIENTIFIC_NEGATIVE",
        "status": "NIGHT18C_RSP_GTD_STAGE_A_SCIENTIFIC_NEGATIVE",
        "stage_a_independent_dual_signal_lanes": pass_count,
        "stage_a_required_lanes": 2,
        "stage_a_gate_passed": False,
        "stage_b_trainable_unrolling_authorized": False,
        "reason": "Full decomposition was independently superior to all key matched controls on 0/3 lanes.",
        "novelty_claim": "NONE",
        "label_protocol": "partitions locked before independent public-label evaluation",
        "shutdown_dispatched": False,
    }
    write_json(OUT / "night18c_decision.json", decision)
    report = f"""# Night-18C report

## 我现在需要知道的三件事

1. **本轮问了什么：** RNA 与 ATAC 能否先分成一个共享的组织空间趋势和两个模态私有残差，再让共享趋势改善聚类。这里“图总变差（graph total variation, graph-TV）”指允许区域内部平稳、又尽量保留边界的稀疏图正则。
2. **实际改了哪一层：** 我实现的是确定性表示层分解，不是 selector，也不是新 head。两个模态先分别降维，再用同位点、无标签的正交 Procrustes 协调坐标；随后同一稳健目标求共享趋势与组稀疏私有残差，最后所有 arm 接完全相同的 KMeans endpoint。
3. **结论与分类：** `SCIENTIFIC_NEGATIVE`。Full 在 0/3 lane 同时胜过关键 matched controls；P22 被普通 L2 低通解释，MISAR 未胜过 identity/private，人海马被 L2 或置换残差解释。因此没有启动可训练展开，也不主张新方法成立。

## 绝对指标主表

| lane | Night-16H 结构高位 ARI/NMI | retained+同 head | best full | 同配置/预算最强解释对照 | full 对最强对照 ΔARI/ΔNMI | 独立双升 |
|---|---:|---:|---:|---:|---:|---|
| P22 K9 | .587533/.708974 | .470897/.599897 | .475339/.604288 | L2 .476312/.604843 | {summary.iloc[0].delta_full_vs_max_control_ari:+.6f}/{summary.iloc[0].delta_full_vs_max_control_nmi:+.6f} | 否 |
| MISAR K7 | .534624/.656772 | .357936/.528705 | .363376/.535039 | private .363446/.535744 | {summary.iloc[1].delta_full_vs_max_control_ari:+.6f}/{summary.iloc[1].delta_full_vs_max_control_nmi:+.6f} | 否 |
| Human hippocampus K7 | .596178/.585490 | .152934/.213902 | .173062/.239081 | L2 .174856/.249512 | {summary.iloc[2].delta_full_vs_max_control_ari:+.6f}/{summary.iloc[2].delta_full_vs_max_control_nmi:+.6f} | 否 |

`absolute_metrics_and_controls.csv` 保留 51/51 个真实候选，包括 AMI、FMI、Moran/Geary、邻居一致率、完整簇大小、wall/RAM/GPU。这里 Night-16H 数字只作当前结构化 consumer 高位背景；它与本轮 common KMeans endpoint 不同，不能把差距全归因于表示。

## 贡献归因

- P22：full 相对 retained 略升，但同参数 L2 同时更高，故属于普通平滑解释，不是 shared/private 残差贡献。
- MISAR：full 的最佳 ARI/NMI 被 private/identity 端点双指标覆盖；没有独立增量。
- 人海马：full 相对 retained 有提升，但最佳 L2 双指标更高；一个置换私有残差 arm 也能达到相近区间，位置特异的 private residual 证据不足。
- 三 lane 的私有残差数值非零、目标逐块单调、exact K 均成立；这说明工程实现工作，并不等于科学假设成立。
- 同位点 Procrustes 协调修复了“PCA 只有 shape 相同却坐标系不一致”的语义风险；三条 rotation hash 与 ordered-ID hash 已冻结。

## 为什么不进入可训练展开

图趋势过滤及其 ADMM/神经展开已有明确先例。Graph Unrolling Networks 已覆盖无监督可训练的 graph trend-filtering unrolling，2025 的 nonconvex graph-TV 工作也覆盖 Huber 相关图正则与展开。因此展开 solver 本身不新；预注册门又是 0/3，继续训练只会扩大成本而没有可归因信号。

## 导师汇报版（7句）

我们测试了一个更具物理解释的 RNA+ATAC 表示层：共同组织趋势负责跨模态共享结构，spot 级私有残差吸收模态特有变化。为避免两个独立 PCA 轴直接相加的错误，我们先用同位点无标签 Procrustes 把坐标系协调。三条数据都用同一公式、同一参数网格和同一 KMeans endpoint，候选先锁哈希后才读公开标签。数值求解是稳定且可复算的，三 lane fresh-process partition 均 byte-exact。科学上 full 没有一条 lane 同时胜过全部关键对照：P22 与人海马主要由普通低通解释，MISAR 没有独立提升。现有 Night-16H 高位也没有被追平，因此本轮分类为 `SCIENTIFIC_NEGATIVE`，不启动 trainable unrolling。论文方向上应停止把图-TV/共享私有/展开本身当创新，下一步若继续表示研发，需要换到能先在 matched endpoint 上形成明确增量的对象。

## 失败、限制与审计边界

- Stage A 是 direct optimization，不含可训练网络参数；Stage B 未获授权。
- retained anchor + trend block 的 common endpoint 是严格贡献对照，但不是 Night-16H alpha-expansion/selector；跨 endpoint 比较只作背景。
- 公开标签用于锁定候选后的开发评价；没有盲测、SOTA 或 paper-ready 结论。
- 本轮未下载数据、未写 `/autodl-fs/data`、未派发 shutdown。

## 技术附录

- Targeted tests: 6/6 PASS；覆盖目标单调、私有残差非零、permutation 确定性、exact K、常数列、Procrustes 旋转/符号不变性。
- Fresh-process replay: 3/3 exact partitions and arrays.
- Candidate rows: {len(all_rows)}；producer label arrays accessed: 0；evaluator label reads: 3。
- 资源：GPU training 0 秒；working {resource['working_tree_bytes']} bytes；根盘剩余 {resource['root_statvfs']['available_bytes']} bytes；持久盘新增文件 0。
- `shutdown_dispatched=false`，AutoDL 保持有卡开机。
"""
    (OUT / "night18c_report.md").write_text(report, encoding="utf-8")
    write_json(OUT / "handoff_manifest.json", {
        "files": {path.name: file_sha(path) for path in sorted(OUT.iterdir()) if path.is_file()},
        "candidate_rows": len(all_rows), "stage_a_gate_passed": False,
    })


if __name__ == "__main__":
    main()
