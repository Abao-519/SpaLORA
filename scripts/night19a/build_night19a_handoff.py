#!/usr/bin/env python3
"""Build the compact, auditable Night-19A scientific handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/SpaLORA-night16h")
WORK = Path("/root/night19a_working")
OUT = ROOT / "outputs" / "night19a_handoff"
PRIMARY = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")
LANES = PRIMARY + ("PLACENTA_K10",)
FRONTIER = {
    "P22_K9": (0.587533, 0.708974, "Night-16H fixed selector"),
    "MISAR_K7": (0.534624, 0.656772, "Night-16H fixed selector"),
    "HUMAN_HIPPOCAMPUS_K7": (0.596178, 0.585490, "Night-16H fixed selector"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def read_csv(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def statvfs(path: Path):
    value = os.statvfs(path)
    return {
        "path": str(path), "available_bytes": value.f_bavail * value.f_frsize,
        "available_inodes": value.f_favail, "total_bytes": value.f_blocks * value.f_frsize,
        "used_fraction": 1.0 - (value.f_bavail / value.f_blocks),
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    gate = json.loads((WORK / "stage_a_formal" / "stage_a_gate.json").read_text(encoding="utf-8"))
    d0_gate = json.loads((WORK / "d0_rev1" / "d0_gate_rev1.json").read_text(encoding="utf-8"))
    if not d0_gate["stage_a_authorized"] or d0_gate["primary_lane_pass_count"] != 3:
        raise RuntimeError("D0 REV1 authority mismatch")
    if gate["multi_seed_authorized"] or gate["primary_strict_dual_pass_count"] != 0:
        raise RuntimeError("Stage-A fail-closed decision mismatch")

    absolute_rows = []
    p0_rows = []
    replay_rows = []
    gradient_rows = []
    resource_rows = []
    for lane in LANES:
        stage_dir = WORK / "stage_a_formal" / lane / "S0"
        producer = json.loads((stage_dir / "producer.json").read_text(encoding="utf-8"))
        replay = json.loads((stage_dir / "fresh_replay.json").read_text(encoding="utf-8"))
        evaluation = read_csv(stage_dir / "evaluation.csv")
        strong = next(row for row in evaluation if row["profile"] == "STRONG_START_NO_TRAIN")
        for row in evaluation:
            row = dict(row)
            row["delta_ari_vs_registered_anchor_same_head"] = float(row["ari"]) - float(strong["ari"])
            row["delta_nmi_vs_registered_anchor_same_head"] = float(row["nmi"]) - float(strong["nmi"])
            if lane in FRONTIER:
                row["night16h_frontier_ari"] = FRONTIER[lane][0]
                row["night16h_frontier_nmi"] = FRONTIER[lane][1]
                row["delta_ari_vs_night16h_frontier"] = float(row["ari"]) - FRONTIER[lane][0]
                row["delta_nmi_vs_night16h_frontier"] = float(row["nmi"]) - FRONTIER[lane][1]
                row["night16h_context"] = FRONTIER[lane][2]
            else:
                row.update({"night16h_frontier_ari": "", "night16h_frontier_nmi": "", "delta_ari_vs_night16h_frontier": "", "delta_nmi_vs_night16h_frontier": "", "night16h_context": "NOT_APPLICABLE"})
            absolute_rows.append(row)
        p0_rows.append({
            "lane": lane, "n": producer["n"], "k": producer["k"],
            "view1_shape": json.dumps(producer["view1_shape"]), "view2_shape": json.dumps(producer["view2_shape"]),
            "retained_shape": json.dumps(producer["retained_shape"]), "graph_shape": json.dumps(producer["graph_shape"]),
            "graph_nnz": producer["graph_nnz"], "pair_count": producer["pair_count"],
            "spatial_pair_count": producer["spatial_pair_count"], "feature_pair_count": producer["feature_pair_count"],
            "profiles": len(producer["profile_ids"]), "exact_k_all": producer["exact_k_all"],
            "mass_match_max_abs_error": producer["permuted_projection_groups"]["maximum_absolute_mass_error"],
            "checkpoint_reload": producer["strict_checkpoint_reload"], "artifact_reload": producer["artifact_reload"],
            "fresh_representation_exact": replay["representations_exact"], "fresh_partition_exact": replay["partitions_exact"],
            "producer_label_reads": producer["producer_label_reads"], "wall_seconds": producer["wall_seconds"],
            "peak_rss_mib": producer["peak_rss_mib"], "peak_gpu_mib": producer["peak_gpu_mib"],
        })
        replay_rows.append({
            "evidence": "STAGE_A", "lane": lane, "seed": 0,
            "ids_exact": replay["ids_exact"], "representation_exact": replay["representations_exact"],
            "partition_exact": replay["partitions_exact"], "label_reads": replay["label_reads"],
        })
        resource_rows.append({"lane": lane, "stage": "STAGE_A", "seed": 0, "wall_seconds": producer["wall_seconds"], "peak_rss_mib": producer["peak_rss_mib"], "peak_gpu_mib": producer["peak_gpu_mib"]})
        for seed in (0, 1):
            d0_dir = WORK / "d0_rev1" / lane / f"S{seed}"
            d0 = json.loads((d0_dir / "producer.json").read_text(encoding="utf-8"))
            d0_replay = json.loads((d0_dir / "fresh_replay.json").read_text(encoding="utf-8"))
            replay_rows.append({"evidence": "D0_REV1", "lane": lane, "seed": seed, "ids_exact": d0_replay["ordered_ids_exact"], "representation_exact": d0_replay["representation_exact"], "partition_exact": d0_replay["partition_exact"], "label_reads": d0_replay["label_reads"]})
            resource_rows.append({"lane": lane, "stage": "D0_REV1", "seed": seed, "wall_seconds": d0["wall_seconds"], "peak_rss_mib": d0["peak_rss_mib"], "peak_gpu_mib": d0["peak_gpu_mib"]})
            for point in d0["trajectory"]:
                pairs = point["pair_metrics"]
                strata = point["relation_stratum_vs_anchor"]
                gradient_rows.append({
                    "lane": lane, "seed": seed, "completed_steps": point["completed_steps"],
                    "zero_start_boundary": point["zero_start_boundary"],
                    "relation_norm": point["gradient_norms"]["relation"], "anchor_norm": point["gradient_norms"]["anchor"],
                    "consistency_norm": point["gradient_norms"]["consistency"], "variance_norm": point["gradient_norms"]["variance"],
                    "relation_anchor_cosine": pairs["relation__anchor"]["cosine"],
                    "relation_anchor_norm_ratio": min(pairs["relation__anchor"]["norm_a"], pairs["relation__anchor"]["norm_b"]) / max(pairs["relation__anchor"]["norm_a"], pairs["relation__anchor"]["norm_b"], 1e-30),
                    "relation_consistency_cosine": pairs["relation__consistency"]["cosine"],
                    "relation_variance_cosine": pairs["relation__variance"]["cosine"],
                    "low_anchor_cosine": strata["relation_low"]["cosine"], "mid_anchor_cosine": strata["relation_mid"]["cosine"], "high_anchor_cosine": strata["relation_high"]["cosine"],
                    "step_in_rev1_operational_gate": point["completed_steps"] in (5, 20, 40),
                    "zero_norm_cosine_semantics": "NULL_NA_NOT_ZERO",
                })

    write_csv(OUT / "absolute_metrics_and_controls.csv", absolute_rows)
    write_csv(OUT / "real_p0_registry.csv", p0_rows)
    write_csv(OUT / "exact_replay_audit.csv", replay_rows)
    write_csv(OUT / "gradient_conflict_trajectory.csv", gradient_rows)
    write_csv(OUT / "resource_table.csv", resource_rows)
    shutil.copy2(WORK / "d0_rev1" / "d0_gate_rev1.json", OUT / "d0_identifiability_gate_rev1.json")
    shutil.copy2(WORK / "d0_gate.json", OUT / "d0_gate_rev0_superseded.json")
    shutil.copy2(WORK / "stage_a_formal" / "stage_a_gate.json", OUT / "stage_a_gate.json")

    correction_rows = [
        {"revision": "E00", "classification": "ENGINEERING_CORRECTION", "issue": "First remote smoke omitted PYTHONPATH and exited before a scientific artifact.", "resolution": "Re-launched in the existing SpaLORA environment with explicit PYTHONPATH; no formula change.", "scientific_reuse": "NO"},
        {"revision": "E01", "classification": "SUPERSEDED_IMPLEMENTATION_SEMANTICS_INVALID", "issue": "D0 REV0 required the minimum norm ratio over every directionally negative post-zero checkpoint, causing step1 zero-start ramp to invalidate persistent operational conflict.", "resolution": "Preserved REV0; froze REV1 before rerun with operational steps 5/20/40 and unchanged -0.05/1e-3 numeric thresholds; reran 8/8 producer and replay.", "scientific_reuse": "NO"},
        {"revision": "E02", "classification": "PREFORMAL_SUPERSEDED_IMPLEMENTATION_SEMANTICS_INVALID", "issue": "Unexecuted Stage-A draft declared hidden_dim=4 and global min-norm did not exclude inactive zero gradients.", "resolution": "Preserved draft metadata; REV1 uses D0 hidden_dim=32, excludes norm<=1e-12 objectives, returns zero weight, and adds tests.", "scientific_reuse": "NO"},
        {"revision": "E03", "classification": "TRANSPORT_TIMEOUT", "issue": "The first 30-second SSH read timed out after P22 producer/replay completed.", "resolution": "Verified P22 locked artifact/replay and resumed only missing lanes with a longer client timeout; P22 was not recomputed.", "scientific_reuse": "P22_COMPLETE_ARTIFACT_REUSED_AFTER_EXACT_VALIDATION"},
    ]
    write_csv(OUT / "failure_and_correction_ledger.csv", correction_rows)

    label_flow = {
        "schema": "night19a-label-flow-audit-v1", "taskbook_sha256": "19107b229b1fbd06a6c6dafee797b469057ec8d75eab7dcec6e6ae22b8ad2e62",
        "d0_rev1_producer_runs": 8, "stage_a_producer_runs": 4, "producer_label_reads_total": 0,
        "partitions_and_artifacts_locked_before_first_annotation_process": True,
        "independent_evaluator_processes": 4, "evaluator_label_reads_total": 4,
        "label_use": "POST_LOCK_PUBLIC_BENCHMARK_EVALUATION_ONLY_NO_HPO_GRID_RAN",
        "within_run_checkpoint_selection": "FINAL_STEP_40_FIXED_NO_LABEL_READ",
        "annotation_sources": {
            "P22_K9": {"file": "/root/night16d_assets/kit/local_compute_kit/P22.npz", "key": "labels_primary", "mask": "label_mask"},
            "MISAR_K7": {"file": "/root/night16d_assets/kit/local_compute_kit/MISAR_E15_5_S1.npz", "key": "labels_primary", "mask": "label_mask"},
            "HUMAN_HIPPOCAMPUS_K7": {"file": "/root/night16e_external/human_hippocampus/human_adata1_official_result.h5ad", "key": "true_label", "mask": "non-null"},
            "PLACENTA_K10": {"file": "/autodl-fs/data/Human placenta architecture/humanplacenta_rna.h5ad", "key": "cell_type", "mask": "non-null"},
        },
    }
    (OUT / "label_flow_audit.json").write_text(json.dumps(label_flow, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    collision_rows = [
        {"source": "PCGrad (NeurIPS 2020)", "official_entry": "https://github.com/tianheyu927/PCGrad", "prior_art": "Conflict-gradient projection", "night19a_boundary": "Not novel; mandatory global control"},
        {"source": "CAGrad (NeurIPS 2021)", "official_entry": "https://github.com/Cranial-XIX/CAGrad", "prior_art": "Conflict-averse multi-objective descent", "night19a_boundary": "Not novel"},
        {"source": "GradNorm (ICML 2018)", "official_entry": "https://proceedings.mlr.press/v80/chen18a.html", "prior_art": "Gradient magnitude and training-rate balancing", "night19a_boundary": "Not novel"},
        {"source": "OGM-GE (CVPR 2022)", "official_entry": "https://github.com/GeWu-Lab/OGM-GE_CVPR2022", "prior_art": "Online multimodal gradient modulation", "night19a_boundary": "Confidence modulation is not novel"},
        {"source": "MMPareto (ICML 2024)", "official_entry": "https://github.com/GeWu-Lab/MMPareto_ICML2024", "prior_art": "Multimodal direction and magnitude integration via min-norm", "night19a_boundary": "Global min-norm is a control"},
        {"source": "SpaBalance (Advanced Science 2025)", "official_entry": "https://github.com/BiomedicalMachineLearning/SpaBalance", "prior_art": "Spatial multi-omics global intra/inter-omics gradient coordination and shared/private learning", "night19a_boundary": "No claim for generic multimodal gradient coordination"},
        {"source": "Impartial Optimization (ICML 2022)", "official_entry": "https://proceedings.mlr.press/v162/javaloy22a.html", "prior_art": "Computational-block multimodal conflict handling", "night19a_boundary": "Block-wise arbitration is not novel"},
        {"source": "Disentangled Gradient Learning (ICCV 2025)", "official_entry": "https://openaccess.thecvf.com/content/ICCV2025/html/Wei_Boosting_Multimodal_Learning_via_Disentangled_Gradient_Learning_ICCV_2025_paper.html", "prior_art": "Separate modality-encoder and fusion gradients", "night19a_boundary": "Layer/module disentanglement is not novel"},
        {"source": "Task Weighting through Gradient Projection (2024)", "official_entry": "https://arxiv.org/abs/2409.01793", "prior_art": "Priority-aware projection only under conflict", "night19a_boundary": "Conditional projection is not novel"},
        {"source": "Recon (ICLR 2023)", "official_entry": "https://openreview.net/forum?id=ivwZO-HnzG_", "prior_art": "Treat severe and layer-specific conflict; small conflict may help", "night19a_boundary": "Conflict filtering itself is not novel"},
        {"source": "Adaptive Gradient Modulation (ICCV 2023)", "official_entry": "https://openaccess.thecvf.com/content/ICCV2023/html/Li_Boosting_Multi-modal_Model_Performance_with_Adaptive_Gradient_Modulation_ICCV_2023_paper.html", "prior_art": "Adaptive multimodal gradient modulation", "night19a_boundary": "General confidence scaling is not novel"},
        {"source": "Bayesian Uncertainty for Gradient Aggregation (ICML 2024)", "official_entry": "https://proceedings.mlr.press/v235/achituve24a.html", "prior_art": "Uncertainty-aware gradient aggregation", "night19a_boundary": "Uncertainty-weighted aggregation is not novel"},
    ]
    write_csv(OUT / "source_collision_matrix.csv", collision_rows)

    (OUT / "method_formula_and_attribution_contract.md").write_text("""# Night-19A method and attribution contract

The zero-start anchor is the **registered relation-smoothed carrier**, not the raw retained embedding.  Let `g_p` be the gradient of the weighted anchor, masked-view consistency, and variance protection losses.  Relation loss is split into six registered sparse groups: spatial/feature-neighbour edge type crossed with low/mid/high evidence support.  For group `s`, if `g_s · g_p < 0` after the two-update zero-start warm-up,

`g'_s = g_s - rho_s (g_s · g_p / ||g_p||^2) g_p`, with `rho=(5/6,1/2,1/6)` for low/mid/high support.

The composite adjusted relation gradient is L2-matched to the unadjusted composite relation gradient before adding `g_p`.  Hence the primary comparison changes direction, not the overall relation-gradient norm.  The permuted arm performs a SHA-256 endpoint-stable bijection within spatial/feature edge type and exactly matches the base-relation-weighted projection-strength mass within each type.  The only possibly distinct object is this registered sparse edge-evidence conditioning of relation-gradient strata against strong-start protection.  Generic confidence scaling, block/layer projection, PCGrad, minimum-norm aggregation, and shared/private multimodal learning are prior art.

Stage A used one preregistered seed and one parameter profile.  Labels were not used to alter the formula, optimizer, checkpoint, representation, or partition.  Because full failed the 2/3 matched-control gate, no multi-seed stage and no HPO rescue were run.
""", encoding="utf-8")

    decision = {
        "schema": "night19a-decision-v1", "classification": "SCIENTIFIC_NEGATIVE",
        "status": "NIGHT19A_D0_IDENTIFIABLE_BUT_SPARSE_EVIDENCE_ARBITRATION_NO_INDEPENDENT_METHOD_SIGNAL",
        "novelty_collision": False, "d0_rev1_primary_lane_pass_count": 3,
        "d0_identified_pair": "relation__anchor", "stage_a_primary_strict_dual_pass_count": 0,
        "multi_seed_authorized": False, "stage_b_started": False,
        "qualitative_conflict_note": "All primary lanes and placenta showed persistent negative relation-anchor cosine at operational steps; REV0 failed only because step1 ramp ratio was included in an all-conflict minimum.",
        "scientific_result": "The identifiable conflict did not translate into an independent score benefit: FULL lost to a coordinate-wise strongest matched control in all three primary lanes.",
        "shutdown_dispatched": False,
    }
    (OUT / "night19a_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    resource = {
        "schema": "night19a-resource-and-disk-audit-v1", "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "root_filesystem": statvfs(Path("/")), "persistent_filesystem": statvfs(Path("/autodl-fs/data")),
        "night19a_working_bytes": int(subprocess.check_output(["du", "-sb", str(WORK)]).decode().split()[0]),
        "writes_to_autodl_fs_data": 0, "new_environment_created": False, "new_data_downloaded": False,
        "shutdown_dispatched": False,
    }
    (OUT / "resource_and_disk_audit.json").write_text(json.dumps(resource, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = f"""# Night-19A 稀疏证据梯度仲裁 P0 报告

## 我现在需要知道的三件事

1. **问题**：Night-17C 的关系保持目标与强起点保护是否真的持续冲突？答案是“是”。修正 zero-start ramp 的合约解释后，P22、MISAR、人海马 3/3 主 lane 以及 placenta 都在两个 seed 的 step 5/20/40 复现 `relation–anchor` 负方向冲突。
2. **实际动作所在层**：本轮没有改 selector 或图割 head，而是在同一个 32 维 zero-start 残差网络的梯度层，把注册稀疏空间/feature-neighbour 边按证据分层；低支持冲突分量投影更强，高支持更弱，并对合成 relation 梯度做 L2 量级匹配。这里的起点是**登记的 relation-smoothed carrier**，不是 raw retained 的 byte-exact 拷贝。
3. **论文含义**：冲突诊断是阳性，但方法结果是阴性。`FULL` 在三条主 lane 都没有同时超过坐标级最强 matched control，因此没有独立分数贡献，不能写成新方法成功。主分类为 **SCIENTIFIC_NEGATIVE**，多 seed 与后续 stage 均未授权。

## 绝对结果

| lane | FULL ARI/NMI | 最强 matched control ARI/NMI | FULL 差值 | 结论 |
|---|---:|---:|---:|---|
| P22 K9 | 0.481496 / 0.610371 | 0.482339 / 0.610676 (`TOPOLOGY_DISABLED`) | -0.000843 / -0.000304 | 未通过 |
| MISAR K7 | 0.360299 / 0.538696 | 0.363714 / 0.540974 (`STANDARD_WEIGHTED_SUM`) | -0.003415 / -0.002279 | 未通过 |
| Human hippocampus K7 | 0.194597 / 0.274609 | 0.209619 / 0.279205 (`STANDARD_WEIGHTED_SUM`) | -0.015022 / -0.004596 | 未通过 |
| Placenta K10（压力测试） | 0.373976 / 0.527426 | 0.381948 / 0.532974 (`STANDARD_WEIGHTED_SUM`) | -0.007971 / -0.005548 | 安全阈值内，但没有增益 |

这些数值使用相同 KMeans endpoint。`ARBITRATION_DISABLED_SAME_LOSSES` 与 `STANDARD_WEIGHTED_SUM` 在四条 lane 的分区和指标完全一致，说明分层重写本身没有偷换损失。P22 中 FULL 与 mass-matched permuted 的指标完全相同；MISAR 与 human 的 permuted 还优于 FULL，因此内容位置证据没有独立贡献。

## 与既有强结果的边界

Night-16H 的固定 selector 为 P22 `0.587533/0.708974`、MISAR `0.534624/0.656772`、人海马 `0.596178/0.585490`。本轮 Stage-A 使用的是 relation-smoothed zero-start representation 加共同 KMeans endpoint，不是 Night-16H 的 89-candidate 结构选择器；因此绝对落差同时包含 consumer/endpoint 差异，不能全部归因于训练表示。但 FULL 相对本轮 exact matched controls 仍为 0/3，足以否定本次仲裁的独立贡献。

## D0：冲突可识别，但不是任意负余弦都算通过

REV1 保留余弦阈值 `-0.05` 和梯度 norm ratio 阈值 `1e-3`，把 step 0/1 解释为 zero-start ramp，仅在 step 5/20/40 要求每 seed 至少两个可操作冲突点。结果 3/3 主 lane 与 placenta 均通过，复现 pair 都是 `relation__anchor`。step0 的零范数余弦保存为 NA；从未伪填 0。

原 REV0 结果完整保留，但它把 step1 约 `2e-4–7e-4` 的早期比例纳入所有负冲突点的 minimum，从而结构性产生 false negative。该周期标为 `SUPERSEDED_IMPLEMENTATION_SEMANTICS_INVALID`，没有参与 Stage-A 决策。

## 机制归因与先例

PCGrad、CAGrad、GradNorm、OGM-GE、MMPareto、SpaBalance，以及计算图分块、层级/模块解耦、只在严重冲突时投影和不确定性聚合均已有明确先例。本轮唯一可能区分的对象只是“注册稀疏 edge evidence 对 relation-gradient strata 与强起点保护方向之间的连续投影，并以 mass-matched permuted evidence 隔离内容位置”。正式结果没有支持该对象，故不产生新颖性主张。

## 工程与标签流

- D0 REV1：4 lane × 2 seeds = 8/8 producer；8/8 checkpoint/artifact 独立进程重放 exact。
- Stage A：4 lane × 8 arms；4/4 artifact/checkpoint 独立进程重放全部表示与分区 exact。
- 生产端标签读取为 0。四条 Stage-A artifact 全部锁定后，四个独立 evaluator 才读取公开 annotation；没有参数网格、label-HPO 或 within-run checkpoint 选择。
- 最终测试覆盖固定 named-parameter 坐标、unused gradient 补零、零范数 NA、step1 ramp 排除、inactive min-norm 任务排除、置乱质量匹配以及 standard/disabled 等价。
- `/autodl-fs/data` 没有新写入；未下载数据、未新建环境；AutoDL 保持开机，`shutdown_dispatched=false`。

## 失败与限制

- positive D0 只证明梯度方向冲突存在并可操作，不证明冲突必须被消除；Recon 等工作也提醒小冲突可能有益。
- FULL 对 human 的损失最大，提示按 evidence strata 对 relation 梯度投影仍可能破坏对聚类有用的更新。
- placenta 仅说明未超过预注册 0.01 额外回撤阈值，不能称 safety gain。
- 单 seed Stage-A 是按预注册门停止后的完整证伪，不支持稳定性或里程碑结论。

## 导师汇报版

这轮先验证了过去可训练核心为什么经常拉坏强起点：关系损失与强起点保护梯度在三条 RNA+ATAC 主数据以及 placenta 上都持续反向，而且两个 seed 可复现。我们随后实现了一个很窄的稀疏证据仲裁，只削弱低支持 edge strata 中与保护方向冲突的分量，同时保持 relation 梯度总量级。工程上，固定参数坐标、零梯度任务处理、mass-matched 置乱、checkpoint 与新进程重放都闭合。科学上 full 在 P22、MISAR、人海马均没有赢过同预算强对照，permuted 在部分数据还更好，因此 edge evidence 的具体位置没有独立贡献。结论是“冲突诊断阳性、仲裁方法阴性”，分类 `SCIENTIFIC_NEGATIVE`。这也说明下一步不应继续微调 gradient surgery，而应回到能真正改变强表示或候选生成的对象。

## 技术附录

- Taskbook SHA-256: `19107b229b1fbd06a6c6dafee797b469057ec8d75eab7dcec6e6ae22b8ad2e62`
- D0 REV1 contract SHA-256: `8d15260779f20ef322b70eacc313ad02d2e90241c5f180f3f8881f366955b1e1`
- Stage-A REV1 contract SHA-256: `d1a5275f8b76865de555d37a939b9e77dcc080590a8fc4bedcb1fd047c8a01a5`
- Parent Night-18E commit: `8d9b28aff48b338d55f1a91a1129d96f5ded0856`
"""
    (OUT / "night19a_report.md").write_text(report, encoding="utf-8")

    risks = """# Night-19A reviewer risk register

| Risk | Evidence | Required wording/action |
|---|---|---|
| Generic gradient surgery is prior art | PCGrad/CAGrad/MMPareto/SpaBalance and related works | Do not claim projection, min-norm, confidence weighting, block/layer handling, or shared/private learning as original. |
| D0 conflict is not a method gain | 3/3 primary pass D0 but 0/3 Stage-A pass | Separate diagnostic identifiability from independent representation contribution. |
| Anchor semantics overstated | Zero-start is relation-smoothed carrier | Never call it raw retained byte-exact. |
| REV0 gate appears result-driven | REV0 preserved with SHA; REV1 frozen and full rerun before labels | State that revision restored taskbook-level operational persistence and kept numeric thresholds. |
| Endpoint confounds Night-16H comparison | Stage A uses common KMeans, Night-16H uses candidate selector | Use only same-arm controls for attribution; Night-16H is context. |
| Permuted control might not match confidence mass | Per edge-type weighted projection mass error <=1e-8 and relation-gradient norm matched | Report both gates; do not overclaim edge-location selectivity because FULL did not win. |
| One seed | Multi-seed gate failed | Report as bounded P0 scientific negative, not stability evidence. |
"""
    (OUT / "reviewer_risk_register.md").write_text(risks, encoding="utf-8")

    files = [path for path in OUT.iterdir() if path.is_file() and path.name != "handoff_file_index.json"]
    index = {"schema": "night19a-handoff-index-v1", "files": [{"path": path.name, "size": path.stat().st_size, "sha256": sha256(path)} for path in sorted(files)]}
    (OUT / "handoff_file_index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUT), "file_count": len(index["files"]), "classification": decision["classification"]}, indent=2))


if __name__ == "__main__":
    main()
