"""Build the compact, evidence-derived Night-23A handoff after both frozen gates."""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path

from SpaLORA.night23a_xbed import file_sha


HISTORICAL = {
    "P22_K9": (0.5875325536381577, 0.7089736211480059, "Night-16H formal selector"),
    "MISAR_K7": (0.5346235433360444, 0.6567718908460013, "Night-16H formal selector"),
    "HUMAN_HIPPOCAMPUS_K7": (0.5961783620969016, 0.5854895180308324, "Night-16H formal selector"),
    "MELANOMA_TUMOR_K2": (0.9757656645125035, 0.9427557787068623, "Night-16H formal selector"),
}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--taskbook-sha", required=True)
    parser.add_argument("--parent-commit", required=True)
    parser.add_argument("--stage-a-freeze", required=True)
    parser.add_argument("--stage-b-freeze", required=True)
    args = parser.parse_args()
    working, repo, output = map(Path, (args.working, args.repo, args.output))
    output.mkdir(parents=True, exist_ok=True)

    stage_a = json.loads((working / "stage_a_summary/stage_a_decision.json").read_text())
    stage_b = json.loads((working / "stage_b_summary/stage_b_decision.json").read_text())
    if not stage_a["stage_b_authorized"] or stage_b["placenta_confirmation_authorized"]:
        raise RuntimeError("handoff builder only accepts observed Stage-A-pass/Stage-B-stop state")

    all_metrics = []
    for path in sorted((working / "stage_b_evaluation").glob("*.csv")):
        all_metrics.extend(read_csv(path))
    write_csv(output / "absolute_metrics_and_matched_controls.csv", all_metrics)
    shutil.copy2(working / "stage_a_summary/edge_transfer_table.csv", output / "edge_transfer_table.csv")
    shutil.copy2(working / "stage_b_summary/matched_partition_result.csv", output / "method_contribution_board.csv")
    shutil.copy2(working / "stage_a_summary/stage_a_decision.json", output / "stage_a_decision.json")
    shutil.copy2(working / "stage_b_summary/stage_b_decision.json", output / "stage_b_decision.json")

    by_lane = {}
    for row in all_metrics:
        row["ari"] = float(row["ari"])
        row["nmi"] = float(row["nmi"])
        by_lane.setdefault(row["lane"], []).append(row)
    frontier_rows = []
    for lane, rows in sorted(by_lane.items()):
        best = sorted(rows, key=lambda value: (-value["ari"], -value["nmi"], value["candidate_id"]))[0]
        old_ari, old_nmi, source = HISTORICAL[lane]
        frontier_rows.append(
            {
                "lane": lane,
                "night23a_best_arm": best["candidate_id"],
                "night23a_best_ari": best["ari"],
                "night23a_best_nmi": best["nmi"],
                "historical_authority_ari": old_ari,
                "historical_authority_nmi": old_nmi,
                "historical_source": source,
                "night23a_refreshes_frontier": best["ari"] > old_ari and best["nmi"] > old_nmi,
                "attribution": "matched control/head context; not XBED contribution" if best["candidate_id"] != "FULL_XBED" else "FULL XBED",
            }
        )
    write_csv(output / "score_frontier_context.csv", frontier_rows)

    teacher_rows, shape_rows = [], []
    for path in sorted((working / "teachers").glob("*.json")):
        item = json.loads(path.read_text())
        validation = item["teacher_validation"]
        teacher_rows.append(
            {
                "lane": item["lane"],
                "teacher_candidate_id": item["teacher_candidate_id"],
                "teacher_file_sha256": item["teacher_file_sha256"],
                "teacher_partition_sha256": item["teacher_partition_sha256"],
                "relation_sha256": item["relation_sha256"],
                "relation_count": item["relation_count"],
                "same_edge_fraction": item["same_edge_fraction"],
                "exact_k": validation["exact_k"],
                "no_singleton": validation["no_singleton"],
                "all_clusters_internal_spatial_support": validation["all_clusters_have_internal_spatial_edge"],
                "teacher_cluster_ids_exported_to_model": item["teacher_cluster_ids_exported_to_model"],
                "benchmark_reference_labels_read": item["benchmark_reference_labels_read"],
            }
        )
    write_csv(output / "teacher_authority_registry.csv", teacher_rows)
    for path in sorted((working / "features").glob("*.json")):
        item = json.loads(path.read_text())
        shape_rows.append(
            {
                "lane": item["lane"],
                "role": item["role"],
                "k": item["k"],
                "n": item["ids_shape"][0],
                "view1_shape": str(item["view1_shape"]),
                "view2_shape": str(item["view2_shape"]),
                "retained_shape": str(item["retained_shape"]),
                "spatial_shape": str(item["spatial_shape"]),
                "spatial_nnz": item["spatial_nnz"],
                "union_edge_count": item["union_edge_count"],
                "carrier_sha256": item["carrier_sha256"],
                "features_sha256": item["features_sha256"],
                "labels_read": item["labels_read"],
                "teacher_files_read": item["teacher_files_read"],
            }
        )
    write_csv(output / "real_path_shape_and_authority.csv", shape_rows)

    collisions = [
        {"method": "ARISE", "prior_object": "RNA-feature/spatial hard graph intersection and hierarchical fusion", "collision": "ARISE-like intersection is a mature atomic control", "remaining_distinction": "cross-study invariant teacher-edge relation with held-out no-teacher inference", "paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC13360277/", "official_source": "https://github.com/XiangxiangWang-code/ARISE", "fixed_head": "fefdd849494c0d08e755052a7a31b20169945e40", "license_or_reuse": "source inspected; no code copied"},
        {"method": "MMSpa", "prior_object": "within-study noisy-edge recognition/removal for spatial domains", "collision": "edge cleaning itself is prior art", "remaining_distinction": "study-transfer model and held-out no-teacher graph weighting", "paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12768284/", "official_source": "paper-linked source audit", "fixed_head": "N/A", "license_or_reuse": "no code copied"},
        {"method": "PRAGA", "prior_object": "adaptive modality graphs and prototype contrastive aggregation", "collision": "learned graph reliability is mature", "remaining_distinction": "teacher relation is label-ID/K invariant and transferred across physical studies", "paper": "https://ojs.aaai.org/index.php/AAAI/article/view/32010", "official_source": "https://github.com/Xubin-s-Lab/PRAGA", "fixed_head": "4adb11c96fc7ddad800fa1787eadcc8b91b42784", "license_or_reuse": "AGPL-3.0; clean-room only"},
        {"method": "SMART / S3RL / SEPAR", "prior_object": "MNN positive-negative relations, metric learning, and spatial multi-omics representation", "collision": "positive/negative local relation learning is not new", "remaining_distinction": "partition-derived boundary edge transfer plus fixed sparse partition consumer", "paper": "official papers and previously frozen source audit", "official_source": "project source-collision authorities", "fixed_head": "see prior final audits", "license_or_reuse": "no code copied"},
        {"method": "SpatialGlue", "prior_object": "within- and cross-modality attention/fusion", "collision": "cross-modal graph fusion is not new", "remaining_distinction": "cross-study held-out edge relation prediction", "paper": "Nature Methods 2024 SpatialGlue", "official_source": "https://github.com/JinmiaoChenLab/SpatialGlue", "fixed_head": "7c976d811d27ace51ce47ae0ad94a068a7d222fa", "license_or_reuse": "no code copied"},
        {"method": "stGuide / stMixer", "prior_object": "reference-to-query or cross-slice label/representation transfer", "collision": "cross-study transfer is mature", "remaining_distinction": "no reference annotation or teacher is available in held-out inference", "paper": "https://pmc.ncbi.nlm.nih.gov/articles/PMC12137301/", "official_source": "https://github.com/YQX-code/stMixer", "fixed_head": "bcb7d94488bc8744f32593e3845369473317e6e3", "license_or_reuse": "no code copied"},
        {"method": "pseudo-label graph self-training", "prior_object": "cluster pseudo-labels supervise node/edge models", "collision": "teacher supervision and edge classification are generic prior art", "remaining_distinction": "only the full joint protocol is narrow: invariant edge target, study-balanced LOSO, no held-out teacher, sparse exact-K consumer", "paper": "broad prior-art class", "official_source": "N/A", "fixed_head": "N/A", "license_or_reuse": "no code copied"},
    ]
    write_csv(output / "source_code_collision_matrix.csv", collisions)

    label_flow = {
        "schema": "night23a-teacher-label-flow-audit-v1",
        "stage_a_training": "source-study Night-16H selector partitions converted only to same/boundary edge bits",
        "model_inputs_exclude": ["dataset name", "tissue name", "observation ID", "teacher cluster ID", "benchmark label"],
        "heldout_prediction_teacher_reads": 0,
        "heldout_prediction_benchmark_label_reads": 0,
        "stage_a_diagnostic_first_teacher_read": "after prediction artifact/checkpoint SHA lock and fresh replay",
        "stage_b_partition_teacher_reads": 0,
        "stage_b_partition_benchmark_label_reads": 0,
        "stage_b_evaluator_reference_reads_per_lane": 1,
        "placenta_labels_read": 0,
        "placenta_reason": "Stage-B gate stopped before confirmation",
        "all_teacher_authority_rows": len(teacher_rows),
    }
    (output / "teacher_and_label_flow_audit.json").write_text(json.dumps(label_flow, indent=2, sort_keys=True), encoding="utf-8")

    failures = [
        {"cycle": "PRE_STAGE_B_DETERMINISM_REVIEW", "status": "SUPERSEDED_BEFORE_REFERENCE_EVALUATION", "issue": "initial sparse eigensolver draft did not pass an explicit deterministic v0", "correction": "froze LINEAR_1_TO_2_UNIT_NORM v0 before any Stage-B partition/reference evaluation", "scientific_retry": False},
        {"cycle": "STAGE_B_LAUNCH_0", "status": "ENGINEERING_FAILURE_NO_PARTITION", "issue": "PYTHONPATH absent caused ModuleNotFoundError before producer computation", "correction": "registered the existing repository root in process environment; source/formula unchanged", "scientific_retry": False},
        {"cycle": "STAGE_A_PRIMARY", "status": "PASS_HETEROGENEOUS", "issue": "logistic transfer failed on human although pooled and 2/3 threshold passed", "correction": "none; retained as evidence; MLP remained diagnostic and did not rescue gate", "scientific_retry": False},
        {"cycle": "STAGE_B_PRIMARY", "status": "SCIENTIFIC_GATE_FAILURE", "issue": "FULL was dominated on all 3 primary studies", "correction": "none; placenta/final training/GSE205055 prohibited", "scientific_retry": False},
    ]
    write_csv(output / "failure_and_correction_ledger.csv", failures)

    disk = shutil.disk_usage("/")
    resource = {
        "schema": "night23a-resource-audit-v1",
        "root_total_bytes": disk.total,
        "root_used_bytes": disk.used,
        "root_free_bytes": disk.free,
        "working_tree_bytes": tree_bytes(working),
        "working_tree_path": str(working),
        "gpu": "NVIDIA GeForce RTX 4080 SUPER 32760 MiB",
        "stage_a_train_device": "cuda",
        "stage_b_partition_device": "cpu",
        "peak_gpu_mib": "not captured; no peak claim",
        "runtime_versions": {"python": "3.8.10", "torch": "2.0.0+cu118", "numpy": "1.22.3", "scipy": "1.8.1", "sklearn": "1.1.1"},
        "stage_b_wall_seconds_by_lane": {path.stem: json.loads(path.read_text())["wall_seconds"] for path in sorted((working / "stage_b_partitions").glob("*.json"))},
        "shutdown_dispatched_at_science_seal": False,
    }
    (output / "resource_audit.json").write_text(json.dumps(resource, indent=2, sort_keys=True), encoding="utf-8")

    test_log = working / "targeted_tests_final.log"
    if not test_log.exists() or "10 passed" not in test_log.read_text(encoding="utf-8"):
        raise RuntimeError("final targeted-test log is absent or incomplete")
    stage_a_replays = [json.loads(path.read_text()) for path in sorted((working / "replays").glob("*.json"))]
    stage_b_replays = [json.loads(path.read_text()) for path in sorted((working / "stage_b_replays").glob("*.json"))]
    if len(stage_a_replays) != 4 or any(item["status"] != "PASS" for item in stage_a_replays):
        raise RuntimeError("Stage-A replay audit incomplete")
    if len(stage_b_replays) != 4 or any(item["status"] != "PASS" or item["candidate_count"] != 10 for item in stage_b_replays):
        raise RuntimeError("Stage-B replay audit incomplete")
    test_replay = {
        "schema": "night23a-test-and-replay-audit-v1",
        "targeted_test_status": "PASS",
        "targeted_test_count": 10,
        "targeted_test_log_sha256": file_sha(test_log),
        "stage_a_fresh_process_replay_lanes": len(stage_a_replays),
        "stage_a_all_pass": True,
        "stage_b_fresh_process_replay_lanes": len(stage_b_replays),
        "stage_b_partitions_replayed": sum(item["candidate_count"] for item in stage_b_replays),
        "stage_b_all_pass": True,
    }
    (output / "test_and_replay_audit.json").write_text(json.dumps(test_replay, indent=2, sort_keys=True), encoding="utf-8")

    decision = {
        "schema": "night23a-final-decision-v1",
        "classification": "RELATIONAL_TRANSFER_WITHOUT_PARTITION_GAIN",
        "stage_a_edge_identifiability_passed": True,
        "stage_b_partition_gate_passed": False,
        "stage_b_main_lane_passes": stage_b["main_lane_passes"],
        "stage_b_macro_delta_ari_vs_coordinatewise_strongest": stage_b["macro_delta_ari_vs_coordinatewise_strongest"],
        "stage_b_macro_delta_nmi_vs_coordinatewise_strongest": stage_b["macro_delta_nmi_vs_coordinatewise_strongest"],
        "placenta_confirmation_run": False,
        "gse205055_authority_closure_run": False,
        "night22_junction_reopened": False,
        "method_line_action": "stop XBED partition line; preserve edge-transfer diagnostic only",
        "score_frontier_advance": False,
        "labels_used": "heldout teacher for post-lock edge diagnostics; benchmark labels for post-lock partition evaluation only",
    }
    (output / "night23a_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    contribution = {row["lane"]: row for row in read_csv(output / "method_contribution_board.csv")}
    report = f"""# Night-23A 跨研究边界边蒸馏报告

## 我现在需要知道的三件事

1. **边关系能迁移，但不是四研究一致。** 共享 logistic 在 P22、MISAR 的 held-out teacher-edge AUROC 分别为 0.8652、0.8685，human 只有 0.4484；三主研究 pooled AUROC 为 0.8229，Stage A 按预注册门通过。这个结果说明训练研究中学到的多模态局部关系含有可迁移信息，但在人海马上的线性关系发生明显域偏移。
2. **可识别的边，不等于可用的分区边权。** 同一 exact-K 稀疏分区器下，FULL 在 P22/MISAR/human 的 ARI/NMI 分别为 0.4210/0.5979、0.1604/0.4047、0.4630/0.4613，三条都被匹配原子对照双指标支配，主门 0/3；macro ΔARI/ΔNMI 为 {stage_b['macro_delta_ari_vs_coordinatewise_strongest']:.4f}/{stage_b['macro_delta_nmi_vs_coordinatewise_strongest']:.4f}。
3. **终态是 `RELATIONAL_TRANSFER_WITHOUT_PARTITION_GAIN`。** 本轮得到的是一个真实、可复算的跨研究 edge-identifiability 诊断，不是新的空间分区方法。按冻结合约停止四研究最终训练、placenta 和 GSE205055，不回到 Night-22 junction，也不追加 head/HPO 挽救。

## 算法实际做了什么

每个研究只从注册 spatial CSR 和 retained/RNA/ATAC 三个数值 view 机械构建稀疏 union edge。30 个特征全部在研究内转为 rank/robust relation statistics；模型看不到数据集名、组织名、观测 ID、teacher cluster 编号或真实标签。训练研究的 Night-16H partition 只转换为“同簇/跨簇”二元边关系，因而对 cluster label permutation 不变，也不依赖不同研究的 K 数值可比。

四折 LOSO 中，预测 artifact、checkpoint 和 edge hashes 先锁定并 fresh-process exact replay，held-out teacher 随后才由诊断 evaluator 打开。Stage B 则完全不读取 held-out teacher：冻结 edge probability 后，十个 matched arms 使用同一 edge union、同一 sparse normalized adjacency、同一 deterministic eigensolver 和同一 KMeans exact-K endpoint。FULL 使用 `c=2|p-0.5|; w=(1-c)+cp`：低置信边退回 raw union，高置信同域边保留，高置信边界边衰减。

## Stage A：跨研究边关系可识别性

| Held-out study | Logistic AUROC | Logistic AUPRC | MLP AUROC | Spatial AUROC | Intersection AUROC | Shuffled AUROC |
|---|---:|---:|---:|---:|---:|---:|
| P22 K9 | 0.8652 | 0.9575 | 0.8320 | 0.6569 | 0.6149 | 0.3421 |
| MISAR K7 | 0.8685 | 0.9457 | 0.8223 | 0.6239 | 0.5894 | 0.5893 |
| Human hippocampus K7 | 0.4484 | 0.8124 | 0.7071 | 0.7160 | 0.5032 | 0.5937 |
| Melanoma K2（secondary） | 0.6213 | 0.9744 | 0.7688 | 0.8350 | 0.5007 | 0.3173 |
| 三主研究 pooled | **0.8229** | **0.9327** | — | 0.6586 | 0.5990 | 0.4219 |

Stage A 的正面证据仅能表述为“跨研究 teacher-edge 统计可部分迁移”。Human 的 logistic 失败、melanoma 的空间距离更强，禁止把 pooled 数字写成普适边界识别器。

## Stage B：绝对分区指标与匹配贡献

| Study | FULL ARI/NMI | strongest matched control ARI/NMI | ΔARI/ΔNMI | FULL 最小簇 | 独立通过 |
|---|---:|---:|---:|---:|---|
| P22 K9 | 0.4210 / 0.5979 | 0.4568 / 0.6213（MLP direct） | -0.0358 / -0.0233 | 242 | 否 |
| MISAR K7 | 0.1604 / 0.4047 | 0.4346 / 0.5588（retained-only） | -0.2742 / -0.1541 | 75 | 否 |
| Human hippocampus K7 | 0.4630 / 0.4613 | 0.5169 / 0.5261（MLP direct） | -0.0540 / -0.0648 | 170 | 否 |
| Melanoma K2（secondary） | 0.9001 / 0.8163 | 0.9001 / 0.8163（多臂等价） | 0 / 0 | 271 | 不计票 |

FULL 不是 no-op：相对 raw union 的最优标签对齐后 changed spots 为 P22 1867、MISAR 479、human 178；melanoma 为 0。它确实改变了分区，却没有带来独立质量增益，因此失败不能归因于“模块没接进去”。所有 exact-K 与 fresh replay 均通过，没有空簇；主失败是 learned reliability 到 partition capacity 的语义不匹配/域偏移，而非工程崩坏。

与 Night-16H 正式结果相比，本轮任何臂都没有刷新 frontier：P22 0.5875/0.7090、MISAR 0.5346/0.6568、human 0.5962/0.5855、melanoma 0.9758/0.9428 仍是对应权威高位。Night-23A 中较好的 MLP-direct/retained-only 数字只是 matched head/context，不是 XBED 贡献。

## 贡献边界与新颖性

ARISE 已覆盖 RNA-feature/空间图硬交集与层级融合；MMSpa 已覆盖研究内 noisy-edge removal；PRAGA 已覆盖动态模态图和 prototype aggregation；stGuide/stMixer 已覆盖 reference/query 或跨切片迁移。teacher supervision、边分类和 pseudo-label graph self-training 也都是成熟对象。因此能够保留的窄对象只能是“cluster-label/K 不变 edge target + study-balanced LOSO + held-out no-teacher inference + sparse exact-K consumer”的联合协议，不能把任何组件单独称原创。

本轮联合协议只通过了 edge diagnostic，没有通过 partition contribution gate，所以暂不值得冻结论文方法名，也不进入 placenta 外部确认。最直接的科学结论是：**用分区 teacher 学到的边概率可以预测另一个研究的 teacher edges，但不能直接视为最优 Potts/spectral capacity。**

## 失败、限制与下一步边界

- Human 的 logistic 域偏移显著，MLP 虽改善 edge AUROC，却仍无法使 FULL 分区优于对照；这提示 calibration 与 partition loss 并不一致。
- teacher 来自 Night-16H selector，属于项目历史 benchmark-development 资产；即使标签没有进入本轮 producer，也不能把整个教师链称为原始数据端到端无标签。
- Stage B 采用 fixed spectral exact-K consumer；结果否定的是当前 reliability-to-capacity 公式与该 consumer 的联合对象，不等价于否定所有跨研究边预测任务。
- 预注册门禁止 placenta/GSE205055，因此不存在外部确认结果，也不得把“没跑”写成负数据集结果。
- 这条 XBED 分区线到此停止。若未来复用，只宜把 edge model 作为独立 boundary diagnostic，不能在没有新数学对象和外部证据时继续调 capacity/head。

## 5–8 句导师汇报版

我们这轮不再改 Night-22 的 junction，而是把 Night-16H 的高分分区压成与类别编号无关的同域/跨域边关系，训练一个跨研究共享边模型。模型在 P22 和 MISAR 上的 held-out AUROC 都约 0.87，三主研究 pooled AUROC 约 0.82，证明局部边关系里确实有可迁移统计信号。可惜在人海马上线性模型 AUROC 只有 0.45，说明跨组织校准不稳定。更关键的是，把这些概率真正接进统一 sparse exact-K 分区器后，FULL 在三个主要研究上全部被原子对照双指标支配，方法门是 0/3。FULL 确实改变了数百到上千个 spot，并且所有重放和 exact-K 检查通过，所以这是科学负结果，不是代码没生效。最终分类是 `RELATIONAL_TRANSFER_WITHOUT_PARTITION_GAIN`：边诊断有信号，但不能写成分区方法贡献。我们按预注册规则没有再跑 placenta、GSE205055 或追加 HPO，也没有复活旧 junction。论文上最多保留为“跨研究边可识别但直接容量映射失败”的负证据和算法诊断资产。

## 技术审计摘要

- Parent: `{args.parent_commit}` / `night22b-final-20260827`
- Taskbook SHA-256: `{args.taskbook_sha}`
- Stage-A freeze: `{args.stage_a_freeze}`
- Stage-B freeze: `{args.stage_b_freeze}`
- Stage-A checkpoint/prediction replay: 4/4 PASS；Stage-B partition replay: 4 lanes × 10 arms PASS。
- Targeted tests at Stage-B freeze: 10/10 PASS。
- Root free space at handoff build: {disk.free / (1024**3):.2f} GiB；working artifacts: {tree_bytes(working) / (1024**2):.2f} MiB。
- Shutdown was not dispatched when scientific handoff was built; final dispatch is an external last-command step after compact verification.
"""
    (output / "night23a_report.md").write_text(report, encoding="utf-8")
    mentor = "\n".join(report.split("## 5–8 句导师汇报版\n\n", 1)[1].split("\n\n## 技术审计摘要", 1)[0].splitlines()) + "\n"
    (output / "mentor_summary.md").write_text(mentor, encoding="utf-8")

    semantics = """# Night-23A formula and selection contract

- Teacher target is only `1[partition_i == partition_j]`; cluster IDs are never model inputs.
- Stage A is study-balanced four-fold LOSO. The primary gate is the frozen logistic model; MLP is diagnostic.
- Stage B uses one locked sparse union and deterministic exact-K spectral partitioner for all arms.
- FULL capacity is `c=2|p-0.5|`, `w=(1-c)+c p`, nonnegative on every edge.
- Partition artifacts and fresh-process replay precede every benchmark-label read.
- Stage-B failure mechanically prohibits final four-teacher training, placenta, GSE205055, post-hoc HPO, and any Night-22 junction revival.
"""
    (output / "method_semantics_and_selection_contract.md").write_text(semantics, encoding="utf-8")

    source_files = [
        repo / "SpaLORA/night23a_xbed.py",
        repo / "configs/night23a/stage_a_contract.json",
        repo / "configs/night23a/stage_b_contract.json",
        *sorted((repo / "scripts/night23a").glob("*.py")),
        repo / "tests/test_night23a_xbed.py",
    ]
    manifest = {
        "schema": "night23a-source-authority-manifest-v1",
        "taskbook_sha256": args.taskbook_sha,
        "parent_commit": args.parent_commit,
        "stage_a_freeze_commit": args.stage_a_freeze,
        "stage_b_freeze_commit": args.stage_b_freeze,
        "files": [
            {"path": str(path.relative_to(repo)).replace(os.sep, "/"), "size": path.stat().st_size, "sha256": file_sha(path)}
            for path in source_files
        ],
    }
    (output / "source_authority_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
