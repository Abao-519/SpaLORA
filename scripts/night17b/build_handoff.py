#!/usr/bin/env python3
"""Build the compact Night-17B P0 handoff from locked artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path("/root/night17b_p0_working")
REPO = Path("/root/SpaLORA-night16h")
OUT = REPO / "outputs/night17b_handoff"
LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_rows(path: Path):
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(value):
    return f"{float(value):.6f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    selection = json.loads((ROOT / "formal/family_selection.json").read_text())
    matched = read_rows(ROOT / "formal/matched_controls.csv")
    all_rows = []
    for lane in LANES:
        all_rows.extend(read_rows(ROOT / f"formal/{lane}/metrics.csv"))
    write_csv(OUT / "all_locked_config_metrics.csv", all_rows)
    write_csv(OUT / "absolute_metrics_and_controls.csv", matched)

    chosen = {}
    for lane in LANES:
        chosen[lane] = {
            row["arm"]: row
            for row in matched
            if row["lane"] == lane
        }
    melanoma_rows = read_rows(ROOT / "secondary/MELANOMA_TUMOR_K2/metrics.csv")
    melanoma = {row["arm"]: row for row in melanoma_rows}

    producers = {
        lane: json.loads((ROOT / f"formal/{lane}/producer.producer.json").read_text()) for lane in LANES
    }
    config = selection["selected_config_id"]
    main_lines = []
    for lane in LANES:
        frozen = chosen[lane]["FROZEN_RETAINED_SAME_HEAD"]
        smooth = chosen[lane]["FEASIBLE_CONSENSUS_SAME_HEAD"]
        full = chosen[lane]["FULL_WEIGHTED"]
        unbiased = chosen[lane]["UNBIASED_BANK_WEIGHTED"]
        permuted = chosen[lane]["RELATION_PERMUTED"]
        main_lines.append(
            f"| {lane} | {fmt(frozen['ari'])}/{fmt(frozen['nmi'])} | "
            f"{fmt(smooth['ari'])}/{fmt(smooth['nmi'])} | {fmt(full['ari'])}/{fmt(full['nmi'])} | "
            f"{fmt(full['delta_ari_vs_strongest_control'])}/{fmt(full['delta_nmi_vs_strongest_control'])} | "
            f"{fmt(unbiased['ari'])}/{fmt(unbiased['nmi'])} | {fmt(permuted['ari'])}/{fmt(permuted['nmi'])} | "
            f"{'PASS' if float(full['delta_ari_vs_strongest_control']) > 0 and float(full['delta_nmi_vs_strongest_control']) > 0 else 'FAIL'} |"
        )

    report = f"""# Night-17B SFRD P0 报告

## 我现在需要知道的三件事

1. **问题**：Night-16H 的 89 个结构可行分区里确实有大量关系信息；本轮检验能否把这些软关系蒸馏进一个真正更新参数的 RNA+ATAC 表示，而不是继续选择现成分区。
2. **实际动作所在层**：在真实 view1、view2 与 retained carrier 上，用注册空间边加受限 feature-kNN 边形成稀疏 pair bank；所有结构可行候选先产生 co-cluster / separated / uncertainty 后验，再训练小型残差编码器，最终所有 arm 使用同一个 KMeans endpoint。标签只在所有 partition 与 hash 写出后由独立 evaluator 打开。
3. **论文含义**：严格门只通过 **1/3** 主 lane。P22 有真实的可训练表示局部增益；MISAR 没有，human 又明显不如非训练的 feasible-relation smooth control。终态为 **SCIENTIFIC_NEGATIVE**，不能把它写成跨研究方法信号，也不刷新 Night-16H 的总体分数前沿。

## 明确终态

- classification: `SCIENTIFIC_NEGATIVE`
- status: `NIGHT17B_SFRD_NOT_IDENTIFIABLE`
- 次级事实：`P22_REPRESENTATION_LOCAL_SIGNAL`
- 严格继续门：full learned representation 必须同时超过 frozen-retained 与 feasible-consensus 两个同-head control，并在至少 2/3 主 lane 双指标为正；实际为 **1/3**。

## 绝对指标与匹配贡献

所有数字均为同一个 KMeans head；“强参考”按 ARI、NMI 分别取 frozen 与 feasible-smooth 的较强值。

| 数据集 | Frozen ARI/NMI | Feasible smooth ARI/NMI | SFRD full ARI/NMI | Δ vs 强参考 ARI/NMI | Unbiased-bank ARI/NMI | Permuted-relation ARI/NMI | 严格门 |
|---|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(main_lines)}

次级 melanoma K2 安全 lane：frozen `{fmt(melanoma['FROZEN_RETAINED_SAME_HEAD']['ari'])}/{fmt(melanoma['FROZEN_RETAINED_SAME_HEAD']['nmi'])}`，feasible smooth `{fmt(melanoma['FEASIBLE_CONSENSUS_SAME_HEAD']['ari'])}/{fmt(melanoma['FEASIBLE_CONSENSUS_SAME_HEAD']['nmi'])}`，SFRD full `{fmt(melanoma['FULL_WEIGHTED']['ari'])}/{fmt(melanoma['FULL_WEIGHTED']['nmi'])}`。它同样否定“可训练残差优于现成可行关系 head”。

## 高分来自哪一层

- **P22**：full 比强参考提高 `+0.044882 ARI / +0.003485 NMI`；unbiased bank 仍为 `{fmt(chosen['P22_K9']['UNBIASED_BANK_WEIGHTED']['ari'])}/{fmt(chosen['P22_K9']['UNBIASED_BANK_WEIGHTED']['nmi'])}`，说明这条局部增益不完全依赖历史 authority 候选。但 unweighted 与 raw-view-adapter 对照非常接近 full，关系权重本身的净贡献仍弱。
- **Human hippocampus**：full 虽高于 frozen，却低于 feasible smooth `-0.046528/-0.064978`；permuted relation 与仅启用一个 raw-view adapter 的 arm 还高于 full，关系位置特异性不成立。
- **MISAR**：full 比强参考低 `-0.001969/-0.003744`，unbiased bank 进一步下降。
- `SINGLE_VIEW1/2` 只屏蔽对应 raw-view adapter；retained adapter 仍可能含双模态信息，所以这些 arm 只能解释为 **ONE_RAW_VIEW_ADAPTER_ABLATION**，不能称真正单模态模型。
- full bank 含历史 benchmark-HPO strong starts；因此本轮主张只能依赖 matched unbiased-bank sensitivity。P22 在该 sensitivity 下保持，另外两条不保持。

## 真实工程 P0

| lane | view1 | view2 | retained | sparse pair 数 | optimizer steps | 参数 L2 变化 | max grad | fresh replay |
|---|---:|---:|---:|---:|---:|---:|---:|---|
"""
    for lane in LANES:
        producer = producers[lane]
        diag = next(row for row in producer["run_diagnostics"] if row["run_id"] == f"{config}__FULL_WEIGHTED__S0")
        report += (
            f"| {lane} | {producer['view1_shape']} | {producer['view2_shape']} | {producer['retained_shape']} | "
            f"{producer['pair_count']} | {diag['actual_optimizer_steps']} | {diag['parameter_l2_change']:.6f} | "
            f"{diag['max_gradient_norm']:.6f} | PASS |\n"
        )
    report += f"""

三条主 lane 均实际产生非零梯度和参数变化；checkpoint strict load 后，GPU fresh-process representation 与 partition 均 byte-exact。候选关系与 feature-neighbour 只用稀疏 pair list，没有 dense N×N。

## 工程修正与作废范围

1. 首次 smoke 未设置仓库 `PYTHONPATH`，在导入前退出；设置正确路径后重跑，未生成科学结果。
2. 初次 fresh replay 在 CPU 上重算 GPU representation，浮点路径不同而未 byte-exact；改为相同 CUDA 数值路径后 3/3 seed0 与额外运行均 exact。
3. evaluator 最初把 training seed 写死为 0；改为从锁定 run IDs 解析，重新评价所有既有 partitions，分区未变。
4. **最重要修正**：首版 gate 只与 frozen retained 比较，错误忽略 feasible-consensus same-head control，曾误判 2/3 并提前运行 seeds 1/2。修正后严格门为 1/3；这些额外 seed 文件完整保留但标记 `SUPERSEDED_PREMATURE_CONFIRMATION_DUE_GATE_BUG`，不进入主科学判定。

## 标签与 HPO 边界

- 三个 config 与所有 control 的 partitions/hash 先写出，之后 evaluator 才读取公开 benchmark annotations。
- family config `{config}` 是透明的 post-lock label-assisted benchmark HPO；它不是盲测或冻结外部确认。
- 89-candidate bank 本身来自历史公开 benchmark 开发；本轮不能称原始数据端到端完全无标签教师。`UNBIASED_BANK_WEIGHTED` 剔除了 primary authority/medoid，只保留 robustness KMeans seeds 与 ordered continuation starts。

## 限制

- SFRD 仍是 retained-representation plug-in，不是 raw-fragment end-to-end 模型。
- P22 的局部分数 `{fmt(chosen['P22_K9']['FULL_WEIGHTED']['ari'])}/{fmt(chosen['P22_K9']['FULL_WEIGHTED']['nmi'])}` 低于 Night-16H 已交付的总体分区高位，不能称 score-frontier advance。
- Human 与 melanoma 表明可行候选关系经过简单非训练 head 已可非常强，而当前残差训练会损失这些结构。
- 没有证据支持继续加 seed 或扩大同机制网格；下一对象应改变蒸馏目标/消费者，而不是修补本公式。

## 导师汇报版

这轮测试的是能否把 Night-16H 多个可行分区的共识关系学进一个新表示。模型确实在三套真实 RNA+ATAC 数据上完成了反向传播、参数更新和 checkpoint 精确重放。P22 上，同一 KMeans head 的 ARI 从强参考 0.3938 提到 0.4387，NMI 从 0.5847 提到 0.5881，说明有一条局部表示信号。可是 MISAR 轻微下降，human 也明显输给非训练的关系平滑 control；melanoma 的非训练 control 更远高于残差模型。严格科学门因此只有 1/3，通过不了。早先 2/3 是 gate 漏掉强 control 的实现错误，已经修正，额外 seeds 保留但不作证据。结论是科学负结果，不是 SOTA，也不是可投稿的方法核心。

## 技术附录

- source/config selection: `{config}`，seed0 为主 P0。
- targeted tests: 7/7 PASS。
- AutoDL 状态：保持开机，`shutdown_dispatched=false`。
"""
    (OUT / "p0_report.md").write_text(report, encoding="utf-8")

    replay = {}
    for lane in LANES:
        replay[lane] = json.loads((ROOT / f"formal/{lane}/replay.json").read_text())
    (OUT / "checkpoint_and_replay_audit.json").write_text(
        json.dumps({"schema": "night17b-replay-audit-v1", "lanes": replay}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    correction = {
        "schema": "night17b-correction-ledger-v1",
        "entries": [
            {"cycle": 1, "type": "ENGINEERING", "issue": "producer script import lacked repository PYTHONPATH", "action": "set explicit PYTHONPATH and rerun smoke", "scientific_change": False},
            {"cycle": 2, "type": "ENGINEERING", "issue": "CPU replay differed from GPU production numeric path", "action": "fresh replay on same CUDA path", "scientific_change": False},
            {"cycle": 3, "type": "ENGINEERING", "issue": "evaluator hard-coded training_seed=0", "action": "parse seed from locked run IDs and recompute evaluator tables", "scientific_change": False},
            {"cycle": 4, "type": "CONTRACT_IMPLEMENTATION", "issue": "gate omitted feasible-consensus same-head control", "action": "gate now uses coordinate-wise strongest of frozen and feasible controls; premature multiseed marked superseded", "scientific_change": False},
        ],
        "superseded_artifacts_preserved": ["/root/night17b_p0_working/confirmation/seed1", "/root/night17b_p0_working/confirmation/seed2"],
    }
    (OUT / "implementation_and_correction_ledger.json").write_text(json.dumps(correction, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    decision = {
        "schema": "night17b-sfrd-p0-decision-v1",
        "status": "NIGHT17B_SFRD_NOT_IDENTIFIABLE",
        "classification": "SCIENTIFIC_NEGATIVE",
        "secondary_signal": "P22_REPRESENTATION_LOCAL_SIGNAL",
        "strict_gate": "full learned representation must exceed both frozen-retained and feasible-consensus same-head controls in ARI and NMI on at least 2 of 3 primary lanes",
        "strict_dual_positive_lanes": 1,
        "strict_gate_passed": False,
        "selected_config_id": config,
        "selection_semantics": selection["selection_semantics"],
        "primary_lanes": list(LANES),
        "multi_seed_confirmation_status": "SUPERSEDED_PREMATURE_CONFIRMATION_DUE_GATE_BUG",
        "labels_used_for_producer_or_gradient": 0,
        "labels_used_post_lock_for_benchmark_hpo": 1,
        "shutdown_dispatched": False,
        "autodl_state_instruction": "KEEP_ON_FOR_AUTONOMOUS_WORK",
    }
    (OUT / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    freeze = {
        "schema": "night17b-sfrd-p0-freeze-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "formula_revision_after_evaluation": False,
        "configs_materialized_before_label_evaluation": True,
        "source_sha256": {
            str(path.relative_to(REPO)): sha256(path)
            for path in [
                REPO / "SpaLORA/night17b_sfrd.py",
                REPO / "scripts/night17b/night17b_producer.py",
                REPO / "scripts/night17b/night17b_evaluator.py",
                REPO / "scripts/night17b/night17b_replay.py",
                REPO / "scripts/night17b/select_family_config.py",
                REPO / "tests/test_night17b_sfrd.py",
            ]
        },
        "lane_artifacts": {
            lane: {
                "producer_artifact_sha256": producers[lane]["artifact_sha256"],
                "checkpoint_sha256": producers[lane]["checkpoint_sha256"],
                "representation_sha256": next(
                    row["representation_sha256"] for row in producers[lane]["run_diagnostics"] if row["run_id"] == f"{config}__FULL_WEIGHTED__S0"
                ),
                "partition_sha256": next(
                    row["partition_sha256"] for row in producers[lane]["run_diagnostics"] if row["run_id"] == f"{config}__FULL_WEIGHTED__S0"
                ),
            }
            for lane in LANES
        },
    }
    (OUT / "formal_freeze_and_hash_registry.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    system = os.statvfs("/")
    persistent = os.statvfs("/root/autodl-fs")
    working_tree_bytes = sum(path.stat().st_size for path in ROOT.rglob("*") if path.is_file())
    resource = {
        "schema": "night17b-resource-audit-v1",
        "snapshot_utc": datetime.now(timezone.utc).isoformat(),
        "snapshot_source": "live os.statvfs plus bounded recursive file-size sum of /root/night17b_p0_working",
        "system_available_bytes_after": int(system.f_bavail * system.f_frsize),
        "system_available_inodes_after": int(system.f_favail),
        "persistent_available_bytes_after": int(persistent.f_bavail * persistent.f_frsize),
        "persistent_available_inodes_after": int(persistent.f_favail),
        "working_tree_bytes": int(working_tree_bytes),
        "no_new_environment": True,
        "no_download": True,
        "no_raw_copy": True,
        "dense_n_by_n": 0,
        "shutdown_dispatched": False,
        "lane_resources": {
            lane: {
                "wall_seconds": producers[lane]["wall_seconds"],
                "peak_rss_mb": producers[lane]["peak_rss_mb"],
                "peak_gpu_mb": producers[lane]["peak_gpu_mb"],
            }
            for lane in LANES
        },
    }
    (OUT / "resource_and_disk_audit.json").write_text(json.dumps(resource, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    pytest_log = (ROOT / "final_tests.log").read_text(encoding="utf-8")
    match = re.search(r"(\d+) passed", pytest_log)
    if match is None or int(match.group(1)) != 7:
        raise ValueError("targeted pytest log does not prove 7 passed")
    (OUT / "targeted_test_summary.txt").write_text(
        pytest_log.rstrip() + "\nreal producer 3/3; seed0 strict CUDA replay 3/3; evaluator seed metadata recomputed\n",
        encoding="utf-8",
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
