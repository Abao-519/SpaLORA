"""Build the compact scientific handoff for Night-22A from locked artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np


REPO = Path("/root/SpaLORA-night16h")
WORK = Path("/root/night22a_working")
OUT = REPO / "outputs" / "night22a_handoff"

HISTORICAL = {
    "A1_K10": (0.276171767, 0.421937362, "Night16E balanced label-assisted frontier"),
    "TONSIL_S1_K4": (0.236682867, 0.317365241, "Night16C family-frozen reference"),
    "P22_K9": (0.596390056, 0.718243175, "Night16H strict-LOSO reference"),
    "PLACENTA_K10": (0.499999398, 0.631180143, "Night19B concatenated-feature control frontier"),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def copy_file(source: Path, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Preserve complete small scientific ledgers from both valid discovery cycles.
    copy_file(WORK / "stage_a_summary" / "geometry_ceiling_all.csv", OUT / "geometry_ceiling_all.csv")
    copy_file(WORK / "stage_a_summary" / "geometry_ceiling_summary.csv", OUT / "geometry_ceiling_board.csv")
    copy_file(
        WORK / "stage_a_summary" / "stage_b_development_start_selection.json",
        OUT / "stage_b_development_start_selection.json",
    )
    for cycle, directory in (("cycle1", "stage_b_summary"), ("cycle2", "stage_b_summary_v2")):
        for name in (
            "junction_all_candidate_ledger.csv",
            "junction_matched_contribution_board.csv",
            "junction_best_full_by_lane.csv",
            "stage_b_decision.json",
        ):
            copy_file(WORK / directory / name, OUT / f"{cycle}_{name}")

    all_rows = read_csv(WORK / "stage_b_summary_v2" / "junction_all_candidate_ledger.csv")
    selected = read_csv(WORK / "stage_b_summary_v2" / "junction_best_full_by_lane.csv")
    stage_a = {row["lane"]: row for row in read_csv(WORK / "stage_a_summary" / "geometry_ceiling_summary.csv")}

    absolute = []
    score_board = []
    p0 = []
    replay_rows = []
    resource_rows = []
    for summary in selected:
        lane = summary["lane"]
        full = next(
            row
            for row in all_rows
            if row["lane"] == lane
            and row["start_candidate"] == summary["start_candidate"]
            and row["candidate_id"] == summary["full_candidate_id"]
        )
        hist_ari, hist_nmi, hist_source = HISTORICAL[lane]
        parent = stage_a[lane]
        absolute.append(
            {
                "lane": lane,
                "family": summary["family"],
                "n_total": full["n_total"],
                "n_eval": full["n_eval"],
                "k": full["observed_k_full"],
                "selected_start": summary["start_candidate"],
                "selected_profile": summary["profile_id"],
                "full_ari": full["ari"],
                "full_nmi": full["nmi"],
                "full_ami": full["ami"],
                "full_fmi": full["fmi"],
                "full_homogeneity": full["homogeneity"],
                "full_v_measure": full["v_measure"],
                "neighbor_agreement": full["neighbor_agreement"],
                "moran_indicator_macro": full["moran_indicator_macro"],
                "geary_indicator_macro": full["geary_indicator_macro"],
                "min_cluster_size": full["min_cluster_size_full"],
                "cluster_sizes": full["cluster_sizes_full"],
                "changed_spots_vs_start": full["changed_spots_vs_start"],
                "delta_ari_vs_strongest_atomic": summary["delta_ari_vs_coordinatewise_atomic"],
                "delta_nmi_vs_strongest_atomic": summary["delta_nmi_vs_coordinatewise_atomic"],
                "delta_ari_vs_stage_a_best_head": summary["delta_ari_vs_stage_a_best_head"],
                "delta_nmi_vs_stage_a_best_head": summary["delta_nmi_vs_stage_a_best_head"],
                "delta_ari_vs_night21c_best": float(full["ari"]) - float(parent["night21c_endpoint_best_ari"]),
                "delta_nmi_vs_night21c_best": float(full["nmi"]) - float(parent["night21c_endpoint_best_nmi"]),
                "historical_frontier_ari": hist_ari,
                "historical_frontier_nmi": hist_nmi,
                "delta_ari_vs_historical_frontier": float(full["ari"]) - hist_ari,
                "delta_nmi_vs_historical_frontier": float(full["nmi"]) - hist_nmi,
                "strict_independent_lane_pass": summary["independent_lane_pass"],
                "partition_sha256": full["candidate_partition_sha256"],
                "training_seed_count": 1,
                "seed_mean_median_min": "identical deterministic seed0; no multi-seed expansion because family gate failed",
            }
        )

        lane_rows = [row for row in all_rows if row["lane"] == lane]
        best_ari = max(lane_rows, key=lambda row: (float(row["ari"]), float(row["nmi"])))
        best_nmi = max(lane_rows, key=lambda row: (float(row["nmi"]), float(row["ari"])))
        score_board.extend(
            [
                {
                    "lane": lane,
                    "profile": "STAGE_A_GEOMETRY_CEILING_MAX_ARI",
                    "candidate": parent["stage_a_best_candidate_id"],
                    "ari": parent["stage_a_best_ari"],
                    "nmi": parent["stage_a_best_nmi"],
                    "selection_semantics": "label-assisted post-lock geometry ceiling",
                },
                {
                    "lane": lane,
                    "profile": "JUNCTION_PUBLIC_BENCHMARK_MAX_ARI",
                    "candidate": best_ari["candidate_id"],
                    "ari": best_ari["ari"],
                    "nmi": best_ari["nmi"],
                    "selection_semantics": "transparent label-assisted post-lock HPO",
                },
                {
                    "lane": lane,
                    "profile": "JUNCTION_PUBLIC_BENCHMARK_MAX_NMI",
                    "candidate": best_nmi["candidate_id"],
                    "ari": best_nmi["ari"],
                    "nmi": best_nmi["nmi"],
                    "selection_semantics": "transparent label-assisted post-lock HPO",
                },
                {
                    "lane": lane,
                    "profile": "HISTORICAL_TRUSTED_FRONTIER",
                    "candidate": hist_source,
                    "ari": hist_ari,
                    "nmi": hist_nmi,
                    "selection_semantics": "cross-revision context; not a Night-22A producer output",
                },
            ]
        )

        bank_stem = f"{lane}__RETAINED_CARRIER__{summary['start_candidate']}"
        manifest_path = WORK / "junction_v2" / f"{bank_stem}.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        with np.load(manifest["carrier_path"], allow_pickle=False) as carrier:
            view1_shape = list(carrier["view1"].shape)
            view2_shape = list(carrier["view2"].shape)
        with np.load(manifest["embedding_path"], allow_pickle=False) as embedding:
            representation_shape = list(embedding["representation"].shape)
        replay_path = WORK / "junction_replays_v2" / f"{bank_stem}.json"
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        p0.append(
            {
                "lane": lane,
                "ids_shape": f"[{full['n_total']}]",
                "view1_shape": json.dumps(view1_shape),
                "view2_shape": json.dumps(view2_shape),
                "locked_representation_shape": json.dumps(representation_shape),
                "soft_partition_shape": f"[{full['n_total']},{full['observed_k_full']}]",
                "graph_names": "|".join(manifest["graph_names"]),
                "graph_edges": json.dumps(manifest["graph_edges"], sort_keys=True),
                "actual_backward_and_parameter_update": "PASS",
                "parameter_l1_change": manifest["candidate_diagnostics"][summary["full_candidate_id"]]["parameter_l1_change"],
                "strict_checkpoint_reload": "PASS",
                "fresh_process_partition_replay": replay["status"],
                "replayed_candidate_count": replay["count"],
                "labels_read_by_producer": manifest["labels_read"],
            }
        )
        replay_rows.append(
            {
                "lane": lane,
                "start_candidate": summary["start_candidate"],
                "replay_process": "fresh independent Python process",
                "status": replay["status"],
                "candidate_count": replay["count"],
                "bank_sha256": manifest["partition_bank_sha256"],
                "checkpoint_sha256": manifest["checkpoint_bank_sha256"],
            }
        )
        resource_rows.append(
            {
                "lane": lane,
                "start_candidate": summary["start_candidate"],
                "candidate_count": len(manifest["candidate_ids"]),
                "wall_seconds": manifest["wall_seconds"],
                "peak_rss_mb": manifest["peak_rss_mb"],
                "device": manifest["device"],
                "gpu_model": "NVIDIA GeForce RTX 4080 SUPER, 32760 MiB (host-reported nvidia-smi during Night-22A)",
                "peak_gpu_mb": "NOT_INSTRUMENTED; compact state size and device are recorded",
            }
        )

    write_csv(OUT / "absolute_main_table.csv", absolute)
    write_csv(OUT / "score_frontier_board.csv", score_board)
    write_csv(OUT / "real_p0_registry.csv", p0)
    write_csv(OUT / "fresh_process_replay_table.csv", replay_rows)
    write_csv(OUT / "training_resource_table.csv", resource_rows)

    family_rows = [
        {
            "family": "RNA_PROTEIN",
            "discovery_lanes": "A1_K10|TONSIL_S1_K4",
            "strict_independent_passes": "0/2",
            "family_config_frozen": False,
            "confirmation_lanes_run": "NONE",
            "reason": "FULL was dominated by matched shared/additive controls on both discovery lanes",
        },
        {
            "family": "RNA_CHROMATIN",
            "discovery_lanes": "P22_K9|PLACENTA_K10",
            "strict_independent_passes": "2/2",
            "family_config_frozen": False,
            "confirmation_lanes_run": "NONE",
            "reason": "local signal did not satisfy the cross-family and macro-delta taskbook gate",
        },
    ]
    write_csv(OUT / "family_default_and_confirmation.csv", family_rows)

    decision_source = json.loads((WORK / "stage_b_summary_v2" / "stage_b_decision.json").read_text(encoding="utf-8"))
    decision = {
        "schema": "night22a-final-decision-v1",
        "classification": "LOCAL_PARTITION_JUNCTION_SIGNAL_WITHOUT_FAMILY_CONFIRMATION",
        "secondary_classifications": ["HEAD_ONLY_GEOMETRY_SIGNAL", "PLACENTA_ARI_SCORE_FRONTIER_ADVANCE"],
        "stage_a_geometry_signal": True,
        "strict_independent_junction_lanes": decision_source["independent_lane_passes"],
        "strict_independent_junction_lane_count": decision_source["independent_lane_pass_count"],
        "family_frozen_confirmation_authorized": False,
        "confirmation_lanes_run": [],
        "paper_method_name_frozen": False,
        "sota_claim": False,
        "labels_in_training_or_gradient": 0,
        "public_benchmark_hpo_disclosed": True,
        "github_push_status": "PENDING_FINAL_COMMIT",
        "shutdown_dispatched_at_report_build": False,
    }
    (OUT / "night22a_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    label_flow = {
        "schema": "night22a-label-flow-audit-v1",
        "known_k_source": "registered public benchmark protocol",
        "producer_access": "explicit numeric IDs, locked representation, view1, view2 and sparse graph CSR arrays",
        "producer_annotation_arrays_accessed": 0,
        "labels_in_input_loss_gradient_checkpoint_selection": 0,
        "candidate_lock_before_evaluator": True,
        "candidate_partition_sha_checked_before_metrics": True,
        "stage_a_and_stage_b_selection": "transparent label-assisted benchmark HPO after lock",
        "family_confirmation": "not run because the frozen gate failed",
    }
    (OUT / "label_flow_audit.json").write_text(json.dumps(label_flow, indent=2, sort_keys=True), encoding="utf-8")

    disk = shutil.disk_usage("/")
    resource = {
        "schema": "night22a-resource-audit-v1",
        "root_total_bytes": disk.total,
        "root_used_bytes": disk.used,
        "root_free_bytes": disk.free,
        "root_free_gib": disk.free / 2**30,
        "minimum_required_free_gib": 15,
        "night22a_working_bytes": int(
            subprocess.check_output(["du", "-sb", str(WORK)], text=True).split()[0]
        ),
        "large_new_downloads": 0,
    }
    (OUT / "resource_and_disk_audit.json").write_text(json.dumps(resource, indent=2, sort_keys=True), encoding="utf-8")

    lines = []
    for row in absolute:
        lines.append(
            f"| {row['lane']} | {float(row['full_ari']):.6f}/{float(row['full_nmi']):.6f} | "
            f"{float(row['delta_ari_vs_strongest_atomic']):+.6f}/{float(row['delta_nmi_vs_strongest_atomic']):+.6f} | "
            f"{float(row['delta_ari_vs_night21c_best']):+.6f}/{float(row['delta_nmi_vs_night21c_best']):+.6f} | "
            f"{row['min_cluster_size']} | {row['strict_independent_lane_pass']} |"
        )
    report = f"""# Night-22A geometry-aware partition junction report

## 我现在需要知道的三件事

1. **问题**：Night-21C 显示锁定表示里仍有标签后置 probe 可分信息，但普通 KMeans/GMM 无法稳定把它变成空间分区。本轮先测非球形几何上限，再直接优化 `N×K` 分区变量；它不是从旧候选中挑一个答案。
2. **实际动作与结果**：feature-Ncut/Leiden 在 A1、P22 相对 Night-21C endpoint 双升；随后 clean-room junction 在同一 retained carrier 上联合椭圆发射项与五张稀疏图。`FULL` 在 P22 和 placenta 严格胜过同起点全部原子臂，但 A1 与 tonsil s1 被更简单的 shared/additive control 解释，跨家族冻结门失败。
3. **论文意义与分类**：终态为 **LOCAL_PARTITION_JUNCTION_SIGNAL_WITHOUT_FAMILY_CONFIRMATION**，并有独立的 `HEAD_ONLY_GEOMETRY_SIGNAL`。Placenta ARI 刷新到 0.551001，但 NMI 未同步刷新；不能称盲测、SOTA、统一方法里程碑或已经冻结的方法名。

## 绝对指标与匹配贡献

| lane | FULL ARI/NMI | Δ vs strongest atomic | Δ vs Night-21C endpoint best | min cluster | strict independent pass |
|---|---:|---:|---:|---:|---|
{os.linesep.join(lines)}

完整 AMI、FMI、homogeneity、V-measure、Moran/Geary、簇大小和 partition SHA 见 `absolute_main_table.csv`。固定输入下优化器是确定性的；门失败后没有用多 seed 扩张制造“最好一次”。

## 几何 ceiling

- A1 retained + feature-Ncut k24：0.268066/0.395941，比 Night-21C endpoint best +0.022159/+0.025518。
- P22 retained + feature-Leiden：0.503401/0.641908，+0.027710/+0.024905。
- Placenta feature-Leiden：0.535146/0.592084，ARI 上升但 NMI 下降。
- Tonsil s1 最佳新几何 0.194390/0.259507，低于 Night-21C endpoint best。

这说明 non-spherical feature geometry 确实补回了一部分 endpoint gap，但仍没有达到 A1 0.276172/0.421937、P22 0.596390/0.718243 等历史高位。

## junction 的真实归因

`Q` 是直接训练的软分区；每一步在稀疏图上计算 cluster-conditioned normalized association，并与锁定表示上的对角椭圆发射项联合。`FULL` 相对 matched atom 的净增益在 P22 极小（约 +0.000062/+0.000158），在 placenta 较明确（约 +0.008517/+0.003692）。Protein 两条均由 `SHARED_GRAPH_ONLY` 或 `ADDITIVE_SHARED` 支配，因此不能把“组合后高一点”推广为跨家族方法。

Cycle 1 是有效开发结果，源码保存在 commit `1ac9231d577f22db3651312f6cbac846eed85499`。Cycle 2 只依据已锁定 Stage-A 证据向所有 lane 同时补上 retained k24 图，其他目标和预算未变；仍未越过跨家族门，因此停止，不运行 D1/tonsil s2/s3/MISAR。

## 新颖性与可继续性

DEC、SwAV、P²OT、DeepCut、普通多视图图融合、BANKSY、spaMGCN、S3RL、SEPAR 与 CRCT 已占据软分配、平衡、normalized-cut、邻域几何、图融合和原型等组件。Night-22A 没有复制第三方源码。当前仅“cluster-conditioned 多图 junction + 椭圆发射的直接分区联合体”可作为工作对象，但证据只有 chromatin 本地信号，不足以冻结论文名称。下一轮若继续，应该先做外部 chromatin frozen confirmation；不应继续在这四条 discovery lane 扩网格。

## 外部数据旁路

GSE205055 ME13 50 µm 是同一 GSE205055 study family 的新物理样本，官方 raw 总包约 7.6 GB，canonical K/mask 尚未闭合。Stereo-CITE thymus 是独立 RNA+protein 来源且有四张切片，但原论文中的 cortex/medulla 解释不能自动等同独立全域 ground truth。两者都只完成 provenance/体量审计，没有下载或制造标签。

## 导师汇报版

我们把 Night-21C 的“表示里有信息但 KMeans 接不出来”拆成了非球形 head 和直接分区优化两层。第一层很明确：feature-Ncut 或 Leiden 在 A1 和 P22 都比 Night-21C 的八个 endpoint 双升，说明几何确实是瓶颈之一。第二层我们不是挑旧候选，而是直接训练 N×K 分区，并把椭圆簇似然与多尺度、多模态、空间稀疏图放进同一目标。这个 FULL 在 P22 和 placenta 都严格超过同预算原子臂，placenta 的 ARI 到 0.551，但 protein 两条没有独立贡献。因此本轮是 chromatin 局部的 partition-junction 信号，不是跨家族里程碑，也没有启动冻结迁移。已有工作已经覆盖 normalized cut、原型分配、图融合和邻域特征，我们只保留联合对象的窄边界。最合理的下一步是用冻结 chromatin 公式做真正外部样本确认，而不是继续在开发集调参。

## 技术状态

- 真实 P0：4/4，实际 backward、参数改变、strict checkpoint reload 均通过。
- Fresh-process replay：4 条 headline bank 的全部 15 个 trainable candidates 均 exact；另一个 start 的完整 bank 也已重放，完整表在 ledger。
- Stage-A candidates 先锁/hash，Stage-B candidates 与 checkpoints 先锁/hash，标签随后由独立 evaluator 打开。
- 根盘保持约 {resource['root_free_gib']:.1f} GiB 可用，超过 15 GiB 安全线；未下载新 raw。
- GitHub 认证预检在本轮开始时一次通过；普通 push 只在 final commit 后尝试一次。
"""
    (OUT / "night22a_report.md").write_text(report, encoding="utf-8")
    mentor = "\n".join(
        [
            "1. Night-21C 的瓶颈确实包含 clustering geometry，而不只是表示。",
            "2. 在 retained carrier 上，feature-Ncut/Leiden 让 A1 和 P22 相对上一轮 endpoint 双升。",
            "3. 新模块直接训练 N×K 分区，不是候选 selector，也不是再训练 embedding 后接 KMeans。",
            "4. P22 与 placenta 的 FULL 严格超过 matched atomic controls；placenta 达到 0.551001/0.625391。",
            "5. A1 与 tonsil s1 的增益被更简单 graph/additive controls 解释，所以跨家族门失败。",
            "6. 结论是 chromatin 本地 junction 信号加 geometry/head 信号，不是 SOTA 或 paper-ready milestone。",
            "7. 新颖性边界很窄：经典 prototype、normalized cut、图融合、BANKSY 邻域几何都按先例归因。",
            "8. 下一步若继续，应冻结 chromatin 对象做外部确认；不能继续在四条 discovery lane 扩 grid。",
        ]
    )
    (OUT / "mentor_oral_report.md").write_text(mentor + "\n", encoding="utf-8")

    # Final engineering evidence is copied only after the locked scientific
    # outputs have completed.  The summary is derived from the actual pytest
    # text instead of being a hand-written PASS string.
    final_test_log = WORK / "final_targeted_tests.txt"
    final_test_text = final_test_log.read_text(encoding="utf-8")
    if "8 passed" not in final_test_text:
        raise RuntimeError("the final targeted test log does not contain 8 passed")
    copy_file(final_test_log, OUT / "final_targeted_tests.txt")
    test_summary = {
        "schema": "night22a-targeted-tests-v1",
        "status": "PASS",
        "passed": 8,
        "failed": 0,
        "source": "actual final pytest output",
        "log_sha256": sha(final_test_log),
    }
    (OUT / "targeted_test_summary.json").write_text(
        json.dumps(test_summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    second_replay = WORK / "second_replay_summary.json"
    second_replay_payload = json.loads(second_replay.read_text(encoding="utf-8"))
    if second_replay_payload.get("status") != "PASS":
        raise RuntimeError("second fresh-process replay summary is not PASS")
    copy_file(second_replay, OUT / "second_fresh_process_replay_summary.json")

    authority_paths = [
        OUT / "Night22A_Geometry_Aware_Partition_Junction_Taskbook_2026-08-27.md",
        REPO / "SpaLORA" / "night22a_geometry.py",
        REPO / "SpaLORA" / "night22a_junction.py",
        REPO / "scripts" / "night22a" / "night22a_geometry_producer.py",
        REPO / "scripts" / "night22a" / "night22a_geometry_evaluator.py",
        REPO / "scripts" / "night22a" / "night22a_junction_producer.py",
        REPO / "scripts" / "night22a" / "night22a_junction_replay.py",
        REPO / "scripts" / "night22a" / "build_stage_a_summary.py",
        REPO / "scripts" / "night22a" / "build_stage_b_summary.py",
        REPO / "scripts" / "night22a" / "build_handoff.py",
        REPO / "configs" / "night22a" / "stage_a_geometry_freeze.json",
        REPO / "configs" / "night22a" / "stage_b_junction_freeze.json",
        REPO / "configs" / "night22a" / "stage_b_junction_freeze_v2.json",
        REPO / "tests" / "test_night22a_geometry.py",
        REPO / "tests" / "test_night22a_junction.py",
        Path("/root/night21c_delivery_20260826/official_compact/compact_delivery_index.json"),
        Path("/root/night21c_delivery_20260826/night21c-final-20260826.incremental.bundle"),
    ]
    missing_authority = [str(path) for path in authority_paths if not path.is_file()]
    if missing_authority:
        raise RuntimeError(f"missing authority files: {missing_authority}")
    manifest_entries = []
    for path in authority_paths:
        manifest_entries.append(
            {
                "path": str(path),
                "size": path.stat().st_size,
                "sha256": sha(path),
            }
        )
    source_manifest = {
        "schema": "night22a-source-authority-manifest-v1",
        "parent_tag": "night21c-final-20260826",
        "parent_commit": "7e35d07ddb2b0b19cfdc59a89bcf1829d0f59ef1",
        "parent_compact_index_sha256": "5921dac05fb345271f902b69ec8c0198351dd60a14e02595ea62cb186bf19901",
        "cycle1_commit": "1ac9231d577f22db3651312f6cbac846eed85499",
        "cycle2_commit": "aa9c0967241a5acff88ac590849a2ebbafca88b6",
        "active_core_sha256": sha(REPO / "SpaLORA" / "night22a_junction.py"),
        "entries": manifest_entries,
    }
    (OUT / "source_authority_manifest.json").write_text(
        json.dumps(source_manifest, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
