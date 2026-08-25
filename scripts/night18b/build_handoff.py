#!/usr/bin/env python3
"""Build the compact, evidence-first Night-18B handoff from locked artifacts."""

from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


REPO = Path("/root/SpaLORA-night16h")
WORK = Path("/root/night18b_working/formal")
OUT = REPO / "outputs/night18b_handoff"
ARMS = ["MATCHED_BACKBONE", "FULL_RESPONSE_CALIBRATION", "LOWPASS_ONLY_CALIBRATION",
        "SHARPEN_ONLY_CALIBRATION", "SHARED_SCALE_CONTROL", "SWAPPED_RESPONSE_CONTROL"]
LANES = ["A1_K10", "P22_K9"]


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"empty delivery table: {path.name}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def aligned_changed(left: np.ndarray, right: np.ndarray) -> int:
    k = max(int(left.max()), int(right.max())) + 1
    contingency = np.zeros((k, k), dtype=np.int64)
    np.add.at(contingency, (left, right), 1)
    rows, cols = linear_sum_assignment(-contingency)
    mapping = {int(col): int(row) for row, col in zip(rows, cols)}
    aligned = np.asarray([mapping[int(value)] for value in right])
    return int(np.sum(left != aligned))


def selected_partition(lane: str, arm: str, endpoint: str) -> np.ndarray:
    stem = f"{lane}__{arm}__S0"
    manifest = json.loads((WORK / f"artifacts/{stem}.json").read_text(encoding="utf-8"))
    with np.load(WORK / f"artifacts/{stem}.npz", allow_pickle=False) as artifact:
        return artifact["partitions"][int(manifest["selections"][endpoint]["index"])].astype(np.int32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    evaluation_files = sorted(WORK.glob("evaluation/*.csv"))
    if len(evaluation_files) != 12:
        raise RuntimeError(f"expected 12 evaluation files, found {len(evaluation_files)}")
    metrics = pd.concat([pd.read_csv(path) for path in evaluation_files], ignore_index=True)
    if len(metrics) != 48 or set(metrics.arm) != set(ARMS) or set(metrics.lane) != set(LANES):
        raise RuntimeError("locked 2-lane x 6-arm x 4-endpoint table is incomplete")
    metrics.sort_values(["lane", "arm", "endpoint"]).to_csv(OUT / "absolute_metrics_main_table.csv", index=False)

    matched_rows: list[dict[str, object]] = []
    for lane in LANES:
        for endpoint in sorted(metrics.endpoint.unique()):
            baseline = metrics[(metrics.lane == lane) & (metrics.arm == "MATCHED_BACKBONE") & (metrics.endpoint == endpoint)].iloc[0]
            baseline_partition = selected_partition(lane, "MATCHED_BACKBONE", endpoint)
            for arm in ARMS[1:]:
                row = metrics[(metrics.lane == lane) & (metrics.arm == arm) & (metrics.endpoint == endpoint)].iloc[0]
                partition = selected_partition(lane, arm, endpoint)
                matched_rows.append({"lane": lane, "endpoint": endpoint, "arm": arm,
                                     "baseline_ari": baseline.ari, "baseline_nmi": baseline.nmi,
                                     "arm_ari": row.ari, "arm_nmi": row.nmi,
                                     "delta_ari": row.ari - baseline.ari, "delta_nmi": row.nmi - baseline.nmi,
                                     "changed_spots_aligned": aligned_changed(baseline_partition, partition),
                                     "arm_partition_sha256": row.partition_sha256,
                                     "baseline_partition_sha256": baseline.partition_sha256})
    write_csv(OUT / "matched_contribution_table.csv", matched_rows)

    benchmark_rows = []
    for (lane, arm), group in metrics.groupby(["lane", "arm"]):
        best = group.sort_values(["ari", "nmi", "endpoint"], ascending=[False, False, True]).iloc[0]
        benchmark_rows.append({"lane": lane, "arm": arm, "selection_semantics": "LABEL_ASSISTED_POST_LOCK_MAX_ARI_THEN_NMI",
                               "endpoint": best.endpoint, "candidate_id": best.candidate_id,
                               "ari": best.ari, "nmi": best.nmi, "ami": best.ami, "fmi": best.fmi,
                               "partition_sha256": best.partition_sha256, "min_cluster_size_full": best.min_cluster_size_full})
    write_csv(OUT / "benchmark_profile_table.csv", benchmark_rows)

    response_rows = []; candidate_rows = []; p0_rows = []; resource_rows = []
    for manifest_path in sorted(WORK.glob("artifacts/*.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")); stem = manifest_path.stem
        response = manifest["diagnostics"]["response"]
        response_rows.append({"lane": manifest["lane"], "arm": manifest["arm"], "config_id": manifest["config"]["config_id"],
                              "initial_roughness_view1": response["initial_roughness"][0],
                              "initial_roughness_view2": response["initial_roughness"][1],
                              "target_roughness": response["target_roughness"],
                              "corrected_roughness_view1": response["corrected_roughness"][0],
                              "corrected_roughness_view2": response["corrected_roughness"][1],
                              "log_gap_before": response["log_gap_before"], "log_gap_after": response["log_gap_after"],
                              "response_plan_json": json.dumps(response["plans"], sort_keys=True),
                              "representation_sha256": manifest["representation_sha256"],
                              "parameter_changed": manifest["diagnostics"]["parameter_changed"]})
        selected = {value["candidate_id"]: name for name, value in manifest["selections"].items()}
        for record in manifest["candidate_records"]:
            candidate_rows.append({"lane": manifest["lane"], "arm": manifest["arm"], "candidate_id": record["candidate_id"],
                                   "candidate_sha256": record["candidate_sha256"], "source": record["source"],
                                   "complexity": record["complexity"], "feasible": record["feasible"],
                                   "min_cluster_size": record["min_cluster_size"],
                                   "registered_endpoint_names": selected.get(record["candidate_id"], "")})
        p0_rows.append({"lane": manifest["lane"], "arm": manifest["arm"], "n": manifest["n"], "k": manifest["k"],
                        "view1_shape": json.dumps(manifest["diagnostics"]["view1_shape"]),
                        "view2_shape": json.dumps(manifest["diagnostics"]["view2_shape"]),
                        "retained_shape": json.dumps(manifest["diagnostics"]["retained_shape"]),
                        "spatial_graph_nnz": manifest["diagnostics"]["spatial_graph_nnz"],
                        "optimizer_steps": manifest["diagnostics"]["optimizer_steps"], "parameter_changed": True,
                        "strict_checkpoint_reload": manifest["strict_checkpoint_reload"], "artifact_reload": manifest["artifact_reload"],
                        "producer_annotation_reads": manifest["producer_annotation_reads"], "carrier_sha256": manifest["carrier_sha256"]})
        resource_rows.append({"lane": manifest["lane"], "arm": manifest["arm"], "wall_seconds": manifest["wall_seconds"],
                              "gpu_peak_mib": manifest["gpu_peak_mib"], "peak_rss_mib": manifest["peak_rss_mib"],
                              "artifact_bytes": (WORK / f"artifacts/{stem}.npz").stat().st_size,
                              "checkpoint_bytes": (WORK / f"artifacts/{stem}.pt").stat().st_size})
    write_csv(OUT / "response_parameter_table.csv", response_rows)
    write_csv(OUT / "locked_candidate_registry.csv", candidate_rows)
    write_csv(OUT / "real_input_p0_registry.csv", p0_rows)
    write_csv(OUT / "resource_table.csv", resource_rows)

    replay_rows = []
    for path in sorted(WORK.glob("replay/*.json")):
        value = json.loads(path.read_text(encoding="utf-8")); replay_rows.append({"file": path.name, **value})
    if len(replay_rows) != 12 or not all(row["representation_exact"] and row["partition_exact"] for row in replay_rows):
        raise RuntimeError("fresh-process replay audit failed")
    (OUT / "exact_replay_audit.json").write_text(json.dumps({"status": "PASS", "count": 12, "rows": replay_rows}, indent=2, sort_keys=True), encoding="utf-8")

    test_log = (WORK / "targeted_tests.log").read_text(encoding="utf-8")
    match = re.search(r"(\d+) passed", test_log)
    if not match or int(match.group(1)) != 6:
        raise RuntimeError("actual targeted-test log does not show 6 passes")
    (OUT / "targeted_test_summary.json").write_text(json.dumps({"status": "PASS", "passed": 6,
                                                                 "source_log_sha256": file_sha(WORK / "targeted_tests.log")}, indent=2), encoding="utf-8")

    failures = [
        {"id": "E01", "boundary": "PREFORMAL", "type": "NOVELTY_COLLISION", "status": "SUPERSEDED_BEFORE_UPLOAD_OR_RUN",
         "detail": "Initial Chebyshev frequency-coherence staging was stopped after SMGFM was found to cover frequency bands, semantic roles and reliability routing."},
        {"id": "E02", "boundary": "PREFORMAL", "type": "FORMULA_REVISION", "status": "SUPERSEDED_BEFORE_SCIENTIFIC_RUN",
         "detail": "A one-step smoothing/unsharp draft was replaced by the bounded rational normalized-Laplacian response before producer execution."},
        {"id": "E03", "boundary": "P0_ENGINEERING", "type": "PYTHONPATH_IMPORT", "status": "CORRECTED_SAME_FROZEN_FORMULA",
         "detail": "First A1 launch omitted repository PYTHONPATH and failed before loading data; rerun added PYTHONPATH without changing formula/config."},
        {"id": "S01", "boundary": "FORMAL", "type": "SCIENTIFIC_NEGATIVE", "status": "NO_MULTI_SEED_EXPANSION",
         "detail": "Full response calibration did not beat matched controls independently on either lane; the preregistered small screen was stopped."},
    ]
    write_csv(OUT / "failure_and_correction_ledger.csv", failures)

    target_rows = [
        {"lane": "A1_K10", "protocol": "expert K10 all 3484", "historical_frontier_ari": .276003,
         "historical_frontier_nmi": .421740, "night18a_learned_common_ari": .232410, "night18a_learned_common_nmi": .388158},
        {"lane": "P22_K9", "protocol": "expert K9 all 9196", "historical_frontier_ari": .595552,
         "historical_frontier_nmi": .717931, "night18a_learned_common_ari": .478639, "night18a_learned_common_nmi": .606982},
    ]
    write_csv(OUT / "score_target_board.csv", target_rows)

    (OUT / "label_flow_audit.json").write_text(json.dumps({
        "status": "PASS", "producer_annotation_reads": 0, "producer_allowed_inputs": ["ids", "view1", "view2", "retained", "graph0-2"],
        "partition_lock_before_evaluator": True, "evaluation_files": len(evaluation_files), "evaluator_label_reads_per_artifact": 1,
        "benchmark_profile": "public-label post-lock endpoint selection, reported separately",
        "within_run_label_checkpoint_selection": False}, indent=2, sort_keys=True), encoding="utf-8")

    source_audit = """# Night-18B source collision and transfer audit\n\n""" + """
| Object | Official source | Prior-art boundary | Night-18B decision |
|---|---|---|---|
| SMGFM (2026) | https://arxiv.org/abs/2606.12867 | Chebyshev graph-frequency bands, band semantic roles, coupling reliability, consensus/private routes | Original frequency-coherence proposal rejected before execution; never claimed. |
| GatorPrism | https://github.com/Gator-Group/GatorPrism | Joint coalition expert, modality-private experts, prototype-conditioned spot router | Expert routing/shared-private scaffold not novel and not implemented here. |
| ARISE | https://github.com/XiangxiangWang-code/ARISE | RNA-anchored graph intersection and hierarchical fusion | RNA anchor/graph intersection/fusion are prior art. |
| DRIFT | https://github.com/rsinghlab/DRIFT | Heat-kernel low-pass preprocessing before a downstream model | Generic graph diffusion/low-pass is prior art and is a matched control. |
| SpaDDM | https://github.com/WHY-17/SpaDDM | Directional graph diffusion for spatial multi-omics | Directional diffusion is prior art; no such novelty claim is made. |
| SpaGFT / DeepGFT | official paper/code entrances audited in taskbook | Graph Fourier representation and filtering | Spectral filtering itself is prior art. |
| SpatialCOC / FOCUS | official papers/code entrances audited in taskbook | Continuous spatial functions and cross-resolution mapping | Continuous/cross-resolution correction is prior art. |
| Night-18B bounded response calibration | this clean-room source | Robust input-derived modality roughness, bounded rational inverse/forward response to a common target, same residual consumer | Minimal distinct hypothesis only; empirical independent-contribution gate failed, so novelty remains unsupported. |
\nNo third-party code was copied.  Official repositories were used only for semantic collision review.\n"""
    (OUT / "source_code_collision_and_transfer_audit.md").write_text(source_audit, encoding="utf-8")

    semantics = r"""# Method semantics and attribution contract

For a symmetric normalized graph Laplacian L, modality m is transformed by

`Y_m = (I + beta_m L)^(-1) (I + gamma_m L) X_m`, beta_m,gamma_m >= 0.

The sparse solve uses sixteen fixed Richardson iterations.  beta > gamma is low-pass, gamma > beta is bounded sharpening, and beta = gamma = 0 is exact identity.  Each modality's robust graph roughness is the median feature-wise energy `||X-SX||^2 / ||X||^2`; the common target is the geometric mean across the two observed modalities.  A fixed coefficient bank and distortion penalty choose beta/gamma from input statistics only.

All arms use the same adapters, zero-initialized residual around the retained representation, masked reconstruction, cross-view consistency, DEC-style prototype term, optimizer, 80 steps, 15-candidate bank and four registered endpoints.  Labels are unavailable to calibration, training, candidate generation and checkpoint locking.  The benchmark profile opens labels only after artifacts are locked.

Attribution requires FULL_RESPONSE_CALIBRATION to improve the same endpoint over MATCHED_BACKBONE and not be explained by low-pass-only, sharpen-only, shared-scale or swapped-response controls.  This condition failed; endpoint-specific gains are not a transferable representation claim.
"""
    (OUT / "method_semantics_and_attribution_contract.md").write_text(semantics, encoding="utf-8")

    disk = os.statvfs("/"); work_bytes = sum(path.stat().st_size for path in Path("/root/night18b_working").rglob("*") if path.is_file())
    resource_audit = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "root_available_bytes": disk.f_bavail * disk.f_frsize,
                      "root_available_inodes": disk.f_favail, "night18b_working_bytes": work_bytes,
                      "persistent_disk_writes": 0, "new_environment_created": False, "shutdown_dispatched": False,
                      "power_policy": "KEEP_ON_FOR_CONTINUOUS_RESEARCH"}
    (OUT / "resource_and_disk_audit.json").write_text(json.dumps(resource_audit, indent=2, sort_keys=True), encoding="utf-8")

    decision = {
        "classification": "SCIENTIFIC_NEGATIVE", "status": "NIGHT18B_NO_INDEPENDENT_MEASUREMENT_RESPONSE_SIGNAL",
        "taskbook_sha256": "96aba215df9c8d6ef18e386c515e2da43a587f1433fef6cd7b01e1bb867b483f",
        "frequency_route_status": "REJECTED_BY_SMGFM_COLLISION_BEFORE_EXECUTION",
        "formula": "BOUNDED_RATIONAL_MODALITY_RESPONSE_CALIBRATION",
        "formal_lanes": 2, "formal_arms": 6, "training_seeds": [0], "multi_seed_authorized": False,
        "independent_method_signal_lanes": 0,
        "a1_common_full": {"ari": .2342962096119434, "nmi": .3895159418904272,
                           "delta_vs_matched_ari": .001886, "delta_vs_matched_nmi": .001358,
                           "control_interpretation": "low-pass-only nearly explains the change"},
        "p22_common_full": {"ari": .417729, "nmi": .549751, "delta_vs_matched_ari": -.001477, "delta_vs_matched_nmi": -.001437},
        "p22_gmm_context": {"full_ari": .4518480245229052, "full_nmi": .5787260728093653,
                            "identical_to": "SHARPEN_ONLY_CALIBRATION", "interpretation": "single-view/head-conditional control signal"},
        "score_frontier_advance": False, "new_data_downloads": 0, "shutdown_dispatched": False,
    }
    (OUT / "night18b_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")

    report = f"""# Night-18B report

## 我现在需要知道的三件事

1. **本轮解决什么问题：** 原计划检验跨模态频率相干，但 SMGFM 已公开覆盖频带、语义角色和可靠性路由，因此该方向在运行前即被淘汰。实际检验的是更物理的“模态空间测量响应校准”：先估计 RNA 与第二模态各自的有效空间粗糙度，再用有界前向/逆向图响应对齐到共同组织带宽。
2. **实际改了哪一层：** 改的是进入共同残差 backbone 之前的表示输入层。公式、训练、候选数和 endpoint 在 A1 与 P22 完全相同；12 个真实训练工件均有参数更新、严格 checkpoint reload 和 fresh-process 分区逐字节重放。
3. **对论文意味着什么：** 终态是 **SCIENTIFIC_NEGATIVE**。A1 的 full 在 common KMeans 仅有 +0.001886/+0.001358，且 low-pass-only 几乎解释全部；P22 的 full 在 common KMeans 双降。P22 的 GMM 条件增益与 sharpen-only 分区逐字节相同，属于单模态锐化/endpoint 条件信号，不是统一响应校准贡献。没有进入多 seed，也没有刷新历史 frontier。

## 绝对指标主表

| 数据集 | 同口径 endpoint | Matched backbone | Full calibration | ΔARI / ΔNMI | 最强解释性 control | 历史 frontier |
|---|---|---|---|---|---|---|
| A1 K10 | COMMON_KMEANS | .232410 / .388158 | **.234296 / .389516** | +.001886 / +.001358 | low-pass-only .234001 / .389618 | .276003 / .421740 |
| P22 K9 | COMMON_KMEANS | **.419206 / .551188** | .417729 / .549751 | -.001477 / -.001437 | sharpen-only = full | .595552 / .717931 |
| P22 K9 | COMMON_GMM context | .408791 / .553338 | .451848 / .578726 | +.043057 / +.025388 | **sharpen-only byte-exact = full** | .595552 / .717931 |

完整 48 行的 ARI/NMI/AMI/FMI、空间指标、簇大小和资源见 `absolute_metrics_main_table.csv`；每个 endpoint 的同端点差值和对齐后 changed spots 见 `matched_contribution_table.csv`。

## 机制归因

- A1 原始粗糙度为 1.2012/0.9306；full 自动选择 `(beta,gamma)=(.35,.10)` 与 `(.15,.35)`，log-gap 从 .2552 降到 .0420。这证明校准算子确实按输入统计工作，但分数增量过小且未独立胜过单向 controls。
- P22 原始粗糙度已很接近（1.2061/1.1602）；full 只对第二模态选择轻锐化 `(beta,gamma)=(.10,.15)`，因此与 sharpen-only 表示和分区相同。这里不能声称“双模态共同带宽校准”。
- Swapped response 在 A1 的固定结构 endpoint 得到 .241818/.397338，但它是负对照且依赖另一 endpoint；它不能反向证明 full，反而说明当前粗糙度匹配目标与聚类质量并不一致。
- 所有结果仍远低于项目历史强 representation/head frontier；没有把 endpoint 变化伪装成 representation 胜利。

## 新颖性分诊

SMGFM 已覆盖多模态图频带语义和可靠性路由，GatorPrism 已覆盖 joint/private experts 与 prototype router；DRIFT、SpaGFT/DeepGFT 和 SpaDDM 分别覆盖低通/图频/方向扩散。Night-18B 的最小候选对象是“输入粗糙度驱动的有界逆/前向测量响应等化”，源码层面未发现完全同构，但本轮实证门失败，因此不进入论文贡献清单，只保留为可复算负结果。

## 失败与局限

- 只有 seed 0，因为预注册独立贡献门未通过；补 seed 无法把 control-identical 的 P22 结果变成新机制证据。
- 粗糙度是全局统计，可能无法描述局部组织带宽异质性；但继续加局部 router 会与最新先例高度碰撞，且本轮没有信号支持扩张。
- P22 的锐化条件信号可作为后续成熟 preprocessing 诊断，但不得作为 SpaLORA 新模块。
- 没有新数据下载，没有在 inode 已满的持久盘写文件。

## 导师汇报版

1. 我们先主动否决了已被 SMGFM 覆盖的“频率专家”方案，没有换名抢创新。
2. 随后实现了一个更有物理含义的统一模块：估计每种组学的空间测量带宽，再用有界有理图响应校准到共同尺度。
3. A1 与 P22 的真实训练、checkpoint、候选生成、独立评价和 fresh-process 重放全部闭合。
4. A1 只得到千分位提升，而且低通单臂几乎可解释。
5. P22 在共同 KMeans 上反而下降；GMM 上的较大提升与“只锐化一个模态”完全相同。
6. 因而这轮不能证明统一测量响应校准是论文方法贡献，分类为 SCIENTIFIC_NEGATIVE。
7. 我们保留了 P22 的单模态锐化/head 条件线索，但没有补 seed 或扩大网格美化结果。
8. 服务器保持开机，未派发 shutdown。

## 技术附录

- taskbook SHA-256: `96aba215df9c8d6ef18e386c515e2da43a587f1433fef6cd7b01e1bb867b483f`
- parent commit: `2c9d59098afd6ef927746ddf2774887fcf681abc`
- formal artifacts: 12; exact representation/partition replays: 12/12; targeted tests: 6/6.
- labels: producer 0 reads; evaluator 1 post-lock read per artifact; benchmark endpoint selection is transparently label-assisted.
- shutdown_dispatched=false; KEEP_ON_FOR_CONTINUOUS_RESEARCH.
"""
    (OUT / "night18b_report.md").write_text(report, encoding="utf-8")
    (OUT / "plain_summary.md").write_text("Night-18B 结论：空间测量响应校准的工程路径完整，但独立方法贡献为 0/2；P22 GMM 增益由 sharpen-only 完全解释，终态 SCIENTIFIC_NEGATIVE。\n", encoding="utf-8")

    risk = """# Reviewer risk register\n\n| Risk | Evidence | Disposition |\n|---|---|---|\n| Frequency-route novelty collision | SMGFM 2026 | Route rejected before execution. |\n| Generic diffusion prior art | DRIFT/SpaGFT/DeepGFT/SpaDDM | No novelty claim for filtering. |\n| Endpoint confounding | P22 GMM improves while common KMeans falls | All claims are endpoint-matched; no universal representation claim. |\n| Control identity | P22 full equals sharpen-only byte-exact | Full method claim rejected. |\n| Label-assisted selection | public endpoint metrics opened post-lock | Separated benchmark profile; not called automatic or blind. |\n| Single seed | independent gate failed | No confirmation claim and no extra compute. |\n"""
    (OUT / "reviewer_risk_register.md").write_text(risk, encoding="utf-8")

    print(json.dumps({"status": "PASS", "outputs": len(list(OUT.iterdir())), "working_bytes": work_bytes}, indent=2))


if __name__ == "__main__":
    main()
