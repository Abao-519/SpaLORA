#!/usr/bin/env python3
"""Build the compact Night-16F scientific handoff from locked artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics


LANES = {
    "P22_K9": {
        "evaluation": "/root/night16f_working/development/P22_K9/evaluation.csv",
        "producer": "/root/night16f_working/development/P22_K9/partitions.producer.json",
        "carrier": "/root/night16f_working/carriers/P22_K9.carrier.json",
        "role": "DEVELOPMENT_HISTORICAL_LABEL_ASSISTED_START",
    },
    "MISAR_K7": {
        "evaluation": "/root/night16f_working/development/MISAR_K7/evaluation.csv",
        "producer": "/root/night16f_working/development/MISAR_K7/partitions.producer.json",
        "carrier": "/root/night16f_working/carriers/MISAR_K7.carrier.json",
        "role": "DEVELOPMENT_HISTORICAL_LABEL_ASSISTED_START",
    },
    "HUMAN_HIPPOCAMPUS_K7": {
        "evaluation": "/root/night16f_working/development/HUMAN_HIPPOCAMPUS_K7/evaluation.csv",
        "producer": "/root/night16f_working/development/HUMAN_HIPPOCAMPUS_K7/partitions.producer.json",
        "carrier": "/root/night16f_working/carriers/HUMAN_HIPPOCAMPUS_K7.carrier.json",
        "role": "DEVELOPMENT_ATTRIBUTION_BYTE_EXACT_PARENT_START_AND_PARENT_NUMERICAL_BOUNDARY",
    },
    "MELANOMA_TUMOR_K2": {
        "evaluation": "/root/night16f_working/formal/MELANOMA_TUMOR_K2/evaluation.csv",
        "producer": "/root/night16f_working/formal/MELANOMA_TUMOR_K2/partitions.producer.json",
        "carrier": "/root/night16f_working/carriers/MELANOMA_TUMOR_K2.carrier.json",
        "role": "FROZEN_INDEPENDENT_TRANSFER",
    },
}

PRIMARY = (
    "DIRECT_BASE",
    "BIMODAL_SUPPORT",
    "UNIFORM_MASS_MATCHED",
    "PERMUTED_SUPPORT",
    "RNA_ONLY_SUPPORT",
    "ATAC_ONLY_SUPPORT",
)
MATCHED_COMPETITORS = PRIMARY[2:]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(row: dict[str, str], key: str) -> float:
    return float(row[key])


def arm_row(rows: list[dict[str, str]], arm: str, start_index: int = 0) -> dict[str, str]:
    values = [row for row in rows if row["arm"] == arm and int(row["start_index"]) == start_index]
    if len(values) != 1:
        raise ValueError(f"expected one {arm} row for start {start_index}, observed {len(values)}")
    return values[0]


def stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def run(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    authority_rows: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    main_rows: list[dict[str, object]] = []
    score_frontier: list[dict[str, object]] = []
    p0_rows: list[dict[str, object]] = []
    resource_rows: list[dict[str, object]] = []
    lane_values: dict[str, list[dict[str, str]]] = {}

    for lane, specification in LANES.items():
        rows = read_csv(Path(specification["evaluation"]))
        lane_values[lane] = rows
        producer = json.loads(Path(specification["producer"]).read_text())
        carrier = json.loads(Path(specification["carrier"]).read_text())
        for row in rows:
            all_rows.append({"lane": lane, "study_role": specification["role"], **row})
        for arm in PRIMARY:
            authority = arm_row(rows, arm)
            authority_rows.append({"lane": lane, "study_role": specification["role"], **authority})
            seed_rows = [
                row for row in rows if row["arm"] == arm and int(row["start_index"]) > 0
            ]
            all_start_rows = [row for row in rows if row["arm"] == arm]
            ari_seed = stats([metric(row, "absolute_ari") for row in seed_rows])
            nmi_seed = stats([metric(row, "absolute_nmi") for row in seed_rows])
            ari_all = stats([metric(row, "absolute_ari") for row in all_start_rows])
            nmi_all = stats([metric(row, "absolute_nmi") for row in all_start_rows])
            summaries.append(
                {
                    "lane": lane,
                    "arm": arm,
                    "authority_ari": metric(authority, "absolute_ari"),
                    "authority_nmi": metric(authority, "absolute_nmi"),
                    "robustness_seed_count": len(seed_rows),
                    **{f"robustness_ari_{key}": value for key, value in ari_seed.items()},
                    **{f"robustness_nmi_{key}": value for key, value in nmi_seed.items()},
                    **{f"all_start_ari_{key}": value for key, value in ari_all.items()},
                    **{f"all_start_nmi_{key}": value for key, value in nmi_all.items()},
                }
            )
        input_row = arm_row(rows, "INPUT_START")
        direct = arm_row(rows, "DIRECT_BASE")
        bimodal = arm_row(rows, "BIMODAL_SUPPORT")
        strongest = max(
            (arm_row(rows, arm) for arm in MATCHED_COMPETITORS),
            key=lambda row: (metric(row, "absolute_ari"), metric(row, "absolute_nmi"), row["arm"]),
        )
        main_rows.append(
            {
                "lane": lane,
                "role": specification["role"],
                "n_total": input_row["n_total"],
                "n_evaluated": input_row["n_evaluated"],
                "k": input_row["k"],
                "input_ari": metric(input_row, "absolute_ari"),
                "input_nmi": metric(input_row, "absolute_nmi"),
                "direct_ari": metric(direct, "absolute_ari"),
                "direct_nmi": metric(direct, "absolute_nmi"),
                "bimodal_ari": metric(bimodal, "absolute_ari"),
                "bimodal_nmi": metric(bimodal, "absolute_nmi"),
                "strongest_mass_matched_control": strongest["arm"],
                "strongest_control_ari": metric(strongest, "absolute_ari"),
                "strongest_control_nmi": metric(strongest, "absolute_nmi"),
                "bimodal_minus_direct_ari": metric(bimodal, "absolute_ari") - metric(direct, "absolute_ari"),
                "bimodal_minus_direct_nmi": metric(bimodal, "absolute_nmi") - metric(direct, "absolute_nmi"),
                "bimodal_minus_strongest_control_ari": metric(bimodal, "absolute_ari") - metric(strongest, "absolute_ari"),
                "bimodal_minus_strongest_control_nmi": metric(bimodal, "absolute_nmi") - metric(strongest, "absolute_nmi"),
                "bimodal_cluster_sizes": bimodal["cluster_sizes_full"],
                "direct_cluster_sizes": direct["cluster_sizes_full"],
                "peak_rss_mib": producer["peak_rss_mib"],
                "producer_wall_seconds": producer["wall_seconds"],
            }
        )
        best = max(
            (row for row in rows if row["status"] == "PASS"),
            key=lambda row: (metric(row, "absolute_ari"), metric(row, "absolute_nmi"), row["candidate_id"]),
        )
        score_frontier.append(
            {
                "lane": lane,
                "candidate_id": best["candidate_id"],
                "start_id": best["start_id"],
                "arm": best["arm"],
                "absolute_ari": metric(best, "absolute_ari"),
                "absolute_nmi": metric(best, "absolute_nmi"),
                "selection_semantics": "PUBLIC_BENCHMARK_LABEL_ASSISTED_BEST_NOT_FORMAL_OUTPUT",
                "formal_selector_changed": False,
            }
        )
        p0_rows.append(
            {
                "lane": lane,
                "n": carrier["n"],
                "k": carrier["k"],
                "view1_shape": json.dumps(carrier["view1_shape"]),
                "view2_shape": json.dumps(carrier["view2_shape"]),
                "retained_shape": json.dumps(carrier["retained_shape"]),
                "graph_shapes_nnz": json.dumps(carrier["graph_shapes_nnz"]),
                "ordered_id_sha256": carrier["ordered_id_sha256"],
                "formal_producer_label_reads": carrier["formal_producer_label_reads"],
                "artifact_reload": carrier["artifact_reload"],
            }
        )
        resource_rows.append(
            {
                "lane": lane,
                "producer_wall_seconds": producer["wall_seconds"],
                "peak_rss_mib": producer["peak_rss_mib"],
                "gpu_time_seconds": producer["gpu_time_seconds"],
                "peak_gpu_mib": producer["peak_gpu_mib"],
                "candidate_rows": len(rows),
            }
        )

    write_csv(output / "matched_attribution_table.csv", all_rows)
    write_csv(output / "authority_start_attribution_table.csv", authority_rows)
    write_csv(output / "attribution_distribution_summary.csv", summaries)
    write_csv(output / "frozen_transfer_main_table.csv", main_rows)
    write_csv(output / "score_frontier_table.csv", score_frontier)
    write_csv(output / "real_input_p0_table.csv", p0_rows)
    write_csv(output / "resource_table.csv", resource_rows)

    protocol = json.loads(
        Path("/root/night16f_working/melanoma_protocol/scp2176_protocol_registry.json").read_text()
    )
    shutil.copy2(
        "/root/night16f_working/melanoma_protocol/scp2176_protocol_registry.json",
        output / "scp2176_protocol_registry.json",
    )
    shutil.copy2(
        "/root/night16f_working/replays/exact_replay_audit.json",
        output / "exact_replay_audit.json",
    )
    shutil.copy2(
        "/root/night16f_working/audit/human_direct_reproduction_audit.json",
        output / "human_direct_reproduction_audit.json",
    )
    shutil.copy2(
        "/root/night16f_working/audit/engineering_correction_scope_audit.json",
        output / "engineering_correction_scope_audit.json",
    )
    shutil.copy2(
        "/root/night16f_working/formal/MELANOMA_TUMOR_K2/start_diagnostic.csv",
        output / "melanoma_start_diagnostic.csv",
    )
    shutil.copy2(
        "/root/night16f_working/formal/MELANOMA_TUMOR_K2/start_diagnostic.json",
        output / "melanoma_start_diagnostic.json",
    )
    for lane, specification in LANES.items():
        shutil.copy2(specification["carrier"], output / f"{lane}_carrier_manifest.json")
        evaluator = Path(specification["evaluation"]).with_suffix(".evaluator.json")
        shutil.copy2(evaluator, output / f"{lane}_evaluator_audit.json")

    independent_relation_specific_gate = all(
        row["bimodal_minus_strongest_control_ari"] > 0
        and row["bimodal_minus_strongest_control_nmi"] > 0
        for row in main_rows
        if row["lane"] in ("HUMAN_HIPPOCAMPUS_K7", "MELANOMA_TUMOR_K2")
    )
    melanoma = next(row for row in main_rows if row["lane"] == "MELANOMA_TUMOR_K2")
    human = next(row for row in main_rows if row["lane"] == "HUMAN_HIPPOCAMPUS_K7")
    human_local_gate = (
        human["bimodal_minus_strongest_control_ari"] > 0
        and human["bimodal_minus_strongest_control_nmi"] > 0
    )
    melanoma_transfer_gate = (
        melanoma["bimodal_minus_strongest_control_ari"] > 0
        and melanoma["bimodal_minus_strongest_control_nmi"] > 0
    )
    melanoma_frontier = next(row for row in score_frontier if row["lane"] == "MELANOMA_TUMOR_K2")
    exact_replay = json.loads((output / "exact_replay_audit.json").read_text())
    decision = {
        "schema": "night16f-decision-v1",
        "status": "NIGHT16F_HUMAN_RELATION_SPECIFIC_SUPPORT_LOCAL_SIGNAL_NO_MELANOMA_TRANSFER",
        "classification": "LOCAL_SIGNAL",
        "primary_hypothesis": "relation-specific bimodal edge support with independent melanoma transfer",
        "primary_hypothesis_passed": independent_relation_specific_gate,
        "primary_hypothesis_result": "LOCAL_SIGNAL_WITHOUT_INDEPENDENT_TRANSFER",
        "development_signal": {
            "type": "HUMAN_HIPPOCAMPUS_RELATION_SPECIFIC_BIMODAL_SUPPORT",
            "passed": human_local_gate,
            "bimodal_ari": human["bimodal_ari"],
            "bimodal_nmi": human["bimodal_nmi"],
            "strongest_control": human["strongest_mass_matched_control"],
            "bimodal_minus_strongest_control_ari": human["bimodal_minus_strongest_control_ari"],
            "bimodal_minus_strongest_control_nmi": human["bimodal_minus_strongest_control_nmi"],
        },
        "independent_transfer": {
            "lane": "MELANOMA_TUMOR_K2",
            "passed": melanoma_transfer_gate,
            "medoid_direct_ari": melanoma["direct_ari"],
            "medoid_direct_nmi": melanoma["direct_nmi"],
            "medoid_bimodal_ari": melanoma["bimodal_ari"],
            "medoid_bimodal_nmi": melanoma["bimodal_nmi"],
            "strongest_control": melanoma["strongest_mass_matched_control"],
        },
        "label_assisted_score_frontier": melanoma_frontier,
        "all_cell_k10_status": protocol["protocols"]["ALL_CELL_K10"]["status"],
        "tumor_k2_status": protocol["protocols"]["TUMOR_ONLY_K2"]["status"],
        "exact_replay": f"{exact_replay['passed_lanes']}/{exact_replay['total_lanes']}",
        "formula_changed_after_freeze": False,
        "engineering_correction_after_freeze": (
            "restored parent OMP/MKL/OPENBLAS=1 numerical boundary; rebuilt all carriers and reran all lanes"
        ),
        "formal_selector_changed_after_melanoma_labels": False,
        "human_parent_start_correction": (
            "fresh recomputed medoid results are retained under superseded/; final attribution uses "
            "the byte-exact Night-16E parent start"
        ),
        "human_direct_reproduction": (
            "PASS: corrected Night-16F DIRECT_BASE byte-exactly reproduces the Night-16E partition; "
            "the unpinned and float64-graph-only outputs are retained as superseded controls"
        ),
        "shutdown_dispatched": False,
        "instance_state_instruction": "KEEP_ON_FOR_NIGHTTIME_CHAINED_TASKS",
    }
    (output / "night16f_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True))

    label_audit = {
        "schema": "night16f-label-flow-audit-v1",
        "formal_producer_label_reads": 0,
        "independent_evaluator_label_read_events": 4,
        "melanoma_labels_used_before_formula_freeze": 0,
        "melanoma_labels_used_to_change_formula_or_selector": 0,
        "melanoma_tumor_membership_source": (
            "public SCP2176 annotation mechanically defines exact tumour_1/tumour_2 membership; "
            "the sanitized producer carrier contains no annotation column"
        ),
        "score_frontier_semantics": "public-label-assisted post-hoc development record only",
    }
    (output / "label_flow_audit.json").write_text(json.dumps(label_audit, indent=2, sort_keys=True))

    resource_audit = {
        "schema": "night16f-resource-audit-v1",
        "producer_candidate_rows": len(all_rows),
        "producer_failed_rows": sum(row.get("status") != "PASS" for row in all_rows),
        "total_producer_wall_seconds": sum(float(row["producer_wall_seconds"]) for row in resource_rows),
        "peak_rss_mib": max(float(row["peak_rss_mib"]) for row in resource_rows),
        "gpu_time_seconds": 0.0,
        "peak_gpu_mib": 0.0,
        "dense_n_by_n_count": 0,
        "new_download_count": 2,
        "all_cell_k10_matrix_download": "BLOCKED_AUTH_REQUIRED",
        "shutdown_dispatched": False,
    }
    (output / "resource_audit.json").write_text(json.dumps(resource_audit, indent=2, sort_keys=True))

    failures = [
        {
            "event": "FIGSHARE_POSITIONAL_IDS_AND_FEATURES",
            "classification": "AUTHORITY_ALIGNMENT_CORRECTION",
            "disposition": (
                "exact row/column numeric identity to the official Apache-2.0 reproduce carrier was "
                "required before biological barcodes/features were transferred"
            ),
            "scientific_formula_changed": False,
        },
        {
            "event": "HUMAN_FRESH_MEDOID_NOT_BYTE_EXACT_PARENT",
            "classification": "START_AUTHORITY_CORRECTION",
            "disposition": (
                "old carrier/results retained under superseded; final human runs use byte-exact parent start "
                "and were rerun twice"
            ),
            "scientific_formula_changed": False,
        },
        {
            "event": "NIGHT16F_UNPINNED_NUMERICAL_BOUNDARY",
            "classification": "ENGINEERING_REPRODUCIBILITY_CORRECTION",
            "disposition": (
                "Night-16E final replay fixed OMP/MKL/OPENBLAS=1; Night-16F first pass omitted it. "
                "All four carriers, outputs, evaluations and replays were retained under superseded, "
                "then rebuilt and rerun with the parent boundary."
            ),
            "scientific_formula_changed": False,
        },
        {
            "event": "GRAPH_FLOAT32_ROOT_CAUSE_HYPOTHESIS",
            "classification": "FALSIFIED_ENGINEERING_HYPOTHESIS",
            "disposition": (
                "restoring only float64 graph data left the unpinned human DIRECT partition byte-exact; "
                "lossless graph dtype storage was nevertheless adopted for carrier fidelity"
            ),
            "scientific_formula_changed": False,
        },
        {
            "event": "SCP2176_ALL_CELL_K10",
            "classification": "DATA_INPUT_BLOCKED_AUTH_REQUIRED",
            "disposition": "K10 not evaluated; no label or matrix was fabricated",
            "scientific_formula_changed": False,
        },
        {
            "event": "RELATION_SPECIFIC_BIMODAL_SUPPORT",
            "classification": "LOCAL_SIGNAL_NO_INDEPENDENT_TRANSFER",
            "disposition": (
                "human hippocampus beat all mass-matched controls, but the same frozen operator failed "
                "to do so on independent melanoma; no v2 or scientific retry followed"
            ),
            "scientific_formula_changed": False,
        },
    ]
    write_csv(output / "failure_and_correction_ledger.csv", failures)

    core_paths = [
        Path("SpaLORA/night16f_support_attribution.py"),
        Path("scripts/night16f/build_numeric_carrier.py"),
        Path("scripts/night16f/night16f_producer.py"),
        Path("scripts/night16f/night16f_evaluator.py"),
        Path("scripts/night16f/run_corrected_formal.sh"),
        Path("scripts/night16f/audit_human_direct_reproduction.py"),
        Path("configs/night16f/night16f_formula_and_ablation_contract.json"),
    ]
    source_audit = [
        "# Night-16F source and formula audit",
        "",
        "The primary attribution controls are a clean-room extension of the inherited Night-15F/16E sparse alpha-expansion energy. No third-party solver source was copied.",
        "",
        "| file | SHA-256 |",
        "|---|---|",
        *[f"| `{path}` | `{sha256(path)}` |" for path in core_paths],
        "",
        "SCP2176 labels are public computational cell-type/tumor-state annotations. The 833-cell K=2 carrier is the exact processed tumor subset; it is not described as expert manual spatial-domain truth.",
    ]
    (output / "source_and_formula_audit.md").write_text("\n".join(source_audit) + "\n")

    def fmt(value: object) -> str:
        return f"{float(value):.6f}"

    table_lines = [
        "| 数据/协议 | N/eval/K | 输入起点 ARI/NMI | Direct ARI/NMI | 双模态 support ARI/NMI | 最强质量匹配对照 ARI/NMI |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in main_rows:
        table_lines.append(
            f"| {row['lane']} | {row['n_total']}/{row['n_evaluated']}/{row['k']} | "
            f"{fmt(row['input_ari'])}/{fmt(row['input_nmi'])} | "
            f"{fmt(row['direct_ari'])}/{fmt(row['direct_nmi'])} | "
            f"{fmt(row['bimodal_ari'])}/{fmt(row['bimodal_nmi'])} | "
            f"{row['strongest_mass_matched_control']} {fmt(row['strongest_control_ari'])}/{fmt(row['strongest_control_nmi'])} |"
        )

    report = f"""# Night-16F report

## 我现在需要知道的三件事

1. 本轮要判断 Night-16E 的大幅提分是不是来自“正确的双模态 support 边位置”，而不只是总体平滑变弱、权重被打乱后仍有效，或单一模态已经足够。
2. 我们冻结同一起点、同一 unary、同一 base 能量、同一 self-return 和同一 alpha-expansion，只替换 support 因子；每个图尺度都精确匹配 `sum(base_edge × factor)`。随后把同一冻结路径一次性移到独立的 SCP2176 melanoma tumour-only K=2 协议。
3. 结论分类是 **LOCAL SIGNAL**：人海马开发单元中，双模态 support 为 {fmt(human['bimodal_ari'])}/{fmt(human['bimodal_nmi'])}，双指标超过所有总质量匹配对照；但冻结到独立 melanoma 后，bimodal {fmt(melanoma['bimodal_ari'])}/{fmt(melanoma['bimodal_nmi'])} 低于 uniform/单模态对照，因而没有独立迁移证据。它不是 confirmed milestone，更不是 SOTA。

## 绝对结果主表

{chr(10).join(table_lines)}

完整 4 lane × 47 rows 在 `matched_attribution_table.csv`；每个 primary arm 的 5 个 KMeans robustness starts 给出 min/mean/median/max，另与 authority/medoid 分开，见 `attribution_distribution_summary.csv`。

## Slide-tags melanoma 数据闭合

- SCP2176 公开 annotation 与 spatial endpoint 均为 2535 cells，类别含 pDC=6；小类是数据事实，不用任意 microcluster 阈值删除。
- 指定的两个官方 Figshare H5AD 经官方 MultiGATE reproduce carrier 逐矩阵核对后均为 **833 tumour cells**，不是 2529。它们恰好对齐 tumour_1=561、tumour_2=272，形成 K=2 协议。
- Figshare 导出将 observation/feature IDs 替换成位置编号；只有在 shape、row/column numeric X 全等后，才从官方 reproduce carrier 恢复 biological IDs/features，并再与 SCP ID/coordinates 对齐。
- all-cell K=10 只有 2535 条公开标签/坐标，匿名接口的完整 RNA/ATAC matrices 需要认证，因此状态为 `DATA_INPUT_BLOCKED_AUTH_REQUIRED`，没有伪造或用 tumor matrix 冒充。

## 机制归因

- `UNIFORM_MASS_MATCHED` 保留相同总 Potts 容量；`PERMUTED_SUPPORT` 保留 support 分布但打乱边位置；RNA-only/ATAC-only 保留单视图位置并精确质量匹配。
- 人海马 authority start 下，bimodal 为 {fmt(human['bimodal_ari'])}/{fmt(human['bimodal_nmi'])}；最强质量匹配对照是 {human['strongest_mass_matched_control']}，为 {fmt(human['strongest_control_ari'])}/{fmt(human['strongest_control_nmi'])}。这支持“边位置有用”的开发期局部信号。
- melanoma authority medoid 下，bimodal 仅略高于 direct，却低于 {melanoma['strongest_mass_matched_control']} {fmt(melanoma['strongest_control_ari'])}/{fmt(melanoma['strongest_control_nmi'])}；uniform/permuted/single-view 也不弱。因此该局部信号没有跨研究迁移。
- `RELATION_STAY_OFF_BASE_STAY_ON` 与 bimodal support 相同，说明本轮数值不支持把增益归给新增 relation stay；base self-return 是继承项。

## Score frontier 与正式输出必须分开

Melanoma 的五个 KMeans starts 中，公开标签事后看见的最高行是 `{melanoma_frontier['candidate_id']}`，ARI/NMI={fmt(melanoma_frontier['absolute_ari'])}/{fmt(melanoma_frontier['absolute_nmi'])}。它只是 **public benchmark label-assisted BEST**，没有替换正式无标签 medoid。常规无标签诊断中 S3 赢 inertia 与 Calinski–Harabasz，但 silhouette、Davies–Bouldin 和 partition centrality 选择别的 starts；当前没有单一机械准则稳健锁定该峰值。

## 重放、资源和局限

- 工程审计定位到 Night-16F 首轮漏掉 Night-16E final replay 已冻结的 `OMP/MKL/OPENBLAS=1`。旧 human 连续表示仅有约 1e-4–3e-4 RMS 漂移，却令离散最小割从移动 5 点放大到 1779 点；所有首轮 carrier/结果已整体标为 superseded。
- 只把旧图从 float32 恢复为 float64并不能改变旧 DIRECT 分区，排除了“graph dtype 是根因”。修复线程边界后，human DIRECT 分区 SHA 与 Night-16E byte-exact，mean retained/view1/view2 weights 和移动 5 点也恢复；carrier 同时改为不再无谓降精度。
- 修复后两个 fresh processes 对 4/4 lanes、188/188 candidate partitions 的 SHA、cluster sizes 与 config exact。P22/MISAR 的 carrier、图、47/47 分区在修复前后本来就 byte-exact；human/melanoma 被完整重建。
- producer 标签读取 0；评价器在 partitions 锁定后分别读取 4 个公开 reference；dense N×N=0；GPU time=0。
- 人海马最终同时恢复 byte-exact Night-16E parent start 和父级单线程数值边界；DIRECT 分区 SHA=`6c6c334d...`，与父级一致。
- 独立迁移门失败后没有补 selector、换公式或用 melanoma 标签重试。

## 导师汇报版

这轮专门拆解了 Night-16E 人海马的大提分来源。我们把双模态 support 与统一衰减、位置置换、RNA-only、ATAC-only 做成总平滑质量完全匹配的对照。人海马上，bimodal support 确实双指标超过全部对照，说明边级双模态位置有开发期局部证据。独立 melanoma K=2 上，同一冻结算子却没有超过 uniform/单模态对照，所以还不能写成可迁移方法。工程上还发现并修复了 BLAS 线程边界遗漏；修复后 human DIRECT 与 Night-16E byte-exact，旧首轮数值全部作废。Melanoma 五个公开 benchmark starts 中有一个达到 {fmt(melanoma_frontier['absolute_ari'])}/{fmt(melanoma_frontier['absolute_nmi'])}，但它是标签事后 BEST；无标签准则没有一致地唯一选择它。下一步应优先解决跨研究校准和初始化选择，而不是把人海马单点信号升格。

## 技术附录摘要

- preformal freeze commit: `1e33dc8ccfb1b9fc5f13de0b7160c502511abe1e`
- exact replay: {exact_replay['passed_lanes']}/{exact_replay['total_lanes']}
- SCP2176 K2 ordered reference SHA: `{protocol['protocols']['TUMOR_ONLY_K2']['ordered_reference_sha256']}`
- shutdown dispatched: `false`；实例按连续协作要求保持在线。
"""
    (output / "night16f_report.md").write_text(report)
    (output / "night16f_plain_summary.md").write_text(
        "Night-16F 在人海马得到关系特异性双模态 support 的局部信号，但没有迁移到独立 melanoma；"
        "KMeans S3 的公开 benchmark 峰值也尚不能被统一无标签准则可靠选择。\n"
    )
    # Keep generated and copied CSVs byte-stable across platforms and clean for
    # Git's whitespace checks.  This is serialization only; no table values are
    # changed.
    for csv_path in output.glob("*.csv"):
        csv_path.write_bytes(csv_path.read_bytes().replace(b"\r\n", b"\n"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
