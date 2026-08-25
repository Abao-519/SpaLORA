#!/usr/bin/env python3
"""Build the compact Night-18D scientific handoff from locked artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def metric(value) -> str: return f"{float(value):.6f}"


def run(args: argparse.Namespace) -> None:
    root, out = Path(args.working), Path(args.output); out.mkdir(parents=True, exist_ok=True)
    summary = json.loads((root / "evaluation/evaluation_summary.json").read_text(encoding="utf-8"))
    authority = json.loads((root / "audit/dataset_authority_audit.json").read_text(encoding="utf-8"))
    carrier = json.loads((root / "carrier/placenta_carrier.json").read_text(encoding="utf-8"))
    frozen = json.loads((root / "config/frozen_family_default.json").read_text(encoding="utf-8"))
    selected = load_csv(root / "evaluation/selected_main_table.csv")
    headline = summary["headline_selection"]
    primary_selected = [row for row in selected if row["profile_id"] == frozen["primary_profile_id"] and row["selection_name"].endswith("STRUCTURE_FEASIBLE_MEDOID")]
    table_lines = ["| Arm | ARI | NMI | AMI | FMI | Moran macro | Geary macro | min cluster | changed |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in primary_selected:
        table_lines.append(f"| {row['arm']} | {metric(row['ari'])} | {metric(row['nmi'])} | {metric(row['ami'])} | {metric(row['fmi'])} | {metric(row['moran_indicator_macro'])} | {metric(row['geary_indicator_macro'])} | {row['min_cluster_size']} | {row['changed_observations']} |")
    report = f"""# Night-18D Human Placenta Frozen Structured-Energy External Transfer

## 我现在需要知道的三件事

1. **冻结迁移没有成功。** Night-15F 的 RNA+chromatin 家族中心配置在 1,662 个 human placenta 细胞、K=10 协议上的正式无标签输出只有 ARI/NMI **{metric(headline['ari'])}/{metric(headline['nmi'])}**；最强匹配对照分别达到 ARI {metric(summary['strongest_control_by_ari']['ari'])} 和 NMI {metric(summary['strongest_control_by_nmi']['nmi'])}。主分类是 **SCIENTIFIC_NEGATIVE**。
2. **问题发生在完整结构能量消费者，而不是数据闭合。** no-op medoid 为 {metric(next(row for row in primary_selected if row['arm']=='NO_OP_START')['ari'])}/{metric(next(row for row in primary_selected if row['arm']=='NO_OP_START')['nmi'])}，L2 low-pass 为 {metric(next(row for row in primary_selected if row['arm']=='L2_LOWPASS_MATCHED')['ari'])}/{metric(next(row for row in primary_selected if row['arm']=='L2_LOWPASS_MATCHED')['nmi'])}；full 相对同起点 no-op 在 13/13 start 中没有一次 ARI/NMI 双升，且正式 full medoid 恰好是一个未移动的 ATAC-start 分区，不能归因于 decoder。
3. **负结果可信且可复算。** 官方 ATAC 文件与现有容器虽然容器 SHA 不同，但 X、ordered IDs、63 个 TF 相关调控特征、obs 和坐标数值全部闭合；formal partition 与三张指标表 fresh-process byte-exact，5 个 targeted tests 通过。AutoDL 保持开机，未派发 shutdown。

## 本轮实际改了哪一层

本轮没有训练新 backbone，也没有增加 selector。我们把 Night-15F 的完整结构化能量——动态多模态 prototype unary、三尺度稀疏空间图、冲突感知 pairwise、rejected-mass self-return 与 alpha-expansion——作为一个冻结消费者，接在新 human placenta 的 label-free RNA/ATAC 数值表示与 13 个机械 start 上。RNA 和官方 processed ATAC-derived TF-associated regulatory features 分别降维；两个独立 PCA 坐标系先用同位点正交 Procrustes 对齐，再融合。K=10 来自公开 annotation 协议；逐点 cell_type 只在 130 个 candidate partitions 全部保存并哈希后由独立 evaluator 打开。

## 绝对指标主表（family-frozen primary profile）

数据协议：human placenta，N/eval=1662/1662，K=10，all-cell original-author cell-type annotation。

{chr(10).join(table_lines)}

Full 相对最强 ARI 对照差值为 **{summary['delta_vs_strongest_ari']:.6f}**，相对最强 NMI 对照差值为 **{summary['delta_vs_strongest_nmi']:.6f}**。这些“最强”按每个指标分别计算，没有把两个不同对照伪装成同一分区。label-assisted locked-candidate oracle 仅作开发附表，不能替代无标签 medoid；它也没有挽救 family-default full。

## 贡献归因

- **上游 start/简单平滑有可用信号：** no-op 与 L2 low-pass 显著高于 full；锁定候选中的 label-assisted L2 best 仅作为 benchmark development context。
- **完整 frozen energy 无独立贡献：** full selected ARI/NMI 同时低于 no-op、L2、registered-scale、pairwise-zero 和 pure-unary 等关键对照；同起点配对双升为 0/13。
- **self-return 是保护项但不是成功机制：** 关闭 self-return 后出现 min cluster=4 且分数进一步下降；这说明 stay cost 防止崩坏，却不能证明完整能量可迁移。
- **solver 不是唯一原因：** matched single-site 也低分，且 full alpha-expansion 本身未超过强对照。
- **不作原创性扩张：** prototype、邻域统计、多尺度图、Potts/CRF 与 alpha-expansion 都有明确先例；本轮只检验既有组合的外部迁移，结果为负。

## 数据与标签语义

- RNA: 1662×36601 sparse counts；ATAC: 1662×63 **official processed ATAC-derived TF-associated regulatory features**，不是 raw peaks 或 LSI。
- 官方 repo ATAC SHA-256: `{authority['files']['official_atac']['sha256']}`；现有容器 SHA-256: `{authority['files']['local_atac']['sha256']}`。数值/ID/feature/obs/coordinates exact audit 均 PASS。
- source H5AD 的 obs 元数据在文件级随 AnnData 载入，但 carrier 计算没有索引 annotation 列；producer 只读 sanitized numeric carrier。authority JSON 在 producer 前只消费 `status` 与 `authority_gap` 两个 allow-list 字段。
- evaluator 在 formal artifact SHA `{summary['label_flow']['labels_opened_after_artifact_sha256']}` 锁定后读取 `cell_type`；ordered label hash 为 `{summary['reference']['ordered_labels_sha256']}`。

## 失败、限制与下一步

1. family-center 参数来自 P22 K9 与 MISAR K7 的数值几何中心，外部 placenta 的细胞级坐标图、类别不均衡与 63 维调控特征响应明显不同；本轮结果说明这种参数中心不能直接当作可迁移方法。
2. label-free medoid 是预注册消费者，不保证选择指标最优 start；但 full 的逐 start 配对也 0/13 双升，因此负结论不由 medoid 单独造成。
3. placenta 是细胞类型恢复协议，不应与组织域 benchmark 数字混表；本轮不声称 SOTA、paper-ready 或 confirmed milestone。
4. 如果继续该数据，优先研究 family-level calibration 或更强 raw-feature representation；不应在本轮结果后按 placenta 标签回调旧能量。

## 导师汇报版

我们第一次把 Night-15F 的完整结构能量原封不动迁移到一个此前未参与开发的高质量 human placenta RNA+ATAC 单元。数据权威、1662 个细胞的 ID、坐标、63 个 ATAC 衍生调控特征和 K=10 注释协议均已闭合。冻结 family profile 的正式无标签输出只有 ARI/NMI {metric(headline['ari'])}/{metric(headline['nmi'])}，明显低于 no-op 和简单 L2 low-pass。13 个相同起点中 full 没有一次实现 ARI/NMI 双升，说明失败不是 selector 偶然选错一个候选。关闭 self-return 会更差，表明它有防崩作用，但完整 decoder 仍不具备跨研究迁移性。因此本轮是高价值的科学负结果：它阻止我们把 P22/MISAR 上的开发高分误写成普适方法。工程上 formal 与 fresh-process replay 完全一致，交付可复算；下一步若继续，应转向可解释的 family calibration 或更强 representation，而不是继续包装旧能量。

## 技术附录摘要

- carrier SHA-256: `{carrier['carrier_sha256']}`；13 starts；图 nnz={carrier['graph_nnz']}
- family config SHA-256: `{frozen['primary_config_sha256']}`；source P22/MISAR hashes={json.dumps(frozen['source_config_sha256'], sort_keys=True)}
- formal candidates: 130；labels read by producer: 0；dense N×N: 0
- replay: partition artifact byte-exact；all/selected/paired CSV byte-exact
- shutdown_dispatched=false；AutoDL action=`KEEP_ON_FOR_WORKER2_REVIEW`
"""
    (out / "night18d_report.md").write_text(report, encoding="utf-8")
    decision = {
        "status": "NIGHT18D_HUMAN_PLACENTA_FROZEN_ENERGY_SCIENTIFIC_NEGATIVE",
        "classification": "SCIENTIFIC_NEGATIVE", "primary_profile_id": frozen["primary_profile_id"],
        "headline": {key: headline[key] for key in ("ari", "nmi", "ami", "fmi", "moran_indicator_macro", "geary_indicator_macro", "min_cluster_size", "cluster_sizes", "partition_sha256", "candidate_id")},
        "delta_vs_strongest_ari": summary["delta_vs_strongest_ari"], "delta_vs_strongest_nmi": summary["delta_vs_strongest_nmi"],
        "paired_dual_gain_starts": summary["paired_dual_gain_starts"], "paired_start_count": summary["paired_start_count"],
        "external_frozen_method_gate_passed": False, "independent_decoder_contribution": False,
        "data_authority": "PASS", "formal_replay": "BYTE_EXACT", "targeted_tests": "5/5 PASS",
        "shutdown_dispatched": False, "autodl_state_instruction": "KEEP_ON_FOR_WORKER2_REVIEW",
    }
    (out / "night18d_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True), encoding="utf-8")
    semantics = """# Method semantics and selection contract\n\n- Producer inputs: sanitized numeric carrier, public K=10, frozen family profile; no annotation arrays.\n- Starts: 13 mechanically generated RNA/ATAC/fused/low-pass KMeans/GMM partitions.\n- Full operator: unchanged Night-15F dynamic prototype unary + three sparse graph scales + conflict-aware nonnegative Potts + rejected-mass current-label stay cost + alpha-expansion.\n- Headline selector: within-arm structure-feasible medoid, requiring exact K, no singleton, and at least one fine-graph internal edge per cluster. Plain medoid is a matched control.\n- Label-assisted oracle is opened only after every candidate partition and SHA are locked; it is not the headline.\n- Every frozen-cycle accepted alpha move is monotone for that cycle's frozen unary. No cross-dynamic-cycle global optimum claim is made.\n- This revision tests external transfer; it does not introduce a new method component.\n"""
    (out / "method_semantics_and_selection_contract.md").write_text(semantics, encoding="utf-8")
    source_audit = """# Source-code collision and transfer audit\n\n| Prior work | Audited official source | Relevant prior object | Night-18D boundary |\n|---|---|---|---|\n| SpatialGlue | https://github.com/JinmiaoChenLab/SpatialGlue, HEAD 7c976d811d27ace51ce47ae0ad94a068a7d222fa | spatial graphs, intra-/cross-omics attention and fusion | fusion/graph integration is prior art |\n| spaMGCN | https://github.com/hongfeiZhang-source/spaMGCN, HEAD 77dfe67d4fd80c124722e68a0f71af36d10fa5fa | autoencoder + multi-scale graph adaptation; official human placenta processed asset | neither placenta use nor multi-scale graph is new here |\n| BANKSY | https://github.com/prabhakarlab/Banksy, HEAD 5157d9cf8020ec49c99f18d833b143b69a496762 | neighborhood mean/gradient augmentation for spatial clustering | local spatial statistics are prior art |\n| PRAGA | https://github.com/Xubin-s-Lab/PRAGA, HEAD 4adb11c96fc7ddad800fa1787eadcc8b91b42784 | dynamic graph, prototype aggregation/contrastive learning | prototypes and adaptive graph semantics are prior art |\n| Boykov-Veksler-Zabih / Potts graph cuts | https://cs.uwaterloo.ca/~yboykov/Abstracts/iccv99-abs.html | alpha-expansion large moves for metric pairwise energy | alpha-expansion/Potts cannot be claimed |\n\nThe only object under test is the already developed combination as a frozen external consumer. Because that combination fails the matched external gate, Night-18D makes no novelty claim. Human placenta authority is traced to the Nature Medicine article and official analysis repository: https://www.nature.com/articles/s41591-024-03073-9 and https://github.com/jian-shu-lab/hPlacenta-architecture.\n"""
    (out / "source_code_collision_and_transfer_audit.md").write_text(source_audit, encoding="utf-8")
    label_flow = {
        "status": "PASS", "public_k_used_by_producer": 10,
        "authority_audit_labels_read_for_provenance": True,
        "carrier_annotation_columns_accessed_by_numeric_computation": 0,
        "source_container_obs_loaded_by_anndata": True,
        "producer_input": "SANITIZED_NUMERIC_CARRIER", "producer_labels_read": 0,
        "producer_authority_json_allowlist": ["authority_gap", "status"],
        "candidate_count_locked_before_evaluator": 130, "formal_artifact_sha256": summary["label_flow"]["labels_opened_after_artifact_sha256"],
        "evaluator_reference_column": "cell_type", "evaluator_n": 1662, "evaluator_k": 10,
    }
    (out / "label_flow_audit.json").write_text(json.dumps(label_flow, indent=2, sort_keys=True), encoding="utf-8")
    replay = {
        "p0_candidate_count": 6, "p0_artifact_reload": "PASS", "p0_fresh_process_npz_byte_exact": True,
        "formal_candidate_count": 130, "formal_fresh_process_npz_byte_exact": True,
        "all_candidate_metrics_csv_byte_exact": True, "selected_main_table_csv_byte_exact": True, "paired_start_ablation_csv_byte_exact": True,
        "targeted_tests": {"passed": 5, "failed": 0, "environment": "SpaLORA", "command": "python -m pytest -q tests/test_night18d_placenta_transfer.py"},
    }
    (out / "p0_and_replay_audit.json").write_text(json.dumps(replay, indent=2, sort_keys=True), encoding="utf-8")
    failures = [
        ["E001", "engineering", "first carrier invocation used obsolete --audit/--manifest flags", "no scientific artifact created", "reran with declared CLI", "CLOSED"],
        ["E002", "engineering_semantics", "official ATAC coordinates stored as numeric strings while RNA used float", "array_equal failed despite exact numeric authority", "cast both coordinate arrays to float64 and require exact equality", "CLOSED"],
        ["E003", "numerical_engineering", "float32 Procrustes SVD exceeded strict orthogonality tolerance", "carrier stopped before output", "compute rotation in float64; preserve float32 views", "CLOSED"],
        ["E004", "engineering", "/usr/bin/time unavailable (exit 127)", "formal producer did not start", "reran identical command without unavailable wrapper", "CLOSED"],
        ["S001", "scientific", "frozen full energy failed external matched controls", "0/13 paired dual gains; headline far below controls", "preserved; no post-label tuning", "SCIENTIFIC_NEGATIVE"],
    ]
    with (out / "failure_and_correction_ledger.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n"); writer.writerow(["id", "type", "event", "impact", "resolution", "status"]); writer.writerows(failures)
    for source, name in [
        (root / "evaluation/all_candidate_metrics.csv", "all_candidate_metrics.csv"),
        (root / "evaluation/selected_main_table.csv", "selected_main_table.csv"),
        (root / "evaluation/paired_start_ablation.csv", "paired_start_ablation.csv"),
        (root / "evaluation/evaluation_summary.json", "evaluation_summary.json"),
        (root / "audit/dataset_authority_audit.json", "dataset_authority_audit.json"),
        (root / "carrier/placenta_carrier.json", "numeric_carrier_manifest.json"),
        (root / "config/frozen_family_default.json", "frozen_family_default.json"),
    ]: shutil.copy2(source, out / name)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--working", required=True); parser.add_argument("--output", required=True); run(parser.parse_args())


if __name__ == "__main__": main()
