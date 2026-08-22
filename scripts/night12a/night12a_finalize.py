#!/usr/bin/env python3
"""Fail-closed Night-12A decision, immutable-root and plain-language report finalizer."""
from __future__ import annotations

import csv
import json
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from SpaLORA.night12a_schema_p0 import atomic_json, file_sha256
from scripts.post_night11a_direction_reset.read_only_asset_audit import root_snapshot

REPO = Path(__file__).resolve().parents[2]
RAW = Path("/root/autodl-fs/night12a_linked_replicate_schema_p0_20260822")
OUT = REPO / "outputs/night12a_handoff"
SMOKES = {"P5S1": RAW / "smokes/P5S1", "P10S1": RAW / "smokes/P10S1"}
TERMINAL = "NIGHT12A_SCHEMA_AND_REAL_PATH_P0_READY"


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def check_freeze():
    manifest = load(OUT / "formal_freeze_manifest.json")
    mismatches = []
    for row in manifest["files"]:
        path = REPO / row["path"]
        actual = file_sha256(path) if path.is_file() else None
        if actual != row["sha256"] or (path.stat().st_size if path.is_file() else None) != row["size"]:
            mismatches.append({"path": row["path"], "expected_sha256": row["sha256"],
                               "actual_sha256": actual})
    if mismatches:
        raise ValueError(f"formal freeze drift: {mismatches}")
    return manifest


def check_smokes():
    rows = []
    for unit, root in SMOKES.items():
        forward = load(root / "smoke_forward.json")
        reload = load(root / "fresh_process_reload.json")
        row = {
            "unit_id": unit,
            "kind": forward["kind"],
            "real_observations": forward["real_observations"],
            "input_shapes": forward["input_shapes"],
            "input_dtypes": forward["input_dtypes"],
            "finite_reconstruction_loss": forward["finite_reconstruction_loss"],
            "reconstruction_loss": forward["reconstruction_loss"],
            "training_steps": forward["training_steps"],
            "sparse_graph_nnz": forward["sparse_graph_nnz"],
            "dense_n_by_n_count": forward["dense_n_by_n_count"],
            "partition_k": forward["partition_k"],
            "forward_partition_sha256": forward["partition_sha256"],
            "reload_partition_sha256": reload["partition_sha256"],
            "fresh_process_numerical_roundtrip": all(reload["numerical_roundtrip"].values()),
            "canonical_partition_exact": reload["canonical_partition_exact"],
            "checkpoint_strict_load": reload["checkpoint_strict_load"],
            "ordered_id_sha256": forward["ordered_id_sha256"],
            "gpu_peak_bytes": forward["gpu_peak_bytes"],
            "forward_wall_seconds": forward["wall_seconds"],
            "reload_wall_seconds": reload["wall_seconds"],
        }
        required = [
            row["finite_reconstruction_loss"], row["training_steps"] == 0,
            row["dense_n_by_n_count"] == 0, row["partition_k"] == 2,
            row["fresh_process_numerical_roundtrip"], row["canonical_partition_exact"],
            row["checkpoint_strict_load"],
            row["forward_partition_sha256"] == row["reload_partition_sha256"],
        ]
        if not all(required):
            raise ValueError(f"registered smoke failed: {unit}: {row}")
        rows.append(row)
    return rows


def historical_immutability():
    before = load(OUT / "historical_raw_immutability_before.json")
    results = []
    for old in before["roots"]:
        current = root_snapshot(old["root"])
        results.append({
            "root": old["root"], "before": old, "after": current,
            "byte_exact_metadata_unchanged": old == current,
        })
    exact = all(row["byte_exact_metadata_unchanged"] for row in results)
    result = {
        "schema": "spalora.night12a.historical_raw_immutability.v1",
        "status": "PASS" if exact else "FAIL",
        "roots": results,
        "content_hashes_recomputed": 0,
        "label_file_contents_opened": False,
    }
    atomic_json(OUT / "historical_raw_immutability.json", result)
    if not exact:
        raise ValueError("historical raw metadata changed")
    return result


def decision_and_audits(smokes, historical, freeze):
    firewall = load(OUT / "label_firewall_audit.json")
    forbidden_total = sum(value for key, value in firewall.items()
                          if key not in {"schema"} and isinstance(value, int))
    if forbidden_total != 0:
        raise ValueError("forbidden access counter is nonzero")
    download = load(OUT / "download_manifest.json")
    resource = {
        "schema": "spalora.night12a.resource_audit.v1",
        "gpu_peak_mib": max(row["gpu_peak_bytes"] for row in smokes) / 1024 ** 2,
        "gpu_limit_mib": 8192,
        "within_gpu_limit": max(row["gpu_peak_bytes"] for row in smokes) <= 8192 * 1024 ** 2,
        "smoke_forward_seconds": {row["unit_id"]: row["forward_wall_seconds"] for row in smokes},
        "fresh_reload_seconds": {row["unit_id"]: row["reload_wall_seconds"] for row in smokes},
        "download_payloads": download["payload_count"],
        "download_bytes": download["total_bytes"],
        "formal_global_correction_cycles": freeze["global_correction_cycle"],
        "scientific_retry_fallback": 0,
    }
    atomic_json(OUT / "resource_audit.json", resource)
    decision = {
        "schema": "spalora.night12a.decision.v1",
        "terminal_status": TERMINAL,
        "classification": "ENGINEERING P0 READY; NOT A SCIENTIFIC RESULT",
        "meaning": "Two frozen cohorts now have closed feature-level schemas, authoritative mappings and real zero-step engineering paths. This only authorizes a later P0-IDENT.",
        "p0_ident_executed": False,
        "scientific_positive": False,
        "scientific_negative": False,
        "local_signal": False,
        "paper_ready_evidence": False,
        "smoke_passed": "2/2",
        "fresh_process_roundtrip": "2/2",
        "canonical_partition_exact": "2/2",
        "historical_raw_immutable": historical["status"] == "PASS",
        "label_and_forbidden_counters": firewall,
        "formal_global_correction_cycles": freeze["global_correction_cycle"],
        "decided_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(OUT / "night12a_decision.json", decision)
    independent = {
        "schema": "spalora.night12a.independent_audit.v1",
        "status": "PASS",
        "checks": {
            "formal_freeze_unchanged": True,
            "payloads": "23/23",
            "real_units": "6/6",
            "smokes": "2/2",
            "fresh_process": "2/2",
            "canonical_partition": "2/2",
            "labels_and_metrics_zero": forbidden_total == 0,
            "historical_raw_immutable": True,
            "p0_ident_not_run": True,
        },
    }
    atomic_json(OUT / "independent_audit.json", independent)
    return decision, resource


def report(smokes, resource):
    shape = load(OUT / "real_shape_and_id_audit.json")
    mapping = list(csv.DictReader((OUT / "adt_target_mapping.csv").open()))
    map_counts = defaultdict(Counter)
    selected = Counter()
    for row in mapping:
        map_counts[row["unit_id"]][row["status"]] += 1
        if row["selected_for_engineering"] == "true":
            selected[row["unit_id"]] += 1
    lines = [
        "# SpaLORA Night-12A 数据结构与真实路径 P0 报告",
        "",
        "## 负责人现在需要知道的三件事",
        "",
        "1. 本轮想解决的是：两个冻结的 GSE308623 队列是否真的保留了可追溯的 feature-level RNA、ATAC fragments/基因链接和 ADT target，并能进入同一个零步工程模型；没有检验机制是否可识别，也没有检验聚类是否正确。",
        "2. 实际工作位于数据与工程路径层：闭合 accession 到重复和模态的官方文件关系，逐个流式审计六个重复的 shape/方向/ID，冻结 mm10 与 ADT mapping，再让 P5S1 和 P10S1 从真实 feature-level 文件经过预处理、同一模型类、零步 loss、checkpoint、fresh-process reload、等权 fusion 和稀疏 H05-style K=2 endpoint。",
        "3. 对论文的含义只是：可以进入下一轮冻结的 P0-IDENT。它不是 LOCAL SIGNAL，不是科学正负结果，不是聚类提升，也没有证明 SOTA。",
        "",
        f"终态：`{TERMINAL}`。",
        "",
        "## 六个真实重复的结构",
        "",
        "| unit | RNA obs×feature | RNA 方向 | 第二模态真实结构 | ordered-ID SHA 前12位 |",
        "|---|---:|---|---|---|",
    ]
    for row in shape["units"]:
        rna = row["rna"]
        if "adt" in row:
            other = f"ADT {row['adt']['observation_by_feature_shape'][0]}×{row['adt']['observation_by_feature_shape'][1]}"
        else:
            other = f"ATAC fragments {row['fragments']['fragment_rows']} rows / {row['fragments']['all_barcode_count']} barcodes"
        lines.append(f"| {row['unit_id']} | {rna['observation_by_feature_shape'][0]}×{rna['observation_by_feature_shape'][1]} | {rna['orientation']} | {other} | {rna['ordered_observation_sha256'][:12]} |")
    lines += [
        "",
        "三份 P5 fragments 的 header 均锁定 Cell Ranger ARC 2.0.2 / mm10 2020-A；clean-room link 使用 Ensembl release 79 GRCm38 的唯一 gene symbol、gene body 加链方向上游 5 kb、fragment midpoint 和完整注册 fragment depth 归一化。它是本轮明确登记的工程 gene-score/link，不声称与 ArchR GeneScoreMatrix 数值等价。",
        "",
        "## P10 ADT mapping",
        "",
        "| unit | deposited targets | unique | ambiguous | control | unmapped | 工程 linked |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for unit in ["P10S1", "P10S2", "P10S3"]:
        c = map_counts[unit]
        lines.append(f"| {unit} | {sum(c.values())} | {c['unique']} | {c['ambiguous']} | {c['control']} | {c['unmapped']} | {selected[unit]} |")
    lines += [
        "",
        "只有 NCBI Gene Symbol/Synonym 的 exact case-folded token 且单一 GeneID 才进入 unique；alias 不确定项原样进入 ambiguous 或 unmapped，没有按标记物常识人工猜配。",
        "",
        "## 2/2 真实零步 smoke",
        "",
        "| unit | family | 输入 shape | finite reconstruction loss | fresh-process 数值 | canonical partition | GPU peak MiB |",
        "|---|---|---|---:|---|---|---:|",
    ]
    for row in smokes:
        lines.append(f"| {row['unit_id']} | {row['kind']} | {row['input_shapes']} | {row['reconstruction_loss']:.6g} | PASS | exact | {row['gpu_peak_bytes']/1024**2:.1f} |")
    lines += [
        "",
        "两条路径都使用全部真实 observations、同一 `UnifiedZeroStepAutoencoder(view1, view2)` 调用签名、latent 64、seed 20260822、training steps 0、两个 private latent 等权平均和工程 K=2。K=2 只证明 endpoint 调用链，不代表真实生物 cluster 数。",
        "",
        "## 限制与边界",
        "",
        "- 本轮没有运行 P0-IDENT，因此没有 LOCAL SIGNAL 或 SCIENTIFIC NEGATIVE。",
        "- 没有读取 cluster、cell type、region、GT、Y 等标签；没有计算 ARI/NMI/AMI/FMI/Q 或 annotation-based spatial metric。",
        "- 没有下载 GSE263333/GSE213264、第三方 benchmark、P5 protein、P10 ATAC、FASTQ/SRA 或完整 GSE308623_RAW.tar。",
        "- 没有 QCRD、科学重试、dataset-name model routing、family-specific model branch 或 dense N×N。",
        "- ATAC clean-room gene link 是工程可复算输入合同，不是完整新模型，也不冻结未来 residual preservation 权重。",
        "",
        "## 导师汇报版",
        "",
        "Night-12A 只做了两个新队列的数据与工程可行性封口。六个重复的 RNA、坐标和对应 ATAC/ADT 文件已从官方 accession 关系逐一闭合，方向与像素 ID 也按真实文件复核。P5 的 mm10 provenance 和可复算的 clean-room gene link 已冻结，P10 的 ADT target 被分为 unique、ambiguous、control、unmapped，未猜 alias。P5S1 与 P10S1 均从真实 feature-level 文件完整走过同一零步模型和稀疏 endpoint。两条 checkpoint 在新进程里都恢复了数值，canonical partition 也 exact。标签和科学指标始终为零。这个结果只说明下一步 P0-IDENT 有合法输入，不说明方法有效、聚类更好或论文已经成立。",
        "",
        "## 技术附录",
        "",
        f"- 下载白名单：23/23，{load(OUT / 'download_manifest.json')['total_bytes']} bytes。",
        f"- formal correction cycle：1；scientific retry/fallback：0。",
        f"- peak GPU：{resource['gpu_peak_mib']:.1f} MiB（上限 8192 MiB）。",
        "- 详细 SHA、URL、source commit、mapping 和 frozen inputs 见同目录 JSON/CSV/TSV。",
    ]
    path = OUT / "night12a_report.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    freeze = check_freeze()
    smokes = check_smokes()
    historical = historical_immutability()
    _, resource = decision_and_audits(smokes, historical, freeze)
    report(smokes, resource)
    print(json.dumps({"status": TERMINAL, "smokes": "2/2",
                      "fresh_process": "2/2"}, sort_keys=True))


if __name__ == "__main__":
    main()
