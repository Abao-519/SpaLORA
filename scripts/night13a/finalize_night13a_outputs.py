#!/usr/bin/env python3
"""Assemble the frozen Night-13A audit, board, and decision artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
from pathlib import Path


REPO = Path("/root/autodl-fs/SpaLORA-night13a")
RAW = Path("/root/autodl-fs/night13a_benchmark_expansion_20260822")
OUT = REPO / "outputs/night13a_handoff"
BOARD_ROOT = RAW / "derived/board"

DATASETS = [
    ("A1 lymph node", "A1", "public development/benchmark", 10),
    ("D1 lymph node", "D1", "public development/benchmark", 10),
    ("tonsil slice 1", "tonsil_s1", "public development benchmark", 4),
    ("tonsil slice 2", "tonsil_s2", "public development benchmark", 4),
    ("tonsil slice 3", "tonsil_s3", "public development benchmark", 4),
    ("P22 mouse brain", "P22", "public development/benchmark", 9),
    ("MISAR E15.5 S1", "MISAR_E15_5_S1", "public schema only", None),
    ("legacy placenta", "placenta", "legacy public development benchmark", None),
]

METHODS = ["simple_standardized_concatenation", "SpatialGlue", "SMART", "ARISE"]

BOARD_COLUMNS = [
    "dataset",
    "method",
    "endpoint",
    "K",
    "absolute_ari",
    "absolute_nmi",
    "ami",
    "fmi",
    "homogeneity",
    "v_measure",
    "morans_i",
    "wall_time_seconds",
    "gpu_time_seconds",
    "peak_gpu_mib",
    "peak_rss_mib",
    "status",
    "exception_summary",
    "artifact_sha256",
    "seed",
    "prior_use",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def dump_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def update_registry() -> None:
    path = OUT / "canonical_dataset_registry.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
        columns = list(rows[0])
    for row in rows:
        if row["physical_section"] == "tonsil_slice3":
            row["shape"] = "4521x18085;4521x35"
        if row["physical_section"] == "human_placenta_architecture":
            row["status"] = "UNSUPPORTED_MODALITY_PROVENANCE"
            row["overlap_status"] = "PROVENANCE_NOT_CLOSED;ATAC_OBJECT_IS_1662x63_GENE_EXPRESSION"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def simple_row(display: str, key: str, prior_use: str, expected_k: int | None) -> dict:
    path = BOARD_ROOT / f"simple_{key}" / "result.json"
    if not path.exists():
        if key == "MISAR_E15_5_S1":
            status = "UNSUPPORTED_NO_CANONICAL_LABEL"
            reason = "No reliable canonical public spatial-domain annotation is registered."
        elif key == "placenta":
            status = "UNSUPPORTED_MODALITY_PROVENANCE"
            reason = "The deposited second H5AD is 1662x63 Gene Expression, has two zero-library rows, and does not close a raw ATAC/LSI provenance path."
        else:
            status = "FAILED_OUTPUT_MISSING"
            reason = "Expected simple-anchor result is absent."
        return {
            "dataset": display,
            "method": METHODS[0],
            "endpoint": "COMMON_KMEANS_SEED0",
            "K": "" if expected_k is None else expected_k,
            "status": status,
            "exception_summary": reason,
            "seed": 0,
            "prior_use": prior_use,
        }
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "dataset": display,
        "method": data["method"],
        "endpoint": data["endpoint"],
        "K": data["k"],
        "absolute_ari": data["absolute_ari"],
        "absolute_nmi": data["absolute_nmi"],
        "ami": data["ami"],
        "fmi": data["fmi"],
        "homogeneity": data["homogeneity"],
        "v_measure": data["v_measure"],
        "morans_i": data["morans_i"],
        "wall_time_seconds": data["wall_seconds"],
        "gpu_time_seconds": data["gpu_seconds"],
        "peak_gpu_mib": data["peak_gpu_mib"],
        "peak_rss_mib": data["peak_rss_mib"],
        "status": data["status"],
        "exception_summary": "",
        "artifact_sha256": data["artifact_sha256"],
        "seed": data["seed"],
        "prior_use": prior_use,
    }


def external_row(display: str, key: str, prior_use: str, expected_k: int | None, method: str) -> dict:
    required_p0 = key in {"A1", "P22"}
    if required_p0:
        if method == "SpatialGlue":
            status = "BLOCKED_DENSE_NXN"
            reason = "Official preprocessing densifies adjacency with toarray() and uses dataset-keyed configuration."
        elif method == "SMART":
            status = "BLOCKED_DENSE_NXN"
            reason = "Official MNN path materializes pairwise_distances(X), and tutorials use dataset-specific/manual settings."
        else:
            status = "BLOCKED_DENSE_NXN_AND_LABEL_DRIVEN_SELECTION"
            reason = "Official path creates dense cosine/adjacency matrices and trains/selects with true labels and ARI/NMI."
    else:
        status = "NOT_RUN_METHOD_LANE_BLOCKED_AT_REQUIRED_P0"
        reason = "The official method failed the mandatory A1/P22 family path boundary before board expansion."
    return {
        "dataset": display,
        "method": method,
        "endpoint": "COMMON_ENDPOINT_NOT_REACHED",
        "K": "" if expected_k is None else expected_k,
        "status": status,
        "exception_summary": reason,
        "seed": 0,
        "prior_use": prior_use,
    }


def write_board() -> list[dict]:
    rows = []
    for display, key, prior_use, expected_k in DATASETS:
        rows.append(simple_row(display, key, prior_use, expected_k))
        for method in METHODS[1:]:
            rows.append(external_row(display, key, prior_use, expected_k, method))
    path = OUT / "development_baseline_board.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BOARD_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in BOARD_COLUMNS})
    return rows


def metadata_snapshot(stage: str) -> dict:
    before = json.loads((OUT / "historical_raw_immutability_before.json").read_text(encoding="utf-8"))
    records = []
    for old in before["roots"]:
        root = Path(old["declared_root"]).resolve()
        files = sorted(path for path in root.rglob("*") if path.is_file())
        h = hashlib.sha256()
        for path in files:
            stat = path.stat()
            rel = path.relative_to(root).as_posix()
            h.update(f"{rel}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
        current = {
            "declared_root": old["declared_root"],
            "resolved_root": str(root),
            "file_count": len(files),
            "total_bytes": sum(path.stat().st_size for path in files),
            "max_mtime_ns": max((path.stat().st_mtime_ns for path in files), default=0),
            "metadata_fingerprint": h.hexdigest(),
        }
        current["byte_exact_metadata_match_before"] = all(
            current[field] == old[field]
            for field in ["resolved_root", "file_count", "total_bytes", "max_mtime_ns", "metadata_fingerprint"]
        )
        records.append(current)
    result = {
        "stage": stage,
        "roots": records,
        "passed": all(record["byte_exact_metadata_match_before"] for record in records),
        "changed_root_count": sum(not record["byte_exact_metadata_match_before"] for record in records),
    }
    dump_json(OUT / "historical_raw_immutability_after.json", result)
    return result


def write_changelog() -> None:
    rows = [
        ["ENG-001", "test compatibility", "Python 3.8 lacks ast.unparse", "Use ast.dump in the targeted AST test", "test_runner_has_no_dataset_name_routing", "PASS"],
        ["ENG-002", "identity-blind audit scope", "AST initially inspected legal dataset metadata in I/O loader", "Constrain audit to model and scientific runner functions", "test_runner_has_no_dataset_name_routing", "PASS"],
        ["ENG-003", "observation contract", "P22 has 19 observations outside the canonical annotation subset", "Add explicit registered-observation subset before numeric preprocessing", "test_registered_observation_filter_is_explicit_in_loader_signature", "PASS"],
        ["ENG-004", "label I/O alignment", "A1 public label IDs carry a deposited s1- format prefix", "Add explicit CLI-only label prefix transform", "test_label_prefix_is_only_an_explicit_io_alignment_option", "PASS"],
        ["ENG-005", "public annotation missingness", "Tonsil slice 2/3 contain 1/61 missing final_annot values", "Filter missing public annotations before the common endpoint, preserving counts", "test_label_prefix_is_only_an_explicit_io_alignment_option", "PASS"],
    ]
    path = OUT / "engineering_changelog.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["correction_id", "boundary", "observed_issue", "correction", "regression_test", "status"])
        writer.writerows(rows)


def write_tests_summary() -> dict:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/night13a/test_night13a_runner.py"],
        cwd=str(REPO),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    text = result.stdout
    path = OUT / "tests_summary.txt"
    path.write_text(text, encoding="utf-8")
    return {"exit_code": result.returncode, "passed": result.returncode == 0, "sha256": sha256(path), "text": text.strip()}


def write_report(rows: list[dict], immutability: dict, tests: dict) -> None:
    passing = [row for row in rows if row["method"] == METHODS[0] and row["status"] == "PASS"]
    table = [
        "| 数据集 | K | ARI | NMI | AMI | FMI | Moran's I | 墙钟(s) | GPU(s) | 峰值GPU(MiB) | 峰值RSS(MiB) | 状态 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in passing:
        table.append(
            f"| {row['dataset']} | {row['K']} | {float(row['absolute_ari']):.4f} | {float(row['absolute_nmi']):.4f} | "
            f"{float(row['ami']):.4f} | {float(row['fmi']):.4f} | {float(row['morans_i']):.4f} | "
            f"{float(row['wall_time_seconds']):.2f} | {float(row['gpu_time_seconds']):.2f} | "
            f"{float(row['peak_gpu_mib']):.1f} | {float(row['peak_rss_mib']):.1f} | {row['status']} |"
        )
    p0 = json.loads((OUT / "real_path_p0_results.json").read_text(encoding="utf-8"))
    report = f"""# Night-13A 数据资产、统一 runner 与一种子开发基线报告

## 我现在需要知道的三件事

1. 本轮解决的是公开开发 benchmark 的数据去重、统一工程入口和绝对指标起点，不是训练新方法。
2. 实际完成了 RNA+protein 与 RNA+ATAC 各一条 feature-level 零步真实路径、严格 checkpoint/reload，以及 6 个有可靠公开标签的数据集上的同一 simple anchor/common endpoint。
3. 结果是 **部分完成**：数据表、2/2 真实路径和一种子 simple baseline 板已闭合，但三个外部强基线的官方实现都违反本轮 sparse/identity-blind/common-endpoint 边界，因此不能诚实地写成完整统一 benchmark board，更不能声称 SOTA。

## 结果分类

`NIGHT13A_PARTIAL_BENCHMARK_BOARD`

## 一种子开发基线主表

以下均为 seed 0、simple standardized concatenation、同一 `COMMON_KMEANS_SEED0` endpoint。K 来自每个 canonical public annotation 的唯一类别数，只读取一次并对方法一致；本轮没有根据分数调 seed、epoch、resolution 或 loss weight。

{os.linesep.join(table)}

完整的 32-cell 板保存在 `development_baseline_board.csv`：6 个 simple 单元通过；MISAR E15.5 因没有可靠 canonical spatial-domain label 而 unsupported；legacy placenta 的第二模态只闭合到 1662x63 Gene Expression 对象且有 2 个零 library observations，不能冒充原始 ATAC/LSI；所有外部失败/未运行单元均保留。

## 数据与真实工程路径

- canonical registry 共 15 行；Zenodo tonsil slice 1/2/3 是 canonical，GSE263617 tonsil A1/D1 因 count signature 不同而只保留为歧义来源审计对象，未重复计数。
- 本轮只有 1 个新下载：Zenodo `data_imputation.zip`，486,654,827 bytes；SPOTS 四个 processed 文件和 P5S1 均复用已有权威原件。
- SPOTS spleen rep1：原始 `2653x32285 RNA + 2653x21 ADT`，2,653 个 byte-exact barcode+tissue paired spots，fresh-process 数值与 partition round-trip 通过。
- P5S1：`7794x32285 RNA + 83,593,412x5 fragment rows`，按 Night-12A Ensembl79 clean-room gene-score 合约得到 `7794x256 + 7794x256`，fresh-process 数值与 partition round-trip 通过。
- 两条 P0 都是 seed 0、optimizer steps 0、finite reconstruction loss、sparse graph、dense N x N 计数 0；P0 峰值 GPU {max(r['forward']['peak_gpu_mib'] for r in p0['rows']):.2f} MiB，峰值 RSS {max(r['forward']['peak_rss_mib'] for r in p0['rows']):.2f} MiB。

## 外部源码与失败 lane

- SpatialGlue：官方 preprocessing 将邻接矩阵 `toarray()`，并按 dataset key 解析配置；A1/P22 lane fail-closed。
- SMART：官方 MNN 路径使用 `pairwise_distances(X)` 构造 dense N x N，tutorial 另有 dataset-specific 和手工 cluster 编辑；A1/P22 lane fail-closed。
- ARISE：官方路径构造 dense cosine/adjacency，并在训练期间读取真标签计算 ARI/NMI 来选 best；A1/P22 lane fail-closed。
- 这些不是把 wrapper 调通就能修的 API 问题；若改变图、loss 或 label-driven selection，就不再是官方复现。因此三个 baseline 没有伪造通过。SpaMV、SpaMode、SpaBalance、CANDIES 仅做 source-transfer audit，不执行完整训练。

## 公开标签与研究边界

本轮按用户新授权把 A1、D1、canonical tonsil 和 P22 明确作为公开 benchmark 的开发/复现实验；不声称 pristine blind test。没有为 SPOTS 或 GSE308623 P5 制造标签。没有结果驱动 HPO，没有新方法候选，没有 SOTA 结论。失败单元和 4 次失败尝试保留。历史 raw 的 7/7 metadata fingerprints 在审计前后完全一致：{immutability['passed']}。 targeted tests：{tests['text'].splitlines()[-1] if tests['text'] else 'no output'}。

## 对论文意味着什么

Night-13A 给后续性能优先路线提供了可审计的数据版本表、统一输出格式、绝对指标和资源基线。它也暴露了一个现实问题：若坚持 sparse、无 dataset routing、统一 endpoint，当前三个外部官方实现不能直接纳入同一公平 runner。下一阶段可以把 simple 板作为 Night-13B 的起点，但必须先决定是接受每个方法的 native engineering boundary，还是另设“数学语义改变后的重实现”组；二者不能混称官方复现。本轮本身不支持新方法有效、聚类提升或 SOTA 的结论。

## 导师汇报版

我们先把重复和来源歧义的数据资产去重，固定了真正计入的物理切片。RNA+protein 与 RNA+ATAC 两条真实零步路径都完成了 checkpoint 和 fresh-process round-trip。随后用同一个 simple concatenation 加 common KMeans endpoint，在 6 个有可靠公开标签的数据集上给出了 seed 0 的绝对 ARI/NMI 与资源基线。这个表是公开 benchmark 的开发结果，不是盲测。SpatialGlue、SMART 和 ARISE 的官方代码分别涉及 dense N x N、dataset-specific 设置或标签驱动选择，与本轮的公平边界冲突，所以没有伪造复现通过。legacy placenta 和 MISAR E15.5 的缺口也被透明保留。最终状态是部分 benchmark board，足够作为 Night-13B 的工程起点，但还不是新方法证据，也不构成 SOTA。

## 技术附录

- parent: `c2ed63d460521c198cea79b4f64c2b126672e493` / `night12b-final-20260822`
- branch: `revision/q2-night13a-benchmark-expansion-unified-runner-20260822`
- final tag: `night13a-final-20260822`
- engineering corrections: 5；preserved failed attempts: 4
- new downloads: 1 file / 486,654,827 bytes；outside whitelist: 0
- historical raw: {len(immutability['roots'])}/{len(immutability['roots'])} unchanged；changed roots: {immutability['changed_root_count']}
- final commit、compact index SHA 和 bundle SHA 由最终 tag/Windows delivery manifest 给出，避免在 commit 内容内制造循环 hash。
"""
    (OUT / "night13a_report.md").write_text(report, encoding="utf-8")


def write_manifest_and_decision(rows: list[dict], immutability: dict, tests: dict) -> None:
    p0 = json.loads((OUT / "real_path_p0_results.json").read_text(encoding="utf-8"))
    downloads = json.loads((OUT / "download_manifest.json").read_text(encoding="utf-8"))
    simple_pass = sum(row["method"] == METHODS[0] and row["status"] == "PASS" for row in rows)
    external_pass = sum(row["method"] != METHODS[0] and row["status"] == "PASS" for row in rows)
    run_manifest = {
        "schema": "night13a-run-manifest-v1",
        "seed": 0,
        "result_driven_hpo_count": 0,
        "new_method_candidate_training_count": 0,
        "dense_n_by_n_count": 0,
        "dataset_name_model_routing_count": 0,
        "canonical_registry": {"rows": 15, "sha256": sha256(OUT / "canonical_dataset_registry.csv")},
        "real_path_p0": {"passed": p0["passed"], "expected": p0["expected"], "sha256": sha256(OUT / "real_path_p0_results.json")},
        "development_board": {"rows": len(rows), "simple_pass": simple_pass, "external_pass": external_pass, "sha256": sha256(OUT / "development_baseline_board.csv")},
        "downloads": downloads,
        "tests": tests,
        "historical_raw_immutability": immutability,
        "failed_attempt_artifacts": [
            str(RAW / "derived/board/simple_A1_attempt1_failed"),
            str(RAW / "derived/board/simple_tonsil_s2_attempt1_failed"),
            str(RAW / "derived/board/simple_tonsil_s3_attempt1_failed"),
            str(RAW / "derived/board/simple_placenta"),
        ],
        "environment": {"python": platform.python_version(), "platform": platform.platform()},
    }
    dump_json(OUT / "run_manifest.json", run_manifest)
    decision = {
        "status": "NIGHT13A_PARTIAL_BENCHMARK_BOARD",
        "classification": "PARTIAL_ENGINEERING_AND_DEVELOPMENT_BENCHMARK",
        "reason": "Canonical registry, 2/2 real-family P0 paths, and a six-dataset simple seed-0 board passed; zero strong external baseline passed both families because official implementations violate frozen sparse/identity-blind/common-endpoint boundaries.",
        "success_gate": {
            "canonical_registry_closed": True,
            "real_family_p0": "2/2",
            "simple_anchor_passed": True,
            "strong_external_baselines_both_families": "0/2 required",
            "development_board_rows": len(rows),
            "failures_visible": True,
            "new_method_or_sota_claim": False,
        },
        "metrics_scope": "public benchmark development/reproduction; not pristine confirmatory",
        "forbidden_or_out_of_scope_counts": {
            "result_driven_hpo": 0,
            "new_method_candidate_training": 0,
            "fabricated_labels": 0,
            "dense_n_by_n_project_runner": 0,
            "dataset_name_model_routing": 0,
            "historical_raw_modifications": immutability["changed_root_count"],
            "outside_whitelist_downloads": 0,
            "force_pushes": 0,
        },
        "engineering_correction_count": 5,
        "preserved_failed_attempt_count": 4,
        "tests_passed": tests["passed"],
    }
    dump_json(OUT / "night13a_decision.json", decision)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    update_registry()
    rows = write_board()
    write_changelog()
    immutability = metadata_snapshot("after")
    tests = write_tests_summary()
    write_manifest_and_decision(rows, immutability, tests)
    write_report(rows, immutability, tests)
    if not immutability["passed"] or not tests["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
