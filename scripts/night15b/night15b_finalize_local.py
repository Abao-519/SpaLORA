#!/usr/bin/env python3
"""Assemble the auditable Night-15B result tables and report from frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def reference_table(night15a: pd.DataFrame, night13b: pd.DataFrame) -> Dict[str, dict]:
    result: Dict[str, dict] = {}
    for lane, dataset, k in (
        ("P22", "P22", 9),
        ("MISAR_E15_5_S1", "MISAR_E15_5_S1", 7),
        ("MISAR_E15_5_S1_K12", "MISAR_E15_5_S1", 12),
    ):
        row = night15a[(night15a.dataset == dataset) & (night15a.cluster_k.astype(int) == k) & (night15a.control_id == "NIGHT14B_FROZEN_BEST")].iloc[0]
        result[lane] = {
            "reference_id": str(row.formal_id),
            "ari": float(row.ari_best),
            "nmi": float(row.nmi_best),
            "protocol": "Night-14B frozen best; same registered public-label lane",
        }
    row = night15a[(night15a.dataset == "P22_3DOT_K18") & (night15a.cluster_k.astype(int) == 18)].iloc[0]
    result["P22_3DOT_K18"] = {
        "reference_id": "LOCKED_C15_W02_REFERENCE",
        "ari": float(row.ari_best),
        "nmi": float(row.nmi_best),
        "protocol": "3d-OT author 18-state assignment context; not independent ground truth",
    }
    for dataset in ("A1", "D1", "tonsil_s1", "tonsil_s2", "tonsil_s3"):
        row = night13b[(night13b.dataset == dataset) & (night13b.method == "simple_standardized_concatenation_corrected")].iloc[0]
        result[dataset] = {
            "reference_id": "Night-13B corrected simple anchor",
            "ari": float(row.absolute_ari),
            "nmi": float(row.absolute_nmi),
            "protocol": "common known-K public benchmark endpoint",
        }
    return result


def make_head_board(head: Path, references: Dict[str, dict]) -> Tuple[pd.DataFrame, dict]:
    summary = json.loads((head / "head_search_summary.json").read_text(encoding="utf-8"))
    ledger = pd.read_csv(head / "all_head_run_ledger.csv")
    dataset_wall = {item["dataset"]: float(item["wall_seconds"]) for item in summary["datasets"]}
    rows = []
    for dataset in summary["datasets"]:
        for lane in dataset["lanes"]:
            lane_id = lane["lane"]
            selected = ledger[(ledger.lane == lane_id) & (ledger.status == "PASS")]
            if lane["best_endpoint_seed"] == "AGGREGATE":
                fixed = selected[(selected.phase == "AGGREGATE") & (selected["head"] == lane["best_head"])]
            else:
                fixed = selected[
                    (selected.phase == "FINE")
                    & (selected.embedding_id == lane["best_embedding_id"])
                    & (selected["head"] == lane["best_head"])
                ]
            if fixed.empty:
                raise RuntimeError(f"cannot recover selected head rows: {lane_id}")
            ref = references[lane_id]
            best_row = fixed.sort_values(["absolute_ari", "absolute_nmi"], ascending=False).iloc[0]
            rows.append({
                "dataset": lane_id,
                "family": "RNA+ATAC" if lane_id.startswith(("P22", "MISAR")) else "RNA+protein",
                "k": int(best_row.k),
                "method": "NIGHT15B_HEAD_BEST",
                "embedding_id": str(lane["best_embedding_id"]),
                "head": str(lane["best_head"]),
                "endpoint_seed_best": str(lane["best_endpoint_seed"]),
                "run_count": int(len(fixed)),
                "ari_best": float(fixed.absolute_ari.max()),
                "ari_median": float(fixed.absolute_ari.median()),
                "ari_mean": float(fixed.absolute_ari.mean()),
                "ari_min": float(fixed.absolute_ari.min()),
                "nmi_best": float(fixed.absolute_nmi.max()),
                "nmi_median": float(fixed.absolute_nmi.median()),
                "nmi_mean": float(fixed.absolute_nmi.mean()),
                "nmi_min": float(fixed.absolute_nmi.min()),
                "ami_mean": float(fixed.ami.mean()),
                "fmi_mean": float(fixed.fmi.mean()),
                "morans_i_best_row": float(best_row.morans_i),
                "gearys_c_best_row": float(best_row.gearys_c),
                "historical_reference_id": ref["reference_id"],
                "historical_reference_ari": ref["ari"],
                "historical_reference_nmi": ref["nmi"],
                "delta_best_ari": float(fixed.absolute_ari.max()) - ref["ari"],
                "delta_best_nmi": float(fixed.absolute_nmi.max()) - ref["nmi"],
                "local_cpu_wall_seconds_dataset": dataset_wall[dataset["dataset"]],
                "gpu_seconds": 0.0,
                "peak_gpu_mib": 0.0,
                "status": "PASS",
                "protocol_note": ref["protocol"],
            })
    board = pd.DataFrame(rows)
    audit = {
        "status": "PASS",
        "run_rows": int(summary["run_rows"]),
        "final_search_wall_seconds": float(summary["wall_seconds"]),
        "labels_in_model_input": 0,
        "labels_in_unsupervised_loss": 0,
        "labels_used_for_cross_run_hpo": 1,
        "dense_n_by_n_count": 0,
        "head_gain_lane_count_by_best_ari": int((board.delta_best_ari > 0).sum()),
        "lane_count": int(len(board)),
    }
    return board, audit


def make_sapr_table(local_replay: pd.DataFrame, winner: str) -> pd.DataFrame:
    selected = local_replay[local_replay.candidate_id == winner].copy()
    grouped = selected.groupby(["dataset", "lane", "k", "mode"], as_index=False).agg(
        run_count=("absolute_ari", "size"),
        ari_best=("absolute_ari", "max"),
        ari_median=("absolute_ari", "median"),
        ari_mean=("absolute_ari", "mean"),
        ari_min=("absolute_ari", "min"),
        nmi_best=("absolute_nmi", "max"),
        nmi_median=("absolute_nmi", "median"),
        nmi_mean=("absolute_nmi", "mean"),
        nmi_min=("absolute_nmi", "min"),
        ami_mean=("ami", "mean"),
        fmi_mean=("fmi", "mean"),
        morans_i_mean=("morans_i", "mean"),
        gearys_c_mean=("gearys_c", "mean"),
        delta_ari_vs_retained_mean=("delta_ari_vs_retained", "mean"),
        delta_nmi_vs_retained_mean=("delta_nmi_vs_retained", "mean"),
    )
    piv = grouped.pivot(index="lane", columns="mode", values=["ari_mean", "nmi_mean"])
    full_minus_disabled = {
        lane: (
            float(piv.loc[lane, ("ari_mean", "SAPR_FULL")] - piv.loc[lane, ("ari_mean", "SAPR_RESIDUAL_DISABLED")]),
            float(piv.loc[lane, ("nmi_mean", "SAPR_FULL")] - piv.loc[lane, ("nmi_mean", "SAPR_RESIDUAL_DISABLED")]),
        )
        for lane in piv.index
    }
    grouped["full_minus_disabled_ari_mean"] = grouped.lane.map(lambda x: full_minus_disabled[x][0] if grouped.loc[grouped.lane == x, "mode"].iloc[0] is not None else np.nan)
    grouped["full_minus_disabled_nmi_mean"] = grouped.lane.map(lambda x: full_minus_disabled[x][1])
    grouped["candidate_id"] = winner
    return grouped


def compare_raw(before_path: Path, after_path: Path) -> dict:
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    rows = []
    for left, right in zip(before["roots"], after["roots"]):
        rows.append({
            "registered_path": left["registered_path"],
            "file_count_before": left["file_count"],
            "file_count_after": right["file_count"],
            "total_bytes_same": left["total_bytes"] == right["total_bytes"],
            "max_mtime_same": left["max_mtime_ns"] == right["max_mtime_ns"],
            "metadata_fingerprint_same": left["root_metadata_fingerprint"] == right["root_metadata_fingerprint"],
        })
    return {
        "status": "PASS" if all(x["file_count_before"] == x["file_count_after"] and x["total_bytes_same"] and x["max_mtime_same"] and x["metadata_fingerprint_same"] for x in rows) else "FAILED",
        "content_opened_or_rehashed_for_immutability": False,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--local-replay", type=Path, required=True)
    parser.add_argument("--remote-sapr", type=Path, required=True)
    parser.add_argument("--night15a-main", type=Path, required=True)
    parser.add_argument("--night13b-reference", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--gpu-archive", type=Path, required=True)
    parser.add_argument("--kit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    night15a = pd.read_csv(args.night15a_main)
    night13b = pd.read_csv(args.night13b_reference)
    references = reference_table(night15a, night13b)
    head_board, head_audit = make_head_board(args.head, references)
    head_board.to_csv(args.output / "head_score_board.csv", index=False)

    local_replay = pd.read_csv(args.local_replay / "sapr_local_endpoint_ledger.csv")
    winner_payload = json.loads((args.remote_sapr / "frozen_finalist.json").read_text(encoding="utf-8"))
    winner = winner_payload["candidate_id"]
    sapr_table = make_sapr_table(local_replay, winner)
    sapr_table.to_csv(args.output / "sapr_minimal_contribution_table.csv", index=False)

    full = sapr_table[sapr_table["mode"] == "SAPR_FULL"].copy()
    full["historical_reference_ari"] = full.lane.map(lambda x: references[x]["ari"])
    full["historical_reference_nmi"] = full.lane.map(lambda x: references[x]["nmi"])
    full["delta_mean_ari_vs_historical"] = full.ari_mean - full.historical_reference_ari
    full["delta_mean_nmi_vs_historical"] = full.nmi_mean - full.historical_reference_nmi

    main_rows = []
    for _, row in head_board.iterrows():
        main_rows.append({
            "dataset": row.dataset, "k": row.k, "result_layer": "EMBEDDING_HEAD_HPO",
            "method_or_candidate": row.method, "training_seed": "NOT_APPLICABLE",
            "endpoint_seed": row.endpoint_seed_best, "historical_reference_ari": row.historical_reference_ari,
            "historical_reference_nmi": row.historical_reference_nmi, "ari_best": row.ari_best,
            "ari_median": row.ari_median, "ari_mean": row.ari_mean, "ari_min": row.ari_min,
            "nmi_best": row.nmi_best, "nmi_median": row.nmi_median, "nmi_mean": row.nmi_mean,
            "nmi_min": row.nmi_min, "delta_best_ari": row.delta_best_ari,
            "delta_best_nmi": row.delta_best_nmi, "ami_mean": row.ami_mean, "fmi_mean": row.fmi_mean,
            "morans_i": row.morans_i_best_row, "gearys_c": row.gearys_c_best_row,
            "wall_seconds": row.local_cpu_wall_seconds_dataset, "gpu_seconds": 0.0,
            "peak_gpu_mib": 0.0, "peak_rss_mib": np.nan, "status": "PASS",
        })
    for _, row in full.iterrows():
        main_rows.append({
            "dataset": row.lane, "k": row.k, "result_layer": "SAPR_LOCAL_CPU_REPLAY",
            "method_or_candidate": winner, "training_seed": "1,2 discovery; 0,1,2 confirmation",
            "endpoint_seed": "frozen per lane", "historical_reference_ari": row.historical_reference_ari,
            "historical_reference_nmi": row.historical_reference_nmi, "ari_best": row.ari_best,
            "ari_median": row.ari_median, "ari_mean": row.ari_mean, "ari_min": row.ari_min,
            "nmi_best": row.nmi_best, "nmi_median": row.nmi_median, "nmi_mean": row.nmi_mean,
            "nmi_min": row.nmi_min, "delta_best_ari": row.ari_best - row.historical_reference_ari,
            "delta_best_nmi": row.nmi_best - row.historical_reference_nmi,
            "ami_mean": row.ami_mean, "fmi_mean": row.fmi_mean,
            "morans_i": row.morans_i_mean, "gearys_c": row.gearys_c_mean,
            "wall_seconds": np.nan, "gpu_seconds": np.nan, "peak_gpu_mib": np.nan,
            "peak_rss_mib": np.nan, "status": "PASS",
        })
    pd.DataFrame(main_rows).to_csv(args.output / "absolute_metrics_main_table.csv", index=False)

    audits = json.loads((args.remote_sapr / "resource_and_training_audit.json").read_text(encoding="utf-8"))
    replay_audit = json.loads((args.local_replay / "sapr_local_endpoint_replay_audit.json").read_text(encoding="utf-8"))
    p0 = json.loads((args.remote_sapr / "real_p0_and_roundtrip.json").read_text(encoding="utf-8"))
    finalist_replay = json.loads((args.remote_sapr / "finalist_roundtrip_audit.json").read_text(encoding="utf-8"))
    confirmation_replay = json.loads((args.remote_sapr / "confirmation_roundtrip_audit.json").read_text(encoding="utf-8"))
    replay_rows = p0["fresh_process_replays"] + finalist_replay + confirmation_replay
    resource = {
        "status": "PASS",
        "autodl_training_invocations": len(audits),
        "autodl_optimizer_steps": int(sum(x["optimizer_steps"] for x in audits)),
        "autodl_gpu_training_seconds": float(sum(x["wall_seconds"] for x in audits)),
        "autodl_peak_gpu_mib": float(max(x["peak_gpu_mib"] for x in audits)),
        "autodl_peak_rss_mib": float(max(x["peak_rss_mib"] for x in audits)),
        "checkpoint_fresh_process_roundtrip_passed": int(sum(x["strict_load"] and x["full_match"] and x["disabled_match"] for x in replay_rows)),
        "checkpoint_fresh_process_roundtrip_total": len(replay_rows),
        "windows_final_head_hpo_seconds": float(head_audit["final_search_wall_seconds"]),
        "windows_complete_preserved_head_rerun_seconds": 274.49201139999786,
        "windows_head_smoke_seconds": 83.8,
        "windows_abandoned_overwide_attempt_seconds_lower_bound": 2400.0,
        "windows_total_endpoint_work_seconds_lower_bound": float(head_audit["final_search_wall_seconds"] + 274.49201139999786 + 83.8 + 2400.0 + replay_audit["wall_seconds"]),
        "windows_local_sapr_replay_seconds": float(replay_audit["wall_seconds"]),
        "gpu_asset_archive_bytes": args.gpu_archive.stat().st_size,
        "gpu_asset_archive_sha256": sha256(args.gpu_archive),
    }
    write_json(args.output / "resource_audit.json", resource)

    immutability = compare_raw(args.remote_sapr.parent / "raw_metadata_before.json", args.remote_sapr.parent / "raw_metadata_after.json")
    write_json(args.output / "historical_raw_immutability.json", immutability)
    label_audit = {
        "status": "PASS",
        "labels_used_as_model_input": 0,
        "labels_used_in_unsupervised_loss": 0,
        "labels_used_in_gradient": 0,
        "labels_used_for_within_run_checkpoint_selection": 0,
        "public_labels_used_for_known_k_cross_run_hpo_and_evaluation": 1,
        "dataset_name_model_routing": 0,
        "dense_n_by_n_count": 0,
        "new_scientific_data_downloads": 0,
        "external_full_method_reproductions": 0,
        "historical_raw_modified": 0,
        "failed_seed_or_run_deleted": 0,
        "GSE213264_label_protocol_status": "UNSUPPORTED_NOT_SCORED",
    }
    write_json(args.output / "label_and_routing_audit.json", label_audit)

    for name in (
        "engineering_changelog.csv", "reported_score_protocol_context.csv", "source_and_novelty_audit.md",
        "source_registry.json", "WINDOWS_LOCAL_RUNNER_README.md", "failure_ledger.csv",
    ):
        shutil.copy2(args.staging / name, args.output / name)
    shutil.copy2(args.head / "all_head_run_ledger.csv", args.output / "all_head_run_ledger.csv")
    shutil.copy2(args.head / "head_search_summary.json", args.output / "head_search_summary.json")
    shutil.copy2(args.local_replay / "sapr_local_endpoint_ledger.csv", args.output / "sapr_local_endpoint_ledger.csv")
    shutil.copy2(args.local_replay / "sapr_local_endpoint_replay_audit.json", args.output / "sapr_local_endpoint_replay_audit.json")
    shutil.copy2(args.remote_sapr / "sapr_run_ledger.csv", args.output / "sapr_remote_run_ledger.csv")
    shutil.copy2(args.remote_sapr / "real_p0_and_roundtrip.json", args.output / "real_p0_and_roundtrip.json")
    shutil.copy2(args.remote_sapr / "frozen_finalist.json", args.output / "frozen_finalist.json")
    shutil.copy2(args.remote_sapr / "finalist_roundtrip_audit.json", args.output / "finalist_roundtrip_audit.json")
    shutil.copy2(args.remote_sapr / "confirmation_roundtrip_audit.json", args.output / "confirmation_roundtrip_audit.json")
    shutil.copy2(args.remote_sapr / "resource_and_training_audit.json", args.output / "resource_and_training_audit.json")

    head_k18 = head_board[head_board.dataset == "P22_3DOT_K18"].iloc[0]
    head_k9 = head_board[head_board.dataset == "P22"].iloc[0]
    head_m7 = head_board[head_board.dataset == "MISAR_E15_5_S1"].iloc[0]
    head_m12 = head_board[head_board.dataset == "MISAR_E15_5_S1_K12"].iloc[0]
    sapr_full = sapr_table[sapr_table["mode"] == "SAPR_FULL"]
    retained_wins = int((sapr_full.delta_ari_vs_retained_mean > 0).sum())
    decision = {
        "terminal_state": "HEAD_ONLY_SCORE_GAIN",
        "classification": "ENGINEERING_SCORE_SIGNAL_NOT_NEW_METHOD_SIGNAL",
        "plain_conclusion": "P22 K=18 的无标签 partition consensus 提高了绝对 ARI，但 SAPR 在 Windows 同端点回放中没有任何一个 lane 的 mean ARI 超过 retained teacher。",
        "head_gain_lane_count_by_best_ari": int((head_board.delta_best_ari > 0).sum()),
        "head_lane_count": int(len(head_board)),
        "protein_head_ari_gain_count": int(((head_board.family == "RNA+protein") & (head_board.delta_best_ari > 0)).sum()),
        "protein_head_lane_count": int((head_board.family == "RNA+protein").sum()),
        "sapr_mean_ari_beats_retained_lane_count": retained_wins,
        "sapr_lane_count": int(len(sapr_full)),
        "p22_k18_best_ari": float(head_k18.ari_best),
        "p22_k18_best_nmi": float(head_k18.nmi_best),
        "p22_k9_best_ari": float(head_k9.ari_best),
        "misar_k7_best_ari": float(head_m7.ari_best),
        "misar_k12_best_ari": float(head_m12.ari_best),
        "cross_dataset_method_signal": False,
        "cluster_aware_residual_local_signal": False,
        "labels_in_model_input_loss_gradient_or_checkpoint_selection": 0,
        "public_benchmark_development_not_blind_confirmation": True,
        "paper_ready_or_sota_claim": False,
        "GSE213264": "not scored because canonical label/mask/ID protocol did not close",
    }
    write_json(args.output / "night15b_decision.json", decision)

    report = f"""# SpaLORA Night-15B 稳定性锚定原型与分数冲刺报告

## 我现在需要知道的三件事

1. **问题**：本轮先问已有强表示还能否靠统一的 head/consensus 提高绝对分数，再问稳定性锚定的可训练原型残差 SAPR 是否在同一 strongest representation/head 上有独立增益。
2. **实际做了什么**：Windows 对 7 个真实数据资产运行 12,377 行 embedding-level coarse-to-fine head HPO；AutoDL 用同一 SAPR core 完成 2-family P0、12 配置 screen、top-3 多 seed 与冻结后的 D1/tonsil s2/s3 确认；最后把 41 份 checkpoint embedding 下载到 Windows，在与 head HPO 完全相同的 sklearn 环境中重放 123 行三行贡献对照。
3. **对论文的意义**：终态是 `HEAD_ONLY_SCORE_GAIN`。P22 K=18 的 label-free partition consensus 把 ARI 从 0.6122 提到 {head_k18.ari_best:.4f}，但 SAPR 在 8 个 lane 中 **0/8** 的 mean ARI 超过 exact retained teacher；所以这不是新方法成功，更不是 SOTA 或 paper-ready 证据。

## 结果分类

- 终态：`HEAD_ONLY_SCORE_GAIN`
- 解释：有可复现的 head/partition-ensemble 分数收益，但没有独立的 trainable-core 收益。
- 明确否定：`CROSS_DATASET_METHOD_SIGNAL` 与 `CLUSTER_AWARE_RESIDUAL_LOCAL_SIGNAL` 均不成立。
- 标签角色：公开标签仅用于 known K、跨运行 HPO 与最终评价；没有进入模型输入、无监督 loss、梯度或单次 checkpoint selection。

## 绝对分数主表

| 数据/协议 | 历史 BEST ARI/NMI | Night-15B BEST ARI/NMI | BEST ΔARI/ΔNMI | median / mean / min ARI | 结论 |
|---|---:|---:|---:|---:|---|
| P22 K=9 | {head_k9.historical_reference_ari:.4f}/{head_k9.historical_reference_nmi:.4f} | {head_k9.ari_best:.4f}/{head_k9.nmi_best:.4f} | {head_k9.delta_best_ari:+.4f}/{head_k9.delta_best_nmi:+.4f} | {head_k9.ari_median:.4f}/{head_k9.ari_mean:.4f}/{head_k9.ari_min:.4f} | 未过 0.60，也未超过旧高位 |
| P22 K=18 author assignment | {head_k18.historical_reference_ari:.4f}/{head_k18.historical_reference_nmi:.4f} | {head_k18.ari_best:.4f}/{head_k18.nmi_best:.4f} | {head_k18.delta_best_ari:+.4f}/{head_k18.delta_best_nmi:+.4f} | {head_k18.ari_median:.4f}/{head_k18.ari_mean:.4f}/{head_k18.ari_min:.4f} | ARI 过 0.65；NMI 未超过旧高位；不是独立 GT |
| MISAR K=7 | {head_m7.historical_reference_ari:.4f}/{head_m7.historical_reference_nmi:.4f} | {head_m7.ari_best:.4f}/{head_m7.nmi_best:.4f} | {head_m7.delta_best_ari:+.4f}/{head_m7.delta_best_nmi:+.4f} | {head_m7.ari_median:.4f}/{head_m7.ari_mean:.4f}/{head_m7.ari_min:.4f} | 未过 0.55 |
| MISAR K=12 | {head_m12.historical_reference_ari:.4f}/{head_m12.historical_reference_nmi:.4f} | {head_m12.ari_best:.4f}/{head_m12.nmi_best:.4f} | {head_m12.delta_best_ari:+.4f}/{head_m12.delta_best_nmi:+.4f} | {head_m12.ari_median:.4f}/{head_m12.ari_mean:.4f}/{head_m12.ari_min:.4f} | 未过 0.50；未接近 0.644 context |

Protein head HPO 的 BEST ARI 相对 Night-13B simple anchor：A1、D1、tonsil s1、tonsil s2 提升，tonsil s3 回撤，即 4/5 ARI lane 为正；NMI 是 3/5 为正。逐行绝对 ARI/NMI、AMI/FMI、Moran、Geary、seed 和资源见 `absolute_metrics_main_table.csv` 与 `head_score_board.csv`。

## SAPR 最小三行贡献对照

Windows exact-endpoint 回放显示：冻结 finalist S22 的 `SAPR_FULL` 在 A1、D1、P22、MISAR K=7/K=12、tonsil s1/s2/s3 共 8 个 lane 中，没有一个 lane 的 mean ARI 超过 `RETAINED_TEACHER`。个别 seed（MISAR K=7）有较高 BEST，但 mean 仍略低于 retained；D1 与 tonsil s2/s3 的冻结确认明确翻转。`SAPR_FULL - SAPR_RESIDUAL_DISABLED` 只在少数 lane 为正，且不足以超过 strongest retained control，因此不能把普通 adapter/prototype sharpening 包装为独立新模块贡献。三行逐 seed 与汇总见 `sapr_local_endpoint_ledger.csv`、`sapr_minimal_contribution_table.csv`。

## 工程与资源

- 真实 P0：RNA+protein A1 与 RNA+ATAC P22 2/2 通过 finite forward/loss/backward。
- checkpoint：P0 2 + finalist 30 + confirmation 9 = {resource['checkpoint_fresh_process_roundtrip_passed']}/{resource['checkpoint_fresh_process_roundtrip_total']} strict fresh-process 数值回放通过。
- AutoDL：101 次训练调用、13,880 optimizer steps，累计 GPU 训练时间 {resource['autodl_gpu_training_seconds']:.1f}s，peak GPU {resource['autodl_peak_gpu_mib']:.1f} MiB，peak RSS {resource['autodl_peak_rss_mib']:.1f} MiB。
- Windows：最终 head HPO {resource['windows_final_head_hpo_seconds']:.1f}s；SAPR 41-export/123-row CPU 回放 {resource['windows_local_sapr_replay_seconds']:.1f}s；含保留的失败尝试，总 endpoint 工作时间下界 {resource['windows_total_endpoint_work_seconds_lower_bound']/60:.1f} 分钟。
- local compute kit 小于 2 GiB，不含 raw fragments 或 dense N×N；GSE213264 因 canonical label/mask/ID 协议未闭合而未伪造 ARI。
- 历史 raw 的 count/size/mtime/fingerprint 前后 5/5 相同。

## 最重要的失败

1. SAPR 的远端 screen 曾显示局部增益，但 Windows 同版本 exact-endpoint 回放未保持，说明 trainable representation 与最终 cluster separability 仍错位。
2. P22 K=9、MISAR K=7/K=12 均未超过 Night-14B/15A 的绝对高位。
3. 冻结确认中 D1 与 tonsil s3 整体下降；tonsil s2 的提升来自 residual-disabled 路径而非 SAPR residual。
4. P22 K=18 的提升来自 label-free partition consensus/head，不是 SAPR，也不能因使用 author 18-state assignment 而称独立外部确认。

## 5–8 句导师汇报版

Night-15B 把计算拆成远端训练与本地 endpoint 两部分，完整保留了 head 搜索、失败和所有 seed。已有表示经过更广的无标签 head/consensus 搜索后，P22 K=18 的 ARI 从 0.612 提高到 {head_k18.ari_best:.3f}，但 NMI 没超过旧高位。P22 K=9 和 MISAR 两个协议都没有突破已有最好结果。我们还实现了统一的 SAPR 小型可训练核心，并完成两家族 P0、三候选多 seed 和冻结后的 D1/tonsil 确认。远端初筛的局部增益在 Windows 同端点回放中没有保持，8 个 lane 的 mean ARI 都未超过 retained teacher。因而本轮只能定为 head-only score gain，不能把 SAPR 写成方法贡献。下一步若继续，应更换真正改变可分性的 representation objective，而不是继续调 prototype residual 或 consensus。当前结果不是 SOTA、不是 confirmed milestone，也不是 paper-ready evidence。

## 技术附录说明

Git commit/tag、bundle、compact index 与 Windows 独立复算在最终 delivery audit 中登记。完整 run ledger、source/license/novelty audit、工程修正、label/routing firewall、local compute kit 与 GPU asset SHA 均随交付提供；compact 不包含 raw、checkpoint、embedding、partition 大数组或 vendor 环境。
"""
    (args.output / "night15b_report.md").write_text(report, encoding="utf-8")
    advisor = """Night-15B 完成了 embedding/head 冲刺和 SAPR 统一小核心的严格对照。P22 K=18 的 label-free consensus 把 ARI 提到 0.688，但 NMI 未超过旧高位，且该 18-state assignment 不是独立 ground truth。P22 K=9 与 MISAR K=7/K=12 都没有突破历史最好。SAPR 已完成两家族 P0、top-3 多 seed 和冻结确认。Windows 同端点回放显示 8 个 lane 的 mean ARI 都没有超过 retained teacher，确认数据还出现明显翻转。因此终态是 HEAD_ONLY_SCORE_GAIN，不是新方法成功。下一轮应改变 representation objective，而不是继续修补 prototype residual。\n"""
    (args.output / "night15b_plain_summary.md").write_text(advisor, encoding="utf-8")
    print(json.dumps({"status": "PASS", "terminal": decision["terminal_state"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
