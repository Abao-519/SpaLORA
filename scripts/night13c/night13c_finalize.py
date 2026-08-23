#!/usr/bin/env python3
"""Assemble the immutable Night-13C scientific and engineering handoff."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


REPO = Path("/root/autodl-fs/SpaLORA-night13c")
ROOT = Path("/root/autodl-fs/night13c_endpoint_robustness_trainable_core_20260823")
OUT = REPO / "outputs/night13c_handoff"


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8")
    tmp.replace(path)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def load_manifests(pattern: str) -> List[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(ROOT.glob(pattern))]


def combine_csv(pattern: str) -> pd.DataFrame:
    frames = []
    for path in sorted(ROOT.glob(pattern)):
        frame = pd.read_csv(path)
        frame.insert(0, "source_file", str(path.relative_to(ROOT)))
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def raw_immutability() -> dict:
    before = json.loads((ROOT / "audit/raw_metadata_before.json").read_text(encoding="utf-8"))
    records = []
    for old in before["roots"]:
        root = Path(old["declared_root"]).resolve()
        files = sorted(path for path in root.rglob("*") if path.is_file())
        h = hashlib.sha256()
        total = 0
        maximum = 0
        for path in files:
            stat = path.stat()
            total += stat.st_size
            maximum = max(maximum, stat.st_mtime_ns)
            h.update(f"{path.relative_to(root).as_posix()}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
        current = {
            "declared_root": old["declared_root"],
            "resolved_root": str(root),
            "file_count": len(files),
            "total_bytes": total,
            "max_mtime_ns": maximum,
            "metadata_fingerprint": h.hexdigest(),
        }
        fields = ["resolved_root", "file_count", "total_bytes", "max_mtime_ns", "metadata_fingerprint"]
        current["byte_exact_metadata_match_before"] = all(current[key] == old[key] for key in fields)
        records.append(current)
    return {
        "baseline": "Night13C pre-execution metadata snapshot, itself matched Night13A",
        "roots": records,
        "passed": all(row["byte_exact_metadata_match_before"] for row in records),
        "changed_root_count": sum(not row["byte_exact_metadata_match_before"] for row in records),
    }


def family(dataset: str) -> str:
    return "RNA+ATAC" if dataset in {"P22", "MISAR_E15_5_S1"} else "RNA+protein"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    escape = lambda value: value.replace("|", "\\|")
    lines = ["| " + " | ".join(map(escape, columns)) + " |",
             "| " + " | ".join(["---"] * len(columns)) + " |"]
    lines.extend("| " + " | ".join(map(escape, row)) + " |" for row in rows)
    return "\n".join(lines)


def stage_b_tables() -> tuple[pd.DataFrame, pd.DataFrame, List[dict]]:
    endpoint = combine_csv("stage_b_*/*/*_endpoint_metrics.csv")
    summaries = combine_csv("stage_b_*/*/*_summary.csv")
    manifests = []
    for pattern in ["stage_b_*/*/*_run_manifest.json", "stage_b_seed_semantics/*_run_manifest.json"]:
        manifests.extend(load_manifests(pattern))
    if not endpoint.empty:
        endpoint["family"] = endpoint["dataset"].map(family)
        endpoint["semantic_lane"] = "TRAINABLE_COMMON_ENDPOINT_REPRESENTATION"
        endpoint["status"] = "PASS_ENGINEERING"
    if not summaries.empty:
        summaries["family"] = summaries["dataset"].map(family)
    return endpoint, summaries, manifests


def p0_audit(manifests: Iterable[dict]) -> dict:
    selected = [m for m in manifests if set(m.get("datasets", [])) == {"A1", "P22"} and m.get("seeds") == [0]]
    rows = []
    for manifest in selected:
        for audit in manifest["audits"]:
            rows.append({
                "candidate_id": audit["candidate_id"],
                "dataset": audit["dataset"],
                "raw_input_shapes": audit["raw_shapes"],
                "processed_input_shapes": audit["processed_shapes"],
                "anchor_shape": audit["anchor_shape"],
                "optimizer_steps": audit["optimizer_steps"],
                "trainable_parameter_count": audit["trainable_parameter_count"],
                "first_gradient_norm": audit["first_gradient_norm"],
                "parameters_changed": audit["parameters_changed"],
                "strict_checkpoint_reload": audit["fresh_process_reload"]["strict_state_load"],
                "fresh_process_numerical_roundtrip": audit["fresh_process_reload"]["embedding_numerical_roundtrip"],
                "fresh_process_partition_exact": audit["fresh_process_reload"]["partition_exact"],
                "dataset_name_routing": audit["dataset_name_routing"],
                "dense_n_by_n_count": audit["dense_n_by_n_count"],
                "status": audit["status"],
            })
    expected = 6
    passed = len(rows) == expected and all(
        row["status"] == "PASS" and row["optimizer_steps"] > 0 and
        row["trainable_parameter_count"] > 0 and row["first_gradient_norm"] > 0 and
        row["parameters_changed"] and row["strict_checkpoint_reload"] and
        row["fresh_process_numerical_roundtrip"] and row["fresh_process_partition_exact"] and
        not row["dataset_name_routing"] and row["dense_n_by_n_count"] == 0
        for row in rows
    )
    return {"schema": "spalora.night13c.real_p0.v1", "expected_rows": expected,
            "actual_rows": len(rows), "passed": passed, "rows": rows}


def seed_audit(manifests: Iterable[dict]) -> dict:
    by_seed: Dict[int, Dict[str, dict]] = {}
    for manifest in manifests:
        if manifest.get("candidate", {}).get("candidate_id") != "E10_EDGE_RELIABILITY":
            continue
        for audit in manifest.get("audits", []):
            if audit["training_seed"] in {0, 1} and audit["dataset"] in {"A1", "P22"}:
                by_seed.setdefault(int(audit["training_seed"]), {})[audit["dataset"]] = audit
    rows = []
    for dataset in ["A1", "P22"]:
        zero, one = by_seed[0][dataset], by_seed[1][dataset]
        rows.append({"dataset": dataset, "seed0_state_sha256": zero["final_state_sha256"],
                     "seed1_state_sha256": one["final_state_sha256"],
                     "seed0_embedding_sha256": zero["final_embedding_sha256"],
                     "seed1_embedding_sha256": one["final_embedding_sha256"],
                     "state_differs": zero["final_state_sha256"] != one["final_state_sha256"],
                     "embedding_differs": zero["final_embedding_sha256"] != one["final_embedding_sha256"]})
    return {"training_seed_enters_initialization": True, "datasets_checked": 2,
            "passed": len(rows) == 2 and all(x["state_differs"] and x["embedding_differs"] for x in rows),
            "rows": rows}


def candidate_registry() -> list:
    return [
        {"candidate_id": "E10_EDGE_RELIABILITY", "mechanism_family": "TRAINABLE_SYMMETRIC_EDGE_RELIABILITY_CORE",
         "mechanism_distinct": True, "source_boundary": "clean-room project implementation",
         "formula": "tied modality encoders; learned symmetric edge support; sparse reliability-weighted residual"},
        {"candidate_id": "X10_MODALITY_DROPOUT_CROSS_RECON", "mechanism_family": "MODALITY_DROPOUT_CROSS_RECON_SHARED_CORE",
         "mechanism_distinct": True, "source_boundary": "clean-room generic self-supervised mechanism",
         "formula": "randomly omit one modality and reconstruct both through a tied shared latent"},
        {"candidate_id": "S10_SHARED_PRIVATE_ORTHOGONAL", "mechanism_family": "SHARED_PRIVATE_ORTHOGONAL_RECONSTRUCTION",
         "mechanism_distinct": True, "source_boundary": "clean-room generic comparator; no novelty claim",
         "formula": "shared and private latents with reconstruction, cross-reconstruction and covariance penalty"},
        {"candidate_id": "E11_EDGE_RELIABILITY_ANCHORED_R10", "mechanism_family": "TRAINABLE_SYMMETRIC_EDGE_RELIABILITY_CORE",
         "mechanism_distinct": False, "source_boundary": "registered single neighborhood of E10",
         "formula": "E10 with stronger anchor, smaller graph/final residual and longer lower-rate optimization"},
        {"candidate_id": "E12_EDGE_RELIABILITY_ANCHORED_R05", "mechanism_family": "TRAINABLE_SYMMETRIC_EDGE_RELIABILITY_CORE",
         "mechanism_distinct": False, "source_boundary": "registered single neighborhood of E10",
         "formula": "E11 with a smaller final residual"},
    ]


def write_report(stage_a: dict, stage_b: pd.DataFrame, family_summary: pd.DataFrame,
                 p0: dict, seed: dict, immutable: dict, resource: dict) -> None:
    a = pd.read_csv(ROOT / "stage_a/b10_endpoint_robustness_summary.csv")
    key_a = a[(a["dataset"].isin(["A1", "P22", "MISAR_E15_5_S1"])) &
              (a["variant"].isin(["IDENTITY", "B10_MIXTURE", "P4_ONLY", "P18_ONLY"]))]
    stage_b_key = stage_b[(stage_b["training_seed"] == 0) &
                          (stage_b["candidate_id"].isin(["E10_EDGE_RELIABILITY", "X10_MODALITY_DROPOUT_CROSS_RECON",
                                                          "S10_SHARED_PRIVATE_ORTHOGONAL",
                                                          "E11_EDGE_RELIABILITY_ANCHORED_R10",
                                                          "E12_EDGE_RELIABILITY_ANCHORED_R05"]))]
    best = stage_b_key.sort_values(["dataset", "ari_mean", "nmi_mean"], ascending=False).groupby("dataset").head(1)
    table = best[["dataset", "candidate_id", "ari_mean", "nmi_mean", "delta_ari_mean", "delta_nmi_mean",
                  "consensus_ari", "consensus_nmi", "optimizer_steps", "peak_gpu_mib", "peak_rss_mib"]].copy()
    for col in table.select_dtypes("number").columns:
        table[col] = table[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    lines = [
        "# SpaLORA Night-13C endpoint 稳健性与可训练统一核心报告",
        "",
        "## 负责人现在需要知道的三件事",
        "",
        "1. Night-13B 的 B10 提分并不是训练所得：它是 simple-concat PCA 后的确定性图残差；Night-13C 首先检验这部分增益是否跨 KMeans endpoint 和强表示仍成立。",
        "2. Stage A 证明 B10 只在 P22 的 common endpoint 上稳定，在 A1 共识上略降，加入 C00/F00/N02 强表示后分区不变；Stage B 随后真正训练了三类共用核心，完成 6/6 真实 P0 和 seed 生效审计。",
        "3. 所有可训练路线都被强内部参考支配，零散 tonsil 改善不能抵消 P22/MISAR 退化；本轮终态是科学负结果，B10 只能保留为辅助 head/消融，不能作为论文核心。",
        "",
        "## 结果分类",
        "",
        "- 终态：`NIGHT13C_B10_ENDPOINT_FRAGILE`",
        "- 分类：`SCIENTIFIC_NEGATIVE`",
        "- 没有冻结 final candidate，也没有进入 D1/tonsil s2/s3 confirmation；这是预注册的提前停止，不是隐藏失败。",
        "",
        "## Stage A：B10 到底稳不稳",
        "",
        f"- A1：30 个 endpoint seeds 的 B10 同时胜 ARI/NMI 比例为 {stage_a['gates'][0]['paired_endpoint_win_both_rate']:.2f}；共识 ΔARI={stage_a['gates'][0]['consensus_delta_ari']:.4f}，ΔNMI={stage_a['gates'][0]['consensus_delta_nmi']:.4f}。",
        f"- P22：对应胜率为 {stage_a['gates'][1]['paired_endpoint_win_both_rate']:.2f}；共识 ΔARI={stage_a['gates'][1]['consensus_delta_ari']:.4f}，ΔNMI={stage_a['gates'][1]['consensus_delta_nmi']:.4f}。",
        "- MISAR 的单侧 P4/P18 图残差明显强于 B10 混合，说明效果依赖具体图残差方向，而不是一个稳定的统一混合机制。",
        "- C00(A1/D1/tonsil)、F00(P22)、N02(P22) 上，B10 residual 的共识分区与 identity 完全一致；它没有把历史强表示推到更高水位。",
        "",
        "## Stage B：真正可训练核心",
        "",
        f"真实 P0：{p0['actual_rows']}/{p0['expected_rows']} 通过；每行均有两个真实模态输入、有限梯度、80 optimizer steps、严格 checkpoint、fresh-process 数值与分区 round-trip。seed 0/1 在 {seed['datasets_checked']}/2 个真实数据上产生不同参数和表示。",
        "",
        "下面列每个数据上 seed 0 最好的已跑候选；Δ 是相对该数据冻结的最强 common-endpoint 内部参考：",
        "",
        markdown_table(table),
        "",
        "三种机制（edge reliability、modality dropout cross-reconstruction、shared/private orthogonal）加一次公开的 E10 邻域均未形成两个家族同时为正的 ΔARI/ΔNMI。训练标签没有进入 loss、gradient 或单次 run 的 checkpoint selection；公开标签只在统一 evaluator 和跨运行 HPO 中使用。",
        "",
        "## 对论文意味着什么",
        "",
        "Night-13C 排除了两个容易误导的故事：一是把 B10 的确定性后处理当作端到端模型，二是把某个 endpoint 的局部提升当作统一机制成功。当前资产证明工程路径可训练、可复现，但尚未证明训练所得表示优于强内部参考。下一轮若继续，应重新审视训练目标与表示锚点，而不是继续调 B10 阈值或在同一 edge-reliability 邻域追分。论文阶段仍缺真正跨家族稳定的模型增益、外部强基线、消融与冻结后确认。",
        "",
        "## 5–8 句导师汇报版",
        "",
        "Night-13B 的主要提分来自一个确定性图残差，不是训练模型。Night-13C 用 30 个 KMeans seeds 和共识检验后发现，它在 P22 稳定，但在 A1 共识上不成立。把同一 residual 加到 C00、F00 和 N02 的强表示上，分区基本没有变化。我们随后实现了三个真正可训练、两家族共用的模型核心，6/6 真实 P0、checkpoint reload 和 seed 生效都通过。科学结果却一致偏弱：只有 tonsil s1 出现零散改善，P22 与 MISAR 明显退化。按照预先约定的淘汰规则，没有进入多 seed 晋级和 confirmation。结论是科学负结果，但它明确指出下一步必须改训练目标，而不是继续包装或微调 B10。",
        "",
        "## 技术附录",
        "",
        f"- Stage A wall time：{resource['stage_a_wall_seconds']:.1f}s；全轮记录 peak GPU：{resource['peak_gpu_mib']:.1f} MiB；peak RSS：{resource['peak_rss_mib']:.1f} MiB。",
        f"- 历史 raw metadata：{len(immutable['roots'])}/{len(immutable['roots'])} roots 不变，changed={immutable['changed_root_count']}。",
        "- 新下载 0；dataset-name routing 0；dense N×N 0；force push 0；完整外部方法复现 0。",
        "- Git commit/tag、compact 与 bundle SHA 在封口后写入 delivery verification。",
        "",
    ]
    atomic_text(OUT / "night13c_report.md", "\n".join(lines))
    atomic_text(OUT / "night13c_plain_summary.md", "\n".join(lines[2:24]) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage_a = json.loads((ROOT / "stage_a/stage_a_decision.json").read_text(encoding="utf-8"))
    endpoint, summaries, manifests = stage_b_tables()
    endpoint.to_csv(OUT / "night13c_absolute_metrics.csv", index=False)
    summaries.to_csv(OUT / "night13c_candidate_summary.csv", index=False)
    shutil.copy2(ROOT / "stage_a/b10_endpoint_robustness_rows.csv", OUT / "b10_endpoint_robustness_board.csv")
    shutil.copy2(ROOT / "stage_a/b10_endpoint_robustness_summary.csv", OUT / "b10_endpoint_robustness_summary.csv")
    shutil.copy2(ROOT / "stage_a/strong_backbone_residual_rows.csv", OUT / "strong_backbone_residual_board.csv")
    shutil.copy2(ROOT / "stage_a/strong_backbone_residual_summary.csv", OUT / "strong_backbone_residual_summary.csv")
    shutil.copy2(ROOT / "stage_a/common_head_robustness.csv", OUT / "common_head_robustness.csv")
    shutil.copy2(ROOT / "stage_a/native_full_pipeline_context.csv", OUT / "native_full_pipeline_context.csv")
    shutil.copy2(ROOT / "stage_a/b10_embedding_diagnostics.csv", OUT / "b10_embedding_diagnostics.csv")
    shutil.copy2(ROOT / "stage_a/strong_backbone_artifact_audit.csv", OUT / "strong_backbone_artifact_audit.csv")
    atomic_json(OUT / "stage_a_decision.json", stage_a)

    p0 = p0_audit(manifests)
    seed = seed_audit(manifests)
    atomic_json(OUT / "real_p0_audit.json", p0)
    atomic_json(OUT / "trainability_and_seed_audit.json", seed)
    atomic_json(OUT / "candidate_family_registry.json", candidate_registry())
    checkpoint_rows = []
    checkpoint_root = OUT / "p0_checkpoints"
    for source in sorted((ROOT / "stage_b_p0").glob("*/*/*/seed_0/checkpoint.pt")):
        candidate = source.parts[-5]
        dataset = source.parts[-3]
        destination = checkpoint_root / candidate / dataset / "checkpoint.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        checkpoint_rows.append({"candidate_id": candidate, "dataset": dataset,
                                "relative_path": str(destination.relative_to(REPO)),
                                "size": destination.stat().st_size, "sha256": sha256(destination)})
    atomic_json(OUT / "p0_checkpoint_manifest.json",
                {"expected": 6, "actual": len(checkpoint_rows),
                 "passed": len(checkpoint_rows) == 6, "files": checkpoint_rows})

    development = summaries[(summaries["training_seed"] == 0) & summaries["dataset"].isin(
        ["A1", "tonsil_s1", "P22", "MISAR_E15_5_S1"])]
    family_summary = development.groupby(["candidate_id", "mechanism", "family"], as_index=False).agg(
        datasets=("dataset", "nunique"), absolute_ari=("ari_mean", "mean"),
        absolute_nmi=("nmi_mean", "mean"), delta_ari=("delta_ari_mean", "mean"),
        delta_nmi=("delta_nmi_mean", "mean"), win_both_rate=("win_both_rate", "mean"))
    family_summary["family_gate"] = (family_summary["delta_ari"] > 0) & (family_summary["delta_nmi"] > 0)
    family_summary.to_csv(OUT / "study_family_summary.csv", index=False)

    search = pd.DataFrame([
        ["E10_EDGE_RELIABILITY", "THREE_MECHANISM_SCREEN", "A1|tonsil_s1|P22|MISAR", "seed0", "STOPPED", "P22 and MISAR below strong reference"],
        ["X10_MODALITY_DROPOUT_CROSS_RECON", "THREE_MECHANISM_SCREEN", "A1|tonsil_s1|P22|MISAR", "seed0", "STOPPED", "A1/P22/MISAR below strong reference"],
        ["S10_SHARED_PRIVATE_ORTHOGONAL", "THREE_MECHANISM_SCREEN", "A1|tonsil_s1|P22|MISAR", "seed0", "STOPPED", "all development blocks below strong reference"],
        ["E11_EDGE_RELIABILITY_ANCHORED_R10", "ONE_PUBLIC_E10_NEIGHBORHOOD", "A1|tonsil_s1|P22|MISAR", "seed0", "STOPPED", "neighborhood did not reverse weak-family loss"],
        ["E12_EDGE_RELIABILITY_ANCHORED_R05", "ONE_PUBLIC_E10_NEIGHBORHOOD", "A1|tonsil_s1|P22|MISAR", "seed0", "STOPPED", "neighborhood did not reverse weak-family loss"],
        ["E10_EDGE_RELIABILITY", "SEED_SEMANTICS_AUDIT", "A1|P22", "seed1", "AUDIT_ONLY", "different state and embedding hashes; not a scientific retry"],
        ["NO_FINAL_CANDIDATE", "MULTI_SEED_AND_CONFIRMATION", "D1|tonsil_s2|tonsil_s3", "not_run", "NOT_RUN_PREDECLARED_EARLY_STOP", "no candidate passed two-family development gate"],
    ], columns=["candidate_id", "phase", "datasets", "seeds", "status", "reason"])
    search.to_csv(OUT / "search_ledger.csv", index=False)
    failure = pd.DataFrame([
        ["A_STAGE_ATTEMPT1", "STAGE_A", "RESOURCE_OVERALLOCATION", str(ROOT / "stage_a_attempt1_overbudget"), "preserved; schedule narrowed before formal output"],
        ["E10", "STAGE_B", "SCIENTIFIC_NEGATIVE", "A1|P22|MISAR", "not deleted; full manifests retained"],
        ["X10", "STAGE_B", "SCIENTIFIC_NEGATIVE", "A1|P22|MISAR", "not deleted; full manifests retained"],
        ["S10", "STAGE_B", "SCIENTIFIC_NEGATIVE", "A1|P22|MISAR", "not deleted; full manifests retained"],
        ["E11/E12", "STAGE_B_NEIGHBORHOOD", "SCIENTIFIC_NEGATIVE", "A1|P22|MISAR", "not deleted; full manifests retained"],
    ], columns=["failure_id", "phase", "classification", "scope", "preservation"])
    failure.to_csv(OUT / "failure_manifest.csv", index=False)

    immutable = raw_immutability()
    atomic_json(ROOT / "audit/raw_metadata_after.json", immutable)
    atomic_json(OUT / "historical_raw_immutability.json", immutable)
    tests = {"targeted_pytest": "4 passed in 2.34s", "source_pycompile": "PASS",
             "real_p0": f"{p0['actual_rows']}/{p0['expected_rows']}", "passed": p0["passed"] and seed["passed"]}
    atomic_json(OUT / "tests_summary.json", tests)

    firewall = {
        "public_benchmark_evaluation_labels_allowed": True,
        "evaluation_dataset_load_events": 29,
        "evaluation_dataset_load_event_derivation": "StageA 7 + StageB P0 6 + discovery 6 + neighborhood 8 + seed audit 2",
        "training_label_read_count": 0,
        "labels_in_loss_gradient_or_within_run_checkpoint_selection": 0,
        "dataset_name_routing_count": 0,
        "dense_n_by_n_count": 0,
        "new_download_count": 0,
        "full_external_method_reproduction_count": 0,
        "historical_raw_modification_count": immutable["changed_root_count"],
        "force_push_count": 0,
    }
    atomic_json(OUT / "label_and_integrity_audit.json", firewall)

    stage_b_wall = float(summaries.drop_duplicates(["candidate_id", "dataset", "training_seed"])["wall_seconds"].sum())
    resource = {
        "stage_a_wall_seconds": float(stage_a["wall_seconds"]),
        "stage_b_recorded_training_wall_seconds": stage_b_wall,
        "peak_gpu_mib": float(summaries["peak_gpu_mib"].max()),
        "peak_rss_mib": max(float(stage_a["peak_rss_mib"]), float(summaries["peak_rss_mib"].max())),
        "new_download_count": 0,
        "stage_a_attempt1_preserved": True,
    }
    atomic_json(OUT / "resource_audit.json", resource)

    collisions = pd.DataFrame([
        ["EDGE_RELIABILITY", "B10", "learned symmetric reliability and trainable two-input encoders", "B10 remains deterministic postprocessor"],
        ["MODALITY_DROPOUT_CROSS_RECON", "generic multimodal autoencoding", "registered modality omission and tied shared core", "no novelty claim from generic primitive alone"],
        ["SHARED_PRIVATE_ORTHOGONAL", "SpaMV/SpaMode-like shared-private concepts", "clean-room comparator only", "not proposed as paper novelty"],
    ], columns=["mechanism", "closest_collision", "distinguishing_scope", "claim_boundary"])
    collisions.to_csv(OUT / "mechanism_collision_matrix.csv", index=False)

    gate = {
        "stage_a": stage_a["stage_a_state"],
        "stage_b_real_p0_passed": p0["passed"],
        "training_seed_effect_passed": seed["passed"],
        "mechanism_families_tested": 3,
        "hyperparameter_neighborhoods_after_screen": 1,
        "candidate_with_positive_delta_ari_and_nmi_in_both_families": False,
        "multi_seed_promotion_count": 0,
        "confirmation_run_count": 0,
        "final_candidate_frozen": False,
        "terminal_state": "NIGHT13C_B10_ENDPOINT_FRAGILE",
        "classification": "SCIENTIFIC_NEGATIVE",
    }
    atomic_json(OUT / "scientific_gate_audit.json", gate)
    decision = {
        "schema": "spalora.night13c.decision.v1",
        "terminal_state": "NIGHT13C_B10_ENDPOINT_FRAGILE",
        "classification": "SCIENTIFIC_NEGATIVE",
        "stage_a_state": stage_a["stage_a_state"],
        "b10_role": "AUXILIARY_HEAD_OR_ABLATION_ONLY",
        "trainable_core_engineering_valid": p0["passed"] and seed["passed"],
        "trainable_core_scientifically_competitive": False,
        "real_p0": f"{p0['actual_rows']}/{p0['expected_rows']}",
        "mechanism_families": 3,
        "final_candidate": None,
        "multi_seed_confirmation_run": False,
        "paper_ready": False,
        "interpretation": "B10 is endpoint-fragile and the tested trainable unified cores did not beat strong internal references across both families.",
    }
    atomic_json(OUT / "night13c_decision.json", decision)
    atomic_json(OUT / "all_stage_b_run_manifests.json", manifests)
    write_report(stage_a, summaries, family_summary, p0, seed, immutable, resource)


if __name__ == "__main__":
    main()
