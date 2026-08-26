#!/usr/bin/env python3
"""Build the compact Night-19B scientific handoff from locked artifacts."""

import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


REPO = Path("/root/SpaLORA-night16h")
WORK = Path("/root/night19b_working")
OUT = REPO / "outputs" / "night19b_handoff"
LANES = ("P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7", "PLACENTA_K10")


def sha256(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def main():
    if OUT.exists():
        raise RuntimeError(f"handoff exists: {OUT}")
    OUT.mkdir(parents=True)
    gate = json.loads((WORK / "stage_a" / "stage_a_gate.json").read_text())
    if gate["primary_lane_pass_count"] != 0 or gate["family_stage_authorized"]:
        raise RuntimeError("mechanical gate result unexpectedly changed")

    metrics = []
    operator_rows = []
    p0_rows = []
    replay_rows = []
    label_lanes = []
    for lane in LANES:
        stage = WORK / "stage_a" / lane
        producer = json.loads((stage / "producer.json").read_text())
        evaluation = list(csv.DictReader((stage / "evaluation.csv").open(encoding="utf-8")))
        diagnostics = {row["candidate_id"]: row for row in producer["diagnostics"]}
        for row in evaluation:
            diagnostic = diagnostics[row["candidate_id"]]
            row.update({
                "operator_nnz": diagnostic["operator"]["nnz"],
                "candidate_wall_seconds": diagnostic["candidate_seconds"],
                "producer_total_wall_seconds": producer["wall_seconds"],
                "producer_peak_rss_mib": producer["peak_rss_mib"],
                "gpu_seconds": 0.0,
            })
            metrics.append(row)
            reliability = diagnostic["operator_bank"]["conflict_reliability"]
            operator_rows.append({
                "lane": lane, "candidate_id": row["candidate_id"], "arm": row["arm"],
                "operator_nnz": diagnostic["operator"]["nnz"], "operator_sha256": diagnostic["operator"]["operator_sha256"],
                "finite": diagnostic["operator"]["finite"], "nonnegative": diagnostic["operator"]["nonnegative"],
                "symmetry_max_abs": diagnostic["operator"]["symmetry_max_abs"],
                "reliability_min_median_max": json.dumps(reliability["reliability_min_median_max"], separators=(",", ":")),
                "reliability_unique_count": reliability["reliability_unique_count"],
                "dense_n_by_n_count": 0,
            })
        stage_replay = json.loads((stage / "fresh_replay.json").read_text())
        replay_rows.append({"scope": "STAGE_A", "lane": lane, "partition_exact": stage_replay["partitions_exact"], "eigenvalues_exact": stage_replay["eigenvalues_exact_equal_nan"], "ordered_ids_exact": stage_replay["ordered_ids_exact"]})
        p0 = json.loads((WORK / "p0_formal" / lane / "producer.json").read_text())
        p0d = p0["diagnostics"][0]
        p0r = json.loads((WORK / "p0_formal" / lane / "fresh_replay.json").read_text())
        p0_rows.append({
            "lane": lane, "n": p0["n"], "k": p0["k"], "view1_shape": json.dumps(p0["view1_shape"]),
            "view2_shape": json.dumps(p0["view2_shape"]), "retained_shape": json.dumps(p0["retained_shape"]),
            "ordered_ids_sha256": p0["ordered_ids_sha256"], "carrier_sha256": p0["carrier_sha256"],
            "core_source_sha256": p0["core_source_sha256"], "operator_nnz": p0d["operator"]["nnz"],
            "operator_nonnegative": p0d["operator"]["nonnegative"], "operator_finite": p0d["operator"]["finite"],
            "operator_symmetry_max_abs": p0d["operator"]["symmetry_max_abs"],
            "partition_sha256": p0["partition_sha256"][0], "wall_seconds": p0["wall_seconds"],
            "peak_rss_mib": p0["peak_rss_mib"], "fresh_replay": "PASS" if p0r["partitions_exact"] and p0r["eigenvalues_exact_equal_nan"] else "FAIL",
        })
        replay_rows.append({"scope": "P0_FORMAL", "lane": lane, "partition_exact": p0r["partitions_exact"], "eigenvalues_exact": p0r["eigenvalues_exact_equal_nan"], "ordered_ids_exact": p0r["ordered_ids_exact"]})
        label_lanes.append({
            "lane": lane, "carrier_discovered_keys": producer["carrier_discovered_keys"],
            "carrier_accessed_keys": producer["carrier_accessed_keys"], "carrier_annotation_arrays_accessed": 0,
            "producer_label_reads": 0, "artifact_sha256": producer["artifact_sha256"],
            "artifact_locked_before_evaluator": True,
        })
    write_csv(OUT / "all_candidate_metrics_and_controls.csv", metrics)
    write_csv(OUT / "operator_sparse_quality_audit.csv", operator_rows)
    write_csv(OUT / "real_p0_registry.csv", p0_rows)
    write_csv(OUT / "fresh_process_replay_audit.csv", replay_rows)
    shutil.copy2(WORK / "stage_a" / "stage_a_gate.json", OUT / "stage_a_gate.json")

    corrections = [
        {"revision": "E00", "classification": "PREFORMAL_SUPERSEDED_SEMANTICS_MISMATCH", "issue": "Draft text said self-tuned while code used one global distance median.", "resolution": "Before real P0, froze per-node kth-neighbour local scaling; preserved superseded draft JSON.", "scientific_reuse": "NO"},
        {"revision": "E01", "classification": "TEST_CORRECTION", "issue": "First duplicate-observation test compared rows without swapping their self-loop columns.", "resolution": "Replaced it with an independent dense local-scale reference.", "scientific_reuse": "NO"},
        {"revision": "E02", "classification": "PREFORMAL_REPLAY_CORRECTION", "issue": "First replay padded eigenvalues using all contract profiles instead of the profiles present in the locked P0 artifact.", "resolution": "Replay now derives padding from locked candidate profiles; no producer formula changed.", "scientific_reuse": "NO"},
        {"revision": "E03", "classification": "PREFORMAL_CONTROL_FLOW_CORRECTION", "issue": "SPATIALLY_ANCHORED_ALTERNATING computed a valid forward operator then fell through to unknown-arm.", "resolution": "Corrected the control branch; marked c497 core P0 as superseded and reran 4/4 P0 with active addcdf core.", "scientific_reuse": "NO"},
        {"revision": "E04", "classification": "TRANSPORT_ASYNC_COMPLETION", "issue": "Local wrapper yielded while the long P22 Stage-A fresh replay remained active.", "resolution": "Monitored the same remote PID/artifact to completion; no candidate was regenerated.", "scientific_reuse": "LOCKED_ARTIFACT_REUSED_AFTER_EXACT_REPLAY"},
    ]
    write_csv(OUT / "failure_and_correction_ledger.csv", corrections)
    label_audit = {
        "schema": "night19b-label-flow-audit-v1", "producer_label_reads": 0,
        "carrier_annotation_arrays_accessed": 0, "candidate_artifacts_locked_before_any_evaluator": True,
        "public_labels_used_only_by_independent_evaluators": True,
        "selection_tier": "TRANSPARENT_PUBLIC_BENCHMARK_HPO", "lanes": label_lanes,
    }
    (OUT / "label_flow_audit.json").write_text(json.dumps(label_audit, indent=2, sort_keys=True) + "\n")

    collision_rows = [
        {"source": "Coifman and Lafon diffusion maps", "entry": "https://doi.org/10.1016/j.acha.2006.04.006", "prior_object": "Markov diffusion eigenvectors and multiscale geometry", "boundary": "Diffusion maps and spectral endpoint are scaffold"},
        {"source": "Nadler et al. spectral clustering", "entry": "https://proceedings.neurips.cc/paper/2005/file/2a0f97f81755e2878b264adf39cba68e-Paper.pdf", "prior_object": "Normalized graph eigenvectors for spectral clustering", "boundary": "Spectral clustering is not novel"},
        {"source": "Katz et al. alternating diffusion maps", "entry": "https://doi.org/10.1016/j.inffus.2018.01.007", "prior_object": "Alternating multimodal diffusion to retain common latent variability", "boundary": "RNA-ATAC alternation is not novel"},
        {"source": "Murphy and Maggioni spectral-spatial diffusion", "entry": "https://arxiv.org/abs/1902.05402", "prior_object": "Spatially regularized diffusion geometry for hyperspectral clustering", "boundary": "Adding a spatial operator to spectral diffusion is prior art"},
        {"source": "Seurat WNN", "entry": "https://satijalab.org/seurat/reference/findmultimodalneighbors", "prior_object": "Cell-specific multimodal neighbour weighting and joint graph", "boundary": "Multimodal graph fusion and local modality utility are prior art"},
        {"source": "SMART", "entry": "https://www.nature.com/articles/s41467-026-70821-5", "prior_object": "Spatial multi-omic graph neural integration and metric learning", "boundary": "Spatial multi-omics graph integration is mature scaffold"},
        {"source": "ARISE", "entry": "https://pubmed.ncbi.nlm.nih.gov/42366683/", "prior_object": "RNA-anchored shared-edge graph and hierarchical multimodal fusion", "boundary": "Graph intersection/anchoring/fusion are not novel"},
    ]
    write_csv(OUT / "source_collision_matrix.csv", collision_rows)

    method = """# Night-19B method and attribution contract\n\n`P_R` and `P_A` are mutual cosine-kNN Markov operators with per-node kth-neighbour local scales; `P_S` is one registered sparse spatial scale. The tested full operator forms `P_R P_S P_A` and `P_A P_S P_R`, top-k prunes and row-normalizes after every sparse multiplication, averages both orders, and attenuates off-diagonal propagation by a content-derived RNA/ATAC/spatial reliability. Rejected propagation mass returns to self before a symmetric nonnegative normalized spectral embedding and one common KMeans endpoint.\n\nAlternating diffusion, diffusion maps, spectral clustering, spatial-spectral diffusion, WNN and graph fusion are prior art. The only tested narrow object was the three-operator order plus content-derived conflict self-return. Stage A produced 0/3 independent method passes, so no novelty or method-success claim is retained. The permutation control uses independent SHA-256 node conjugation per modality: it preserves global row-degree/weight multisets and spectrum, not the same-ID row degree; this is an explicit residual attribution limitation.\n"""
    (OUT / "method_formula_and_attribution_contract.md").write_text(method, encoding="utf-8")
    risks = """# Night-19B reviewer risk register\n\n- The full operator lost substantially to simple averaging or spatial-only controls on every primary lane; no parameter refinement is scientifically authorized.\n- The permutation null preserves global graph statistics and spectrum but not per-ID degree, so it is not a perfect conditional randomization control. This limitation cannot rescue the already negative full-vs-simple controls.\n- The common KMeans spectral endpoint differs from Night-16H's 89-candidate selector. Absolute gaps to Night-16H mix representation/operator and consumer effects; the independent negative conclusion rests on exact matched Stage-A controls.\n- Conflict reliabilities were continuous and non-degenerate but mostly low, so repeated self-return may over-localize the cross path. This is a mechanism diagnosis, not authorization for post-result formula changes.\n- Public annotations were used for post-lock benchmark HPO/evaluation; results are neither blind nor SOTA evidence.\n"""
    (OUT / "reviewer_risk_register.md").write_text(risks, encoding="utf-8")

    stat = os.statvfs("/")
    disk = {
        "schema": "night19b-resource-disk-audit-v1", "root_available_bytes": stat.f_bavail * stat.f_frsize,
        "root_available_inodes": stat.f_favail,
        "working_bytes": int(subprocess.check_output(["du", "-sb", str(WORK)], text=True).split()[0]),
        "autodl_fs_data_new_files": 0, "new_environment": False, "downloads": 0, "gpu_seconds": 0.0,
    }
    (OUT / "resource_and_disk_audit.json").write_text(json.dumps(disk, indent=2, sort_keys=True) + "\n")
    test_log = WORK / "targeted_tests.log"
    text = test_log.read_text(encoding="utf-8")
    test_summary = {"schema": "night19b-targeted-tests-v1", "command": "pytest -q tests/test_night19b_csad.py", "passed": 7, "failed": 0, "status": "PASS", "log_sha256": sha256(test_log)}
    if "7 passed" not in text:
        raise RuntimeError("targeted test log does not prove 7 passes")
    shutil.copy2(test_log, OUT / "targeted_tests.log")
    (OUT / "targeted_test_summary.json").write_text(json.dumps(test_summary, indent=2, sort_keys=True) + "\n")

    best_full = {}
    for lane in LANES:
        rows = [row for row in metrics if row["lane"] == lane and row["arm"] == "CSAD_FULL"]
        best_full[lane] = max(rows, key=lambda row: (float(row["ari"]), float(row["nmi"])))
    decision = {
        "schema": "night19b-decision-v1", "classification": "SCIENTIFIC_NEGATIVE",
        "status": "NIGHT19B_CROSS_MODAL_SPATIAL_ALTERNATING_DIFFUSION_NO_INDEPENDENT_SIGNAL",
        "primary_lane_pass_count": 0, "family_stage_authorized": False, "trainable_unfolding_authorized": False,
        "reason": "CSAD_FULL lost to coordinate-wise strongest same-config controls in all three primary lanes and failed the third-lane safety condition.",
        "secondary_score_frontier": {
            "lane": "PLACENTA_K10",
            "classification": "MATCHED_CONTROL_BACKBONE_HEAD_SIGNAL_NOT_CSAD_CONTRIBUTION",
            "S01_CONCATENATED_FEATURE_KNN_ari_nmi": [0.4999993975293688, 0.6311801428733832],
            "S02_CONCATENATED_FEATURE_KNN_ari_nmi": [0.4671057793331236, 0.6357721471971621]
        },
        "producer_label_reads": 0, "shutdown_dispatched": False,
    }
    (OUT / "night19b_decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    lane_lines = []
    gate_by_lane = {row["lane"]: row for row in gate["lane_results"]}
    for lane in LANES:
        full = best_full[lane]; selected = gate_by_lane[lane]
        lane_lines.append(f"| {lane} | {float(full['ari']):.6f}/{float(full['nmi']):.6f} | {selected['coordinate_max_control_ari']:.6f}/{selected['coordinate_max_control_nmi']:.6f} | {selected['delta_ari']:+.6f}/{selected['delta_nmi']:+.6f} | {full['min_cluster_size']} |")
    report = f"""# Night-19B 跨模态空间交替扩散报告\n\n## 我现在需要知道的三件事\n\n1. **问题**：RNA、ATAC 与空间图按有序交替扩散并在冲突处回流 self，能否产生比普通扩散/图平均更可分的空间域？答案是否定的。\n2. **实际动作所在层**：本轮实现了真实三稀疏算子表示生成器，不是 selector 或关系蒸馏；所有稀疏乘积立即 top-k，最终用同一谱嵌入和 KMeans endpoint。\n3. **论文含义**：P0 工程 4/4 闭合，但 Stage A 独立方法门为 0/3；FULL 在三条主 lane 都明显输给同预算简单控制。因此分类 `SCIENTIFIC_NEGATIVE`，停止 refine、family seeds 与可训练展开。\n\n## 绝对指标与 matched attribution\n\n| lane | best FULL ARI/NMI | gate 坐标强控制 ARI/NMI | gate FULL 差值 | best FULL 最小簇 |\n|---|---:|---:|---:|---:|\n{os.linesep.join(lane_lines)}\n\nNight-16H strict LOSO 的历史强结果为 P22 `0.596390/0.718243`、MISAR `0.535306/0.658265`、human `0.596178/0.585490`。它依赖已锁 89-candidate selector，不是本轮 same-endpoint control；因此本轮绝对差距不能全归因于 CSAD。但 FULL 对本轮 exact matched controls 仍是 0/3，足以否定独立信号。\n\n## 机制诊断\n\n四条真实 P0 的 RNA/ATAC/retained shapes、ordered ID、三尺度稀疏图与 source SHA 均重新登记；active core `addcdf944...` 下 P0 4/4 和 Stage-A 4/4 fresh-process partition/eigenvalue replay exact。冲突可靠性连续且非退化，但 median 很低；self-return 使约三至五成传播质量留在本节点。结果表明该组合更像过度局部化，而不是稳定提取共同组织几何。\n\n## 标签、失败与资源\n\n四个 20-candidate artifact 先锁 SHA 并新进程重放，之后独立 evaluator 才打开公开 annotations；producer label read 为 0。最初 global-median/self-tuned 语义不一致、P0 replay padding 和 spatial-anchor 控制流错误都在 formal label evaluation 前修正并保留 superseded ledger。全程未下载数据、未新建环境、未向 `/autodl-fs/data` 写文件，working 远低于 150 MiB。\n\n## 导师汇报版\n\n这轮换了一个真正的新表示层对象，不再修 selector 或关系后验。我们分别构建 RNA、ATAC 和空间的稀疏扩散算子，并测试双向有序传播与冲突质量回流。四条真实路径、稀疏性、谱分解和新进程重放全部闭合。科学结果却很明确：FULL 在 P22、MISAR 和人海马均显著低于简单图平均、空间单轴或其他 matched control。冲突门没有提供独立增益，反而可能让传播过度局部化。因此本轮分类是 `SCIENTIFIC_NEGATIVE`，不追加 refine、多 seed 或神经展开。这个结果也排除了把普通 alternating diffusion 加空间门包装成论文核心的路线。\n\n## 技术附录\n\n- Taskbook SHA-256: `4083713ea11280a9cfeea2882380da4352f0d76687a4cdd3ae795fea2ebdf096`\n- Formula contract SHA-256: `{sha256(REPO / 'configs/night19b/formula_freeze.json')}`\n- Active core SHA-256: `{sha256(REPO / 'SpaLORA/night19b_csad.py')}`\n- Parent commit: `e58da0eb307235553fa8359f75e82b527cb735a3`\n- shutdown_dispatched at science seal: `false`\n"""
    secondary = """\n\n## 次级 score frontier（不归因于 CSAD）\n\nPlacenta K10 的 matched control `CONCATENATED_FEATURE_KNN` 在 S01 达到 `0.499999/0.631180`，S02 为 `0.467106/0.635772`，高于 Night-18D matched-control context 约 `0.370597/0.533451`。这是可复算的 **backbone/head control score-frontier**，不是 CSAD_FULL 的方法贡献。Human 的 simple average 最高 ARI 为 `0.373640`、spatial-only 最高 NMI 为 `0.404234`，仍明显低于 Night-16H 高位，故不登记 human frontier。"""
    report = report.replace("\n\nNight-16H strict LOSO", secondary + "\n\nNight-16H strict LOSO", 1)
    (OUT / "night19b_report.md").write_text(report, encoding="utf-8")

    entries = []
    index_path = OUT / "handoff_file_index.json"
    for path in sorted(OUT.iterdir()):
        if path.is_file() and path != index_path:
            entries.append({"name": path.name, "size": path.stat().st_size, "sha256": sha256(path)})
    index_path.write_text(json.dumps({"schema": "night19b-handoff-index-v1", "files": entries, "file_count": len(entries)}, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"classification": decision["classification"], "primary_lane_pass_count": 0, "handoff_files": len(entries) + 1, "output": str(OUT)}, indent=2))


if __name__ == "__main__":
    main()
