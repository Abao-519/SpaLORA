"""Build the compact Night-23B fail-closed Stage-A handoff from locked artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


REPO = Path("/root/SpaLORA-night16h")
FORMAL = Path("/root/night23b_working/stage_a_formal")
OUT = REPO / "outputs/night23b_handoff"
LANES = ["P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7", "MELANOMA_TUMOR_K2"]
PRIMARY = set(LANES[:3])


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty table: {name}")
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    artifact_dir = OUT / "artifacts/oracle_banks"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    decision = json.loads((FORMAL / "stage_a_decision.json").read_text())
    if decision["stage_b_authorized"] or decision["primary_identifiable_count"] != 1:
        raise RuntimeError("formal decision is inconsistent with the fail-closed handoff")
    selected = decision["selected_consumer"]
    all_rows: list[dict] = []
    main_rows: list[dict] = []
    connectivity_rows: list[dict] = []
    replays = []
    runtime = 0.0
    peak_rss = 0.0
    for lane in LANES:
        rows = list(csv.DictReader((FORMAL / f"{lane}.csv").open(encoding="utf-8")))
        all_rows.extend(rows)
        chosen = [r for r in rows if r["mode"] == selected["mode"] and float(r["relation_scale"]) == float(selected["relation_scale"])]
        carrier = [r for r in rows if r["mode"] == "CARRIER_ONLY"]
        if len(chosen) != 1 or len(carrier) != 1:
            raise RuntimeError(f"candidate authority mismatch: {lane}")
        summary = json.loads((FORMAL / f"{lane}.json").read_text())
        replay = json.loads((FORMAL / f"{lane}_replay.json").read_text())
        if not replay["ids_candidate_ids_partitions_exact"]:
            raise RuntimeError(f"replay failed: {lane}")
        replays.append(replay)
        for suffix in (".npz", ".json", "_replay.json"):
            shutil.copy2(FORMAL / f"{lane}{suffix}", artifact_dir / f"{lane}{suffix}")
        runtime += float(summary["wall_seconds"])
        peak_rss = max(peak_rss, float(summary["peak_rss_mb"]))
        c, x = carrier[0], chosen[0]
        gain = min(float(x["teacher_recovery_ari"]) - float(c["teacher_recovery_ari"]), float(x["teacher_recovery_nmi"]) - float(c["teacher_recovery_nmi"]))
        lane_pass = float(x["teacher_recovery_ari"]) >= .90 and float(x["teacher_recovery_nmi"]) >= .90 and gain >= .20
        main_rows.append({
            "lane": lane, "role": "PRIMARY" if lane in PRIMARY else "SECONDARY", "N": summary["n"],
            "K": summary["k"], "union_edges": summary["union_edge_count"],
            "carrier_recovery_ari": c["teacher_recovery_ari"], "carrier_recovery_nmi": c["teacher_recovery_nmi"],
            "oracle_consumer": x["mode"], "relation_scale": x["relation_scale"],
            "oracle_recovery_ari": x["teacher_recovery_ari"], "oracle_recovery_nmi": x["teacher_recovery_nmi"],
            "dual_gain_over_carrier": gain, "edge_disagreement": x["edge_disagreement"],
            "observed_k": x["observed_k"], "min_cluster_size": x["min_cluster_size"],
            "partition_sha256": x["partition_sha256"], "lane_identifiable": lane_pass,
            "wall_seconds": summary["wall_seconds"], "peak_rss_mb": summary["peak_rss_mb"], "gpu_used": False,
        })
        con = summary["connectivity"]
        connectivity_rows.append({
            "lane": lane, "N": summary["n"], "K": summary["k"], "positive_components": con["positive_component_count"],
            "isolated_nodes": con["isolated_node_count"], "cluster_split_counts_json": json.dumps(con["teacher_cluster_split_counts"], sort_keys=True),
            "cross_teacher_cluster_positive_component": con["cross_teacher_cluster_positive_component"],
            "same_edges": summary["teacher_same_edge_count"], "boundary_edges": summary["teacher_boundary_edge_count"],
            "registered_spatial_edges": summary["registered_spatial_edge_count"],
            "registered_spatial_same": summary["registered_spatial_same_count"],
            "registered_spatial_boundary": summary["registered_spatial_boundary_count"],
        })
    write_csv("oracle_consumer_identifiability_table.csv", main_rows)
    write_csv("oracle_all_candidates.csv", all_rows)
    write_csv("teacher_connectivity_and_coverage.csv", connectivity_rows)
    shutil.copy2(FORMAL / "oracle_consumer_summary.csv", OUT / "oracle_gate_summary.csv")
    shutil.copy2(FORMAL / "stage_a_decision.json", OUT / "stage_a_decision.json")
    write_json("fresh_process_replay_summary.json", {"schema": "night23b-replay-summary-v1", "passed": len(replays), "total": len(LANES), "records": replays})
    write_json("decision.json", {
        "schema": "night23b-decision-v1", "main_classification": "SCIENTIFIC_NEGATIVE",
        "narrow_classification": "RELATION_CONSUMER_NOT_IDENTIFIABLE", "stage_a_primary_passes": 1,
        "stage_a_primary_total": 3, "stage_b_authorized": False, "placenta_confirmation_authorized": False,
        "benchmark_reference_labels_read": 0, "night23_edge_distillation_mainline": "CLOSED",
        "reason": "Two of three primary studies failed the frozen oracle recovery-and-gain gate; a learned/calibrated bridge cannot repair a non-identifiable fixed consumer.",
    })
    write_json("teacher_and_label_flow_audit.json", {
        "schema": "night23b-label-flow-v1", "stage_executed": "STAGE_A_ORACLE_DIAGNOSTIC_ONLY",
        "producer_inputs": ["locked retained carrier", "Night-23A union edge features", "Night-16H teacher partition", "Night-23A teacher relation"],
        "teacher_use": "oracle consumer identifiability diagnostic only", "benchmark_reference_labels_read": 0,
        "benchmark_metrics_computed": False, "stage_b_predictor_trained": False, "placenta_opened": False,
        "selection_inputs": "teacher recovery ARI/NMI under a pre-frozen diagnostic gate; not benchmark labels",
    })
    collision_rows = [
        {"object": "signed spectral clustering / signed Laplacian", "primary_authority": "Kunegis et al., WWW 2010, DOI 10.1145/1772690.1772769", "status": "MATURE PRIOR ART", "night23b_use": "atomic oracle consumer scaffold only", "novelty_claim": "none"},
        {"object": "must-link / cannot-link constrained clustering", "primary_authority": "Wagstaff et al., ICML 2001, Constrained K-means Clustering with Background Knowledge", "status": "MATURE PRIOR ART", "night23b_use": "alternative consumer considered but not implemented", "novelty_claim": "none"},
        {"object": "probability calibration and risk thresholding", "primary_authority": "Niculescu-Mizil and Caruana, ICML 2005, Predicting Good Probabilities with Supervised Learning", "status": "MATURE PRIOR ART", "night23b_use": "conditional Stage B concept, not executed", "novelty_claim": "none"},
        {"object": "boundary-affinity segmentation", "primary_authority": "Boykov and Jolly, ICCV 2001, DOI 10.1109/ICCV.2001.937505", "status": "MATURE PRIOR ART", "night23b_use": "negative-edge semantics only", "novelty_claim": "none"},
        {"object": "cross-study invariant teacher relation + nested source calibration + carrier-preserving tri-state sparse consumer", "primary_authority": "Night-23B joint protocol", "status": "JOINT OBJECT NOT TESTED", "night23b_use": "Stage B forbidden by Stage A", "novelty_claim": "no positive claim"},
    ]
    write_csv("source_collision_audit.csv", collision_rows)
    write_csv("failure_and_correction_ledger.csv", [
        {"cycle": "PREFORMAL_AUTHORITY_FIX", "issue": "Night-23A relation NPZ contains relation only", "action": "added explicit Night-16H teacher partition input and file/array/candidate SHA checks", "scientific_results_used": False, "status": "CORRECTED_BEFORE_FREEZE"},
        {"cycle": "PREFORMAL_REAL_SMOKE", "issue": "single P22 real route used to verify authority and replay before formula freeze", "action": "preserved under /root/night23b_working/p0_smoke and fully regenerated in formal output after freeze", "scientific_results_used": False, "status": "SUPERSEDED_BY_FORMAL"},
        {"cycle": "FORMAL_SUMMARY_CLI", "issue": "first summary call used ambiguous output option names", "action": "reran summary only with explicit --output-table/--output-decision; producer banks unchanged", "scientific_results_used": True, "status": "ENGINEERING_INVOCATION_CORRECTED"},
        {"cycle": "STAGE_A_GATE", "issue": "oracle consumer gate passed only 1/3 primary studies", "action": "stopped Stage B and placenta exactly as preregistered", "scientific_results_used": True, "status": "SCIENTIFIC_NEGATIVE"},
    ])
    contract_text = """# Night-23B method semantics and selection contract\n\n- Stage A is an oracle diagnostic, not method evidence. It uses no benchmark reference annotation.\n- The fixed consumer concatenates a robust retained-carrier geometry block with a sparse signed-relation eigen-embedding, followed by deterministic farthest-first exact-K KMeans.\n- Positive teacher edges enter as attraction; teacher boundary edges enter with negative sign; absent edges remain absent.\n- The consumer and scale grid were frozen at commit `005e3427ea28e0102668ef4feaeac6d98b271d59` before formal oracle banks.\n- The selected diagnostic is the primary-study worst-recovery leader, with predeclared tie breaks.\n- Per-lane identifiability requires ARI>=0.90, NMI>=0.90, and minimum dual gain over carrier>=0.20. At least 2/3 primary lanes were required.\n- Observed result: 1/3. Stage B, benchmark evaluation, and placenta confirmation are forbidden.\n- Therefore no calibrated bridge, tri-state predictor, or method contribution is claimed.\n"""
    (OUT / "method_semantics_and_selection_contract.md").write_text(contract_text, encoding="utf-8")
    source_files = [REPO / "SpaLORA/night23b_signed_bridge.py", REPO / "configs/night23b/stage_a_oracle_contract.json",
                    REPO / "scripts/night23b/run_oracle_consumer_audit.py", REPO / "scripts/night23b/replay_oracle_consumer_audit.py",
                    REPO / "scripts/night23b/summarize_oracle_gate.py", REPO / "scripts/night23b/build_handoff.py",
                    REPO / "tests/test_night23b_signed_bridge.py"]
    write_json("source_authority_manifest.json", {
        "schema": "night23b-source-authority-v1", "parent_commit": "7ca7fa2907354e0e1c6856fcf11984ecb89caf67",
        "parent_tag": "night23a-final-20260827", "parent_compact_index_sha256": "8ef3b131047aa5a4c6b5b502478ede7e6dc09024fca5801870e15034302169eb",
        "taskbook_sha256": "00be7e3d5f3469bcc7f255c6f5f64d65fd1024fe6ddcae7838eceb12c65e1d1e",
        "stage_a_freeze_commit": "005e3427ea28e0102668ef4feaeac6d98b271d59", "source_files": {str(p.relative_to(REPO)): sha(p) for p in source_files},
        "formal_banks": {lane: {"bank_sha256": sha(FORMAL / f"{lane}.npz"), "audit_sha256": sha(FORMAL / f"{lane}.json")} for lane in LANES},
    })
    root_usage = shutil.disk_usage("/")
    repo_bytes = int(subprocess.check_output(["du", "-sb", str(REPO)]).split()[0])
    work_bytes = int(subprocess.check_output(["du", "-sb", "/root/night23b_working"]).split()[0])
    write_json("resource_audit.json", {
        "schema": "night23b-resource-audit-v1", "captured_utc": datetime.now(timezone.utc).isoformat(),
        "root_total_bytes": root_usage.total, "root_used_bytes": root_usage.used, "root_free_bytes": root_usage.free,
        "repo_bytes": repo_bytes, "night23b_working_bytes": work_bytes, "formal_wall_seconds_sum": runtime,
        "formal_peak_rss_mb_max": peak_rss, "gpu_used": False,
    })
    report_rows = "\n".join(
        f"| {r['lane']} | {r['N']} | {r['K']} | {float(r['carrier_recovery_ari']):.6f}/{float(r['carrier_recovery_nmi']):.6f} | {float(r['oracle_recovery_ari']):.6f}/{float(r['oracle_recovery_nmi']):.6f} | {float(r['dual_gain_over_carrier']):+.6f} | {r['min_cluster_size']} | {'PASS' if r['lane_identifiable'] else 'FAIL'} |"
        for r in main_rows)
    report = f"""# Night-23B 结果报告\n\n## 我现在需要知道的三件事\n\n1. **问题**：Night-23A 已能跨研究预测部分边关系，但这些关系没有变成好分区。本轮先问一个更基础的问题：即使把近乎完美的 teacher 同域/边界关系直接交给固定消费者，它能否恢复 teacher 分区。\n2. **实际动作所在层**：实现的是保留 retained carrier 节点几何的稀疏 signed consumer：carrier emission 与正负关系谱坐标共同进入同一个 exact-K endpoint。它不是候选 selector，也没有训练新的 predictor。\n3. **论文意义与分类**：严格 oracle gate 仅 1/3 primary 通过，所以问题首先卡在固定消费者可识别性，而非仅是 calibration。终态为 **SCIENTIFIC_NEGATIVE / RELATION_CONSUMER_NOT_IDENTIFIABLE**；Stage B 与 placenta 均未授权，Night-23 edge-distillation 主线到此关闭。\n\n## Oracle 消费者上限（诊断，不是方法成绩）\n\n下表 ARI/NMI 是输出分区相对 teacher partition 的 recovery，不是相对生物学 benchmark annotation。整个 Night-23B 没有打开 benchmark labels。\n\n| lane | N | K | carrier recovery ARI/NMI | signed oracle recovery ARI/NMI | strict dual gain | min cluster | gate |\n|---|---:|---:|---:|---:|---:|---:|---|\n{report_rows}\n\n统一选择的原子消费者为 `ORACLE_SIGNED_RELATION`, relation scale=1。P22 几乎精确恢复；MISAR 的恢复仍只有 0.755/0.834 且 dual gain 0.153；human 的 ARI 高但 NMI 0.884 未达冻结门。primary 严格通过数为 1/3。melanoma 是 secondary，虽达到 1.0/1.0，不参与主门。\n\n## 根因归因\n\n- **coverage**：Night-23A union edge 上同时存在 teacher 同域和边界边，P22/MISAR/melanoma 的正边连通分量恰等于 K；human 有 12 个正连通分量、3 个孤立点。\n- **consumer**：signed relation 显著优于 carrier-only，但统一固定 consumer 仍不能在至少 2/3 primary 同时满足高恢复与足够增益；因此消费者上限本身不稳健。\n- **predictor/calibration**：本轮没有进入 Stage B，不能新增归因；Night-23A 的 predictor 排序信号不能补偿这个 oracle consumer 缺口。\n- **三态机制**：正边吸引、负边排斥、缺边擦除的数值语义已由性质测试闭合，但没有成为方法证据，因为 oracle gate 先失败。\n\n## 贡献边界与新颖性\n\nSigned Laplacian、must/cannot-link、概率校准和 boundary affinity 都是成熟原子先例。本来只有“跨研究置换不变 relation + nested source calibration + carrier-preserving tri-state sparse consumer”的联合对象可能检验；由于 Stage B 被禁止，这个联合对象没有获得正面证据，也不冻结论文名称。\n\n## 失败与限制\n\n- 当前 fixed consumer 只探索了预冻结的小型 sparse signed-embedding family；科学结论是该消费者族不可识别，不等同于数学上所有 possible signed partition objective 均不可能。\n- Stage A 使用 teacher 作为 oracle 诊断，因此任何 recovery 高分都不能写入方法主表。\n- 没有 benchmark ARI/NMI、AMI/FMI、Moran/Geary；这是 label firewall 的主动结果，不是漏报。edge disagreement 是对 teacher relation 的稀疏结构指标。\n- 未运行 GPU、MLP、LOSO Stage B 或 placenta。\n\n## 导师汇报版\n\nNight-23A 的边分类器在部分数据上 AUROC 不低，但分区失败的根因一直不清楚。Night-23B 先把预测误差拿掉，直接用 teacher 同域/边界边测试一个保留强分子表示的 signed 稀疏消费者。P22 和 melanoma 能近乎完整恢复 teacher，说明实现和关系对象并非完全无效；然而 MISAR 只有 0.755/0.834，human 的 NMI 也只有 0.884，严格门只有 1/3。按照预注册规则，我们没有继续训练校准桥，也没有读取 benchmark labels 或跑 placenta。这说明当前瓶颈不只是边预测精度，而是“局部边关系如何唯一确定全局 K 分区”的消费者可识别性。Signed clustering、约束聚类和概率校准都有充分先例，因此本轮不做新颖性包装。项目层面应关闭 Night-23 edge-distillation 主线，而不是继续调整阈值。\n\n## 技术审计\n\n- Stage-A freeze commit: `005e3427ea28e0102668ef4feaeac6d98b271d59`\n- 四条 formal bank 均 fresh-process exact replay。\n- Targeted tests 由最终封口时的真实 pytest 输出登记。\n- GPU: 未使用；formal producer peak RSS 最大约 {peak_rss:.1f} MiB。\n- `shutdown_dispatched` 在 sealed science handoff 中记录封口时事实；最终远端命令另行本地登记。\n"""
    (OUT / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"output": str(OUT), "files": len(list(OUT.iterdir())), "classification": "RELATION_CONSUMER_NOT_IDENTIFIABLE"}, indent=2))


if __name__ == "__main__":
    main()
