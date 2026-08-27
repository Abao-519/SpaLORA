"""Build Night-23C report and compact handoff tables from locked formal evidence."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path("/root/SpaLORA-night16h")
WORK = Path("/root/night23c_working")
FORMAL = WORK / "formal"
EVAL = WORK / "evaluation"
OUT = REPO / "outputs/night23c_handoff"
LANES = ["P22_K9", "MISAR_K7", "HUMAN_HIPPOCAMPUS_K7"]
FULL = "FULL_NESTED_CALIBRATED_TRISTATE_SIGNED"
CARRIER = "CARRIER_ONLY"


def sha(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()


def write_json(name: str, value: object) -> None:
    (OUT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True),encoding="utf-8")


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows: raise RuntimeError(f"empty table {name}")
    with (OUT/name).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True,exist_ok=True)
    decision=json.loads((EVAL/"stage_b_decision.json").read_text())
    if decision["primary_independent_passes"] != 0 or decision["placenta_confirmation_authorized"]:
        raise RuntimeError("formal gate inconsistent with negative handoff")
    absolute=[]; contribution=[]; relation=[]; calibration=[]; p0=[]; replays=[]
    artifacts=OUT/"artifacts"; artifacts.mkdir(exist_ok=True)
    formal_out=artifacts/"formal"; formal_out.mkdir(exist_ok=True)
    eval_out=artifacts/"evaluation"; eval_out.mkdir(exist_ok=True)
    wall=0.0; peak_rss=0.0
    for lane in LANES:
        rows=list(csv.DictReader((EVAL/f"{lane}.csv").open(encoding="utf-8")))
        absolute.extend(rows)
        by={r["candidate_id"]:r for r in rows}; full,carrier=by[FULL],by[CARRIER]
        controls=[r for k,r in by.items() if k!=FULL]
        best_ari=max(float(r["ari"]) for r in controls); best_nmi=max(float(r["nmi"]) for r in controls)
        contribution.append({"lane":lane,"full_ari":full["ari"],"full_nmi":full["nmi"],"carrier_ari":carrier["ari"],"carrier_nmi":carrier["nmi"],
                             "delta_ari_vs_carrier":float(full["ari"])-float(carrier["ari"]),"delta_nmi_vs_carrier":float(full["nmi"])-float(carrier["nmi"]),
                             "coordinate_best_control_ari":best_ari,"coordinate_best_control_nmi":best_nmi,
                             "delta_ari_vs_coordinate_best":float(full["ari"])-best_ari,"delta_nmi_vs_coordinate_best":float(full["nmi"])-best_nmi,
                             "independent_dual_win":False,"min_cluster_size":full["min_cluster_size_full"],"morans_i_macro":full["morans_i_macro"],
                             "gearys_c_macro":full["gearys_c_macro"],"neighbor_agreement":full["neighbor_agreement"],"partition_sha256":full["partition_sha256"]})
        rel=json.loads((EVAL/f"{lane}_relation.json").read_text()); mlp=[x for x in rel["metrics"] if x["model"]=="mlp_probability"][0]
        relation.append({"lane":lane,"mlp_auroc":mlp["auroc"],"mlp_auprc":mlp["auprc"],"mlp_brier":mlp["brier"],**rel["tri_state"]})
        manifest=json.loads((FORMAL/f"{lane}.json").read_text()); replay=json.loads((FORMAL/f"{lane}_replay.json").read_text())
        if replay["status"]!="PASS" or not replay["partition_bank_exact"]: raise RuntimeError(f"replay failed {lane}")
        replays.append(replay); wall+=manifest["wall_seconds"]; peak_rss=max(peak_rss,manifest["peak_rss_mb"])
        cal=manifest["calibration"]["primary"]
        calibration.append({"heldout_lane":lane,"source_lanes":"|".join(manifest["source_lanes"]),"method":"STUDYWISE_MIDRANK",
                            "lower":cal["lower"],"upper":cal["upper"],"worst_selective_utility":cal["worst_selective_utility"],
                            "mean_selective_utility":cal["mean_selective_utility"],"worst_purity":cal["worst_purity"],"worst_coverage":cal["worst_coverage"]})
        p0.append({"lane":lane,"feature_shape":json.dumps(manifest["feature_shape"]),"carrier_shape":json.dumps(manifest["carrier_shape"]),
                   "union_edges":manifest["union_edge_count"],"source_lanes":"|".join(manifest["source_lanes"]),"checkpoint_reload":manifest["same_process_checkpoint_reload"],
                   "fresh_process_replay":replay["status"],"exact_k_all_arms":all(r["exact_k"] for r in manifest["rows"]),
                   "mlp_parameter_l1_change":manifest["mlp_parameter_l1_change"],"train_device":manifest["train_device"],
                   "wall_seconds":manifest["wall_seconds"],"peak_rss_mb":manifest["peak_rss_mb"]})
        for path in FORMAL.glob(f"{lane}*"):
            if path.is_file(): shutil.copy2(path,formal_out/path.name)
        for path in EVAL.glob(f"{lane}*"):
            if path.is_file(): shutil.copy2(path,eval_out/path.name)
    write_csv("absolute_metrics_and_controls.csv",absolute); write_csv("method_contribution_board.csv",contribution)
    write_csv("relation_transfer_diagnostics.csv",relation); write_csv("nested_calibration_table.csv",calibration); write_csv("real_p0_registry.csv",p0)
    shutil.copy2(EVAL/"method_contribution_table.csv",OUT/"mechanical_gate_table.csv"); shutil.copy2(EVAL/"stage_b_decision.json",OUT/"stage_b_decision.json")
    write_json("decision.json",{"schema":"night23c-decision-v1","main_classification":"SCIENTIFIC_NEGATIVE",
               "narrow_classification":"NO_CALIBRATED_RELATION_BRIDGE_SIGNAL","night23b_erratum":"IMPLEMENTATION_FAILURE__DECISION_SEMANTICS_INVALID",
               "oracle_evidence":"ORACLE_CONSUMER_IDENTIFIABLE_2_OF_3","stage_b_primary_passes":0,"stage_b_primary_total":3,
               "placenta_confirmation_authorized":False,"gse205055_downloaded":False,"night23_mainline":"PERMANENTLY_CLOSED",
               "benchmark_labels_used_only_after_partition_and_replay_lock":True})
    write_json("fresh_process_replay_summary.json",{"schema":"night23c-replay-summary-v1","passed":3,"total":3,"records":replays})
    write_json("teacher_and_label_flow_audit.json",{"schema":"night23c-label-flow-v1","producer_benchmark_labels_read":0,
               "producer_heldout_teacher_files_read":0,"source_teacher_use":"loss, inner cross-fit calibration and thresholds on outer source studies only",
               "heldout_teacher_use":"relation diagnostic after all primary banks and fresh replays locked","benchmark_reference_use":"independent evaluation after locks",
               "outer_sources":{"P22_K9":["MISAR_K7","HUMAN_HIPPOCAMPUS_K7"],"MISAR_K7":["P22_K9","HUMAN_HIPPOCAMPUS_K7"],"HUMAN_HIPPOCAMPUS_K7":["P22_K9","MISAR_K7"]},
               "heldout_teacher_or_reference_in_predictor_calibration_threshold_consumer":False})
    write_csv("failure_and_correction_ledger.csv",[
        {"cycle":"NIGHT23B_READONLY_ERRATUM","issue":"worst-first selection followed by pass-count gate blocked valid scale2 2/3 oracle evidence","action":"preserved Night23B artifacts; reclassified decision layer only","labels_used":False,"status":"DECISION_SEMANTICS_INVALID"},
        {"cycle":"PREFORMAL_PLATT_PURITY","issue":"no threshold pair met per-source 0.80 state purity","action":"no partition produced; replaced with source-only selective-utility rule","labels_used":False,"status":"SUPERSEDED"},
        {"cycle":"PREFORMAL_PLATT_UTILITY","issue":"Platt produced negative worst-source utility and boundary collapse","action":"replaced by label-free per-study midrank; no heldout teacher/reference opened","labels_used":False,"status":"SUPERSEDED"},
        {"cycle":"FORMAL_MIDRANK","issue":"FULL failed 0/3 and was dominated by controls","action":"stopped placenta and permanently closed Night23 mainline","labels_used":True,"status":"SCIENTIFIC_NEGATIVE"},
    ])
    write_csv("source_collision_audit.csv",[
        {"object":"signed Laplacian/signed spectral clustering","prior_status":"mature","night23c_role":"fixed atomic consumer","claim":"none"},
        {"object":"must-link/cannot-link constrained clustering","prior_status":"mature","night23c_role":"semantic precedent","claim":"none"},
        {"object":"Platt/rank probability calibration","prior_status":"mature","night23c_role":"source-only calibration scaffold","claim":"none"},
        {"object":"boundary-affinity segmentation","prior_status":"mature","night23c_role":"negative-edge semantic precedent","claim":"none"},
        {"object":"cross-study invariant relation + nested source calibration + carrier-preserving tri-state consumer","prior_status":"joint object evaluated","night23c_role":"FULL","claim":"negative evidence; no method name frozen"},
    ])
    source_files=[REPO/"SpaLORA/night23b_signed_bridge.py",REPO/"SpaLORA/night23c_tristate_bridge.py",REPO/"configs/night23c/stage_b_contract.json",
                  REPO/"scripts/night23c/produce_loso_tristate_bridge.py",REPO/"scripts/night23c/replay_loso_tristate_bridge.py",
                  REPO/"scripts/night23c/evaluate_relation_after_lock.py",REPO/"scripts/night23c/summarize_stage_b_gate.py",
                  REPO/"scripts/night23c/build_handoff.py",REPO/"scripts/night23a/evaluate_stage_b_partitions.py",
                  REPO/"tests/test_night23b_signed_bridge.py",REPO/"tests/test_night23c_tristate_bridge.py"]
    write_json("source_authority_manifest.json",{"schema":"night23c-source-authority-v1","parent_commit":"8cd67b652df145d6f618dcb33ad01bcff18b6183",
               "parent_tag":"night23b-final-20260827","parent_compact_index_sha256":"551a7163646a4ec356f81e1da80ac57a552d89f9ad64e32d53c43c0aa7804645",
               "taskbook_sha256":"79276b7e0c69e2f08efbea2f38770f00e64f8819f0efe8296845d1792cbf0450",
               "formula_freeze_commit":"a976c722669bc6e621fb436664c5a9f34eaa3565","evaluator_freeze_commit":"7693b94beaa8304eae138db8440a18244e8226d2",
               "source_files":{str(p.relative_to(REPO)):sha(p) for p in source_files},
               "formal_banks":{lane:sha(FORMAL/f"{lane}_partitions.npz") for lane in LANES}})
    usage=shutil.disk_usage("/"); work_bytes=int(subprocess.check_output(["du","-sb",str(WORK)]).split()[0])
    write_json("resource_audit.json",{"schema":"night23c-resource-audit-v1","captured_utc":datetime.now(timezone.utc).isoformat(),
               "root_total_bytes":usage.total,"root_used_bytes":usage.used,"root_free_bytes":usage.free,"night23c_working_bytes":work_bytes,
               "formal_wall_seconds_sum":wall,"formal_peak_rss_mb_max":peak_rss,"gpu_used":True,"train_device":"cuda",
               "peak_gpu_memory_mb":"NOT_INSTRUMENTED__LIMITATION"})
    method=("# Night-23C 方法与选择合同\n\n- Night-23B 数值有效，唯一下游勘误是决策顺序；scale=2 固定。\n"
            "- 每个 outer fold 只用另外两个 primary studies 的 teacher relations；inner leave-one-study-out 产生跨研究 score。\n"
            "- MLP 为 primary，logistic 为解释性对照；calibration 是每研究 score midrank，阈值最大化最差 source selective utility。\n"
            "- WITHIN 吸引，BOUNDARY 排斥，FULL 的 UNKNOWN 权重严格为零；另有 retained-support unknown 原子臂。\n"
            "- carrier geometry 与 signed relation embedding 拼接，relation scale 固定 2，deterministic exact-K endpoint。\n"
            "- 所有 primary bank/checkpoint 先锁定并 fresh replay，之后 evaluator 才打开 heldout teacher/reference。\n"
            "- FULL 必须在至少 2/3 lane 严格双胜每个 matched control，macro 双增且安全门通过。实际 0/3，placenta 禁止。\n")
    (OUT/"method_semantics_and_selection_contract.md").write_text(method,encoding="utf-8")
    rows_md="\n".join(f"| {r['lane']} | {float(r['carrier_ari']):.6f}/{float(r['carrier_nmi']):.6f} | {float(r['full_ari']):.6f}/{float(r['full_nmi']):.6f} | {r['coordinate_best_control_ari']:.6f}/{r['coordinate_best_control_nmi']:.6f} | {r['delta_ari_vs_carrier']:+.6f}/{r['delta_nmi_vs_carrier']:+.6f} | {r['min_cluster_size']} | FAIL |" for r in contribution)
    report=f"""# Night-23C 结果报告\n\n## 我现在需要知道的三件事\n\n1. **问题**：Night-23B 错误地用 worst-first 规则阻断了 scale=2 的 2/3 oracle 可识别证据。本轮先勘误，再真实检验 source-only learned tri-state bridge。\n2. **实际层级**：共享 MLP 预测 held-out union edges；inner study-wise cross-fit 只用 source teacher 做 midrank calibration 和阈值；WITHIN 吸引、BOUNDARY 排斥、UNKNOWN 擦除，随后进入固定 scale=2 carrier-preserving signed consumer。\n3. **论文意义与分类**：oracle consumer 确实可识别 2/3，但 learned bridge 为 0/3，macro 相对 carrier 为 ΔARI {decision['study_balanced_delta_ari_vs_carrier']:+.6f}、ΔNMI {decision['study_balanced_delta_nmi_vs_carrier']:+.6f}。终态 **SCIENTIFIC_NEGATIVE / NO_CALIBRATED_RELATION_BRIDGE_SIGNAL**；Night-23 主线永久关闭。\n\n## Night-23B 只读勘误\n\n原候选、指标和 replay 保持有效；其 narrow decision 改判为 `IMPLEMENTATION_FAILURE / DECISION_SEMANTICS_INVALID`。scale=2 的 P22/human 通过冻结 oracle gate，证据为 `ORACLE_CONSUMER_IDENTIFIABLE_2_OF_3`。这只授权本轮 Stage B，不是方法成绩。\n\n## 绝对 benchmark 指标与匹配贡献\n\n| lane | carrier ARI/NMI | FULL ARI/NMI | coordinate-wise strongest control | FULL-carrier Δ | min cluster | independent gate |\n|---|---:|---:|---:|---:|---:|---|\n{rows_md}\n\nP22/MISAR 的最强 matched control 均是 `RETAINED_ONLY_UNSIGNED`；human 最强是 `MLP_DIRECT_NONNEGATIVE`。FULL 在三条 lane 都没有同时超过全部原子臂。human 虽高于弱 carrier endpoint，但远低于 MLP direct，不能算独立贡献。\n\n## Relation 与三态诊断\n\n- P22/MISAR held-out MLP AUROC 为 {relation[0]['mlp_auroc']:.3f}/{relation[1]['mlp_auroc']:.3f}，说明排序信号仍在；human 仅 {relation[2]['mlp_auroc']:.3f}。\n- source-only threshold 在 P22/MISAR 将约 96%/92% 边判为 BOUNDARY，但真实 boundary purity 仅约 0.199/0.264；负边被大规模误用，signed consumer 分区崩坏。\n- human 的 boundary purity 仅约 0.127；FULL 也被 MLP direct 原子臂显著支配。\n- UNKNOWN 在三条 lane 约 2%，且 FULL 中确实擦除，没有恢复为 raw-union 强连接。三态实现正确，但 calibration/transfer 科学对象失败。\n\n## 标签流与真实性\n\n三条 checkpoint、prediction 和 9-arm banks 全部先物化/hash，并完成 3/3 fresh-process exact replay。producer 没有打开 held-out teacher 或 benchmark annotation；held-out teacher 只用于锁定后的 relation AUROC/AUPRC，benchmark labels 只由独立 evaluator 读取。MLP 参数真实更新、checkpoint strict reload 通过、所有 arms exact-K。\n\n## 失败、限制与新颖性\n\nPlatt purity cycle 在 source-only 阶段无可行阈值；Platt utility cycle 发生 boundary collapse；两者均在任何 held-out teacher/reference 打开前被 supersede。正式 midrank 公式只跑一次完整 LOSO，未按分数修改。Signed clustering、约束聚类、概率校准和 boundary affinity 都是成熟先例；联合对象得到负证据，不冻结方法名。GPU 峰值显存没有在 producer 中登记，这是资源审计限制；CPU peak RSS 与 wall time完整保留。\n\n## 导师汇报版\n\nNight-23B 的数值没有错，错的是决策顺序：scale=2 实际能在 P22 和 human 两条 primary 上恢复 teacher，因此我们先发布只读勘误。Night-23C 随后真正执行了原本被阻断的 learned Stage B，每折只用另外两个研究的 teacher relations 训练和校准。工程路径、三态作用、checkpoint 与 fresh replay都成立。P22/MISAR 的 MLP 边排序仍有约 0.84 AUROC，但跨研究阈值把绝大多数边错判成 boundary，导致 signed partition 大幅下降。三条 primary 的 FULL 均未双胜全部 matched controls，宏观 ARI/NMI 也显著为负。human 仅相对弱 carrier KMeans endpoint 上升，但被 MLP direct 原子臂解释。按冻结门 placenta 没有运行，GSE205055 没有下载。结论是 oracle 边关系可以被消费者利用，但当前 learned relation-to-partition bridge不能跨研究校准；Night-23 路线到此永久结束。\n\n## 技术摘要\n\n- Formula freeze: `a976c722669bc6e621fb436664c5a9f34eaa3565`\n- Evaluator gate freeze: `7693b94beaa8304eae138db8440a18244e8226d2`\n- Formal replay: 3/3 exact\n- GPU: used; peak memory not instrumented\n- Max RSS: {peak_rss:.1f} MiB; formal producer wall sum: {wall:.1f} s\n"""
    (OUT/"report.md").write_text(report,encoding="utf-8")
    print(json.dumps({"output":str(OUT),"classification":"NO_CALIBRATED_RELATION_BRIDGE_SIGNAL","files":len(list(OUT.rglob('*')))},indent=2))


if __name__=="__main__": main()
