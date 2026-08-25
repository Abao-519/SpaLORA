#!/usr/bin/env python3
"""Build the sealed Night-17G scientific-negative handoff from locked artifacts."""
from __future__ import annotations
import csv,hashlib,json,os,shutil,subprocess,time
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path("/root/SpaLORA-night16h")
WORK=Path("/root/night17g_working")
OUT=ROOT/"outputs/night17g_handoff"
LANES=("P22_K9","MISAR_K7","HUMAN_HIPPOCAMPUS_K7")
N16H={"P22_K9":(0.5875325536381577,0.7089736211480059),"MISAR_K7":(0.5346235433360444,0.6567718908460013),"HUMAN_HIPPOCAMPUS_K7":(0.5961783620969016,0.5854895180308324)}

def sha(path:Path)->str:
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def write_json(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False)+"\n",encoding="utf-8")

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 frames=[]
 for lane in LANES: frames.append(pd.read_csv(WORK/f"stage_a/{lane}/evaluation.csv"))
 ledger=pd.concat(frames,ignore_index=True); ledger.to_csv(OUT/"all_stage_a_candidate_ledger.csv",index=False)
 contribution=[]; main_rows=[]
 for lane in LANES:
  d=ledger[ledger.lane==lane].copy(); full=d[d.arm=="FULL_CSBO"].sort_values(["absolute_ari","absolute_nmi"],ascending=False).iloc[0]
  cfg=full.config_id; same=d[d.config_id==cfg]; backbone=same[same.arm=="BACKBONE_NO_CSBO"].iloc[0]
  controls=same[same.arm.isin(["PERMUTED_EDGE_STATES","UNSIGNED_ONLY","BOUNDARY_TO_ABSTAIN","CONFLICT_AS_POSITIVE"])]
  control_double_wins=int(np.sum((full.absolute_ari>controls.absolute_ari)&(full.absolute_nmi>controls.absolute_nmi)))
  control_gate=control_double_wins==len(controls)
  backbone_gate=bool(full.absolute_ari>backbone.absolute_ari and full.absolute_nmi>backbone.absolute_nmi)
  contribution.append({"lane":lane,"selected_full_config":cfg,"full_ari":full.absolute_ari,"full_nmi":full.absolute_nmi,
    "matched_backbone_ari":backbone.absolute_ari,"matched_backbone_nmi":backbone.absolute_nmi,"delta_vs_backbone_ari":full.absolute_ari-backbone.absolute_ari,
    "delta_vs_backbone_nmi":full.absolute_nmi-backbone.absolute_nmi,"double_better_than_backbone":backbone_gate,"matched_control_double_wins":control_double_wins,
    "matched_control_count":len(controls),"independent_control_gate":control_gate,"lane_gate_pass":bool(backbone_gate and control_gate),
    "night16h_strong_ari":N16H[lane][0],"night16h_strong_nmi":N16H[lane][1],"delta_vs_night16h_ari":full.absolute_ari-N16H[lane][0],"delta_vs_night16h_nmi":full.absolute_nmi-N16H[lane][1]})
  selected=[d[d.candidate_id=="INPUT_STRONG_START"].iloc[0],d[d.candidate_id=="FROZEN_RETAINED_SAME_HEAD"].iloc[0],backbone,full]
  strongest_control=controls.sort_values(["absolute_ari","absolute_nmi"],ascending=False).iloc[0]; selected.append(strongest_control)
  labels=["NIGHT16H_INPUT_PARTITION_AUTHORITY","FROZEN_RETAINED_KMEANS_ENDPOINT","MATCHED_BACKBONE_NO_CSBO","FULL_CSBO_PUBLIC_DEVELOPMENT_BEST","STRONGEST_MATCHED_CONTROL_BY_ARI"]
  for label,row in zip(labels,selected):
   main_rows.append({"lane":lane,"profile":label,"candidate_id":row.candidate_id,"config_id":row.config_id,"arm":row.arm,"absolute_ari":row.absolute_ari,"absolute_nmi":row.absolute_nmi,
    "ami":row.ami,"fmi":row.fmi,"homogeneity":row.homogeneity,"v_measure":row.v_measure,"morans_i_macro":row.morans_i_macro,"gearys_c_macro":row.gearys_c_macro,"neighbor_agreement":row.neighbor_agreement,
    "n_total":row.n_total,"n_evaluated":row.n_evaluated,"k":row.k,"min_cluster_size":row.min_cluster_size,"cluster_sizes":row.cluster_sizes,"changed_from_strong_start":row.changed_from_strong_start,
    "partition_sha256":row.partition_sha256,"representation_sha256":row.representation_sha256,"wall_seconds":row.wall_seconds,"peak_gpu_mib":row.peak_gpu_mib,"peak_rss_mib":row.peak_rss_mib})
 pd.DataFrame(contribution).to_csv(OUT/"matched_contribution_and_gate.csv",index=False)
 pd.DataFrame(main_rows).to_csv(OUT/"absolute_metrics_main_table.csv",index=False)
 gate_pass=sum(x["lane_gate_pass"] for x in contribution)
 p0=[]; replay=[]
 for lane in LANES:
  m=json.loads((WORK/f"stage_a/{lane}/producer.json").read_text()); p0.append({"lane":lane,"n":m["n"],"k":m["k"],**m["shapes"],"candidate_count":m["candidate_count"],"wall_seconds":m["wall_seconds"],"peak_gpu_mib":m["peak_gpu_mib"],"peak_rss_mib":m["peak_rss_mib"],"edge_state":m["edge_state"],"authority":m["authority"]})
  for name in ("replay1.json","replay2.json"):
   r=json.loads((WORK/f"stage_a/{lane}/{name}").read_text()); replay.append({"lane":lane,"replay":name,"count":r["count"],"all_exact":r["all_exact"]})
 write_json(OUT/"p0_real_path_registry.json",p0); write_json(OUT/"exact_replay_audit.json",{"rows":replay,"all_exact":all(x["all_exact"] for x in replay)})
 write_json(OUT/"label_flow_audit.json",{"producer_label_reads":0,"evaluator_label_reads":3,"candidate_checkpoints_and_partitions_locked_before_evaluation":True,"public_annotations_used_for_post_lock_benchmark_evaluation_and_profile_selection":True,"within_run_checkpoint_selection_used_labels":False})
 collision=[
  ["CoMo","https://github.com/Lab-Xu/CoMo","70965cba0ca7df33e426f6a1dd5626347978cdae","NO_LICENSE_FILE_DETECTED","CoMo/model.py; CoMo/contrast.py; CoMo/CoMo_pyG.py","cross-attention GAE; reconstruction; neighbor and cluster contrastive","No explicit dual-modal consensus-boundary versus conflict-abstention edge simplex"],
  ["SpaAlign","https://github.com/VitaIntelli-CQU/SpaAlign","0513e78319b8d99a227d32a755edadbac8903c62","NO_LICENSE_FILE_DETECTED","SpaAlign/model.py; nets.py; contrastive_loss.py","per-view MLP AE; cross-view interactions; DEC refinement","Uses spatial attraction and cross-view contrast, not registered-edge three-state signed semantics"],
  ["ARISE","https://github.com/XiangxiangWang-code/ARISE","fefdd849494c0d08e755052a7a31b20169945e40","NO_LICENSE_FILE_DETECTED","ARISE/model.py; train.py; process.py","RNA-anchored graph intersection; hierarchical GCN fusion; neighbor/non-neighbor contrast","Negative set is generic non-neighbor complement; no conflict abstention"],
  ["CRCT","https://doi.org/10.1609/aaai.v40i9.37646","AAAI-26 DOI 10.1609/aaai.v40i9.37646","NO_OFFICIAL_CODE_LOCATED","AAAI paper Methods","adaptive graph; prototype refinement; ISDA rare-cluster augmentation","No explicit three-state dual-modal registered-edge objective in paper"],
  ["PRAGA","https://github.com/Xubin-s-Lab/PRAGA","4adb11c96fc7ddad800fa1787eadcc8b91b42784","AGPL-3.0","PRAGA/model.py; Train_model.py","learnable graph; reconstruction; dynamic prototype/BGMM contrast","Prototype/dynamic graph are prior art; no same-boundary/conflict split"],
  ["stDGCC","https://github.com/TimE9527/stDGCC","1c9de6cd8da076080926768907c93c73ee2c0da8","NO_LICENSE_FILE_DETECTED","stDGCC/model.py; run.py","single-modality positive/negative graph DGI and reconstruction","Positive/negative graph prior art, but negatives are corruptions rather than cross-modal consensus boundaries"],
 ]
 with (OUT/"source_code_collision_matrix.csv").open("w",newline="",encoding="utf-8") as f: w=csv.writer(f,lineterminator="\n"); w.writerow(["method","official_source","commit_or_version","license","source_paths_read","prior_art_scaffold","difference_or_collision_boundary"]); w.writerows(collision)
 (OUT/"source_code_collision_and_transfer_audit.md").write_text("""# Night-17G source collision audit\n\nWe inspected model/loss/graph source, not README alone. Autoencoding, cross-attention, DEC/prototypes, learned graphs, generic positive/negative contrast, RNA-anchored graph intersection and spatial attraction are mature prior art. The only tested Night-17G object is the threshold-free three-state edge simplex from two modality-specific local similarity ranks: shared similarity attracts, shared dissimilarity repels, and discordance abstains. ARISE is the closest source-level collision because it combines neighbor attraction with non-neighbor separation, but its negative set is the generic spatial complement and it does not distinguish a dual-modal consensus boundary from modality conflict. This P0 does not establish novelty: independent numerical contribution failed, so the combination must not be promoted as a method claim. Repositories without an explicit license were read only; no third-party code was copied. PRAGA is AGPL-3.0 and was used only for semantic comparison.\n""",encoding="utf-8")
 configs={"formula":"s1,s2 are per-lane empirical-CDF midranks of dual-modal cosine similarity on registered graph0 edges; a=s1*s2; b=(1-s1)*(1-s2); c=s1*(1-s2)+(1-s1)*s2; a+b+c=1; a attracts, b repels by normalized cosine-margin loss, c abstains","configs":[x["config"] for x in json.loads((WORK/"stage_a/P22_K9/producer.json").read_text())["rows"] if x["arm"]=="FULL_CSBO"],"arms":["BACKBONE_NO_CSBO","FULL_CSBO","PERMUTED_EDGE_STATES","UNSIGNED_ONLY","BOUNDARY_TO_ABSTAIN","CONFLICT_AS_POSITIVE"],"endpoint_alias":{"artifact_name":"FROZEN_RETAINED_SAME_HEAD","display_name":"FROZEN_RETAINED_KMEANS_ENDPOINT","reason":"This endpoint is prototype-initialized KMeans, not the Night-16H alpha-expansion graph-cut head."},"permuted_control":{"base_weighted_state_masses_matched":True,"loss_mass_normalization":"Attraction and boundary losses are separately divided by their active base-weighted mass, removing global channel scale. It does not remove location-dependent weighting; the permuted control tests that remaining location effect."}}
 write_json(OUT/"formula_config_and_ablation_contract.json",configs)
 corrections=[
  {"id":"E01","status":"SUPERSEDED_PREFORMAL","issue":"Ordinal rank split exact similarity ties by sparse edge order.","correction":"Replaced by deterministic midrank before formal rerun; preserved old artifacts under superseded_ordinal_rank_preformal."},
  {"id":"E02","status":"CORRECTED_BEFORE_FORMAL","issue":"np.isclose identity comparison and attraction-only mass audit.","correction":"Used boolean negation and audited attraction/boundary/conflict base-weighted mass."},
  {"id":"E03","status":"CORRECTED_BEFORE_FORMAL","issue":"Permuted states did not preserve base-weighted channel capacity.","correction":"Deterministic within-weight-stratum permutation plus exact per-channel base-weighted mass matching."},
  {"id":"E04","status":"DISPLAY_ALIAS_ONLY","issue":"FROZEN_RETAINED_SAME_HEAD could be misread as the Night-16H energy head.","correction":"Report alias is FROZEN_RETAINED_KMEANS_ENDPOINT; artifacts remain immutable."},
 ]
 pd.DataFrame(corrections).to_csv(OUT/"failure_and_correction_ledger.csv",index=False)
 write_json(OUT/"source_clone_cleanup_audit.json",{"scope":"/root/night17g_working/source_audit only","before_bytes_approx":743440384,"after_bytes_observed":30408704,"recorded_commits_before_cleanup":True,"removed_task_clone_payloads":["SpaAlign/data","ARISE/ARISE/data","stDGCC/model","stDGCC/embedding","stDGCC/generated_data","all five task-clone .git/objects"],"historical_repo_or_raw_touched":False,"reason":"system disk protection; retain only audited source text and metadata"})
 stat=os.statvfs("/"); data_stat=os.statvfs("/autodl-fs/data")
 write_json(OUT/"resource_and_disk_audit.json",{"timestamp_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"root_available_bytes":stat.f_bavail*stat.f_frsize,"root_available_inodes":stat.f_favail,"persistent_available_bytes":data_stat.f_bavail*data_stat.f_frsize,"persistent_available_inodes":data_stat.f_favail,"night17g_working_bytes":sum(p.stat().st_size for p in WORK.rglob("*") if p.is_file()),"new_environment_created":False,"new_data_downloaded":False,"shutdown_dispatched":False})
 decision={"schema":"night17g-decision-v1","classification":"SCIENTIFIC_NEGATIVE","status":"NIGHT17G_CSBO_STAGE_A_SCIENTIFIC_NEGATIVE","stage_a_lane_gate_pass_count":gate_pass,"stage_a_lane_gate_total":3,"multi_seed_authorized":False,"confirmation_run":False,"scientific_decision":"STOP_CSBO_AND_DO_NOT_EXPAND_GRID","secondary_observation":"Human hippocampus C01 full improved over matched backbone but did not simultaneously beat permuted/unsigned controls.","relation_distillation_chain_reopened":False,"shutdown_dispatched":False}
 write_json(OUT/"night17g_decision.json",decision)
 report=f"""# Night-17G report: Cross-modal Signed Boundary Objective\n\n## 我现在需要知道的三件事\n\n1. **问题**：我们检验了真正可训练的统一 RNA+ATAC 深度聚类核心，而不是继续做 selector 或候选共识。核心把空间边分为共同同域吸引、共同边界排斥和模态冲突弃权。\n2. **实际动作所在层**：两模态 adapter、低秩逐元素交互、重构/DEC scaffold 和 retained residual 都真实训练；三态 CSBO 直接进入 embedding loss。三条真实 lane 均完成梯度、参数更新、严格 checkpoint reload 和两次新进程精确重放。\n3. **论文意义与分类**：结果为 **SCIENTIFIC_NEGATIVE**。full 只在人海马相对 matched backbone 双升，但没有同时胜过 permuted/unsigned 控制；P22 与 MISAR 均下降。预注册门为 0/3，因此不进入多 seed，也不把三态组合写成论文贡献。\n\n## 绝对指标主结论\n\n| 数据 | Night-16H 输入 authority ARI/NMI | 本轮 matched backbone | 本轮 full CSBO | full - backbone | 最强匹配控制 | 门 |\n|---|---:|---:|---:|---:|---:|---:|\n"""
 for x in contribution:
  d=ledger[(ledger.lane==x["lane"]) & (ledger.config_id==x["selected_full_config"]) & ledger.arm.isin(["PERMUTED_EDGE_STATES","UNSIGNED_ONLY","BOUNDARY_TO_ABSTAIN","CONFLICT_AS_POSITIVE"])].sort_values(["absolute_ari","absolute_nmi"],ascending=False).iloc[0]
  report+=f"| {x['lane']} | {x['night16h_strong_ari']:.6f}/{x['night16h_strong_nmi']:.6f} | {x['matched_backbone_ari']:.6f}/{x['matched_backbone_nmi']:.6f} | {x['full_ari']:.6f}/{x['full_nmi']:.6f} | {x['delta_vs_backbone_ari']:+.6f}/{x['delta_vs_backbone_nmi']:+.6f} | {d.absolute_ari:.6f}/{d.absolute_nmi:.6f} ({d.arm}) | FAIL |\n"
 report+="""\n`INPUT_STRONG_START` 是 Night-16H 已锁定 partition authority；`FROZEN_RETAINED_KMEANS_ENDPOINT` 是本轮在 frozen retained 上采用相同 prototype-initialized KMeans 端点的显示别名。两者的差距同时包含端点变化，不能归因于表示。CSBO 的独立判断只使用 full 与同一配置、同一端点、同一训练预算的 backbone/结构控制。\n\n## 机制归因\n\n边状态本身数值可识别：三条 lane 的冲突质量均约 0.43--0.47，吸引与边界各约 0.26--0.28，单纯形误差小于 5e-8。失败不是全零状态或实现未训练。P22 和 MISAR 中，unsigned 或 boundary-to-abstain 比 full 更好，说明显式共同边界排斥没有带来独立收益；人海马的 full 相对 backbone 有局部增益，但 NMI 被 permuted/unsigned 控制超过，关系位置特异性不足。\n\nPermuted 控制逐通道匹配 base-weighted 质量。训练中 attraction 与 boundary loss 各自再除以有效质量，因此移除了全局通道强度；仍保留并检验的是边位置对应的相对权重。\n\n## 最重要失败与限制\n\n- 三态由输入模态的局部相似秩构造，不使用标签，但“共同低相似”并不可靠等价于真实域边界。\n- 可训练 residual 的绝对结果远低于 Night-16H 强 partition；安全 anchor 未能同时保留强起点与获得新边界信息。\n- retained representation 和输入 authority 来自历史公开 benchmark 开发链，本轮是 transparent public benchmark development，不是盲测。\n- 本轮只跑 seed 0，因为预注册 Stage-A 门失败后明确禁止多 seed。\n\n## 导师汇报版\n\n我们这轮第一次把跨模态三态边直接放进可训练表示，而不是继续修聚类后端。三条真实 RNA+ATAC 数据都完成了真实梯度、参数更新、checkpoint 回放和独立标签评价。三态权重不是数值退化：吸引、边界和冲突都有足够质量。但 full CSBO 只在人海马相对普通 backbone 上涨，且未同时胜过置换和 unsigned 对照；P22、MISAR 都更差。按预注册规则，方法门是 0/3，因此结论是科学负结果，不补 seed、不扩参数网格。这个结果说明“共同低相似边直接作为排斥”过于粗糙，不能作为论文核心。成熟 backbone 零件和正负图已有充分先例，本轮组合也没有获得独立数值支持。下一方向应换信息来源或更强原生表示，而不是继续修同一三态损失。\n\n## 技术审计\n\n- targeted tests: 7/7 PASS；包含 midrank tie、三态闭合、edge-order invariance、base-weighted permutation mass、真实微型 train/reload。\n- final producer: 3 lanes × 14 rows；全部 finite、exact K、真实 optimizer steps。\n- fresh-process replay: 2 × 3 lanes，全部 checkpoint representation/partition SHA exact。\n- label flow: producer 0 label reads；三个 partition bank 锁定后由 evaluator 各读取一次公开 reference。\n- preformal ordinal-rank 工件保留为 superseded，不进入科学主表。\n- AutoDL 保持开机；`shutdown_dispatched=false`。\n"""
 (OUT/"night17g_report.md").write_text(report,encoding="utf-8")
 (OUT/"night17g_plain_summary.md").write_text("Night-17G 把双模态共同同域、共同边界和冲突弃权直接写进可训练表示，但严格门只有 0/3。工程闭合、科学失败；不扩网格、不进入多 seed，AutoDL 保持开机。\n",encoding="utf-8")

if __name__=="__main__": main()
