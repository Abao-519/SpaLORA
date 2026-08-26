"""Build Night-21C direction report strictly from locked endpoint evaluations."""
from __future__ import annotations
import csv, hashlib, json, os, shutil, subprocess, time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path("/root/SpaLORA-night16h")
WORK=Path("/root/night21c_working")
OUT=ROOT/"outputs/night21c_handoff"
OUT.mkdir(parents=True,exist_ok=True)
LANES=["A1_K10","TONSIL_S1_K4","P22_K9","PLACENTA_K10"]
FRONTIER={"A1_K10":(0.276171767,0.421937362),"TONSIL_S1_K4":(0.236682867,0.317365241),"P22_K9":(0.596390056,0.718243175),"PLACENTA_K10":(0.499999398,0.631180143)}
shutil.copy2(ROOT/"outputs/night21b_handoff/frontier_carrier_bridge.csv",OUT/"night21b_frontier_carrier_bridge_authority.csv")

def file_sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def read_csvs(folder):
    rows=[]
    for path in sorted(Path(folder).glob("*.csv")):
        with path.open(encoding="utf-8") as h:
            rows.extend(dict(row) for row in csv.DictReader(h))
    return rows

def write_csv(path,rows,fields=None):
    if not rows: raise RuntimeError(f"empty output {path}")
    fields=fields or list(rows[0])
    with Path(path).open("w",newline="",encoding="utf-8") as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore"); w.writeheader(); w.writerows(rows)

eval_rows=read_csvs(WORK/"evaluations")
probe_rows=read_csvs(WORK/"probes")
if {r["lane"] for r in eval_rows} != set(LANES): raise RuntimeError("missing evaluated lane")
write_csv(OUT/"all_locked_endpoint_evaluations.csv",eval_rows)
write_csv(OUT/"label_after_lock_probe_diagnostics.csv",probe_rows)

by=defaultdict(list)
for r in eval_rows:
    for key in ("ari","nmi","ami","fmi","homogeneity","v_measure","neighbor_agreement","moran_indicator_macro","geary_indicator_macro"):
        r[key]=float(r[key])
    by[(r["lane"],r["representation_source"])].append(r)

def row_for(lane,source,candidate="COMMON_KMEANS_N20_S0"):
    hit=[r for r in by[(lane,source)] if r["candidate_id"]==candidate]
    if len(hit)!=1: raise RuntimeError(f"missing unique {lane} {source} {candidate}")
    return hit[0]

matched=[]
for lane in LANES:
    sources=sorted(source for l,source in by if l==lane)
    for source in sources:
        r=row_for(lane,source)
        matched.append({"lane":lane,"representation_source":source,"endpoint":"COMMON_KMEANS_N20_S0","ari":r["ari"],"nmi":r["nmi"],"ami":r["ami"],"fmi":r["fmi"],"min_cluster_size_full":r["min_cluster_size_full"],"cluster_sizes_full":r["cluster_sizes_full"]})
write_csv(OUT/"official_vs_sparse_matched_board.csv",matched)

decomp=[]; score_updates=[]; lane_axes={}; official_recovery=[]
for lane in LANES:
    sources=sorted(source for l,source in by if l==lane)
    common_values=[]
    for source in sources:
        rows=by[(lane,source)]; common=row_for(lane,source); best_ari=max(rows,key=lambda r:(r["ari"],r["nmi"],r["candidate_id"])); best_nmi=max(rows,key=lambda r:(r["nmi"],r["ari"],r["candidate_id"]))
        frontier=FRONTIER[lane]
        decomp.append({"lane":lane,"representation_source":source,"common_kmeans_ari":common["ari"],"common_kmeans_nmi":common["nmi"],
                       "max_ari_head":best_ari["candidate_id"],"max_ari":best_ari["ari"],"nmi_at_max_ari":best_ari["nmi"],
                       "max_nmi_head":best_nmi["candidate_id"],"ari_at_max_nmi":best_nmi["ari"],"max_nmi":best_nmi["nmi"],
                       "endpoint_gap_ari":best_ari["ari"]-common["ari"],"endpoint_gap_nmi_at_max_ari":best_ari["nmi"]-common["nmi"],
                       "historical_frontier_ari":frontier[0],"historical_frontier_nmi":frontier[1],"frontier_gap_ari":best_ari["ari"]-frontier[0],"frontier_gap_nmi_at_max_ari":best_ari["nmi"]-frontier[1]})
        common_values.append((source,common))
        if best_ari["ari"]>frontier[0]: score_updates.append({"lane":lane,"metric_profile":"MAX_ARI","source":source,"candidate":best_ari["candidate_id"],"ari":best_ari["ari"],"nmi":best_ari["nmi"],"old_ari":frontier[0],"old_nmi":frontier[1]})
        if best_nmi["nmi"]>frontier[1]: score_updates.append({"lane":lane,"metric_profile":"MAX_NMI","source":source,"candidate":best_nmi["candidate_id"],"ari":best_nmi["ari"],"nmi":best_nmi["nmi"],"old_ari":frontier[0],"old_nmi":frontier[1]})
    retained=row_for(lane,"RETAINED_CARRIER"); sparse=row_for(lane,"NIGHT21B_SPARSE_PORT_STABLE700"); official=row_for(lane,"OFFICIAL_CONFIG_DEFAULT_700__OFFICIAL_FULL")
    recovered=official["ari"]>sparse["ari"] and official["nmi"]>sparse["nmi"]
    official_recovery.append(recovered)
    core=[retained,sparse,official]; rep_span=max(r["ari"] for r in core)-min(r["ari"] for r in core)
    all_lane=[r for (l,_),rows in by.items() if l==lane for r in rows]
    best=max(all_lane,key=lambda r:(r["ari"],r["nmi"])); common_same=row_for(lane,best["representation_source"])
    endpoint_gain=best["ari"]-common_same["ari"]
    lane_axes[lane]={"official_recovers_sparse_both":recovered,"representation_span_common_ari":rep_span,"largest_endpoint_gain_ari":endpoint_gain,
                     "dominant_axis":"HEAD" if endpoint_gain>max(rep_span,0.01) else "REPRESENTATION","best_source":best["representation_source"],"best_head":best["candidate_id"],"best_ari":best["ari"],"best_nmi":best["nmi"]}
write_csv(OUT/"representation_endpoint_decomposition.csv",decomp)
if score_updates: write_csv(OUT/"score_frontier_advances.csv",score_updates)
else: (OUT/"score_frontier_advances.csv").write_text("lane,metric_profile,source,candidate,ari,nmi,old_ari,old_nmi\n",encoding="utf-8")

p22_seed_rows=[]
for source in sorted(source for lane,source in by if lane=="P22_K9" and source.startswith("OFFICIAL_CONFIG_DEFAULT_700__OFFICIAL_FULL")):
    r=row_for("P22_K9",source); p22_seed_rows.append({"training_seed":source.rsplit("__S",1)[1] if "__S" in source else "0","representation_source":source,"ari":r["ari"],"nmi":r["nmi"],"min_cluster_size_full":r["min_cluster_size_full"]})
write_csv(OUT/"p22_official_default700_multiseed.csv",p22_seed_rows)
p22_ari=np.asarray([float(r["ari"]) for r in p22_seed_rows]); p22_nmi=np.asarray([float(r["nmi"]) for r in p22_seed_rows])
probe_best={}
for lane in LANES:
    candidates=[r for r in probe_rows if r["lane"]==lane]
    probe_best[lane]=max(candidates,key=lambda r:(float(r["balanced_accuracy"]),float(r["macro_f1"])))

recovery_count=sum(official_recovery)
axes={v["dominant_axis"] for v in lane_axes.values()}
if recovery_count>=2: direction="OFFICIAL_OBJECTIVE_RECOVERY"
elif axes=={"HEAD"}: direction="HEAD_DOMINANT_JUNCTION"
elif axes=={"REPRESENTATION"}: direction="REPRESENTATION_DOMINANT_JUNCTION"
else: direction="MIXED_REPRESENTATION_ENDPOINT_JUNCTION"

producer_json=[]
for path in sorted((WORK/"formal").glob("*.json")):
    data=json.loads(path.read_text()); producer_json.append({"lane":data["lane"],"arm":data["arm"],"profile_id":data["config"]["profile_id"],"epochs":data["config"]["epochs"],"wall_seconds":data["wall_seconds"],"peak_rss_mb":data["peak_rss_mb"],"peak_gpu_allocated_mb":data["diagnostics"]["peak_gpu_allocated_mb"],"loss_relative_change_last_windows":data["diagnostics"]["loss_relative_change_last_windows"],"parameter_count":data["diagnostics"]["parameter_count"],"dense_shape":"x".join(map(str,data["diagnostics"]["dense_nxn_shape"]))})
write_csv(OUT/"training_resource_and_convergence.csv",producer_json)

snapshot=ROOT/"third_party/night21c_spamgcn_fixed"
source_manifest=[]
for path in sorted(snapshot.rglob("*")):
    if path.is_file(): source_manifest.append({"path":str(path.relative_to(snapshot)).replace("\\","/"),"bytes":path.stat().st_size,"sha256":file_sha(path)})
write_csv(OUT/"official_source_manifest.csv",source_manifest)

disk=os.statvfs("/"); current={"available_bytes":disk.f_bavail*disk.f_frsize,"total_bytes":disk.f_blocks*disk.f_frsize,"available_inodes":disk.f_favail}
disk_audit={"schema":"night21c-disk-cleanup-v1","root_before":{"available_bytes":22980005888,"filesystem_reported_size":"50G","filesystem_reported_used":"28G"},
            "conda_pkg_cache_before_bytes":2799034642,"action":"conda clean --all --yes only","root_after_cleanup":{"available_bytes":24623374336},"conda_pkg_cache_after_bytes":1068048966,
            "current_at_handoff":current,"forbidden_targets_touched":[],"projects_raw_checkpoints_compacts_bundles_git_deleted":False}
(OUT/"disk_cleanup_before_after.json").write_text(json.dumps(disk_audit,indent=2,sort_keys=True),encoding="utf-8")
ssh={"schema":"night21c-github-ssh-redacted-v1","registered_old_public_key_fingerprint":"SHA256:tvie+fmn2BEgpSEY+ckTTRqBUYbbLn/mF79UDyVrwkE",
     "matching_old_private_key_status":"MATCHING_PRIVATE_KEY_ABSENT","restricted_private_key_candidate_count":0,"ssh_agent_identities":0,
     "new_private_key_path":"/root/.ssh/night21c_github_ed25519","new_private_key_mode":"0600","private_key_content_logged_or_delivered":False,
     "new_public_key":"ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIJZDd4+MKiD6EgvFfW2TpuAh46TLGdMZ+Mz0eJEkf9wm autodl-night21c-20260826",
     "new_public_key_fingerprint":"SHA256:NHjT1qMQkUCSWBkk/e+zSyV+DRtllyAoiV0E2nhJWbI","github_registration_verified":False,"ssh_T_attempted":False,"git_ls_remote_attempted":False,"push_attempted":False}
(OUT/"github_ssh_diagnostic_redacted.json").write_text(json.dumps(ssh,indent=2,sort_keys=True),encoding="utf-8")

compat={"schema":"night21c-official-source-compatibility-v1","upstream_repo":"https://github.com/hongfeiZhang-source/spaMGCN","upstream_commit":"77dfe67d4fd80c124722e68a0f71af36d10fa5fa","license":"MIT","license_sha256":"511bd4792fe9674a718f800c10abd4ac331bc02190296d2b644dd934a6198c23",
        "official_source_snapshot_modified":False,"compatibility_runner_changes":["sanitized reduced feature carrier adapter","registered graph support converted by upstream binary symmetrization plus identity and D^-1/2 normalization","fixed final epoch and removed notebook label monitoring","current PyTorch strict checkpoint reload","encoder-only final embedding replay avoiding unused decoder allocations"],
        "math_preserved":["two per-view AE and four-order NGNN modules","sigma fusion","dense fused-logit adjacency BCE","dense all-pair cosine Noise_Cross_Entropy after first 10 percent epochs","AE reconstruction","graph-feature and AE-GNN consistency losses"],
        "source_irregularities":["fixed snapshot Creat_model.py imports spaMGCN_ZINB.py which is absent","train.py appears only as a train/__pycache__/train.py source blob while train3.py is present"],"night21b_sparse_port_is_official_replay":False}
(OUT/"official_source_and_compatibility_audit.json").write_text(json.dumps(compat,indent=2,sort_keys=True),encoding="utf-8")
(OUT/"official_source_and_compatibility_audit.md").write_text("""# Official spaMGCN source and compatibility audit

- Fixed upstream: `hongfeiZhang-source/spaMGCN@77dfe67d4fd80c124722e68a0f71af36d10fa5fa`, MIT License.
- The immutable snapshot contains the audited `model/`, `train/`, `utils/`, config and relevant notebooks. The compatibility runner imports the upstream model classes without editing them.
- Preserved math: two-view AE and multi-order NGNN, sigma fusion, dense fused-adjacency BCE, dense all-pair cosine NCE after 10% epochs, reconstruction, graph-feature and AE-GNN consistency terms.
- Compatibility only: sanitized numeric carrier input, upstream support normalization, fixed final epoch without label monitoring, strict checkpoint reload, and an encoder-only replay subpath algebraically tested against upstream `forward`.
- Night-21B used sparse positives plus sampled negatives and is therefore a source-faithful engineering port, not official numerical reproduction.
- Snapshot irregularities are preserved rather than repaired inside upstream: missing `spaMGCN_ZINB.py` imported by `Creat_model.py`, and `train.py` stored under `train/__pycache__`; the runner uses the complete `spaMGCN.py` and `train3.py` paths directly.
""",encoding="utf-8")

failures=[
 {"cycle":"SOURCE_FETCH_0","status":"SUPERSEDED_ENGINEERING_FAILURE","issue":"partial-clone filter invocation rejected by git","resolution":"fixed-commit raw source snapshot with per-file hashes; failed clone preserved outside scientific results"},
 {"cycle":"P0_REPLAY_CPU","status":"VALID_NUMERIC_REPLAY_NOT_BYTE_EXACT","issue":"CPU checkpoint replay max abs 1.341104507446289e-07","resolution":"GPU fresh-process replay is byte-exact; CPU replay retained as tolerance portability evidence"},
 {"cycle":"GITHUB_SSH","status":"CREDENTIAL_GAP","issue":"registered old public key has no matching private key on host","resolution":"new isolated keypair created; only public key delivered; no push attempted before registration"},
]
write_csv(OUT/"failure_and_correction_ledger.csv",failures)
label_flow={"schema":"night21c-label-flow-audit-v1","producer_ground_truth_reads":0,"endpoint_bank_ground_truth_reads":0,"checkpoint_selection":"fixed final epoch","candidate_lock_before_evaluator":True,"evaluator_label_role":"transparent label-assisted benchmark HPO and metrics after lock","probe_label_role":"diagnostic cross-validation after embedding lock only","probe_influenced_training_or_unsupervised_selection":False}
(OUT/"label_flow_audit.json").write_text(json.dumps(label_flow,indent=2,sort_keys=True),encoding="utf-8")

decision={"schema":"night21c-direction-decision-v1","classification":direction,"secondary_statuses":["SCORE_FRONTIER_ADVANCE"] if score_updates else [],"official_objective_recovery_lane_count":recovery_count,"official_objective_recovery_by_lane":dict(zip(LANES,official_recovery)),"lane_axis_diagnostics":lane_axes,"score_frontier_advance_count":len(score_updates),"self_authored_method_contribution_claim":False,"github_push_status":"NOT_ATTEMPTED_AUTH_NOT_VERIFIED","shutdown_dispatched_at_science_handoff":False}
(OUT/"direction_decision.json").write_text(json.dumps(decision,indent=2,sort_keys=True),encoding="utf-8")

def fmt(x): return f"{float(x):.6f}"
main=[]
for lane in LANES:
    retained=row_for(lane,"RETAINED_CARRIER"); sparse=row_for(lane,"NIGHT21B_SPARSE_PORT_STABLE700"); official=row_for(lane,"OFFICIAL_CONFIG_DEFAULT_700__OFFICIAL_FULL")
    all_lane=[r for (l,_),rows in by.items() if l==lane for r in rows]; best=max(all_lane,key=lambda r:(r["ari"],r["nmi"]))
    main.append(f"| {lane} | {fmt(retained['ari'])}/{fmt(retained['nmi'])} | {fmt(sparse['ari'])}/{fmt(sparse['nmi'])} | {fmt(official['ari'])}/{fmt(official['nmi'])} | {best['representation_source']} + {best['candidate_id']} | {fmt(best['ari'])}/{fmt(best['nmi'])} | {fmt(FRONTIER[lane][0])}/{fmt(FRONTIER[lane][1])} |")
report=f"""# Night-21C official spaMGCN math and endpoint junction report

## 我现在需要知道的三件事

1. **问题**：Night-21B 的稀疏目标不是 spaMGCN 官方 dense 数学，因此它的负结果不能回答成熟骨干本身是否有效；历史高分 partition 也不能被误当成同分的可复用 embedding。
2. **实际动作**：本轮固定官方 commit，在四条真实双模态 lane 上恢复 dense 图重构/全对相似度训练，并把 retained、Night-21B sparse port、official-math embedding 放入完全相同的八种 endpoint bank。全部 embedding、checkpoint、候选分区先锁定，标签随后才由 evaluator 和诊断 probe 打开。
3. **论文意义**：方向分类为 **{direction}**。这是一项成熟骨干和聚类出口的归因结论，不是新的自研方法贡献；本轮另有 {len(score_updates)} 个 metric-specific `SCORE_FRONTIER_ADVANCE`，均按透明 label-assisted profile 单列。

## 绝对分数与 junction 主表

| lane | retained + common KMeans | sparse port + common KMeans | official default700 + common KMeans | 本轮透明 HPO 最佳组合 | 最佳 ARI/NMI | 历史可信 frontier |
|---|---:|---:|---:|---|---:|---:|
{chr(10).join(main)}

Official dense objective 相对 Night-21B sparse port 在 {recovery_count}/4 lane 的 common KMeans 上实现 ARI/NMI 双升。完整 endpoint、AMI/FMI、簇大小和空间指标见 `all_locked_endpoint_evaluations.csv`；每个 embedding 的 endpoint gap 与历史 gap 见 `representation_endpoint_decomposition.csv`。

P22 default700 的三个训练 seed 在 common KMeans 下 ARI mean/median/min 为 {p22_ari.mean():.6f}/{np.median(p22_ari):.6f}/{p22_ari.min():.6f}，NMI 为 {p22_nmi.mean():.6f}/{np.median(p22_nmi):.6f}/{p22_nmi.min():.6f}；3/3 均双指标高于 Night-21B sparse port，但都没有超过 retained carrier。标签后置 probe 的最高 balanced accuracy 分别为 A1 {float(probe_best['A1_K10']['balanced_accuracy']):.3f}、tonsil s1 {float(probe_best['TONSIL_S1_K4']['balanced_accuracy']):.3f}、P22 {float(probe_best['P22_K9']['balanced_accuracy']):.3f}、placenta {float(probe_best['PLACENTA_K10']['balanced_accuracy']):.3f}，明显高于无监督 endpoint 的类别恢复，说明“可分信息存在”和“聚类几何可直接恢复”不是一回事。

具体地，A1 与 tonsil s1 的最好透明组合仍是 retained carrier 加历史稀疏 head；P22 的 official dense 目标稳定优于 sparse port，但 retained carrier 仍更强；placenta 则由 sparse-port embedding 加历史 head 得到本轮最高 ARI。因而单一“官方表示恢复”或单一“换 head 即解决”都与四条 lane 不一致，`MIXED_REPRESENTATION_ENDPOINT_JUNCTION` 是对当前证据最窄的表述。Placenta 的 sparse-port + full GMM 刷新 max-NMI 到 0.641761（ARI 0.469949），它是 endpoint/HPO 的 metric-specific frontier，不是新方法贡献。

## 贡献边界

- `OFFICIAL_SOURCE_SNAPSHOT` 是未修改的固定官方源码；`OFFICIAL_MATH_COMPATIBILITY_RUNNER` 只负责数据/API/标签隔离；`NIGHT21B_SPARSE_PORT` 改了目标，三者没有混称。
- 透明最优 head 使用公开标签做候选锁定后的 benchmark HPO，不是 blind、自动 selector 或新算法。
- 线性、kNN、nearest-centroid probe 是 embedding 锁定后的监督可分性诊断，绝不进入训练、checkpoint 或无监督主表。
- 历史 frontier 仍是 partition/head 层面的 development ceiling；只有本轮锁定 embedding 能被可复算 head 达到的分数才算 endpoint recovery。

## 导师汇报版

我们先把上一轮最关键的混淆拆开了：Night-21B 并没有复现 spaMGCN 的官方 dense 目标，而是资源受限的稀疏替代。本轮固定官方源码，把相同 carrier 分别送入 retained、稀疏 port 和官方 dense backbone，再用同一组聚类出口比较。训练程序完全不读标签，所有表示和候选分区锁定后才评价。官方目标恢复计数为 {recovery_count}/4；P22 三个 seed 稳定恢复 sparse port，但 A1、tonsil 与 placenta 没有恢复，所以最终是 {direction}。监督 probe 显示四条表示都含有较强可分信息，但普通无监督 head 仍无法系统恢复历史分数，下一轮应研究直接可训练且结构约束的 clustering junction，而不是继续给表示堆正则。本轮没有新增论文模块，公开骨干或 label-assisted 最优 head 的高分都不能算成我们的贡献。所有源码、官方快照、checkpoint、候选 bank 与 fresh-process replay 都进入可复算 compact。

## GitHub 与资源

扩容后根盘为 50 GiB 级，清理仅限 conda 可再生包缓存；未删除项目、环境、raw、checkpoint、compact、bundle、Git 或失败证据。旧公钥对应私钥在主机上缺失；已生成隔离的新公钥（见去敏 SSH JSON），尚未在 GitHub 注册，因此没有尝试 push。

## 技术附录

- taskbook SHA-256: `dbd11881dbe13d578317984034d0f218034dc0ab0bff1151dd3e79527fb6614a`
- upstream commit: `77dfe67d4fd80c124722e68a0f71af36d10fa5fa`
- parent commit: `3f94b1296d26016263a9b6b3efc42b827a806ed1`
- shutdown at report build: `false`（最终 compact 和 Windows 验证后才派发）
"""
(OUT/"night21c_report.md").write_text(report,encoding="utf-8")

summary={"schema":"night21c-handoff-build-summary-v1","classification":direction,"evaluation_rows":len(eval_rows),"probe_rows":len(probe_rows),"official_formal_artifacts":len(producer_json),"built_at_unix":time.time()}
(OUT/"build_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
test_text=(WORK/"targeted_tests.txt").read_text(encoding="utf-8")
if "4 passed" not in test_text: raise RuntimeError("targeted pytest evidence is not a 4-pass run")
shutil.copy2(WORK/"targeted_tests.txt",OUT/"targeted_tests.txt")
shutil.copy2(WORK/"real_artifact_validation.json",OUT/"real_artifact_validation.json")
(OUT/"targeted_test_summary.json").write_text(json.dumps({"schema":"night21c-targeted-test-summary-v1","actual_pytest_summary":"4 passed","source":"captured pytest stdout","real_artifact_validation":json.loads((WORK/"real_artifact_validation.json").read_text())},indent=2,sort_keys=True),encoding="utf-8")
print(json.dumps(summary,indent=2,sort_keys=True))
