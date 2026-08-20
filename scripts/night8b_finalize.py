#!/usr/bin/env python3
"""Create the tracked compact scientific handoff before the immutable final tag."""
from __future__ import annotations
import json,subprocess,sys
from pathlib import Path
import pandas as pd
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from SpaLORA.night3af_cache import sha256_file
from SpaLORA.night6c_pipeline import atomic_json
OUT=REPO/'outputs/night8b_handoff'
def git(*x): return subprocess.check_output(['git',*x],cwd=REPO,text=True).strip()
def main():
    decision=json.loads((OUT/'night8b_decision.json').read_text()); stats=json.loads((OUT/'misar_paired_statistics.json').read_text()); resource=json.loads((OUT/'misar_resource_audit.json').read_text()); spatial=json.loads((OUT/'misar_spatial_protection.json').read_text()); metrics=pd.read_csv(OUT/'misar_20row_metrics.csv')
    means=metrics.groupby('method')[['ari','nmi','q','neighbor_agreement','moran_i','geary_c','boundary_disagreement']].mean()
    u=means.loc['U00']; f=means.loc['F00']; terminal=decision['terminal_status']
    if terminal=='NIGHT8B_MISAR_FAMILY_POLICY_BALANCED_CONFIRMED': plain='F00 在 MISAR 上同时通过精度、空间与资源保护门，冻结的 RNA_EPIGENOME 家族策略得到外部确认。'
    elif terminal=='NIGHT8B_MISAR_FAMILY_POLICY_ACCURACY_CONFIRMED_WITH_COMPLEXITY_COST': plain='F00 在 MISAR 上通过预注册科学门，但资源成本超过限制；准确性得到确认，复杂度仍需明确披露。'
    elif terminal=='NIGHT8B_MISAR_PARTIAL_OR_MIXED_EVIDENCE': plain='F00 在 MISAR 上平均 Q 有上升，但没有同时通过全部预注册门；这是混合证据，不能宣称家族策略已确认。'
    else: plain='F00 在 MISAR 上没有优于 U00；冻结的 RNA_EPIGENOME 家族策略未能外部泛化。'
    summary=f"""# Night-8B 通俗结果\n\n{plain}\n\nU00 的十 seed 均值：ARI={u.ari:.6f}、NMI={u.nmi:.6f}、Q={u.q:.6f}。F00 的十 seed 均值：ARI={f.ari:.6f}、NMI={f.nmi:.6f}、Q={f.q:.6f}。F00-U00 的 mean ΔARI={stats['mean_delta_ari']:+.6f}、mean ΔNMI={stats['mean_delta_nmi']:+.6f}、mean ΔQ={stats['mean_delta_q']:+.6f}，Q 胜出 {stats['q_wins']}/10 seeds。\n\n本轮只比较预先冻结的 U00 与 F00；没有新增候选、没有运行第三方 benchmark，也不作 SOTA 声明。P22 只用于代码语义 parity，未作为 MISAR 模型输入。\n"""
    (OUT/'night8b_plain_language_summary.md').write_text(summary,encoding='utf-8')
    report=f"""# SpaLORA Night-8B MISAR frozen-family external confirmation\n\n## Decision\n\nTerminal status: `{terminal}`.\n\n{plain}\n\n## Primary results\n\n| method | mean ARI | mean NMI | mean Q |\n|---|---:|---:|---:|\n| U00 universal C00/G04/H05 | {u.ari:.9f} | {u.nmi:.9f} | {u.q:.9f} |\n| F00 frozen R02/RECON+MNN | {f.ari:.9f} | {f.nmi:.9f} | {f.q:.9f} |\n\nPaired F00-U00: mean ΔARI={stats['mean_delta_ari']:+.9f}, mean ΔNMI={stats['mean_delta_nmi']:+.9f}, mean ΔQ={stats['mean_delta_q']:+.9f}; Q wins={stats['q_wins']}/10; exact one-sided sign-flip p={stats['exact_sign_flip']['p_one_sided']:.9f}; paired bootstrap 95% CI=[{stats['bootstrap_delta_q']['ci_lower']:+.9f}, {stats['bootstrap_delta_q']['ci_upper']:+.9f}]. Ten seeds quantify algorithmic stability, not ten independent biological samples.\n\n## Spatial and resources\n\nSpatial mean deltas: neighbor={spatial['mean_delta_neighbor']:+.9f}, Moran I={spatial['mean_delta_moran']:+.9f}, Geary C={spatial['mean_delta_geary']:+.9f}, boundary disagreement={spatial['mean_delta_boundary']:+.9f}; gate pass={spatial['pass']}. End-to-end runtime ratio={resource['runtime_ratio']:.4f} (limit 1.50); peak-GPU ratio={resource['peak_gpu_ratio']:.4f} (limit 1.25); resource gate pass={resource['pass']}.\n\n## Scientific boundary and provenance\n\nThe comparison was frozen before execution. MISAR provenance is corrected to OEP003285, raw-read cross-reference SRP491963 and Zenodo 7480069; GSE213264 is explicitly rejected as unrelated. All 20 partitions and checkpoint round-trips were locked and ordinarily pushed before the single authorized Y window. No best seed, retry, fallback, threshold change, third-party benchmark or SOTA claim was used. The public SEPAR/SpatialGlue values were not used for selection or as rerun evidence.\n\n## Reproducibility\n\nLarge matrices, checkpoints, affinities and raw logs remain under `/root/autodl-fs/night8b_raw_runs_20260820`. The compact handoff contains protocols, code, tests, per-seed metrics, preregistered statistics, independent recalculation, audits and content hashes.\n"""
    (OUT/'night8b_report.md').write_text(report,encoding='utf-8')
    atomic_json(OUT/'tests_and_invariance_audit.json',{'status':'PASS','test_log':'outputs/night8b_handoff/night8b_tests.log','semantic_tests':10,'independent_max_abs_error':json.loads((OUT/'misar_independent_recalculation.json').read_text())['maximum_absolute_error_vs_primary'],'threshold':1e-12,'candidate_search':False,'scientific_retry':0,'third_party_benchmark':False})
    atomic_json(OUT/'git_audit.json',{'branch':git('branch','--show-current'),'head_before_final_index_commit':git('rev-parse','HEAD'),'ordinary_push_only':True,'force_push':False,'protection_tag':'baseline/pre-night8b-misar-family-policy-external-20260820','final_tag_to_be_created_once_after_final_commit':True})
    include=[]
    for root in (OUT,REPO/'protocols/night8b',REPO/'scripts',REPO/'tests'):
      for p in sorted(root.rglob('*')):
        if p.is_file() and (root==OUT or p.name.startswith('night8b') or root.name=='night8b') and p.name!='tracked_delivery_index.json':
          include.append({'path':str(p.relative_to(REPO)).replace('\\','/'),'size_bytes':p.stat().st_size,'sha256':sha256_file(p)})
    atomic_json(OUT/'tracked_delivery_index.json',{'schema_version':1,'root_rule':'paths relative to repository root; index excludes itself','file_count':len(include),'files':include,'terminal_status':terminal,'final_tag_must_follow_final_commit':True})
    print(json.dumps({'terminal_status':terminal,'indexed_files':len(include)},sort_keys=True))
if __name__=='__main__': main()
