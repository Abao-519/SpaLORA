#!/usr/bin/env python3
"""Fail-closed Night-8B handoff when a fixed partition cannot total-lock."""
from __future__ import annotations
import csv,json,subprocess,sys
from collections import Counter
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from SpaLORA.night3af_cache import sha256_file
from SpaLORA.night6c_pipeline import atomic_json
from SpaLORA.night8b_pipeline import OUT,RAW
def git(*x): return subprocess.check_output(['git',*x],cwd=REPO,text=True).strip()
def main():
    state_path=RAW/'formal_runner_state.json'; state=json.loads(state_path.read_text()); failures=[x for x in state['attempts'].values() if x['status']!='success']
    if state.get('status')!='incomplete' or len(failures)!=1 or failures[0].get('kind')!='transform_U00' or failures[0].get('seed')!=6: raise RuntimeError('unexpected formal failure pattern')
    log=Path(failures[0]['log']); text=log.read_text(errors='replace')
    if 'NumericalHeadFailure: partition did not produce fixed K' not in text: raise RuntimeError('fixed numerical failure signature drift')
    base=list((RAW/'formal/base').glob('seed_*/attempt_001/base_unit_manifest.json')); adapters=list((RAW/'formal/adapter').glob('formal/seed_*/attempt_001/adapter_unit_manifest.json')); transforms=list((RAW/'formal/transforms').glob('*/seed_*/transform_manifest.json'))
    if len(base)!=10 or len(adapters)!=10 or len(transforms)!=19: raise RuntimeError('20 training/19 transform evidence cardinality drift')
    raw_rows=[]
    for p in sorted((RAW/'formal').rglob('*')):
      if p.is_file(): raw_rows.append({'path':str(p),'size_bytes':p.stat().st_size,'sha256':sha256_file(p)})
    with (OUT/'raw_artifact_manifest.csv').open('w',newline='') as h:
      w=csv.DictWriter(h,fieldnames=['path','size_bytes','sha256']); w.writeheader(); w.writerows(raw_rows)
    incomplete={'schema_version':1,'status':'PRELABEL_TOTAL_LOCK_INCOMPLETE','terminal_status':'INFRASTRUCTURE_BLOCKED','reason_code':'FIXED_H05_NUMERICAL_HEAD_FAILURE_K_NOT_12',
      'scientific_training_units_success':'20/20','checkpoint_roundtrips_pass':'20/20','formal_transforms_success':'19/20','failed_primary_key':{'method':'U00_UNIVERSAL_C00','seed':6,'head':'H05','K_required':12},
      'failure_log':str(log),'failure_log_sha256':sha256_file(log),'scientific_retry':0,'fallback':0,'labels_Y_read':False,'evaluation_run':False,'candidate_search':False,
      'successful_transform_manifest_sha256':[{'path':str(p),'sha256':sha256_file(p)} for p in sorted(transforms)],'raw_artifact_manifest_sha256':sha256_file(OUT/'raw_artifact_manifest.csv')}
    atomic_json(OUT/'prelabel_incomplete_manifest.json',incomplete)
    atomic_json(OUT/'budget_and_access_audit.json',{'status':'FAIL_CLOSED','scientific_training':'20/20','checkpoint_roundtrips':'20/20','transforms':'19/20','scientific_retry':0,'fallback':0,'global_prelabel_corrections':'3/4','Y_values_read':False,'authorized_label_window_opened':False,'third_party_benchmark':False})
    atomic_json(OUT/'night8b_decision.json',{'terminal_status':'INFRASTRUCTURE_BLOCKED','reason_code':incomplete['reason_code'],'scientific_conclusion_available':False,'family_policy_generalization_evaluated':False,'label_firewall_preserved':True,'Y_values_read':False,'claim_sota':False})
    summary="# Night-8B 通俗结果\n\n本轮不能给出 F00 是否优于 U00 的 MISAR 科学结论。20 个正式训练单元和 checkpoint reload 全部成功，但固定 U00/H05 的 seed 6 只产生了非预注册的簇数，因此 20 个 partition 只能完成 19 个。任务书禁止重跑坏 seed、换 solver 或 fallback，所以我保留该失败并在打开标签之前停止。MISAR 的 Y 从未读取；ARI、NMI、Q 和空间指标均未计算。\n\n这不是“F00 泛化成功”或“失败”，而是一次严格的标签前数值端点阻塞。P22 artifacts 未进入 MISAR 模型输入，也没有运行第三方 benchmark。\n"
    (OUT/'night8b_plain_language_summary.md').write_text(summary,encoding='utf-8')
    report=f"""# SpaLORA Night-8B blocked prelabel report\n\n## Terminal state\n\n`INFRASTRUCTURE_BLOCKED` with reason `{incomplete['reason_code']}`. This is not a scientific comparison result.\n\n## Completed work\n\n- Authority, provenance, 1,949/1,949 mapping, immutable cache, P22 code-only parity and CUDA P0: PASS.\n- Formal base training: 10/10, fresh-process checkpoint reload: 10/10.\n- Formal R02 adapter training and reload: 10/10.\n- Fixed transforms: 19/20 success. U00/H05 seed 6 failed because the frozen spectral endpoint did not produce K=12.\n- Scientific retry: 0; fallback: 0; post-hoc K/solver/seed changes: 0.\n- MISAR Y values read: 0. No ARI/NMI/Q/spatial or external-family conclusion was computed.\n\n## Interpretation\n\nThe frozen training routes were deployable on MISAR, but the universal comparator failed its fixed partition contract for one seed. The protocol requires all 20 partitions to be locked and ordinarily pushed before label access. Retrying only seed 6 or changing the solver would violate the preregistration, so evaluation was not opened. This result neither confirms nor rejects R02 generalization.\n\n## Preservation\n\nRaw runs, checkpoints, affinities and logs remain under `/root/autodl-fs/night8b_raw_runs_20260820` and are indexed by SHA-256 in `raw_artifact_manifest.csv`. The failing log SHA is `{incomplete['failure_log_sha256']}`. GSE213264 remains explicitly excluded from MISAR provenance; valid identifiers are OEP003285, SRP491963 and Zenodo 7480069.\n"""
    (OUT/'night8b_report.md').write_text(report,encoding='utf-8')
    atomic_json(OUT/'tests_and_invariance_audit.json',{'status':'PASS','semantic_tests':10,'test_log':'outputs/night8b_handoff/night8b_tests.log','no_retry_test':True,'label_window_one_way_test':True,'independent_metrics_test':'synthetic_only_because_authorized_label_window_not_opened','candidate_search':False})
    atomic_json(OUT/'git_audit.json',{'branch':git('branch','--show-current'),'head_before_final_delivery_commit':git('rev-parse','HEAD'),'ordinary_push_only':True,'force_push':False,'final_tag_to_be_created_once_after_final_commit':True})
    include=[]
    for root in (OUT,REPO/'protocols/night8b',REPO/'scripts',REPO/'tests'):
      for p in sorted(root.rglob('*')):
        if p.is_file() and p.name!='tracked_delivery_index.json' and (root==OUT or root.name=='night8b' or p.name.startswith('night8b')): include.append({'path':str(p.relative_to(REPO)).replace('\\','/'),'size_bytes':p.stat().st_size,'sha256':sha256_file(p)})
    atomic_json(OUT/'tracked_delivery_index.json',{'schema_version':1,'root_rule':'repository-relative paths; index excludes itself','terminal_status':'INFRASTRUCTURE_BLOCKED','file_count':len(include),'files':include,'final_tag_must_follow_final_commit':True})
    print(json.dumps({'terminal_status':'INFRASTRUCTURE_BLOCKED','training':'20/20','transforms':'19/20','Y_values_read':False,'indexed_files':len(include)},sort_keys=True))
if __name__=='__main__': main()
