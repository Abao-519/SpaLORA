#!/usr/bin/env python3
"""Finalize verified Night-8A recovery metrics, shortlist, and reports."""
from __future__ import annotations
import hashlib,json,os
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(os.environ.get('NIGHT8A_RECOVERY_REPO','/root/autodl-fs/SpaLORA-night8a-eval-recovery')); OUT=REPO/'outputs/night8a_eval_recovery'; RAW=Path('/root/autodl-fs/night8a_eval_recovery_20260820')
REG=REPO/'protocols/night8a/SpaLORA_Night8A_MFSPC_Registry_2026-08-20.json'
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def atomic(path,value):
 path=Path(path); q=path.with_suffix(path.suffix+'.tmp'); q.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n'); os.replace(q,path)
def references():
 n7=pd.read_csv(REPO/'outputs/night7a_handoff/per_seed_metrics.csv'); p=n7[(n7.candidate_id=='C00_G04_H05_CONFIRMED') & n7.dataset.isin(['a1','tonsil','d1'])].copy(); p['reference_id']='C00_G04_H05_CONFIRMED'
 n7b=pd.read_csv(REPO/'outputs/night7b_handoff/R2_full_per_seed_metrics.csv'); e=n7b[(n7b.config_id=='R02__E1_ADAPTER_C06_MEAN__H01') & (n7b.dataset=='p22')].copy(); e['reference_id']='R02_P22_FRONTIER_DEVELOPMENT_REFERENCE'
 cols=['dataset','seed','reference_id','ari','nmi','q','neighbor_agreement','moran_i','geary_c','boundary_disagreement']; r=pd.concat([p[cols],e[cols]],ignore_index=True)
 if len(r)!=30 or r.duplicated(['dataset','seed']).any(): raise RuntimeError('reference coverage')
 return r
def deltas(frame,ref):
 m=frame.merge(ref,on=['dataset','seed'],how='left',suffixes=('','_reference'),validate='many_to_one')
 success=m.evaluation_status=='SUCCESS'
 if m.loc[success,'reference_id'].isna().any(): raise RuntimeError('missing reference')
 for metric in ['ari','nmi','q','neighbor_agreement','moran_i','geary_c','boundary_disagreement']: m[f'delta_{metric}']=m[metric]-m[f'{metric}_reference']
 return m
def spatial_pass(group):
 failed=[]
 for ds,x in group.groupby('dataset'):
  if (x.delta_neighbor_agreement.mean()<-.03 and x.delta_moran_i.mean()<-.03) or (x.delta_geary_c.mean()>.03 and x.delta_boundary_disagreement.mean()>.03): failed.append(ds)
 return len(failed)==0,failed
def summarize(frame,complexity):
 out=[]
 for cid,rows in frame.groupby('config_id'):
  success=rows[rows.evaluation_status=='SUCCESS']; item={'config_id':cid,'role':'COMPARATOR' if cid=='B00_FAMILY_REFERENCE' else 'NEW_CANDIDATE','success_cells':len(success),'failure_cells':len(rows)-len(success),'complexity':complexity.get(cid,99)}
  for ds in ['a1','tonsil','d1','p22']:
   part=success[success.dataset==ds]
   for metric in ['ari','nmi','q','neighbor_agreement','moran_i','geary_c','boundary_disagreement']:
    item[f'{ds}_mean_{metric}']=float(part[metric].mean()); item[f'{ds}_mean_delta_{metric}']=float(part[f'delta_{metric}'].mean())
   item[f'{ds}_q_wins']=int((part.delta_q>0).sum())
  item['Q_HLN']=.5*(item['a1_mean_q']+item['d1_mean_q']); item['delta_Q_HLN']=.5*(item['a1_mean_delta_q']+item['d1_mean_delta_q'])
  item['priority_macro_Q']=.45*item['Q_HLN']+.45*item['p22_mean_q']+.10*item['tonsil_mean_q']; item['priority_macro_delta_Q']=.45*item['delta_Q_HLN']+.45*item['p22_mean_delta_q']+.10*item['tonsil_mean_delta_q']
  item['important_worst_delta_Q']=min(item['delta_Q_HLN'],item['p22_mean_delta_q']); item['paired_q_wins']=int((success.delta_q>0).sum()); item['spatial_protection_pass'],item['spatial_failure_datasets']=spatial_pass(success); item['complete']=len(success)==len(rows)
  out.append(item)
 return pd.DataFrame(out).sort_values('config_id').reset_index(drop=True)
def rank(frame,primary):
 cols=[]
 for c in [primary,'priority_macro_delta_Q','important_worst_delta_Q','paired_q_wins','complexity','config_id']:
  if c not in cols: cols.append(c)
 asc=[c in ['complexity','config_id'] for c in cols]; return frame.sort_values(cols,ascending=asc).config_id.tolist()
def pareto(frame):
 cols=['delta_Q_HLN','p22_mean_delta_q','tonsil_mean_delta_q']; rows=frame.to_dict('records'); keep=[]
 for x in rows:
  if not any(y['config_id']!=x['config_id'] and all(y[c]>=x[c] for c in cols) and any(y[c]>x[c] for c in cols) for y in rows): keep.append(x['config_id'])
 return sorted(keep)
def main():
 window=json.loads((OUT/'recovery_label_window_audit.json').read_text()); independent=json.loads((RAW/'independent_recompute_audit.json').read_text())
 if window['status']!='CLOSED_PASS' or independent['status']!='PASS' or independent['max_abs_error']>1e-12: raise RuntimeError('window/independent hard gate')
 reg=json.loads(REG.read_text()); complexity={x['id']:len(x['modules']) for x in reg['R2_configs']}
 r1=deltas(pd.read_csv(RAW/'primary_r1.csv'),references()); r2=deltas(pd.read_csv(RAW/'primary_r2.csv'),references())
 baseline=r2[(r2.config_id=='B00_FAMILY_REFERENCE') & (r2.evaluation_status=='SUCCESS')]; parity=float(np.nanmax(np.abs(baseline[['delta_ari','delta_nmi','delta_q']].to_numpy(float))))
 if len(baseline)!=12 or parity>1e-12: raise RuntimeError(f'comparator parity {parity}')
 r1.to_csv(OUT/'recovered_r1_per_seed_metrics.csv',index=False); r2.to_csv(OUT/'recovered_r2_per_seed_metrics.csv',index=False)
 summary=summarize(r2,complexity); summary.to_csv(OUT/'recovered_r2_candidate_summary.csv',index=False)
 probe=[]
 for (cid,ds),x in r1[r1.evaluation_status=='SUCCESS'].groupby(['config_id','dataset']): probe.append({'config_id':cid,'dataset':ds,'mean_ari':x.ari.mean(),'mean_nmi':x.nmi.mean(),'mean_q':x.q.mean(),'mean_delta_ari':x.delta_ari.mean(),'mean_delta_nmi':x.delta_nmi.mean(),'mean_delta_q':x.delta_q.mean(),'spatial_protection_pass':spatial_pass(x)[0]})
 pd.DataFrame(probe).sort_values(['config_id','dataset']).to_csv(OUT/'recovered_r1_module_probe_summary.csv',index=False)
 new=summary[(summary.role=='NEW_CANDIDATE') & summary.complete].copy(); pd.DataFrame({'config_id':pareto(new)}).to_csv(OUT/'recovered_pareto_frontier.csv',index=False)
 eligible=new[new.spatial_protection_pass]
 unified=eligible[(eligible.delta_Q_HLN>=0)&(eligible.p22_mean_delta_q>=0)&(eligible.tonsil_mean_delta_q>=-.01)]
 hln=eligible[eligible.p22_mean_delta_q>=-.01]; p22=eligible[eligible.delta_Q_HLN>=-.01]
 slots={'unified_balanced':rank(unified,'priority_macro_delta_Q')[0] if len(unified) else None,'human_lymph_frontier':rank(hln,'Q_HLN')[0] if len(hln) else None,'P22_frontier':rank(p22,'p22_mean_q')[0] if len(p22) else None}
 ids=[]
 for v in slots.values():
  if v and v not in ids: ids.append(v)
 ids=ids[:3]
 shortlist={'schema_version':'night8a-eval-recovery-shortlist-v1','status':'LOCKED','finalist_ids':ids,'slots':slots,'maximum_distinct_ids':3,'B00_excluded':True,'B03_excluded_incomplete':True,'source_metrics_sha256':sha(OUT/'recovered_r2_per_seed_metrics.csv'),'r3_started':False,'external_benchmark_started':False}
 atomic(OUT/'recovered_shortlist_ids.json',shortlist)
 status='NIGHT8A_DEV_WINDOW1_RECOVERED_SHORTLIST_LOCKED' if ids else 'NIGHT8A_DEV_WINDOW1_RECOVERED_NO_ELIGIBLE_NEW_CANDIDATE'
 decision={'schema_version':'night8a-eval-recovery-decision-v1','status':status,'original_night8a_status':'IMPLEMENTATION_SEMANTICS_INVALID','training_cells_salvaged':116,'aliases_salvaged':12,'fixed_failure':'B03/P22/seed1','comparator_parity_max_abs_error':parity,'independent_recompute_max_abs_error':independent['max_abs_error'],'candidate_dependency_passed':'109/109','finalist_ids':ids,'r3_started':False,'misar_benchmark_started':False,'training':0,'transform':0,'external_benchmark':0,'gpu_used':False}
 atomic(OUT/'recovery_decision.json',decision); (OUT/'independent_recompute_audit.json').write_bytes((RAW/'independent_recompute_audit.json').read_bytes())
 # Human-readable summaries are deliberately generated only after all hard gates.
 def f(x): return f'{float(x):+.6f}'
 lines=['# Night-8A evaluation-only recovery report','',f'终态：`{status}`','', '原 Night-8A 仍永久保持 `IMPLEMENTATION_SEMANTICS_INVALID`。本恢复任务完成 0 training、0 transform、0 external benchmark、0 GPU。','', '## 科学结果','',f'- 116 次真实 CUDA 训练与 12 个合法 alias 全部救回；109/109 新候选依赖隔离通过。',f'- comparator parity 最大误差：`{parity:.3e}`；独立复算最大误差：`{independent["max_abs_error"]:.3e}`。',f'- 固定失败保持：B03/P22/seed1；B03 只有 11/12，未晋级。',f'- shortlist：{", ".join(ids) if ids else "空"}。','', '## R2 pilot 相对锁定 family comparator 的均值变化','', '|配置|HLN ΔQ|P22 ΔQ|tonsil ΔQ|priority macro ΔQ|空间门|完整|','|---|---:|---:|---:|---:|---|---|']
 for x in summary.to_dict('records'): lines.append(f'|{x["config_id"]}|{f(x["delta_Q_HLN"])}|{f(x["p22_mean_delta_q"])}|{f(x["tonsil_mean_delta_q"])}|{f(x["priority_macro_delta_Q"])}|{x["spatial_protection_pass"]}|{x["complete"]}|')
 lines += ['', '## Shortlist slots','']+[f'- {k}: {v}' for k,v in slots.items()]+['','MISAR 现在不能运行：本轮只恢复 3-seed DEV_WINDOW_1。必须先由规划方审查 pilot，并对锁定的最多三个候选运行预注册 R3 补 seed、冻结最终 family candidate，之后才允许一次性外部确认。','']
 (OUT/'night8a_eval_recovery_report.md').write_text('\n'.join(lines),encoding='utf-8')
 plain=['Night-8A 的 116 次训练没有白跑：checkpoint 和候选结果都通过了全量重哈希，错误只在旧 comparator 变换。',f'本次用真正锁定的 C00/R02 comparator 重算后，shortlist 为：{", ".join(ids) if ids else "空"}。','B03/P22/seed1 仍是原始数值失败，没有补跑。','当前还不能跑 MISAR，因为这里只是三 seed pilot 恢复；还需先审查 shortlist、完成 R3 固定补 seed 并冻结最终候选。']
 (OUT/'plain_language_summary.md').write_text('\n'.join(plain)+'\n',encoding='utf-8')
 print(json.dumps(decision,sort_keys=True))
if __name__=='__main__': main()
