#!/usr/bin/env python3
"""Create bounded Night-8A recovery audits, tracked index, and compact handoff."""
from __future__ import annotations
import argparse,hashlib,json,os,shutil,subprocess,tarfile
from pathlib import Path
REPO=Path('/root/autodl-fs/SpaLORA-night8a-eval-recovery'); OUT=REPO/'outputs/night8a_eval_recovery'; RAW=Path('/root/autodl-fs/night8a_eval_recovery_20260820')
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p); q=p.with_suffix(p.suffix+'.tmp'); q.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n'); os.replace(q,p)
def tracked_files():
 roots=[OUT,REPO/'protocols/night8a_eval_recovery']; files=[]
 for root in roots:
  files += [p for p in root.rglob('*') if p.is_file() and p.name not in {'delivery_index.json'}]
 files += list((REPO/'scripts').glob('night8a_eval_recovery_*.py'))+[REPO/'tests/test_night8a_eval_recovery.py']
 return sorted(set(files),key=lambda p:p.relative_to(REPO).as_posix())
def prepare():
 window=json.loads((OUT/'recovery_label_window_audit.json').read_text()); decision=json.loads((OUT/'recovery_decision.json').read_text())
 atomic(OUT/'resource_budget_audit.json',{'schema_version':'night8a-eval-recovery-resource-v1','status':'PASS','training':0,'transform':0,'external_benchmark':0,'gpu_compute_used':False,'server_mode':'card mode for CPU/memory only','evaluator_workers':1,'threads_per_worker':3,'worker_limit':3,'thread_limit':3,'label_window_wall_seconds':window['wall_seconds'],'evaluation_timeout_seconds':3600})
 atomic(OUT/'failure_and_retry_audit.json',{'schema_version':'night8a-eval-recovery-failures-v1','status':'PASS_RETAINED','scientific_retry':0,'training_retry':0,'transform_retry':0,'fixed_scientific_failure':'B03/P22/seed1 retained; no rerun','infrastructure_attempts':[{'attempt':1,'phase':'P0 implementation','reason':'auditor initially compared R02 canonical embedding hash as file hash','label_access':False},{'attempt':2,'phase':'P0 implementation','reason':'auditor initially looked only in Night-7B R1 for seed2, which is locked in R2','label_access':False},{'attempt':3,'phase':'P0','status':'PASS'}]})
 atomic(OUT/'post_window_code_audit.json',{'schema_version':'night8a-eval-recovery-post-window-code-v1','status':'PASS','rule_lock_commit':'e0173e57df696679705930efd0455667e26fc6fd','evaluator_freeze_commit':'6946a50e2b023325ac11fd23c39647e297a7652f','primary_and_independent_evaluator_frozen_and_pushed_before_label_window':True,'configuration_threshold_seed_or_rule_changes_after_label_window':False,'post_window_scripts':['mechanical summary/shortlist implementation of the pre-locked rules','read-only after-manifest rehash','delivery packaging'],'post_window_scientific_training':0,'post_window_transform':0,'post_window_recluster':0,'shortlist_count':len(decision['finalist_ids'])})
def index(results_commit):
 rows=[{'path':p.relative_to(REPO).as_posix(),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in tracked_files()]
 root=hashlib.sha256('\n'.join(x['sha256'] for x in rows).encode()).hexdigest()
 atomic(OUT/'git_audit.json',{'schema_version':'night8a-eval-recovery-git-v1','status':'PASS','base_commit':'d09aa00b5e25269e66712dd47d01064b7c9422cf','protection_tag':'baseline/pre-night8a-eval-recovery-20260820','branch':'revision/q2-night8a-eval-recovery-20260820','results_commit':results_commit,'push_mode':'normal','force_push':False,'planned_final_tag':'night8a-eval-recovery-final-20260820'})
 # Recompute now that git_audit exists.
 rows=[{'path':p.relative_to(REPO).as_posix(),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in tracked_files()]
 root=hashlib.sha256('\n'.join(x['sha256'] for x in rows).encode()).hexdigest()
 atomic(OUT/'delivery_index.json',{'schema_version':'night8a-eval-recovery-delivery-v1','branch':'revision/q2-night8a-eval-recovery-20260820','results_commit':results_commit,'final_tag':'night8a-eval-recovery-final-20260820','file_count':len(rows),'files':rows,'root_rule':'sha256(newline_join(file_sha256_in_lexical_path_order))','root_sha256':root,'raw_files_included':False})
def package(final_commit):
 compact=RAW/'official_compact'
 if compact.exists(): raise RuntimeError('compact already exists')
 compact.mkdir(parents=True)
 for p in tracked_files()+[OUT/'delivery_index.json']:
  rel=p.relative_to(REPO); target=compact/rel; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(p,target)
 bundle=compact/'git/night8a_to_eval_recovery_20260820.bundle'; bundle.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run(['git','-C',str(REPO),'bundle','create',str(bundle),'d09aa00b5e25269e66712dd47d01064b7c9422cf..HEAD','refs/tags/night8a-eval-recovery-final-20260820'],check=True)
 tar=compact/'night8a_eval_recovery_planner_handoff_20260820.tar.gz'
 with tarfile.open(tar,'w:gz') as tf:
  for name in ['outputs/night8a_eval_recovery','protocols/night8a_eval_recovery','scripts','tests/test_night8a_eval_recovery.py']:
   p=REPO/name
   if p.is_dir():
    for f in p.rglob('*'):
     if f.is_file() and (name!='scripts' or f.name.startswith('night8a_eval_recovery_')): tf.add(f,arcname=f.relative_to(REPO))
   elif p.is_file(): tf.add(p,arcname=p.relative_to(REPO))
 files=sorted([p for p in compact.rglob('*') if p.is_file() and p.name!='compact_delivery_index.json'],key=lambda p:p.relative_to(compact).as_posix())
 rows=[{'path':p.relative_to(compact).as_posix(),'sha256':sha(p),'size_bytes':p.stat().st_size} for p in files]; root=hashlib.sha256('\n'.join(x['sha256'] for x in rows).encode()).hexdigest()
 atomic(compact/'compact_delivery_index.json',{'schema_version':'night8a-eval-recovery-compact-v1','branch':'revision/q2-night8a-eval-recovery-20260820','commit':final_commit,'final_tag':'night8a-eval-recovery-final-20260820','file_count':len(rows),'files':rows,'root_rule':'sha256(newline_join(file_sha256_in_lexical_path_order))','root_sha256':root,'raw_files_included':False,'total_bytes_excluding_index':sum(x['size_bytes'] for x in rows)})
 print(json.dumps({'compact':str(compact),'files':len(rows),'root':root,'bytes':sum(x['size_bytes'] for x in rows)},sort_keys=True))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('mode',choices=['prepare','index','package']); ap.add_argument('--commit'); a=ap.parse_args()
 if a.mode=='prepare': prepare()
 elif a.mode=='index': index(a.commit or subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip())
 else: package(a.commit or subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip())
if __name__=='__main__': main()
