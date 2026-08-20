#!/usr/bin/env python3
"""Open exactly one bounded recovery label window and run both evaluators."""
from __future__ import annotations
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
REPO=Path(os.environ.get('NIGHT8A_RECOVERY_REPO','/root/autodl-fs/SpaLORA-night8a-eval-recovery')); OUT=REPO/'outputs/night8a_eval_recovery'; RAW=Path('/root/autodl-fs/night8a_eval_recovery_20260820')
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def write(v):
 p=OUT/'recovery_label_window_audit.json'; q=p.with_suffix('.tmp'); q.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n'); os.replace(q,p)
def main():
 start=time.time(); audit={'schema_version':'night8a-eval-recovery-label-window-v1','status':'OPEN','window_count':1,'rule_lock_sha256':sha(OUT/'recovery_rule_lock.json'),'rule_lock_commit':'e0173e57df696679705930efd0455667e26fc6fd','opened_after_rule_lock_push':True,'training':0,'transform':0,'external_benchmark':0,'gpu_used':False,'authorized_roles':['night8a_eval_recovery_primary','night8a_eval_recovery_independent']}; write(audit)
 env=dict(os.environ,OMP_NUM_THREADS='3',MKL_NUM_THREADS='3',OPENBLAS_NUM_THREADS='3',CUDA_VISIBLE_DEVICES='')
 try:
  for script,log in [('night8a_eval_recovery_primary.py','primary.log'),('night8a_eval_recovery_independent.py','independent.log')]:
   remain=max(1,3600-(time.time()-start));
   with (RAW/log).open('wb') as f: subprocess.run([sys.executable,str(REPO/'scripts'/script)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=remain)
  independent=json.loads((RAW/'independent_recompute_audit.json').read_text())
  audit.update({'status':'CLOSED_PASS','closed':True,'wall_seconds':time.time()-start,'independent_status':independent['status'],'independent_max_abs_error':independent['max_abs_error'],'configuration_or_code_changes_after_labels':False})
 except Exception as e:
  audit.update({'status':'RECOVERY_SEMANTICS_INVALID','closed':True,'wall_seconds':time.time()-start,'error':repr(e),'configuration_or_code_changes_after_labels':False}); write(audit); raise
 write(audit); print(json.dumps(audit,sort_keys=True))
if __name__=='__main__': main()
