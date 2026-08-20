#!/usr/bin/env python3
"""Rehash the exact P0 file set without rewriting any original evidence."""
import hashlib,json,os
from pathlib import Path
REPO=Path(os.environ.get('NIGHT8A_RECOVERY_REPO','/root/autodl-fs/SpaLORA-night8a-eval-recovery')); OUT=REPO/'outputs/night8a_eval_recovery'; RAW=Path('/root/autodl-fs/night8a_eval_recovery_20260820')
def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 before=json.loads((OUT/'original_artifact_manifest_before.json').read_text()); rows=[]
 for old in before['files']:
  p=Path(old['path']); actual=sha(p) if p.is_file() else None; rows.append({**old,'actual_sha256':actual,'exists':p.is_file(),'size_bytes':p.stat().st_size if p.is_file() else None,'match':actual==old['expected_sha256']})
 failures=[x for x in rows if not x['match']]; after={'schema_version':'night8a-eval-recovery-original-artifacts-v1','status':'PASS' if not failures else 'RECOVERY_BLOCKED_ARTIFACT_MISMATCH','file_count':len(rows),'verified_count':len(rows)-len(failures),'failure_count':len(failures),'failures':failures,'files':rows}
 (OUT/'original_artifact_manifest_after.json').write_text(json.dumps(after,indent=2,sort_keys=True)+'\n'); (RAW/'original_artifact_manifest_after.json').write_bytes((OUT/'original_artifact_manifest_after.json').read_bytes())
 identical=len(rows)==len(before['files']) and all(a['path']==b['path'] and a['actual_sha256']==b['actual_sha256'] and a['size_bytes']==b['size_bytes'] for a,b in zip(rows,before['files']))
 audit={'schema_version':'night8a-eval-recovery-invariance-v1','status':'PASS' if identical and not failures else 'RECOVERY_BLOCKED_ARTIFACT_MISMATCH','before_files':len(before['files']),'after_files':len(rows),'byte_identical':identical,'failures':len(failures),'original_repo_head':'d09aa00b5e25269e66712dd47d01064b7c9422cf','original_night8a_status':'IMPLEMENTATION_SEMANTICS_INVALID'}
 (OUT/'original_tree_invariance_audit.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n'); print(json.dumps(audit,sort_keys=True));
 if audit['status']!='PASS': raise SystemExit(2)
if __name__=='__main__': main()
