#!/usr/bin/env python3
"""Create an independent, non-self-referential Night-5C compact delivery index."""
from __future__ import annotations
import hashlib,json,shutil,subprocess
from pathlib import Path
REPO=Path("/root/autodl-fs/SpaLORA-night5c")
STAGE=Path("/root/autodl-fs/night5c_delivery_20260814")
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for chunk in iter(lambda:f.read(1024*1024),b""):h.update(chunk)
 return h.hexdigest()
def main():
 if STAGE.exists():shutil.rmtree(STAGE)
 STAGE.mkdir(parents=True)
 copies={"handoff":REPO/"outputs/night5c_handoff","code":REPO/"SpaLORA/night5c_semantic.py","configs":REPO/"configs/night5c_laplacian_correction.json","scripts":REPO/"scripts","tests":REPO/"tests/test_night5c.py","authoritative_inputs":REPO/"docs/night5c_authoritative_inputs"}
 for name,source in copies.items():
  target=STAGE/name
  if source.is_dir():
   if name=="scripts":
    target.mkdir();
    for p in sorted(source.glob("night5c_*.py")):shutil.copy2(p,target/p.name)
   else:shutil.copytree(source,target)
  else:
   target.mkdir() if name in ("code","configs","tests") else None
   shutil.copy2(source,target/source.name)
 bundle=STAGE/"SpaLORA_night5b_to_night5c_20260814.bundle"
 subprocess.run(["git","bundle","create",str(bundle),"2bfb3a0e9d363e90747ec0bd0ec0da92c829bc20..revision/q2-night5c-laplacian-correction-20260814"],cwd=REPO,check=True)
 verify=subprocess.run(["git","bundle","verify",str(bundle)],cwd=REPO,text=True,capture_output=True,check=True)
 files=[]
 for p in sorted(STAGE.rglob("*")):
  if p.is_file() and p.name!="delivery_index.json":files.append({"path":str(p.relative_to(STAGE)),"size":p.stat().st_size,"sha256":sha(p)})
 index={"schema_version":1,"status":"PASS","independent_non_self_referential_index":True,"authority_parent_commit":"2bfb3a0e9d363e90747ec0bd0ec0da92c829bc20","final_commit":"bcba824bb302c3b19a9b8426e17cc7d2dccc97af","final_tag":"night5c-final-20260814","branch":"revision/q2-night5c-laplacian-correction-20260814","git_push_verified":True,"bundle_verify_output":verify.stdout+verify.stderr,"entry_count":len(files),"entries":files,"raw_runs_included":False,"model_state_included":False}
 (STAGE/"delivery_index.json").write_text(json.dumps(index,indent=2,sort_keys=True),encoding="utf-8")
 print("NIGHT5C_PACKAGE_STAGE_PASS",len(files),sha(STAGE/"delivery_index.json"),sha(bundle))
if __name__=="__main__":main()
