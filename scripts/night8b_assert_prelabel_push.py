#!/usr/bin/env python3
"""Assert that the total-lock commit is present on the ordinary remote branch."""
import json,subprocess,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(REPO))
from SpaLORA.night3af_cache import sha256_file
from SpaLORA.night6c_pipeline import atomic_json
OUT=REPO/"outputs/night8b_handoff"; BRANCH="revision/q2-night8b-misar-family-policy-external-20260820"
def git(*x): return subprocess.check_output(["git",*x],cwd=REPO,text=True).strip()
head=git("rev-parse","HEAD"); remote=git("ls-remote","origin",f"refs/heads/{BRANCH}").split()[0]
lock=OUT/"locked_misar_training_and_prediction_manifest.json"
if head!=remote or json.loads(lock.read_text()).get("status")!="TOTAL_LOCKED_BEFORE_LABEL_ACCESS":
    raise RuntimeError("ordinary prelabel push/total lock gate failed")
atomic_json(OUT/"prelabel_push_audit.json",{"status":"PASS","commit":head,"remote_commit":remote,
    "ordinary_push":True,"force_push":False,"lock_manifest_sha256":sha256_file(lock),"Y_values_read":False})
print(json.dumps({"status":"PASS","prelabel_commit":head}))
