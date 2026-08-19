#!/usr/bin/env python3
"""Create the <25 MiB official compact handoff after the immutable final tag."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
RAW=Path("/root/autodl-fs/night7c_replay_recovery_20260818")
DEST=RAW/"official_compact"
BRANCH="revision/q2-night7c-replay-portability-recovery-20260818"
TAG="night7c-replay-recovery-final-20260818"


def sha(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()


def copy(rel:str):
    src=REPO/rel; dst=DEST/rel
    if src.is_dir(): shutil.copytree(src,dst)
    else: dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)


def main():
    if DEST.exists(): raise RuntimeError("official compact destination exists")
    DEST.mkdir(parents=True)
    for rel in ["outputs/night7c_replay_recovery_handoff","protocols/night7c_replay_recovery","protocols/night7c",
                "SpaLORA/night7c_conflict.py","SpaLORA/night7c_firewall.py"]:
        copy(rel)
    for path in sorted((REPO/"tests").glob("test_night7c*.py")):
        dst=DEST/"tests"/path.name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dst)
    for path in sorted((REPO/"scripts").glob("night7c*.py")):
        dst=DEST/"scripts"/path.name; dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,dst)
    copy("scripts/w00_boundary_guard.py")
    bundle=DEST/"git/night7c_to_replay_recovery_20260819.bundle"; bundle.parent.mkdir(parents=True)
    subprocess.check_call(["git","bundle","create",str(bundle),"^night7c-final-20260818",BRANCH,TAG],cwd=REPO)
    files=[]; index=DEST/"compact_delivery_index.json"
    for p in sorted(x for x in DEST.rglob("*") if x.is_file() and x!=index):
        files.append({"path":p.relative_to(DEST).as_posix(),"sha256":sha(p),"size_bytes":p.stat().st_size})
    root=hashlib.sha256("\n".join(x["sha256"] for x in sorted(files,key=lambda x:x["path"])).encode()).hexdigest()
    value={"schema_version":1,"branch":BRANCH,"commit":subprocess.check_output(["git","rev-parse",BRANCH],cwd=REPO,text=True).strip(),
           "final_tag":TAG,"final_tag_peel":subprocess.check_output(["git","rev-parse",TAG+"^{}"],cwd=REPO,text=True).strip(),
           "file_count":len(files),"files":files,"root_rule":"sha256(newline_join(file_sha256_in_lexical_path_order))",
           "root_sha256":root,"raw_files_included":False}
    index.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    total=sum(p.stat().st_size for p in DEST.rglob("*") if p.is_file())
    if total>=25*1024*1024: raise RuntimeError(f"compact exceeds 25 MiB: {total}")
    print(json.dumps({"index_sha256":sha(index),"root_sha256":root,"file_count":len(files),"total_size_bytes":total,"bundle_sha256":sha(bundle)},sort_keys=True))


if __name__=="__main__": main()
