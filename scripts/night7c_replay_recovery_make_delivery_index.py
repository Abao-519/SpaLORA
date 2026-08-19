#!/usr/bin/env python3
"""Write the non-self-referential tracked handoff delivery index."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO))
OUT=REPO/"outputs/night7c_replay_recovery_handoff"


def sha(path:Path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1048576),b""): h.update(b)
    return h.hexdigest()


def main():
    target=OUT/"delivery_index.json"
    files=[]
    for path in sorted(p for p in OUT.rglob("*") if p.is_file() and p!=target):
        files.append({"path":path.relative_to(REPO).as_posix(),"sha256":sha(path),"size_bytes":path.stat().st_size})
    root=hashlib.sha256("\n".join(x["sha256"] for x in sorted(files,key=lambda x:x["path"])).encode()).hexdigest()
    value={"schema_version":1,"status":json.loads((OUT/"gate_audit.json").read_text())["terminal_status"],
           "file_count":len(files),"files":files,"root_rule":"sha256(newline_join(file_sha256_in_lexical_path_order))",
           "root_sha256":root,"self_referential_fields":False,"raw_files_included":False}
    target.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
    print(json.dumps({"file_count":len(files),"root_sha256":root,"index_sha256":sha(target)},sort_keys=True))


if __name__=="__main__": main()
