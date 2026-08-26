"""Independent Windows verifier for an extracted Night-21C compact."""
from __future__ import annotations
import argparse,hashlib,json,platform,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

p=argparse.ArgumentParser(); p.add_argument("--compact",required=True); p.add_argument("--output",required=True); p.add_argument("--archive"); p.add_argument("--bundle"); a=p.parse_args()
root=Path(a.compact).resolve(); index_path=root/"compact_delivery_index.json"; index=json.loads(index_path.read_text(encoding="utf-8")); expected={item["path"]:item for item in index["files"]}
actual={path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.name!="compact_delivery_index.json"}
missing=sorted(set(expected)-actual); extras=sorted(actual-set(expected)); size=[]; hashes=[]
for relative,item in expected.items():
    path=root/relative
    if not path.is_file(): continue
    if path.stat().st_size!=item["bytes"]: size.append(relative)
    if sha(path)!=item["sha256"]: hashes.append(relative)
result={"schema":"night21c-windows-independent-verification-v2","platform":platform.platform(),"verified_at_unix":time.time(),"indexed":len(expected),"actual_indexed_payloads":len(actual),"missing":missing,"size_mismatches":size,"sha256_mismatches":hashes,"extras":extras,"index_sha256":sha(index_path),"status":"PASS" if not (missing or size or hashes or extras) else "FAIL","self_index_excluded":True,"verification_file_outside_index":True}
for field,value in (("archive",a.archive),("bundle",a.bundle)):
    if value:
        artifact=Path(value).resolve()
        if not artifact.is_file():
            result["status"]="FAIL"
            result[f"{field}_error"]="MISSING"
        else:
            result[f"{field}_bytes"]=artifact.stat().st_size
            result[f"{field}_sha256"]=sha(artifact)
Path(a.output).write_text(json.dumps(result,indent=2,sort_keys=True),encoding="utf-8")
print(json.dumps(result,indent=2,sort_keys=True))
if result["status"]!="PASS": raise SystemExit(2)
