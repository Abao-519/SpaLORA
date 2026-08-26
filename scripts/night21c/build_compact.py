"""Build a compact Night-21C delivery from sealed repo and reproducibility artifacts."""
from __future__ import annotations
import hashlib,json,shutil,tarfile
from pathlib import Path

REPO=Path("/root/SpaLORA-night16h")
WORK=Path("/root/night21c_working")
DELIVERY=Path("/root/night21c_delivery_20260826")
COMPACT=DELIVERY/"official_compact"

def sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def copy_file(source,relative):
    target=COMPACT/relative; target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,target)

DELIVERY.mkdir(parents=True,exist_ok=True)
if COMPACT.exists(): raise RuntimeError(f"compact already exists: {COMPACT}")
archive=DELIVERY/"night21c_official_compact.tar.gz"
if archive.exists(): raise RuntimeError(f"archive already exists: {archive}")
COMPACT.mkdir(parents=True)
repo_paths=[
    "SpaLORA/night21c_official_math.py","configs/night21c","scripts/night21c","tests/test_night21c_official_math.py",
    "third_party/night21c_spamgcn_fixed","outputs/night21c_handoff",
]
for relative in repo_paths:
    source=REPO/relative; target=COMPACT/relative
    if source.is_dir(): shutil.copytree(source,target)
    else: copy_file(source,relative)
for folder in ("p0","formal","replays","endpoint_banks","evaluations","probes"):
    source=WORK/folder
    if source.exists(): shutil.copytree(source,COMPACT/"working"/folder)
bundle=DELIVERY/"night21c-final-20260826.incremental.bundle"
if not bundle.is_file(): raise RuntimeError("incremental bundle must exist before compact build")
copy_file(bundle,"git/night21c-final-20260826.incremental.bundle")
entries=[]
for path in sorted(COMPACT.rglob("*")):
    if path.is_file(): entries.append({"path":path.relative_to(COMPACT).as_posix(),"bytes":path.stat().st_size,"sha256":sha(path)})
index={"schema":"night21c-root-relative-compact-index-v1","indexed_file_count":len(entries),"excluded_self":["compact_delivery_index.json"],"files":entries}
index_path=COMPACT/"compact_delivery_index.json"; index_path.write_text(json.dumps(index,indent=2,sort_keys=True),encoding="utf-8")
with tarfile.open(archive,"w:gz") as tar: tar.add(COMPACT,arcname="official_compact")
verification={"schema":"night21c-remote-compact-build-v1","indexed":len(entries),"index_sha256":sha(index_path),"archive_sha256":sha(archive),"archive_bytes":archive.stat().st_size,"bundle_sha256":sha(bundle),"bundle_bytes":bundle.stat().st_size}
(DELIVERY/"remote_delivery_build.json").write_text(json.dumps(verification,indent=2,sort_keys=True),encoding="utf-8")
print(json.dumps(verification,indent=2,sort_keys=True))
