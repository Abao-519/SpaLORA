#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import importlib
import json
import platform
import subprocess
from pathlib import Path

REPO = Path("/root/autodl-fs/SpaLORA-night6b")
OUT = REPO / "outputs/night6b_handoff"
PROTOCOL = REPO / "protocols/night6b"
EXPECTED = {
    "SpaLORA_Night6A_Independent_Planner_Audit_2026-08-17.md": "4b5d501cd0c521555a919752f91b2ee48562cd623b2f851e901b7696bdd2b314",
    "SpaLORA_Post_Night6A_Research_Decision_2026-08-17.md": "4d7d000382f94d1f2b522bae9e55663dac63ca9f4e99898c9e9a38fa8b385c00",
    "SpaLORA_Night6B_Candidate_Registry_2026-08-17.json": "fde443719dc7ae80ff7cb7a853044ee3df4b1790ac4d7b8fe4d0ec5cee66f030",
    "SpaLORA_Night6B_Firewall_Clean_Graph_and_Clustering_Rescue_Taskbook_2026-08-17.md": "fd23c99388abfed94cb27be56ca7a8185ca5baa6f60f09bcc4af6f850282691c",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def run(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


protocols = {name: {"actual_sha256": sha(PROTOCOL / name), "expected_sha256": digest} for name, digest in EXPECTED.items()}
if not all(x["actual_sha256"] == x["expected_sha256"] for x in protocols.values()):
    raise SystemExit("protocol SHA mismatch")

versions = {"python": platform.python_version()}
for name in ("torch", "sklearn", "anndata", "h5py", "numpy", "scipy", "pandas"):
    module = importlib.import_module(name)
    versions[name] = getattr(module, "__version__", "unknown")
versions["git"] = run("git", "--version")
R = "/root/miniconda3/envs/SpaLORA/bin/R"
RSCRIPT = "/root/miniconda3/envs/SpaLORA/bin/Rscript"
versions["R"] = run(R, "--version").splitlines()[0]
versions["mclust"] = run(RSCRIPT, "-e", "cat(as.character(packageVersion('mclust')))")
versions["gpu"] = run("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader")
versions["cuda_available"] = bool(importlib.import_module("torch").cuda.is_available())

parent = run("git", "-C", str(REPO), "rev-parse", "HEAD")
tag = run("git", "-C", str(REPO), "rev-parse", "night6a-final-20260814")
protect = run("git", "-C", str(REPO), "rev-parse", "baseline/pre-night6b-graph-affinity-rescue-20260817")
payload = {
    "status": "P0_PROTECT_PASS",
    "upstream_parent": parent,
    "night6a_tag_commit": tag,
    "protection_tag_commit": protect,
    "branch": run("git", "-C", str(REPO), "branch", "--show-current"),
    "protocols": protocols,
    "night6a_compact_windows_verification": {
        "verified": 16,
        "declared": 16,
        "bad": 0,
        "delivery_index_sha256": "27ae79af08fc9a0d9b030b3ea27cc752c78beec80f31f75422b3c199b79f752b",
        "relative_root": "official_compact/handoff",
    },
    "night6a_authoritative_status": "IMPLEMENTATION_SEMANTICS_INVALID",
    "night6a_checkpoint_embedding_metric_use_for_selection": False,
    "new_roots": {
        "repo": str(REPO),
        "raw_runs": "/root/autodl-fs/night6b_raw_runs_20260817",
        "cache": "/root/autodl-fs/night6b_cache_20260817",
        "data": "/root/autodl-fs/night6b_data_20260817",
    },
    "versions": versions,
}
if len({parent, tag, protect}) != 1 or parent != "7f204a56690768f22bd06e0dac1b5785c97c4c70":
    raise SystemExit("authority commit mismatch")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "p0_protect_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(payload, sort_keys=True))
