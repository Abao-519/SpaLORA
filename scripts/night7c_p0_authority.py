#!/usr/bin/env python3
"""Night-7C authority, immutable-input, GPU, and firewall preflight."""
from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import sklearn
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night6c_pipeline import array_sha, sparse_sha  # noqa: E402
from SpaLORA.night7a_consensus import canonical_partition, sha256_file  # noqa: E402


RAW = Path("/root/autodl-fs/night7b_score_rnd_20260818")
COMPACT = RAW / "official_compact"
HANDOFF = COMPACT / "handoff"
OUT = REPO / "outputs/night7c_handoff"
PROTOCOL = REPO / "protocols/night7c"
PARENT = "32d6ed947b313423805ee0f80c9dada06bb6a28d"
PARENT_TAG = "night7b-final-20260818"
COMPACT_INDEX_SHA = "032da4baf8bde8f3c7abbfaebf6830c3546a21618c1c32d0d5309b34bff93626"
COMPACT_ROOT_SHA = "eb52cc09e58722fb0a0f7c083d4099b74e526afbf71f1af750244194f3136eb0"
AUTHORITY_SHA = {
    "SpaLORA_Night7C_Planning_Delivery_Index_2026-08-18.json":
        "a0aad701b405f5d60084c4d0d459cdc6d6b381105860227e25bdb4e73653a2b6",
    "SpaLORA_Night7B_Independent_Audit_and_Night7C_Decision_2026-08-18.md":
        "df8fa4ade3543f0816c4e67de7f7a6e12c3b93188072e26895412c2605179e7d",
    "SpaLORA_Night7C_Source_Code_Transfer_Audit_2026-08-18.md":
        "3cc2817d7dbf3dec5516dc39a422498d73eb6c03fcceb345f342c338a89a63aa",
    "SpaLORA_Night7C_Conflict_Gated_Affinity_Registry_2026-08-18.json":
        "1a13c62e11daf1e8357c3755368d13c115c1a1e679d9a8500741610d2f985b20",
    "SpaLORA_Night7C_Conflict_Gated_Routing_and_Runtime_Acceleration_Taskbook_2026-08-18.md":
        "24b494b099d605658a132a74ff53cfe70c999c06251b979b8d502cd04983414e",
    "README_FIRST_Night7C_Experimental_Codex_Prompt_2026-08-18.md":
        "c5d61e8d52350d187cc6bd5f3135c89860810061b7a439e2d0487b9fb33ca925",
}


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def compact_audit() -> dict:
    index_path = COMPACT / "compact_delivery_index.json"
    require(digest(index_path) == COMPACT_INDEX_SHA, "Night-7B compact index SHA mismatch")
    index = json.loads(index_path.read_text())
    failures = []
    for row in index["files"]:
        path = COMPACT / row["path"]
        if (not path.is_file() or path.stat().st_size != int(row["size_bytes"])
                or digest(path) != row["sha256"]):
            failures.append(row["path"])
    root = hashlib.sha256("\n".join(
        row["sha256"] for row in sorted(index["files"], key=lambda x: x["path"])
    ).encode()).hexdigest()
    require(not failures, "Night-7B compact member mismatch: %r" % failures)
    require(root == COMPACT_ROOT_SHA == index["root_sha256"], "Night-7B compact root mismatch")
    return {"verified": len(index["files"]), "expected": 67, "root_sha256": root}


def protocol_audit() -> dict:
    observed = {}
    for name, expected in AUTHORITY_SHA.items():
        path = PROTOCOL / name
        value = digest(path)
        require(value == expected, "authority SHA mismatch: %s" % name)
        observed[name] = value
    planning = json.loads((PROTOCOL / "SpaLORA_Night7C_Planning_Delivery_Index_2026-08-18.json").read_text())
    for row in planning["files"]:
        path = PROTOCOL / row["path"]
        require(path.stat().st_size == int(row["size_bytes"]), "planning size mismatch: %s" % row["path"])
        require(digest(path) == row["sha256"], "planning SHA mismatch: %s" % row["path"])
    return {"verified": len(observed), "sha256": observed}


def git_audit() -> dict:
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True).strip()
    head = git("rev-parse", "HEAD")
    parent = git("rev-parse", PARENT_TAG + "^{}")
    status = git("status", "--short", "--branch")
    require(head == PARENT, "worktree is not at authoritative parent")
    require(parent == PARENT, "parent tag peel mismatch")
    require(status.splitlines()[0].strip() == "## revision/q2-night7c-conflict-gated-mnn-rnd-20260818",
            "Night-7C branch mismatch")
    return {"head": head, "tag_peel": parent, "status_header": status.splitlines()[0]}


def source_audit() -> dict:
    rows = list(csv.DictReader((HANDOFF / "source_unit_index.csv").open(newline="")))
    require(len(rows) == 30, "source unit cardinality mismatch")
    adapter = json.loads((HANDOFF / "adapter_input_lock.json").read_text())
    require(adapter["status"] == "LOCKED_PRE_LABEL" and not adapter["label_access"],
            "adapter lock is not pre-label")
    require(len(adapter["units"]) == 30, "adapter input cardinality mismatch")
    by_unit = {row["unit_id"]: row for row in rows}
    verified_files = 0
    view_shapes = {}
    for lock in adapter["units"]:
        row = by_unit[lock["unit_id"]]
        path = Path(lock["worker_input_path"])
        require(digest(path) == lock["worker_input_sha256"], "adapter worker SHA mismatch")
        contract = json.loads(path.read_text())
        require(set(contract) == {"unit_id", "K", "observation_count", "ordered_observation_sha256",
                                  "g00_views", "g04_views", "observation_ids", "s00", "s04",
                                  "pseudo_partition", "pseudo_affinity"},
                "adapter worker schema is not fail-closed")
        require(not any(str(v).lower().endswith(".h5ad") for v in contract.values()),
                "adapter worker contains h5ad path")
        require(digest(Path(contract["g00_views"])) == row["g00_views_sha256"], "G00 view SHA mismatch")
        require(digest(Path(contract["g04_views"])) == row["g04_views_sha256"], "G04 view SHA mismatch")
        require(sparse_sha(sp.load_npz(contract["s00"])) == row["s00_sha256"], "S00 canonical SHA mismatch")
        require(sparse_sha(sp.load_npz(contract["s04"])) == row["s04_sha256"], "S04 canonical SHA mismatch")
        ids = np.asarray([x.strip() for x in Path(contract["observation_ids"]).read_text().splitlines() if x.strip()])
        ordered_sha = hashlib.sha256("\n".join(ids.tolist()).encode("utf-8")).hexdigest()
        require(ordered_sha == row["ordered_observation_sha256"], "observation order SHA mismatch")
        for key in ("g00_views", "g04_views"):
            with np.load(contract[key], allow_pickle=False) as value:
                require(set(value.files) == {"emb_latent_omics1", "emb_latent_omics2", "SpaLORA_fused",
                                             "alpha_omics1", "alpha_omics2", "alpha_cross"},
                        "six-view schema mismatch")
                require(all(value[name].shape[0] == len(ids) for name in value.files),
                        "six-view observation mismatch")
                view_shapes[lock["unit_id"] + "/" + key] = {name: list(value[name].shape) for name in value.files}
        require(array_sha(np.load(contract["pseudo_partition"], allow_pickle=False).astype(np.int64))
                == lock["pseudo_partition_sha256"], "pseudo partition SHA mismatch")
        require(sparse_sha(sp.load_npz(contract["pseudo_affinity"]))
                == lock["pseudo_affinity_sha256"], "pseudo affinity SHA mismatch")
        verified_files += 9
    return {"units": 30, "verified_artifact_checks": verified_files,
            "six_view_archives": 60, "example_shapes": view_shapes["u000/g00_views"]}


def training_and_specialist_audit() -> dict:
    locked = {}
    training = []
    transforms = []
    for stage in ("R1", "R2"):
        value = json.loads((HANDOFF / ("locked_%s_manifest.json" % stage)).read_text())
        require(value["status"] == "LOCKED_PRE_LABEL" and not value["label_access"],
                "%s is not locked pre-label" % stage)
        locked[stage] = value
        for row in value["training_cells"]:
            training.append((stage, row))
        for row in value["transforms"]:
            transforms.append((stage, row))
    require(len(training) == 124, "formal training cardinality mismatch")
    gpu_values, file_checks = [], 0
    train_lookup = {}
    for stage, cell in training:
        require(cell["status"] == "success" and cell["scientific_training"], "invalid training cell")
        require(not cell["label_access"] and not cell["fallback"] and not cell["retry"],
                "training firewall/retry mismatch")
        manifest = cell["training_manifest"]
        require(manifest["gpu_model"] == "NVIDIA GeForce RTX 4080", "historical GPU model mismatch")
        require(float(manifest["peak_gpu_mib"]) > 0, "historical GPU peak is not positive")
        require(cell["reload_audit"]["status"] == "PASS" and cell["reload_audit"]["fresh_process"],
                "checkpoint reload did not pass")
        require(cell["reload_audit"]["embedding_exact"] and cell["reload_audit"]["gate_exact"],
                "checkpoint output is not exact")
        checkpoint = Path(manifest["checkpoint_path"])
        worker = checkpoint.parent
        checks = [
            (Path(cell["config_path"]), cell["config_sha256"]),
            (worker / "training_manifest.json", cell["training_manifest_sha256"]),
            (checkpoint, manifest["checkpoint_sha256"]),
            (Path(manifest["embedding_path"]), sha256_file(Path(manifest["embedding_path"]))),
            (Path(manifest["gate_path"]), sha256_file(Path(manifest["gate_path"]))),
            (worker / "loss_curve.csv", manifest["loss_curve_sha256"]),
            (worker / "fixed_indices.json", manifest["fixed_indices_sha256"]),
            (worker / "reload_forward_audit.json", cell["reload_audit_sha256"]),
        ]
        # Array fields use canonical SHA, not byte SHA.
        require(array_sha(np.load(manifest["embedding_path"], allow_pickle=False)) == manifest["embedding_sha256"],
                "training embedding canonical SHA mismatch")
        require(array_sha(np.load(manifest["gate_path"], allow_pickle=False)) == manifest["gate_sha256"],
                "training gate canonical SHA mismatch")
        for path, expected in checks:
            require(path.is_file(), "missing training artifact: %s" % path)
            if path.name not in {"embedding.npy", "gate_weights.npy"}:
                require(digest(path) == expected, "training artifact SHA mismatch: %s" % path)
            file_checks += 1
        gpu_values.append(float(manifest["peak_gpu_mib"]))
        train_lookup[(stage, cell["recipe_id"], cell["unit_id"])] = cell

    specialist = [x for x in transforms if x[1]["recipe_id"] in {"R02", "R08"}
                  and x[1]["endpoint"] == "E1_ADAPTER_C06_MEAN"
                  and x[1]["head_id"] in {"H01", "H02"}]
    require(len(specialist) == 120, "specialist transform cardinality mismatch")
    specialist_lookup = {}
    for stage, row in specialist:
        require(row["status"] == "success" and not row["label_access"]
                and not row["fallback"] and not row["retry"], "specialist transform invalid")
        train = train_lookup[(stage, row["recipe_id"], row["unit_id"])]
        attempt = Path(train["training_manifest"]["checkpoint_path"]).parents[1]
        target = attempt / "transforms" / row["endpoint"] / row["head_id"]
        affinity = target / "affinity.npz"
        clusters = target / "clusters.csv"
        reload_audit = target / "fresh_transform_reload_audit.json"
        require(digest(affinity) == row["affinity_file_sha256"], "specialist affinity file SHA mismatch")
        require(digest(clusters) == row["clusters_file_sha256"], "specialist cluster file SHA mismatch")
        require(digest(reload_audit) == row["fresh_transform_reload_audit_sha256"],
                "specialist reload audit SHA mismatch")
        require(sparse_sha(sp.load_npz(affinity)) == row["canonical_affinity_sha256"],
                "specialist canonical affinity SHA mismatch")
        labels = pd.read_csv(clusters)["cluster"].to_numpy(dtype=np.int64)
        require(array_sha(canonical_partition(labels)) == row["canonical_partition_sha256"],
                "specialist canonical partition SHA mismatch")
        specialist_lookup[(row["recipe_id"], row["unit_id"], row["head_id"])] = row
    duplicate_equal = 0
    for recipe in ("R02", "R08"):
        for unit in sorted({x[1]["unit_id"] for x in specialist if x[1]["recipe_id"] == recipe}):
            a = specialist_lookup[(recipe, unit, "H01")]
            b = specialist_lookup[(recipe, unit, "H02")]
            require(a["canonical_affinity_sha256"] == b["canonical_affinity_sha256"],
                    "H01/H02 affinity differs")
            require(a["canonical_partition_sha256"] == b["canonical_partition_sha256"],
                    "H01/H02 partition differs")
            duplicate_equal += 1
    return {
        "training_cells": len(training), "checkpoint_reload_pass": len(training),
        "gpu_model_exact": len(gpu_values), "positive_peak_gpu": len(gpu_values),
        "peak_gpu_mib_min": min(gpu_values), "peak_gpu_mib_max": max(gpu_values),
        "training_artifact_file_checks": file_checks,
        "specialist_transforms": len(specialist), "H01_H02_exact_pairs": duplicate_equal,
    }


def current_environment() -> dict:
    require(torch.cuda.is_available(), "current CUDA is unavailable")
    torch.cuda.reset_peak_memory_stats()
    model = torch.nn.Linear(4, 3).cuda()
    x = torch.arange(20, dtype=torch.float32, device="cuda").reshape(5, 4)
    loss = model(x).square().mean()
    loss.backward()
    require(next(model.parameters()).is_cuda and x.is_cuda and loss.is_cuda,
            "CUDA smoke tensor placement mismatch")
    r_version = subprocess.check_output([
        "bash", "-lc",
        "export R_HOME=/root/miniconda3/envs/SpaLORA/lib/R; "
        "/root/miniconda3/envs/SpaLORA/bin/Rscript -e \"cat(as.character(packageVersion('mclust')))\"",
    ], text=True).strip()
    try:
        import igraph
        igraph_version = igraph.__version__
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("igraph unavailable") from exc
    memory_kib = 0
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemTotal:"):
            memory_kib = int(line.split()[1])
    return {
        "python": platform.python_version(), "torch": torch.__version__,
        "torch_cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0),
        "cuda_smoke": {"model": str(next(model.parameters()).device), "views": str(x.device),
                       "loss": str(loss.device),
                       "peak_gpu_mib": torch.cuda.max_memory_allocated() / 1048576.0},
        "cpu_count": os.cpu_count(), "ram_gib": memory_kib / 1048576.0,
        "scipy": scipy.__version__, "sklearn": sklearn.__version__,
        "igraph": igraph_version, "r_mclust": r_version,
    }


def static_firewall_audit() -> dict:
    source = (REPO / "SpaLORA/night7c_conflict.py").read_text()
    tree = ast.parse(source)
    forbidden = {"dataset", "tissue", "organism", "platform", "technology", "modality",
                 "file_name", "ground_truth", "ari", "nmi", "metric"}
    hits = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.If, ast.IfExp)):
            for part in ast.walk(node.test):
                if isinstance(part, ast.Constant) and isinstance(part.value, str):
                    lower = part.value.lower()
                    if any(token in lower for token in forbidden):
                        hits.append({"line": node.lineno, "value": part.value})
    require(not hits, "identity/evaluation branch detected: %r" % hits)
    trainer = (REPO / "scripts/night7b_train.py").read_text()
    require("anndata" not in trainer and "read_h5ad" not in trainer,
            "base trainer can deserialize original h5ad")
    return {"identity_branch_hits": hits, "base_trainer_imports_anndata": False,
            "prelock_original_h5ad_allowed": False, "label_access": False}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1, "status": "PASS", "label_access": False,
        "authority": protocol_audit(), "compact": compact_audit(), "git": git_audit(),
        "source": source_audit(), "historical": training_and_specialist_audit(),
        "environment": current_environment(), "firewall_static": static_firewall_audit(),
        "budgets": {"routing_new_transforms": 240, "weighted_mnn_pilot_training": 48,
                    "weighted_mnn_pilot_transforms": 48, "scientific_retry": 0,
                    "implementation_corrections_max": 8, "total_training_attempts_max": 56},
    }
    target = OUT / "p0_authority_audit.json"
    target.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "PASS", "output": str(target),
                      "training_cells": report["historical"]["training_cells"],
                      "specialist_transforms": report["historical"]["specialist_transforms"],
                      "six_view_archives": report["source"]["six_view_archives"],
                      "gpu": report["environment"]["gpu"]}, sort_keys=True))


if __name__ == "__main__":
    main()
