#!/usr/bin/env python3
"""Capture the immutable Night 1 software, hardware, Git, and data audit."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse


DATASETS = {
    "a1": {
        "rna": "/root/autodl-fs/Human lymph node/A1/humanlymphnode_rna.h5ad",
        "modality2": "/root/autodl-fs/Human lymph node/A1/humanlymphnode_adt.h5ad",
        "ground_truth": "/root/autodl-fs/Human lymph node/A1/A1_groundtruth.csv",
    },
    "placenta": {
        "rna": "/root/autodl-fs/Human placenta architecture/humanplacenta_rna.h5ad",
        "modality2": "/root/autodl-fs/Human placenta architecture/humanplacenta_atac.h5ad",
        "ground_truth": "obs[cell_type]",
    },
    "p22": {
        "rna": "/root/autodl-fs/P22 mouse brain coronal section/mousebrain_rna.h5ad",
        "modality2": "/root/autodl-fs/P22 mouse brain coronal section/mousebrain_atac.h5ad",
        "ground_truth": "/root/autodl-fs/P22 mouse brain coronal section/MouseBrain_groundtruth.csv",
    },
}


def command(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT).strip()


def sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matrix_audit(matrix) -> dict:
    if sparse.issparse(matrix):
        values = matrix.data
        storage = matrix.getformat()
    else:
        values = np.asarray(matrix).ravel()
        storage = "dense"
    sample = values[: min(values.size, 1_000_000)]
    return {
        "storage": storage,
        "dtype": str(matrix.dtype),
        "nonzero_values": int(values.size),
        "sample_min": None if sample.size == 0 else float(np.nanmin(sample)),
        "sample_max": None if sample.size == 0 else float(np.nanmax(sample)),
        "sample_integer_like": bool(sample.size == 0 or np.allclose(sample, np.rint(sample))),
    }


def h5ad_audit(path: str) -> dict:
    obj = ad.read_h5ad(path, backed="r")
    result = {
        "path": path,
        "sha256": sha256(path),
        "shape": [int(obj.n_obs), int(obj.n_vars)],
        "obs_names_unique": bool(obj.obs_names.is_unique),
        "var_names_unique": bool(obj.var_names.is_unique),
        "obs_columns": list(map(str, obj.obs.columns)),
        "obsm_keys": list(map(str, obj.obsm.keys())),
    }
    obj.file.close()
    loaded = ad.read_h5ad(path)
    result["x"] = matrix_audit(loaded.X)
    return result


def csv_label_audit(path: str) -> dict:
    table = pd.read_csv(path)
    label_col = table.columns[-1]
    id_col = table.columns[0]
    return {
        "path": path,
        "sha256": sha256(path),
        "rows": int(table.shape[0]),
        "id_column": str(id_col),
        "label_column": str(label_col),
        "class_counts": {str(k): int(v) for k, v in table[label_col].value_counts().sort_index().items()},
    }


def package_versions(executable: str) -> dict:
    code = (
        "import importlib, json, sys; "
        "mods=['torch','numpy','pandas','scipy','sklearn','scanpy','anndata','rpy2']; "
        "out={'python':sys.version.split()[0]}; "
        "[(out.__setitem__(m, getattr(importlib.import_module(m),'__version__','unknown'))) "
        "if importlib.util.find_spec(m) else out.__setitem__(m,None) for m in mods]; "
        "print(json.dumps(out))"
    )
    return json.loads(command(executable, "-c", code))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default=".")
    parser.add_argument("--output", default="reports/night1_environment.json")
    args = parser.parse_args()
    repo = Path(args.repo).resolve()

    audited = {}
    for name, cfg in DATASETS.items():
        rna = ad.read_h5ad(cfg["rna"])
        mod2 = ad.read_h5ad(cfg["modality2"])
        entry = {
            "rna": h5ad_audit(cfg["rna"]),
            "modality2": h5ad_audit(cfg["modality2"]),
            "paired_obs_identical": bool(rna.obs_names.equals(mod2.obs_names)),
            "spatial_coordinates_identical": bool(
                "spatial" in rna.obsm and "spatial" in mod2.obsm
                and np.array_equal(rna.obsm["spatial"], mod2.obsm["spatial"])
            ),
        }
        if name == "placenta":
            counts = rna.obs["cell_type"].astype(str).value_counts().sort_index()
            entry["ground_truth"] = {
                "source": "rna.obs[cell_type]",
                "classes": int(counts.size),
                "class_counts": {str(k): int(v) for k, v in counts.items()},
            }
        else:
            entry["ground_truth"] = csv_label_audit(cfg["ground_truth"])
            gt = pd.read_csv(cfg["ground_truth"])
            data_ids = set(map(str, rna.obs_names))
            gt_ids = set(map(str, gt.iloc[:, 0]))
            if name == "a1":
                data_ids = {x[3:] if x.startswith("s1-") else x for x in data_ids}
            entry["evaluation_alignment"] = {
                "matched": len(data_ids & gt_ids),
                "data_only": len(data_ids - gt_ids),
                "ground_truth_only": len(gt_ids - data_ids),
                "a1_rule": "strip leading s1- from h5ad observation IDs" if name == "a1" else None,
            }
        audited[name] = entry

    gpu_fields = command(
        "nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"
    ).split(", ")
    output = {
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "host": {"hostname": platform.node(), "platform": platform.platform()},
        "git": {
            "branch": command("git", "-C", str(repo), "branch", "--show-current"),
            "commit": command("git", "-C", str(repo), "rev-parse", "HEAD"),
            "status_porcelain": command("git", "-C", str(repo), "status", "--porcelain"),
            "baseline_tag": "baseline/pre-q2-revision-20260808",
            "original_worktree_preservation": "/root/autodl-fs/night1_preexisting_20260808",
        },
        "gpu": {
            "name": gpu_fields[0],
            "memory_mib": int(gpu_fields[1]),
            "driver": gpu_fields[2],
            "host_cuda": command("nvidia-smi").split("CUDA Version:", 1)[1].split()[0],
        },
        "environments": {
            "corrected": {
                "path": "/root/miniconda3/envs/SpaLORA",
                **package_versions("/root/miniconda3/envs/SpaLORA/bin/python"),
            },
            "legacy": {
                "path": "/root/miniconda3/envs/SpaLORA_torch112",
                **package_versions("/root/miniconda3/envs/SpaLORA_torch112/bin/python"),
            },
            "r": command("/root/miniconda3/envs/SpaLORA/bin/R", "--version").splitlines()[0],
            "mclust": command(
                "/root/miniconda3/envs/SpaLORA/bin/Rscript",
                "-e",
                "cat(as.character(packageVersion('mclust')))",
            ),
        },
        "datasets": audited,
    }

    target = repo / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
