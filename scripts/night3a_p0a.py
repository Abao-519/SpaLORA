#!/usr/bin/env python3
"""Night-3A P0A input, corrected-sparse preprocessing, and hash-lock audit."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy
import scipy.sparse as sp
import sklearn
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night1_pipeline import prepare_corrected
from SpaLORA.night3a_ige import input_sha256, tensor_sha256
from SpaLORA.preprocess import pca


CONFIG_PATH = REPO / "configs/night3a_ige_feasibility.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_string_sha256(values: Sequence[object]) -> str:
    digest = hashlib.sha256()
    for value in map(str, values):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "little"))
        digest.update(encoded)
    return digest.hexdigest()


def array_sha256(values: np.ndarray) -> str:
    values = np.asarray(values)
    digest = hashlib.sha256()
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
    if values.dtype.kind in "OUS":
        digest.update(ordered_string_sha256(values.ravel()).encode("ascii"))
    else:
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def matrix_audit(matrix) -> dict:
    sparse = sp.issparse(matrix)
    values = matrix.data if sparse else np.asarray(matrix).ravel()
    finite = np.isfinite(values)
    return {
        "shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "dtype": str(matrix.dtype),
        "sparse": bool(sparse),
        "sparse_format": matrix.getformat() if sparse else None,
        "nnz": int(matrix.nnz if sparse else np.count_nonzero(matrix)),
        "nan_count": int(np.isnan(values).sum()),
        "inf_count": int(np.isinf(values).sum()),
        "negative_count": int(np.sum(values < 0)),
        "all_finite": bool(finite.all()),
    }


def h5ad_audit(path: Path) -> tuple:
    obj = sc.read_h5ad(str(path))
    payload = matrix_audit(obj.X)
    payload.update(
        {
            "absolute_path": str(path.resolve()),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
            "n_obs": int(obj.n_obs),
            "n_vars": int(obj.n_vars),
            "observation_order_sha256": ordered_string_sha256(obj.obs_names.astype(str)),
            "feature_order_sha256": ordered_string_sha256(obj.var_names.astype(str)),
            "spatial_present": "spatial" in obj.obsm,
            "spatial_order_sha256": (
                array_sha256(np.asarray(obj.obsm["spatial"])) if "spatial" in obj.obsm else None
            ),
            "spatial_shape": (
                list(map(int, np.asarray(obj.obsm["spatial"]).shape)) if "spatial" in obj.obsm else None
            ),
            "raw_available": obj.raw is not None,
            "raw_shape": list(map(int, obj.raw.shape)) if obj.raw is not None else None,
            "layers": sorted(map(str, obj.layers.keys())),
            "counts_available_in_X": True,
            "x_integer_like": bool(
                np.allclose((obj.X.data if sp.issparse(obj.X) else np.asarray(obj.X)),
                            np.rint(obj.X.data if sp.issparse(obj.X) else np.asarray(obj.X)))
            ),
        }
    )
    return obj, payload


def canonical_ids(values: Sequence[object], rule: str) -> pd.Index:
    values = pd.Index(values).astype(str)
    if rule == "strip_s1_prefix":
        return pd.Index([value[3:] if value.startswith("s1-") else value for value in values])
    if rule == "identity":
        return values
    raise ValueError("Unknown ID rule: %s" % rule)


def graph_summary(value: torch.Tensor) -> dict:
    if not value.is_sparse:
        raise AssertionError("Corrected graph was densified")
    value = value.coalesce()
    return {
        "shape": list(map(int, value.shape)),
        "nnz": int(value._nnz()),
        "dtype": str(value.dtype),
        "is_sparse": True,
        "sha256": tensor_sha256(value),
        "finite": bool(torch.isfinite(value.values()).all()),
    }


def preprocessing_config(config: dict) -> dict:
    pre = config["preprocessing"]
    return {
        "min_cells": int(pre["min_cells"]),
        "alpha": float(pre["alpha_compatibility_only"]),
        "rescue_non_hvg": int(pre["rescue_non_hvg_compatibility_only"]),
        "moran_shrinkage_tau": float(pre["moran_shrinkage_tau_compatibility_only"]),
        "feature_graph": {
            "k": int(pre["feature_graph_k"]),
            "metric": pre["feature_graph_metric"],
        },
    }


def run_order(config: dict) -> list:
    rows = [
        {"dataset": dataset, "variant": variant, "seed": seed}
        for dataset in config["datasets"]
        for variant in config["variants"]
        for seed in config["seeds"]
    ]
    random.Random(int(config["run_order"]["seed"])).shuffle(rows)
    if len(rows) != 60 or len({(x["dataset"], x["variant"], x["seed"]) for x in rows}) != 60:
        raise AssertionError("Preregistered run permutation is not the exact 60-cell factorial")
    return rows


def environment_payload() -> dict:
    def command(*args: str) -> str:
        try:
            return subprocess.check_output(list(args), text=True, stderr=subprocess.STDOUT).strip()
        except Exception as exc:
            return "ERROR: %r" % exc

    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "pandas": pd.__version__,
        "anndata": ad.__version__,
        "scanpy": sc.__version__,
        "sklearn": sklearn.__version__,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": command("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        "r": command("R", "--version"),
        "mclust": command("Rscript", "-e", "cat(as.character(packageVersion('mclust')))"),
        "git_head": command("git", "-C", str(REPO), "rev-parse", "HEAD"),
        "git_branch": command("git", "-C", str(REPO), "branch", "--show-current"),
        "git_status": command("git", "-C", str(REPO), "status", "--short"),
        "disk": command("df", "-h", str(REPO)),
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    if config["parent_commit"] != "49764b6d1b9d16510126a879e58f38973b562ce8":
        failures.append("parent commit drift")
    if config["variants"] != list(("C0", "C1", "IGE", "ILN")) or config["seeds"] != [0, 1, 2, 3, 4]:
        failures.append("factorial definition drift")

    order_path = REPO / config["run_order"]["manifest"]
    order_payload = {
        "schema_version": 1,
        "generated_before_label_access": True,
        "algorithm": config["run_order"]["algorithm"],
        "seed": config["run_order"]["seed"],
        "runs": run_order(config),
    }
    atomic_json(order_path, order_payload)

    manifest_rows = []
    dataset_audits: Dict[str, dict] = {}
    pre_cfg = preprocessing_config(config)
    for dataset, cfg in config["datasets"].items():
        rna_path, mod2_path = Path(cfg["rna"]), Path(cfg["modality2"])
        rna, rna_audit = h5ad_audit(rna_path)
        mod2, mod2_audit = h5ad_audit(mod2_path)
        for role, audit, expected in (
            ("rna", rna_audit, cfg["rna_sha256"]),
            ("modality2", mod2_audit, cfg["modality2_sha256"]),
        ):
            manifest_rows.append({"dataset": dataset, "role": role, **audit})
            if audit["sha256"] != expected:
                failures.append("%s %s SHA drift" % (dataset, role))
            if not audit["all_finite"] or audit["negative_count"]:
                failures.append("%s %s has nonfinite/negative values" % (dataset, role))
        if not rna.obs_names.equals(mod2.obs_names):
            failures.append("%s paired observation order mismatch" % dataset)
        if rna_audit["spatial_order_sha256"] != mod2_audit["spatial_order_sha256"]:
            failures.append("%s paired spatial order mismatch" % dataset)
        if not rna_audit["x_integer_like"]:
            failures.append("%s RNA X is not raw count-like" % dataset)

        # Audit only ground-truth identifiers/schema here; never read label values.
        gt_audit: Dict[str, object]
        if dataset == "placenta":
            gt_audit = {
                "source": "RNA obs schema only",
                "label_column_exists": cfg["ground_truth_label_column"] in rna.obs.columns,
                "identifier_count": int(rna.n_obs),
                "identifier_order_sha256": rna_audit["observation_order_sha256"],
                "label_values_read": False,
            }
            if not gt_audit["label_column_exists"]:
                failures.append("placenta obs[cell_type] schema missing")
        else:
            gt_path = Path(cfg["ground_truth"])
            gt_ids = pd.read_csv(gt_path, usecols=[cfg["ground_truth_id_column"]])[
                cfg["ground_truth_id_column"]
            ].astype(str)
            raw_ids = canonical_ids(rna.obs_names, cfg.get("ground_truth_id_rule", "identity"))
            gt_audit = {
                "absolute_path": str(gt_path.resolve()),
                "size_bytes": int(gt_path.stat().st_size),
                "sha256": sha256_file(gt_path),
                "identifier_count": int(len(gt_ids)),
                "identifier_unique": bool(gt_ids.is_unique),
                "identifier_order_sha256": ordered_string_sha256(gt_ids),
                "raw_input_identifier_set_equal": set(raw_ids) == set(gt_ids),
                "raw_input_identifier_order_equal": list(raw_ids) == list(gt_ids),
                "label_values_read": False,
            }
            if gt_audit["sha256"] != cfg["ground_truth_sha256"]:
                failures.append("%s ground-truth file SHA drift" % dataset)
            if not gt_audit["identifier_unique"]:
                failures.append("%s ground-truth IDs not unique" % dataset)

        prepared = prepare_corrected(dataset, cfg, pre_cfg, "corrected_unweighted")
        pdata = prepared.data
        selected_genes = list(map(str, pdata["selected_gene_names"]))
        prepared_ids = prepared.obs_names.astype(str)
        if dataset != "placenta":
            gt_ids_index = pd.Index(gt_ids.astype(str))
            training_canonical = canonical_ids(prepared_ids, cfg.get("ground_truth_id_rule", "identity"))
            gt_audit["training_identifier_set_equal"] = set(training_canonical) == set(gt_ids_index)
            gt_audit["training_identifier_order_equal"] = list(training_canonical) == list(gt_ids_index)
            if not gt_audit["training_identifier_set_equal"]:
                failures.append("%s post-QC/ground-truth ID set mismatch" % dataset)
        else:
            gt_audit["training_identifier_set_equal"] = set(prepared_ids) == set(rna.obs_names.astype(str))
            gt_audit["training_identifier_order_equal"] = list(prepared_ids) == list(rna.obs_names.astype(str))

        rna_pca_components = 50 if dataset == "p22" else mod2.n_vars - 1
        rna_pca = pca(ad.AnnData(np.asarray(pdata["features_omics1"])), n_comps=rna_pca_components)
        pca_summary = {
            "rna_pca": {
                "shape": list(map(int, rna_pca.shape)),
                "dtype": str(rna_pca.dtype),
                "finite": bool(np.isfinite(rna_pca).all()),
                "sha256": array_sha256(rna_pca),
            },
            "modality2_model_features": {
                "shape": list(map(int, np.asarray(pdata["features_omics2"]).shape)),
                "dtype": str(np.asarray(pdata["features_omics2"]).dtype),
                "finite": bool(np.isfinite(pdata["features_omics2"]).all()),
                "sha256": array_sha256(np.asarray(pdata["features_omics2"])),
            },
        }
        graphs = {
            name: graph_summary(pdata[name])
            for name in (
                "adj_spatial_omics1", "adj_spatial_omics2",
                "adj_feature_omics1", "adj_feature_omics2",
            )
        }
        if not all(value["finite"] and value["is_sparse"] for value in graphs.values()):
            failures.append("%s corrected graph sparse/finite contract failed" % dataset)
        if not pca_summary["rna_pca"]["finite"] or not pca_summary["modality2_model_features"]["finite"]:
            failures.append("%s PCA/model feature nonfinite" % dataset)
        dataset_audits[dataset] = {
            "rna": rna_audit,
            "modality2": mod2_audit,
            "ground_truth_identifier_audit": gt_audit,
            "prepared_n_obs": int(len(prepared_ids)),
            "prepared_observation_order_sha256": ordered_string_sha256(prepared_ids),
            "selected_gene_count": int(len(selected_genes)),
            "selected_gene_order_sha256": ordered_string_sha256(selected_genes),
            "model_input_sha256": input_sha256(pdata, prepared_ids, selected_genes),
            "pca": pca_summary,
            "graphs": graphs,
            "counts_immutable_shape": list(map(int, pdata["counts_immutable"].shape)),
            "xlog_immutable_shape": list(map(int, pdata["xlog_immutable"].shape)),
            "obs_columns_after_label_free_load": [],
        }
        del prepared, pdata, rna, mod2

    manifest_path = output / "data_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in manifest_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest_rows)

    environment = environment_payload()
    atomic_json(output / "night3a_environment.json", environment)

    missing_sources = [name for name in config["source_lock_files"] if not (REPO / name).is_file()]
    if missing_sources:
        failures.append("missing source lock files: %r" % missing_sources)
    source_hashes = {
        name: sha256_file(REPO / name) for name in config["source_lock_files"] if (REPO / name).is_file()
    }
    data_hashes = {}
    for cfg in config["datasets"].values():
        for key in ("rna", "modality2"):
            data_hashes[cfg[key]] = sha256_file(Path(cfg[key]))
        if cfg["ground_truth"].startswith("/"):
            data_hashes[cfg["ground_truth"]] = sha256_file(Path(cfg["ground_truth"]))
    protected_manifest = Path(config["paths"]["protected_manifest"])
    if not protected_manifest.is_file():
        failures.append("protected Night-2C manifest missing")
    taskbook_path = REPO / "docs/SpaLORA_Night3A_Codex_Taskbook_2026-08-10.md"
    if taskbook_path.is_file() and sha256_file(taskbook_path) != config["taskbook_sha256"]:
        failures.append("taskbook SHA drift")

    p0a = {
        "schema_version": 1,
        "stage": "P0A",
        "passed": len(failures) == 0,
        "failures": failures,
        "label_values_read": False,
        "ground_truth_identifier_only_audit": True,
        "night1_corrected_sparse_contract": True,
        "datasets": dataset_audits,
        "run_order_sha256": sha256_file(order_path),
        "data_manifest_sha256": sha256_file(manifest_path),
    }
    atomic_json(output / "night3a_p0a.json", p0a)
    if failures:
        atomic_json(output / "night3a_p0a_failure.json", p0a)
        atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": [p0a]})
        print(json.dumps(p0a, indent=2))
        raise SystemExit(2)

    lock = {
        "schema_version": 1,
        "locked_after_p0a_pass": True,
        "locked_before_semantic_label_access": True,
        "config_sha256": sha256_file(CONFIG_PATH),
        "data_manifest_sha256": sha256_file(manifest_path),
        "run_order_sha256": sha256_file(order_path),
        "p0a_sha256": sha256_file(output / "night3a_p0a.json"),
        "source_sha256": source_hashes,
        "data_sha256": data_hashes,
        "protected_manifest_sha256": sha256_file(protected_manifest),
    }
    atomic_json(output / "config_lock.json", lock)
    atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": []})
    print("P0A_PASS", json.dumps({
        "datasets": {name: value["prepared_n_obs"] for name, value in dataset_audits.items()},
        "config_lock_sha256": sha256_file(output / "config_lock.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
