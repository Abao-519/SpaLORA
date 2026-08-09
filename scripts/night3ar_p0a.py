#!/usr/bin/env python3
"""Night-3A-R P0A-R fresh data/preprocessing audit and source lock."""

from __future__ import annotations

import csv
import json
import os
import shutil
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night1_pipeline import prepare_corrected
from SpaLORA.night3a_ige import input_sha256
from SpaLORA.night3ar_protocol import atomic_json, integrity_read, record_integrity_reads, sha256_file
from SpaLORA.preprocess import pca
from scripts.night3a_p0a import (
    array_sha256, canonical_ids, environment_payload, graph_summary, h5ad_audit,
    ordered_string_sha256, preprocessing_config,
)


CONFIG_PATH = REPO / "configs/night3ar_ige_feasibility.json"
OLD_CONFIG_PATH = REPO / "configs/night3a_ige_feasibility.json"


def comparison_view(audit: dict) -> dict:
    return {
        "prepared_n_obs": audit["prepared_n_obs"],
        "prepared_observation_order_sha256": audit["prepared_observation_order_sha256"],
        "selected_gene_count": audit["selected_gene_count"],
        "selected_gene_order_sha256": audit["selected_gene_order_sha256"],
        "model_input_sha256": audit["model_input_sha256"],
        "counts_immutable_shape": audit["counts_immutable_shape"],
        "xlog_immutable_shape": audit["xlog_immutable_shape"],
        "pca": audit["pca"],
        "graphs": audit["graphs"],
    }


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    failures = []
    if sha256_file(OLD_CONFIG_PATH) != config["old_config_sha256"]:
        failures.append("old Night-3A config SHA drift")
    old_p0a_path = REPO / config["paths"]["old_output_root"] / "night3a_p0a.json"
    if sha256_file(old_p0a_path) != config["old_p0a_sha256"]:
        failures.append("old Night-3A P0A SHA drift")
    old_p0a = json.loads(old_p0a_path.read_text(encoding="utf-8"))

    protocol_amendment = {
        "schema_version": 1,
        "previous_night3a_status": config["previous_night3a_status"],
        "old_night3a_report_and_failure_retained": True,
        "reason": "Old P0B classified integrity-only SHA byte reads as semantic label leakage",
        "allowed_change_a": "Integrity reads complete before scientific window; semantic parsing/use remains forbidden",
        "allowed_change_b": "Mechanism collapse gate uses weighted-gradient influence rather than scalar loss fraction",
        "unchanged": [
            "datasets", "variants", "seeds", "epochs", "optimizer", "preprocessing", "graphs",
            "clustering", "metrics", "IGE formula", "ILN formula", "60-cell order", "ASR freeze",
        ],
        "old_config_sha256": config["old_config_sha256"],
        "new_config_sha256": sha256_file(CONFIG_PATH),
        "taskbook_sha256": config["taskbook_sha256"],
        "semantic_label_access_before_manifest_lock_allowed": False,
    }
    atomic_json(output / "protocol_amendment.json", protocol_amendment)
    (output / "label_flow_audit.md").write_text(
        "# Night-3A-R label flow audit\n\n"
        "1. Before the scientific window, immutable files are opened only as bytes to verify SHA-256; no content is returned.\n"
        "2. P0A-R alone parses only the ground-truth identifier column to verify spot order/set; `label_values_read=false`.\n"
        "3. The scientific-window hook and `pandas.read_csv` guard are installed before preprocessing/model code.\n"
        "4. A1/P22 ground-truth CSV opens, label parsers, and evaluator imports are forbidden until `locked_60_run_manifest.json` is fsynced.\n"
        "5. Placenta RNA h5ad is a required input, but `_load_label_free` removes all `obs` columns before preprocessing output enters the trainer; `cell_type` is never accessed.\n"
        "6. The independent evaluator starts only after 60/60 completion and a PASS training firewall.\n",
        encoding="utf-8",
    )
    with (output / "label_flow_audit.md").open("rb") as handle:
        os.fsync(handle.fileno())

    integrity_reads = []
    for dataset, cfg in config["datasets"].items():
        for key, expected in (("rna", cfg["rna_sha256"]), ("modality2", cfg["modality2_sha256"])):
            integrity_reads.append(integrity_read(Path(cfg[key]), expected, "%s %s input" % (dataset, key)))
        if str(cfg["ground_truth"]).startswith("/"):
            integrity_reads.append(integrity_read(
                Path(cfg["ground_truth"]), cfg["ground_truth_sha256"],
                "%s ground-truth integrity only" % dataset,
            ))
    record_integrity_reads(output, "p0ar_prelock", integrity_reads)
    if not all(row["match"] for row in integrity_reads):
        failures.append("one or more input integrity reads mismatch")

    old_order_path = REPO / config["run_order"]["old_manifest"]
    new_order_path = REPO / config["run_order"]["manifest"]
    old_order = json.loads(old_order_path.read_text(encoding="utf-8"))
    if len(old_order.get("runs", [])) != 60:
        failures.append("old run order is not 60 cells")
    atomic_json(new_order_path, old_order)
    new_order = json.loads(new_order_path.read_text(encoding="utf-8"))
    old_cells = [(i + 1, row["dataset"], row["variant"], row["seed"]) for i, row in enumerate(old_order["runs"])]
    new_cells = [(i + 1, row["dataset"], row["variant"], row["seed"]) for i, row in enumerate(new_order["runs"])]
    if old_cells != new_cells:
        failures.append("Night-3A-R run order differs from original dataset/variant/seed/ordinal")

    pre_cfg = preprocessing_config(config)
    manifest_rows = []
    audits = {}
    comparisons = {}
    for dataset, cfg in config["datasets"].items():
        rna, rna_audit = h5ad_audit(Path(cfg["rna"]))
        mod2, mod2_audit = h5ad_audit(Path(cfg["modality2"]))
        label_column_exists = cfg["ground_truth_label_column"] in rna.obs.columns if dataset == "placenta" else None
        rna.obs = pd.DataFrame(index=rna.obs_names.copy())
        mod2.obs = pd.DataFrame(index=mod2.obs_names.copy())
        for role, audit in (("rna", rna_audit), ("modality2", mod2_audit)):
            manifest_rows.append({"dataset": dataset, "role": role, **audit})
        if not rna.obs_names.equals(mod2.obs_names):
            failures.append("%s paired observation order mismatch" % dataset)
        if rna_audit["spatial_order_sha256"] != mod2_audit["spatial_order_sha256"]:
            failures.append("%s paired spatial order mismatch" % dataset)
        gt_audit = {"label_values_read": False, "access_class": "identifier_only_p0ar"}
        if dataset == "placenta":
            gt_audit.update({
                "source": "RNA obs schema only", "label_column_exists": bool(label_column_exists),
                "identifier_count": int(rna.n_obs),
                "identifier_order_sha256": rna_audit["observation_order_sha256"],
            })
            if not label_column_exists:
                failures.append("placenta obs[cell_type] schema missing")
        else:
            gt_ids = pd.read_csv(cfg["ground_truth"], usecols=[cfg["ground_truth_id_column"]])[
                cfg["ground_truth_id_column"]
            ].astype(str)
            raw_ids = canonical_ids(rna.obs_names, cfg.get("ground_truth_id_rule", "identity"))
            gt_audit.update({
                "identifier_count": int(len(gt_ids)), "identifier_unique": bool(gt_ids.is_unique),
                "identifier_order_sha256": ordered_string_sha256(gt_ids),
                "raw_input_identifier_set_equal": set(raw_ids) == set(gt_ids),
                "raw_input_identifier_order_equal": list(raw_ids) == list(gt_ids),
            })

        prepared = prepare_corrected(dataset, cfg, pre_cfg, "corrected_unweighted")
        pdata = prepared.data
        prepared_ids = prepared.obs_names.astype(str)
        selected_genes = list(map(str, pdata["selected_gene_names"]))
        if any("label" in str(key).lower() or "ground" in str(key).lower() for key in pdata):
            failures.append("%s prepared payload contains label-like key" % dataset)
        if dataset != "placenta":
            training_ids = canonical_ids(prepared_ids, cfg.get("ground_truth_id_rule", "identity"))
            gt_audit["training_identifier_set_equal"] = set(training_ids) == set(gt_ids)
            gt_audit["training_identifier_order_equal"] = list(training_ids) == list(gt_ids)
        else:
            gt_audit["training_identifier_set_equal"] = set(prepared_ids) == set(rna.obs_names.astype(str))
            gt_audit["training_identifier_order_equal"] = list(prepared_ids) == list(rna.obs_names.astype(str))
        rna_pca_components = 50 if dataset == "p22" else mod2.n_vars - 1
        rna_pca = pca(ad.AnnData(np.asarray(pdata["features_omics1"])), n_comps=rna_pca_components)
        audit = {
            "rna": rna_audit,
            "modality2": mod2_audit,
            "ground_truth_identifier_audit": gt_audit,
            "prepared_n_obs": int(len(prepared_ids)),
            "prepared_observation_order_sha256": ordered_string_sha256(prepared_ids),
            "selected_gene_count": int(len(selected_genes)),
            "selected_gene_order_sha256": ordered_string_sha256(selected_genes),
            "model_input_sha256": input_sha256(pdata, prepared_ids, selected_genes),
            "pca": {
                "rna_pca": {"shape": list(map(int, rna_pca.shape)), "dtype": str(rna_pca.dtype),
                            "finite": bool(np.isfinite(rna_pca).all()), "sha256": array_sha256(rna_pca)},
                "modality2_model_features": {
                    "shape": list(map(int, np.asarray(pdata["features_omics2"]).shape)),
                    "dtype": str(np.asarray(pdata["features_omics2"]).dtype),
                    "finite": bool(np.isfinite(pdata["features_omics2"]).all()),
                    "sha256": array_sha256(np.asarray(pdata["features_omics2"])),
                },
            },
            "graphs": {name: graph_summary(pdata[name]) for name in (
                "adj_spatial_omics1", "adj_spatial_omics2", "adj_feature_omics1", "adj_feature_omics2"
            )},
            "counts_immutable_shape": list(map(int, pdata["counts_immutable"].shape)),
            "xlog_immutable_shape": list(map(int, pdata["xlog_immutable"].shape)),
            "obs_columns_after_label_free_load": [],
        }
        audits[dataset] = audit
        old_view, new_view = comparison_view(old_p0a["datasets"][dataset]), comparison_view(audit)
        comparisons[dataset] = {"exact_match": old_view == new_view, "old": old_view, "new": new_view}
        if old_view != new_view:
            failures.append("%s P0A-R preprocessing fingerprint differs from Night-3A" % dataset)
        del prepared, pdata, rna, mod2

    manifest_path = output / "data_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in manifest_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(manifest_rows)
        handle.flush(); os.fsync(handle.fileno())
    atomic_json(output / "night3ar_environment.json", environment_payload())

    missing = [name for name in config["source_lock_files"] if not (REPO / name).is_file()]
    if missing:
        failures.append("missing source lock files: %r" % missing)
    source_hashes = {name: sha256_file(REPO / name) for name in config["source_lock_files"] if (REPO / name).is_file()}
    data_hashes = {}
    for cfg in config["datasets"].values():
        data_hashes[cfg["rna"]] = sha256_file(Path(cfg["rna"]))
        data_hashes[cfg["modality2"]] = sha256_file(Path(cfg["modality2"]))
        if str(cfg["ground_truth"]).startswith("/"):
            data_hashes[cfg["ground_truth"]] = sha256_file(Path(cfg["ground_truth"]))

    p0ar = {
        "schema_version": 1, "stage": "P0A-R", "passed": not failures,
        "failures": failures, "label_values_read": False,
        "identifier_only_audit": True, "datasets": audits,
        "night3a_comparison": comparisons,
        "all_preprocessing_fingerprints_exact": all(row["exact_match"] for row in comparisons.values()),
        "run_order_exact": old_cells == new_cells,
        "old_config_sha256": config["old_config_sha256"],
        "new_config_sha256": sha256_file(CONFIG_PATH),
        "taskbook_sha256": config["taskbook_sha256"],
    }
    atomic_json(output / "night3ar_p0a.json", p0ar)
    if failures:
        atomic_json(output / "night3ar_p0a_failure.json", p0ar)
        atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": [p0ar]})
        print(json.dumps(p0ar, indent=2)); raise SystemExit(2)

    lock = {
        "schema_version": 1, "locked_after_p0ar_pass": True,
        "locked_before_performance_metric_access": True,
        "config_sha256": sha256_file(CONFIG_PATH),
        "data_manifest_sha256": sha256_file(manifest_path),
        "run_order_sha256": sha256_file(new_order_path),
        "p0ar_sha256": sha256_file(output / "night3ar_p0a.json"),
        "source_sha256": source_hashes, "data_sha256": data_hashes,
        "old_night3a_manifest_sha256": sha256_file(Path(config["paths"]["protected_night3a_manifest"])),
        "old_night2c_manifest_sha256": sha256_file(Path(config["paths"]["protected_night2c_manifest"])),
    }
    atomic_json(output / "config_lock.json", lock)
    atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": []})
    print("P0A_R_PASS", json.dumps({
        "datasets": {name: audit["prepared_n_obs"] for name, audit in audits.items()},
        "old_fingerprints_exact": True, "run_order_exact": True,
        "config_lock_sha256": sha256_file(output / "config_lock.json"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
