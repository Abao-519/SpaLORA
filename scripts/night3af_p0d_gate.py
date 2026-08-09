#!/usr/bin/env python3
"""Compare independent P0D builds, publish one immutable cache, and run the hard tests."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night3af_cache import atomic_json, load_cache, sha256_file, verify_cache


CONFIG_PATH = REPO / "configs/night3af_deterministic_pca.json"


def protected_check(root: str, manifest: str) -> dict:
    result = subprocess.run(["sha256sum", "-c", manifest], cwd=root, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return {"passed": result.returncode == 0, "line_count": len(lines),
            "failures": [line for line in lines if not line.endswith(": OK")]}


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    builds = Path(config["paths"]["p0d_temp_root"])
    process_a = json.loads((output / "p0d_process_a.json").read_text(encoding="utf-8"))
    process_b = json.loads((output / "p0d_process_b.json").read_text(encoding="utf-8"))
    failures = []
    comparisons = {}
    for dataset in config["datasets"]:
        a, b = process_a["datasets"][dataset], process_b["datasets"][dataset]
        fields = {
            "canonical_model_input_sha256_exact": a["canonical_model_input_sha256"] == b["canonical_model_input_sha256"],
            "canonical_cache_content_sha256_exact": a["canonical_cache_content_sha256"] == b["canonical_cache_content_sha256"],
            "all_file_sha256_exact": a["file_sha256"] == b["file_sha256"],
            "all_rng_states_unchanged": all(a["rng"].values()) and all(b["rng"].values()),
        }
        comparisons[dataset] = fields
        if not all(fields.values()): failures.append("%s cross-process mismatch: %r" % (dataset, fields))
    if failures:
        atomic_json(output / "p0d_cross_process_comparison.json", {
            "schema_version": 1, "passed": False, "datasets": comparisons, "failures": failures,
        })
        raise SystemExit(2)

    cache_root = REPO / config["paths"]["cache_root"]
    if cache_root.exists() and any(cache_root.iterdir()):
        raise RuntimeError("Refusing to overwrite published cache")
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_index = {}
    for dataset in config["datasets"]:
        shutil.copytree(builds / "process_a" / dataset, cache_root / dataset)
        manifest = verify_cache(cache_root / dataset)
        loaded = load_cache(cache_root / dataset)
        cache_index[dataset] = {
            "directory": str((cache_root / dataset).relative_to(REPO)),
            "manifest_sha256": sha256_file(cache_root / dataset / "manifest.json"),
            "canonical_cache_content_sha256": manifest["canonical_cache_content_sha256"],
            "canonical_model_input_sha256": manifest["canonical_model_input_sha256"],
            "n_observations": len(loaded.obs_names),
            "n_selected_genes": len(loaded.data["selected_gene_names"]),
            "graph_nnz": {name: int(loaded.data[name]._nnz()) for name in (
                "adj_spatial_omics1", "adj_spatial_omics2", "adj_feature_omics1", "adj_feature_omics2"
            )},
        }
    cache_manifest = {
        "schema_version": 1, "published_from": "independent process A after A/B byte-exact comparison",
        "immutable": True, "pca_svd_solver": "randomized", "pca_random_state": 0,
        "datasets": cache_index,
    }
    atomic_json(output / "preprocessing_cache_manifest.json", cache_manifest)
    atomic_json(output / "p0d_cross_process_comparison.json", {
        "schema_version": 1, "passed": True, "datasets": comparisons,
        "byte_exact_dataset_count": 3, "dataset_count": 3,
        "external_rng_seeds_different": process_a["external_rng_seed"] != process_b["external_rng_seed"],
        "process_a_sha256": sha256_file(output / "p0d_process_a.json"),
        "process_b_sha256": sha256_file(output / "p0d_process_b.json"),
    })

    old_p0a = json.loads((REPO / config["paths"]["old_night3a_output"] / "night3a_p0a.json").read_text(encoding="utf-8"))
    old_ar = json.loads((REPO / config["paths"]["old_night3ar_output"] / "night3ar_p0a.json").read_text(encoding="utf-8"))
    diagnosis = {}
    for dataset in config["datasets"]:
        diagnosis[dataset] = {
            "night3a_random_pca_sha256": old_p0a["datasets"][dataset]["pca"]["rna_pca"]["sha256"],
            "night3ar_random_pca_sha256": old_ar["datasets"][dataset]["pca"]["rna_pca"]["sha256"],
            "deterministic_pca_file_sha256": process_a["datasets"][dataset]["file_sha256"]["rna_pca_scores.npy"],
            "night3a_rna_feature_graph_nnz": old_p0a["datasets"][dataset]["graphs"]["adj_feature_omics1"]["nnz"],
            "night3ar_rna_feature_graph_nnz": old_ar["datasets"][dataset]["graphs"]["adj_feature_omics1"]["nnz"],
            "deterministic_rna_feature_graph_nnz": cache_index[dataset]["graph_nnz"]["adj_feature_omics1"],
            "old_new_hash_difference_expected": True,
            "reason": "legacy PCA used randomized SVD without an explicit local random_state",
            "edge_overlap": "edge_overlap_not_available",
        }
    atomic_json(output / "pca_old_new_diagnosis.json", {
        "schema_version": 1, "diagnostic_only_not_a_gate": True, "datasets": diagnosis,
    })
    atomic_json(output / "deterministic_pca_protocol.json", {
        "schema_version": 1, "svd_solver": "randomized", "random_state": 0,
        "legacy_pca_default_unchanged": True, "corrected_pipeline_explicit": True,
        "two_independent_processes": True, "external_process_seeds": [
            process_a["external_rng_seed"], process_b["external_rng_seed"]
        ],
        "old_random_pca_hash_is_not_a_gate": True,
        "taskbook_sha256": config["taskbook_sha256"],
    })
    (output / "label_flow_audit.md").write_text(
        "# Night-3A-F label flow audit\n\n"
        "Raw input and ground-truth files are byte-hashed before each scientific window. "
        "The two deterministic builders read only paired h5ad training inputs; `prepare_corrected` "
        "drops every `obs` column before any feature operation. No label CSV is parsed. The published "
        "cache contains IDs, genes, arrays, sparse graphs, coordinates and PCA metadata, but no labels. "
        "P0B-F and all 60 runs load only that cache. The independent evaluator is forbidden until the "
        "locked 60-run manifest is fsynced.\n",
        encoding="utf-8",
    )

    old_order = json.loads((REPO / config["run_order"]["old_manifest"]).read_text(encoding="utf-8"))
    atomic_json(REPO / config["run_order"]["manifest"], old_order)
    source_missing = [name for name in config["source_lock_files"] if not (REPO / name).is_file()]
    if source_missing: failures.append("missing source files: %r" % source_missing)
    source_hashes = {name: sha256_file(REPO / name) for name in config["source_lock_files"] if (REPO / name).is_file()}
    data_hashes = {}
    for cfg in config["datasets"].values():
        data_hashes[cfg["rna"]] = sha256_file(Path(cfg["rna"]))
        data_hashes[cfg["modality2"]] = sha256_file(Path(cfg["modality2"]))
        if str(cfg["ground_truth"]).startswith("/"):
            data_hashes[cfg["ground_truth"]] = sha256_file(Path(cfg["ground_truth"]))
    lock = {
        "schema_version": 1, "locked_after_p0d_cross_process_pass": True,
        "locked_before_performance_metric_access": True,
        "config_sha256": sha256_file(CONFIG_PATH), "source_sha256": source_hashes,
        "data_sha256": data_hashes,
        "run_order_sha256": sha256_file(REPO / config["run_order"]["manifest"]),
        "preprocessing_cache_manifest_sha256": sha256_file(output / "preprocessing_cache_manifest.json"),
        "protected_manifest_sha256": {
            "night3ar": sha256_file(Path(config["paths"]["protected_night3ar_manifest"])),
            "night3a": sha256_file(Path(config["paths"]["protected_night3a_manifest"])),
            "night2c": sha256_file(Path(config["paths"]["protected_night2c_manifest"])),
        },
    }
    atomic_json(output / "config_lock.json", lock)
    preliminary = {
        "schema_version": 1, "stage": "P0D", "passed": False,
        "cross_process_byte_exact": True, "byte_exact_dataset_count": 3,
        "published_cache_verified": True, "tests_pending": True, "failures": failures,
    }
    atomic_json(output / "night3af_p0d.json", preliminary)

    test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_night3af.py"],
                          cwd=REPO, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          env={**os.environ, "R_HOME": "/opt/R/4.0.3/lib/R"})
    (output / "p0d_tests.log").write_text(test.stdout, encoding="utf-8")
    match = re.search(r"(?:(\d+) failed, )?(\d+) passed", test.stdout)
    failed = int(match.group(1) or 0) if match else -1
    passed = int(match.group(2)) if match else -1
    protections = {
        name: protected_check(config["paths"]["protected_%s_root" % name],
                              config["paths"]["protected_%s_manifest" % name])
        for name in ("night3ar", "night3a", "night2c")
    }
    if test.returncode or failed != 0: failures.append("P0D tests failed: %d" % failed)
    if not all(row["passed"] for row in protections.values()): failures.append("protected files changed")
    p0d = {
        "schema_version": 1, "stage": "P0D", "passed": not failures,
        "cross_process_byte_exact": True, "byte_exact_dataset_count": 3,
        "published_cache_verified": True, "tests_passed": passed, "tests_failed": failed,
        "protections": protections, "failures": failures,
        "semantic_label_values_read": False,
        "config_lock_sha256": sha256_file(output / "config_lock.json"),
    }
    atomic_json(output / "night3af_p0d.json", p0d)
    atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": [] if not failures else [p0d]})
    if failures:
        atomic_json(output / "night3af_p0d_failure.json", p0d)
        print(test.stdout); raise SystemExit(2)
    for directory in cache_root.rglob("*"):
        os.chmod(directory, 0o444 if directory.is_file() else 0o555)
    os.chmod(cache_root, 0o555)
    print("P0D_PASS datasets=3 byte_exact=3 tests=%d failed=0" % passed, flush=True)


if __name__ == "__main__":
    main()
