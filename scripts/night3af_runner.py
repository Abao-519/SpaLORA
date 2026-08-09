#!/usr/bin/env python3
"""Night-3A-F locked 60-run runner; it can only load the published cache."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


os.environ.setdefault("R_HOME", "/opt/R/4.0.3/lib/R")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path: sys.path.insert(0, str(REPO))

from SpaLORA.night3af_cache import load_cache
from SpaLORA.night3af_protocol import (
    ScientificWindow, assert_training_payload_label_free, atomic_json,
    ground_truth_csv_paths, load_cache_index, sha256_file, training_cfg,
    verify_night3af_lock,
)


CONFIG_PATH = REPO / "configs/night3af_deterministic_pca.json"


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    lock = json.loads((output / "config_lock.json").read_text(encoding="utf-8"))
    verify_night3af_lock(REPO, CONFIG_PATH, config, lock, output, "training_60")
    gate = json.loads((output / "night3af_gate_status.json").read_text(encoding="utf-8"))
    if not (gate.get("p0d_pass") and gate.get("p0bf_pass") and gate.get("main_60_authorized")):
        raise RuntimeError("P0D/P0B-F did not authorize main training")
    order_path = REPO / config["run_order"]["manifest"]
    order = json.loads(order_path.read_text(encoding="utf-8"))["runs"]
    if len(order) != 60 or len({(x["dataset"], x["variant"], x["seed"]) for x in order}) != 60:
        raise AssertionError("Locked order is not the exact 60-cell factorial")
    cache_index = load_cache_index(output)
    window = ScientificWindow(config, output, "training_60").install()
    completed = []
    try:
        from SpaLORA.night3a_ige import input_sha256
        from scripts.night3ar_runner import run_dir, run_one

        prepared_cache, load_seconds = {}, {}
        forbidden = ground_truth_csv_paths(config)
        for dataset, cfg in config["datasets"].items():
            row = cache_index["datasets"][dataset]
            prepared = load_cache(REPO / row["directory"], row["manifest_sha256"])
            assert_training_payload_label_free(prepared.data, training_cfg(cfg), forbidden)
            observed = input_sha256(prepared.data, prepared.obs_names, prepared.data["selected_gene_names"])
            if observed != row["canonical_model_input_sha256"]:
                raise RuntimeError("Published deterministic cache input hash mismatch")
            prepared_cache[dataset] = prepared; load_seconds[dataset] = 0.0
            print("CACHE_LOADED %s n=%d hash=%s" % (dataset, len(prepared.obs_names), observed), flush=True)

        for ordinal, cell in enumerate(order, 1):
            dataset, variant, seed = cell["dataset"], cell["variant"], int(cell["seed"])
            path = run_one(config, lock, output, prepared_cache, load_seconds,
                           dataset, variant, seed, ordinal)
            manifest = json.loads(path.read_text(encoding="utf-8"))
            cache_row = cache_index["datasets"][dataset]
            if manifest["locked_input_sha256"] != cache_row["canonical_model_input_sha256"]:
                raise RuntimeError("Run did not consume the locked cache")
            manifest.update({
                "night3af_deterministic_cache": True,
                "deterministic_cache_manifest_sha256": cache_row["manifest_sha256"],
                "deterministic_cache_content_sha256": cache_row["canonical_cache_content_sha256"],
                "pca_svd_solver": config["preprocessing"]["pca_svd_solver"],
                "pca_random_state": config["preprocessing"]["pca_random_state"],
                "versions": {
                    "numpy": __import__("numpy").__version__, "torch": __import__("torch").__version__,
                    "sklearn": __import__("sklearn").__version__, "scanpy": __import__("scanpy").__version__,
                    "scipy": __import__("scipy").__version__,
                },
            })
            atomic_json(path, manifest); completed.append(path)
        if len(completed) != 60: raise AssertionError("Expected 60 completed runs")

        locked_rows = []
        dataset_hashes = {dataset: set() for dataset in config["datasets"]}
        for ordinal, cell in enumerate(order, 1):
            path = run_dir(output, cell["dataset"], cell["variant"], int(cell["seed"])) / "run_manifest.json"
            manifest = json.loads(path.read_text(encoding="utf-8"))
            dataset_hashes[cell["dataset"]].add(manifest["locked_input_sha256"])
            locked_rows.append({
                "ordinal": ordinal, "dataset": cell["dataset"], "variant": cell["variant"],
                "seed": int(cell["seed"]), "run_manifest": str(path.relative_to(REPO)),
                "run_manifest_sha256": sha256_file(path), "artifact_sha256": manifest["artifact_sha256"],
                "locked_input_sha256": manifest["locked_input_sha256"],
            })
        if any(len(values) != 1 for values in dataset_hashes.values()):
            raise RuntimeError("Four variants/five seeds did not use one cache hash")
        locked_path = output / "locked_60_run_manifest.json"
        atomic_json(locked_path, {
            "schema_version": 1, "locked_before_any_semantic_label_access": True,
            "run_count": 60, "failures": 0,
            "preregistered_order_sha256": sha256_file(order_path),
            "config_lock_sha256": sha256_file(output / "config_lock.json"),
            "dataset_cache_hashes": {key: list(values)[0] for key, values in dataset_hashes.items()},
            "runs": locked_rows,
        })
        firewall = window.close(passed=True)
        if not firewall["passed"]: raise RuntimeError("Training label firewall failed")
        atomic_json(output / "training_complete.json", {
            "schema_version": 1, "training_complete": True, "run_count": 60, "failure_count": 0,
            "locked_60_run_manifest_sha256": sha256_file(locked_path),
            "scientific_window_firewall_sha256": sha256_file(output / "scientific_window_label_firewall.json"),
            "semantic_label_values_read": False,
        })
        atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": []})
        print("TRAINING_LOCKED 60/60 manifest=%s" % sha256_file(locked_path), flush=True)
    except Exception:
        try: window.close(passed=False)
        except Exception: pass
        failures = sorted(str(path.relative_to(output)) for path in output.rglob("failure.json"))
        atomic_json(output / "failure_index.json", {"schema_version": 1, "failures": failures})
        raise


if __name__ == "__main__":
    main()
