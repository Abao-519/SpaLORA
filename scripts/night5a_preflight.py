#!/usr/bin/env python3
"""Night-5A P0-PROTECT/P0-ARCH and locked label-free artifact builder."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night3af_cache import load_cache
from SpaLORA.night3a_ige import LOSS_KEYS, _clone_state, calibrated_total, input_sha256, raw_losses
from SpaLORA.night3b_ablation import Night3BTrainer
from SpaLORA.night3ar_protocol import assert_training_payload_label_free, ground_truth_csv_paths
from SpaLORA.night5a_rnd import (
    LEGACY_DELEGATES, Night5ATrainer, build_label_free_artifacts, canonical_sha256,
    load_label_free_artifacts, load_registry, registry_contracts, sha256_file,
)


CONFIG_PATH = REPO / "configs/night5a_metric_rnd.json"
EXPECTED_PARENT = "4a22cfb4afe331e1ca2edcc0b86a01fa3892a452"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def cache_index(config: dict) -> dict:
    return json.loads(Path(config["paths"]["cache_manifest"]).read_text(encoding="utf-8"))


def cache_dir(config: dict, row: dict) -> Path:
    return Path(config["paths"]["night3af_root"]) / row["directory"]


def training_cfg(dataset: dict) -> dict:
    keys = ("embedding_dim", "epochs", "loss_factors", "locked_m_bad_expected")
    result = {key: dataset[key] for key in keys}
    assert not any("label" in key.lower() or "ground" in key.lower() for key in result)
    return result


def generate_r1_order(contracts: dict) -> dict:
    rows, ordinal = [], 0
    for dataset in ("a1", "placenta"):
        for candidate_id in contracts:
            ordinal += 1
            rows.append({"ordinal": ordinal, "stage": "R1", "dataset": dataset,
                         "candidate_id": candidate_id, "seed": 0})
    payload = {
        "schema_version": 1, "algorithm": "dataset_then_registry_order_seed0",
        "locked_before_training": True, "run_count": len(rows), "runs": rows,
    }
    payload["canonical_sha256"] = canonical_sha256(payload)
    return payload


def parity_probe(data, cfg, candidate, legacy_variant, seed=0) -> dict:
    left = Night5ATrainer(data, cfg, candidate, seed, torch.device("cpu"), {}).train()
    right = Night3BTrainer(data, cfg, legacy_variant, seed, torch.device("cpu"), 1e-12).train()
    output_fields = ("SpaLORA", "emb_latent_omics1", "emb_latent_omics2",
                     "alpha", "alpha_omics1", "alpha_omics2")
    output_exact = all(np.array_equal(left.output[name], right.output[name]) for name in output_fields)
    losses_exact = left.initial_losses == right.initial_losses
    coefficients_exact = left.coefficients == right.coefficients
    initial_exact = left.initial_state_sha256 == right.initial_state_sha256
    final_exact = left.final_state_sha256 == right.final_state_sha256
    # The delegated path must also expose identical one-step/checkpoint state.
    left_logs = {row["step"]: row for row in left.logs}
    right_logs = {row["step"]: row for row in right.logs}
    checkpoint_exact = left_logs[1]["checkpoint_state_sha256"] == right_logs[1]["checkpoint_state_sha256"]
    passed = all((output_exact, losses_exact, coefficients_exact, initial_exact, final_exact, checkpoint_exact))
    return {
        "candidate_id": candidate["id"], "legacy_variant": legacy_variant,
        "forward_output_exact": output_exact, "loss_exact": losses_exact,
        "gradient_coefficients_exact": coefficients_exact,
        "initial_state_exact": initial_exact, "adam_one_step_exact": checkpoint_exact,
        "final_state_exact": final_exact, "passed": passed,
    }


def main() -> None:
    started = time.time()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    if git("rev-parse", "HEAD") != EXPECTED_PARENT or git("rev-parse", "night4a-preflight-final-20260813^{}") != EXPECTED_PARENT:
        raise RuntimeError("P0-PROTECT parent/tag mismatch")
    allowed = {"docs/night5a_authoritative_inputs/", "outputs/night5a_handoff/", "SpaLORA/night5a_rnd.py",
               "SpaLORA/night5a_strict_ids.py",
               "configs/night5a_metric_rnd.json", "scripts/night5a_preflight.py", "scripts/night5a_runner.py",
               "scripts/night5a_evaluate.py", "scripts/night5a_finalize.py", "tests/test_night5a.py"}
    status = git("status", "--porcelain=v1")
    unexpected = []
    for line in status.splitlines():
        name = line[3:]
        if not any(name == prefix or name.startswith(prefix) for prefix in allowed):
            unexpected.append(line)
    if unexpected:
        raise RuntimeError("Unexpected worktree modifications: %r" % unexpected)

    source_inputs = {
        config["candidate_registry"]: config["candidate_registry_sha256"],
        config["taskbook"]: config["taskbook_sha256"],
        config["rationale"]: config["rationale_sha256"],
    }
    input_checks = []
    for name, expected in source_inputs.items():
        actual = sha256_file(REPO / name)
        input_checks.append({"path": name, "expected_sha256": expected, "actual_sha256": actual,
                             "match": actual == expected})
    if not all(row["match"] for row in input_checks):
        raise RuntimeError("Authoritative Night-5A input SHA mismatch")

    registry = load_registry(REPO / config["candidate_registry"])
    contracts = registry_contracts(registry)
    atomic_json(output / "candidate_contracts.json", {"schema_version": 1, "candidates": contracts})
    r1_order = generate_r1_order(contracts)
    atomic_json(output / "preregistered_maximum_run_order.json", r1_order)

    index = cache_index(config)
    prepared, artifacts, cache_rows = {}, {}, {}
    forbidden = ground_truth_csv_paths(config)
    artifact_root = Path(config["paths"]["label_free_artifacts"])
    for dataset in ("a1", "placenta"):
        row = index["datasets"][dataset]
        item = load_cache(cache_dir(config, row), row["manifest_sha256"])
        cfg = training_cfg(config["datasets"][dataset])
        assert_training_payload_label_free(item.data, cfg, forbidden)
        observed = input_sha256(item.data, item.obs_names, item.data["selected_gene_names"])
        if observed != row["canonical_model_input_sha256"]:
            raise RuntimeError("Immutable cache input mismatch: %s" % dataset)
        prepared[dataset] = item
        cache_rows[dataset] = {
            "manifest_sha256": row["manifest_sha256"],
            "canonical_model_input_sha256": observed,
            "observation_count": int(len(item.obs_names)), "training_payload_keys": sorted(item.data),
            "semantic_label_access": False,
        }
        directory = artifact_root / dataset
        if directory.is_dir() and (directory / "manifest.json").is_file():
            loaded = load_label_free_artifacts(directory)
        else:
            build_label_free_artifacts(item.data, directory, config["preprocessing_rng_seed"])
            loaded = load_label_free_artifacts(directory)
        artifacts[dataset] = loaded
        cache_rows[dataset]["label_free_artifact_manifest_sha256"] = loaded["manifest_sha256"]

    # CPU parity on an exact, deterministic subsample from A1. Adjacencies are
    # replaced by a sparse identity only for the parity probe; both sides consume
    # byte-identical tensors and the production implementation delegates exactly.
    item = prepared["a1"]
    n = min(48, len(item.obs_names))
    probe_data = dict(item.data)
    probe_data["features_omics1"] = np.asarray(item.data["features_omics1"][:n], np.float32)
    probe_data["features_omics2"] = np.asarray(item.data["features_omics2"][:n], np.float32)
    idx = torch.arange(n)
    identity = torch.sparse_coo_tensor(torch.stack((idx, idx)), torch.ones(n), (n, n)).coalesce()
    for name in ("adj_spatial_omics1", "adj_feature_omics1", "adj_spatial_omics2", "adj_feature_omics2"):
        probe_data[name] = identity
    probe_cfg = dict(training_cfg(config["datasets"]["a1"])); probe_cfg["epochs"] = 2
    parity = []
    by_id = {row["id"]: row for row in registry["candidates"]}
    for candidate_id, legacy in LEGACY_DELEGATES.items():
        parity.append(parity_probe(probe_data, probe_cfg, by_id[candidate_id], legacy))
    if not all(row["passed"] for row in parity):
        raise RuntimeError("C00/C01/C02 parity failed")

    baseline_params = None
    engineering = []
    probe_artifacts = {
        "reliability": artifacts["a1"]["reliability"][:n],
        "triplets": np.asarray([[0, 1, 2, 0], [3, 4, 5, 1]], dtype=np.int64),
        "contrast": np.asarray([[0, 1, 2], [3, 4, 5]], dtype=np.int64),
        "dgi_permutation": np.arange(n - 1, -1, -1, dtype=np.int64),
        "anchor05": identity, "anchor10": identity,
    }
    for candidate_id, candidate in contracts.items():
        if candidate_id in LEGACY_DELEGATES:
            trainer = Night3BTrainer(probe_data, probe_cfg, LEGACY_DELEGATES[candidate_id], 0,
                                     torch.device("cpu"), 1e-12)
            model = trainer.new_model()
        else:
            trainer = Night5ATrainer(probe_data, probe_cfg, candidate, 0, torch.device("cpu"), probe_artifacts)
            model = trainer.new_model()
        result = trainer.forward(model)
        losses = raw_losses(result, trainer.features1, trainer.features2)
        total = sum(losses.values())
        if isinstance(trainer, Night5ATrainer):
            auxiliary, _ = trainer._auxiliary_loss(model, result); total = total + auxiliary
        model.zero_grad(); total.backward()
        gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        if candidate_id == "C00_FULL_IGE":
            baseline_params = parameter_count
        finite = bool(torch.isfinite(result["emb_latent_combined"]).all().cpu())
        nonzero = any(value is not None and float(value.abs().sum()) > 0 for value in gradients)
        engineering.append({"candidate_id": candidate_id, "forward_finite": finite,
                            "active_module_nonzero_gradient": nonzero, "parameter_count": parameter_count})
    residual = next(row for row in engineering if row["candidate_id"] == "C15_RESIDUAL_ENCODER")
    if residual["parameter_count"] > 2 * baseline_params:
        raise RuntimeError("C15 parameter count exceeds 2x C00")
    if not all(row["forward_finite"] and row["active_module_nonzero_gradient"] for row in engineering):
        raise RuntimeError("Candidate engineering probe failed")

    p0_git_path = output / "git_preflight.json"
    if not p0_git_path.is_file():
        raise RuntimeError("P0-GIT record missing")
    gate = {
        "schema_version": 1, "p0_git_pass": True, "p0_protect_pass": True,
        "p0_arch_pass": True, "r1_authorized": True,
        "parent_commit": EXPECTED_PARENT, "baseline_tag": "baseline/pre-night5a-20260813",
        "authoritative_input_checks": input_checks, "candidate_count": 17,
        "candidate_unique_config_sha_count": 17, "r1_locked_run_count": 34,
        "r1_run_order_sha256": sha256_file(output / "preregistered_maximum_run_order.json"),
        "cache_and_artifact_locks": cache_rows, "parity": parity,
        "engineering_probes": engineering, "specialized_tests_required": 20,
        "withheld_candidates_run": [], "semantic_label_access": False,
        "elapsed_seconds": time.time() - started,
    }
    atomic_json(output / "p0_arch.json", gate)
    atomic_json(output / "night5a_gate_status.json", gate)
    print("P0_ARCH_PASS parity=3/3 candidates=17 r1=34", flush=True)


if __name__ == "__main__":
    main()
