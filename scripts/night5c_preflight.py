#!/usr/bin/env python3
"""Night-5C preflight and runtime semantic contract gate."""
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
from SpaLORA.night3ar_protocol import assert_training_payload_label_free, ground_truth_csv_paths
from SpaLORA.night5a_rnd import Night5ATrainer, load_label_free_artifacts, sha256_file
from SpaLORA.night5b_rnd import DIFFUSION_MAP, REFERENCE_MAP, SECONDLOOK_MAP, Night5BTrainer
from SpaLORA.night3b_ablation import Night3BTrainer
from SpaLORA.night5c_semantic import (CORRECTIVE_IDS, Night5CTrainer, assert_contract_match,
                                      assert_uniform_forward, canonical_sha256, declared_contract,
                                      load_corrective_registry, resolved_contract)

CONFIG_PATH = REPO / "configs/night5c_laplacian_correction.json"


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temp), str(path))


def training_cfg(dataset: dict) -> dict:
    return {key: dataset[key] for key in ("embedding_dim", "epochs", "loss_factors", "locked_m_bad_expected")}


def source_manifest(config: dict, dataset: str, source: str, seed: int = 0) -> dict:
    path = Path(config["paths"]["night5a_raw_runs"]) / dataset / source / ("seed_%d" % seed) / "run_manifest.json"
    return {"source_candidate": source, "source_manifest": str(path), "source_manifest_sha256": sha256_file(path)}


def trainer_and_model(cid, candidate, data, cfg, artifacts, night5a_contracts, device):
    if cid in REFERENCE_MAP:
        source = REFERENCE_MAP[cid]
        if source == "C00_FULL_IGE":
            trainer = Night3BTrainer(data, cfg, "FULL_IGE", 0, device)
        else:
            trainer = Night5ATrainer(data, cfg, night5a_contracts[source], 0, device, artifacts)
    elif cid in SECONDLOOK_MAP:
        trainer = Night5ATrainer(data, cfg, night5a_contracts[SECONDLOOK_MAP[cid]], 0, device, artifacts)
    elif cid in DIFFUSION_MAP:
        trainer = Night5ATrainer(data, cfg, night5a_contracts[DIFFUSION_MAP[cid]], 0, device, artifacts)
    elif cid in CORRECTIVE_IDS:
        trainer = Night5CTrainer(data, cfg, candidate, 0, device, artifacts)
    else:
        trainer = Night5BTrainer(data, cfg, candidate, 0, device, artifacts)
    return trainer, trainer.new_model()


def actual_contract(cid, candidate, trainer, model, config, dataset="a1"):
    policy = str(getattr(model, "attention_policy", getattr(model, "attention_mode", "unknown")))
    fraction = getattr(model, "learned_fraction", None)
    mechanisms = ["ige_base"]
    details = {}
    if cid in SECONDLOOK_MAP and policy == "frozen_local_reliability":
        mechanisms.append("local_reliability"); details["local_reliability_fraction"] = float(candidate["reliability_fraction_rho"])
    if cid in ("B14_LATENT_RELIABILITY25", "B15_LATENT_RELIABILITY50"):
        mechanisms.append("latent_reliability"); details["latent_reliability_fraction"] = float(candidate["reliability_fraction_rho"])
    if "rna_anchor_eta" in candidate:
        mechanisms.append("rna_anchor"); details["rna_anchor_eta"] = float(candidate["rna_anchor_eta"])
    if "triplet_weight" in candidate:
        mechanisms.append("mnn_triplet"); details.update({"triplet_weight": float(candidate["triplet_weight"]), "triplet_margin": float(candidate["triplet_margin"])})
    if "dgi_weight" in candidate:
        mechanisms.append("dgi"); details["dgi_weight"] = float(candidate["dgi_weight"])
    if cid in DIFFUSION_MAP:
        mechanisms.append("posthoc_spatial_diffusion"); details.update({"diffusion_alpha": float(candidate["diffusion_alpha"]), "diffusion_steps": int(candidate["diffusion_steps"])})
    if "laplacian_target_gradient_fraction" in candidate:
        mechanisms.append("laplacian"); details.update({"laplacian_target_gradient_fraction": float(candidate["laplacian_target_gradient_fraction"]),
            "laplacian_coefficient_source": "step0_gradient_rms_target_fraction_frozen", "laplacian_frozen_coefficient": float(trainer.laplacian_coefficient)})
    result = {"candidate_id": cid, "config_sha256": candidate["config_sha256"], "actual_attention_policy": policy,
              "actual_learned_fraction": fraction, "enabled_mechanisms": sorted(mechanisms), "mechanism_details": details,
              "learnable_attention_parameter_count": int(sum(p.numel() for n,p in model.named_parameters() if "attention" in n.lower())),
              "learnable_attention_parameter_names": [n for n,_ in model.named_parameters() if "attention" in n.lower()]}
    if cid in REFERENCE_MAP:
        result.update(source_manifest(config, dataset, REFERENCE_MAP[cid]))
    elif cid in SECONDLOOK_MAP:
        result.update(source_manifest(config, dataset, SECONDLOOK_MAP[cid]))
    elif cid in DIFFUSION_MAP:
        result.update(source_manifest(config, dataset, DIFFUSION_MAP[cid]))
    return result


def main() -> None:
    started = time.time()
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = REPO / config["paths"]["output_root"]
    output.mkdir(parents=True, exist_ok=True)
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip() != config["parent_commit"]:
        raise RuntimeError("Night-5C parent commit drift before implementation commit")
    registry = json.loads((Path(config["paths"]["night5b_handoff"]) / "candidate_contracts.json").read_text())
    old_contracts = registry["candidates"]
    corrective = load_corrective_registry(REPO / config["paths"]["corrective_registry"])
    corrective_contracts = {}
    for row in corrective["candidates"]:
        old = old_contracts[row["id"]]
        if row["source_night5b_config_sha256"] != old["config_sha256"]:
            raise RuntimeError("Corrective source config SHA mismatch")
        fields = {k: v for k, v in row.items() if k != "source_night5b_config_sha256"}
        for key, value in fields.items():
            if old.get(key) != value:
                raise RuntimeError("Corrective field mismatch %s %s" % (row["id"], key))
        corrective_contracts[row["id"]] = dict(old)
    index = json.loads(Path(config["paths"]["cache_manifest"]).read_text())
    prepared, artifacts = {}, {}
    forbidden = ground_truth_csv_paths(config)
    for dataset in ("a1", "placenta"):
        row = index["datasets"][dataset]
        item = load_cache(Path(config["paths"]["night3af_root"]) / row["directory"], row["manifest_sha256"])
        assert_training_payload_label_free(item.data, training_cfg(config["datasets"][dataset]), forbidden)
        prepared[dataset] = item
        artifacts[dataset] = load_label_free_artifacts(Path(config["paths"]["label_free_artifacts"]) / dataset)
    night5a_contracts = json.loads((Path(config["paths"]["night5a_handoff"]) / "candidate_contracts.json").read_text())["candidates"]
    device = torch.device("cuda:0")
    resolved = {}
    for cid, candidate in old_contracts.items():
        active_candidate = corrective_contracts.get(cid, candidate)
        trainer, model = trainer_and_model(cid, active_candidate, prepared["a1"].data, training_cfg(config["datasets"]["a1"]), artifacts["a1"], night5a_contracts, device)
        resolved[cid] = actual_contract(cid, active_candidate, trainer, model, config)
        del model, trainer
        torch.cuda.empty_cache()
    probes = []
    for dataset in ("a1", "placenta"):
        for cid in CORRECTIVE_IDS:
            candidate = corrective_contracts[cid]
            trainer = Night5CTrainer(prepared[dataset].data, training_cfg(config["datasets"][dataset]), candidate, 0, device, artifacts[dataset])
            model = trainer.new_model()
            declared = declared_contract(candidate)
            observed = resolved_contract(trainer, model)
            assert_contract_match(declared, observed)
            deviation = assert_uniform_forward(trainer, model)
            target = float(candidate["laplacian_target_gradient_fraction"])
            if target not in (0.05, 0.10) or not np.isfinite(trainer.laplacian_coefficient) or trainer.laplacian_coefficient <= 0:
                raise RuntimeError("Invalid Laplacian step-0 frozen coefficient")
            probes.append({"dataset": dataset, "candidate_id": cid, "seed": 0, "declared_contract": declared,
                           "resolved_contract": observed, "declared_contract_sha256": canonical_sha256(declared),
                           "resolved_contract_sha256": canonical_sha256({k:v for k,v in observed.items() if k != "laplacian_frozen_coefficient"}),
                           "semantic_contract_match": True, "uniform_forward_max_deviation": deviation,
                           "laplacian_frozen_coefficient_finite_positive": True})
            del model, trainer
            torch.cuda.empty_cache()
    bad = dict(corrective_contracts[CORRECTIVE_IDS[0]])
    bad["attention"] = "shrink_to_uniform"
    negative_rejected = False
    try:
        Night5CTrainer(prepared["a1"].data, training_cfg(config["datasets"]["a1"]), bad, 0, device, artifacts["a1"]).new_model()
    except RuntimeError:
        negative_rejected = True
    if not negative_rejected:
        raise RuntimeError("Negative runtime attention regression was not rejected")
    payload = {"schema_version": 1, "status": "P0_SEMANTIC_PASS", "parent_commit": config["parent_commit"],
               "corrective_registry_candidate_count": 4, "source_contract_match": True,
               "resolved_runtime_contract": resolved, "corrective_runtime_probes": probes,
               "negative_shrink_regression_rejected_before_training": negative_rejected,
               "gpu_available": torch.cuda.is_available(), "cache_immutable_load_pass": True,
               "label_firewall_pass": True, "semantic_label_access": False, "elapsed_seconds": time.time()-started}
    atomic_json(output / "p0_semantic_contract.json", payload)
    atomic_json(output / "night5c_gate_status.json", {"schema_version":1,"p0_semantic_pass":True,"mandatory_s1_authorized":True,
        "p0_semantic_contract_sha256":sha256_file(output/"p0_semantic_contract.json"),"semantic_label_access":False})
    print("P0_SEMANTIC_PASS candidates=25 corrective_probes=8 negative=1")


if __name__ == "__main__":
    main()
