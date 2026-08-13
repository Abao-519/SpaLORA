#!/usr/bin/env python3
"""Run locked Night-5B S1/S2 units without opening semantic labels."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("R_HOME", "/root/miniconda3/envs/SpaLORA/lib/R")
REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from SpaLORA.night3af_cache import load_cache
from SpaLORA.night3a_ige import input_sha256
from SpaLORA.night3ar_protocol import ScientificWindow, assert_training_payload_label_free, ground_truth_csv_paths
from SpaLORA.night5a_rnd import Night5ATrainer, load_label_free_artifacts, sha256_file
from SpaLORA.night5b_rnd import (DIFFUSION_MAP, REFERENCE_MAP, SECONDLOOK_MAP, Night5BTrainer,
                                 assert_development_dataset, load_registry, registry_contracts,
                                 single_step_diffusion)
from scripts.night3a_runner import cluster_exact, validate_attention

CONFIG_PATH = REPO / "configs/night5b_secondlook_rescue.json"
REQUIRED_NEW = ("embedding.npz", "attention.npz", "clusters.csv", "observation_ids.csv",
                "loss_trajectory.csv", "coefficient_probe.json", "model_state.pt", "run_manifest.json")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def write_csv(path: Path, rows: list) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())


def training_cfg(dataset: dict) -> dict:
    return {key: dataset[key] for key in ("embedding_dim", "epochs", "loss_factors", "locked_m_bad_expected")}


def raw_path(config, dataset, candidate, seed):
    return Path(config["paths"]["raw_runs"]) / dataset / candidate / ("seed_%d" % seed)


def night5a_path(config, dataset, candidate, seed):
    return Path(config["paths"]["night5a_raw_runs"]) / dataset / candidate / ("seed_%d" % seed)


def verify_source(directory: Path) -> dict:
    manifest_path = directory / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, expected in manifest["artifact_sha256"].items():
        if sha256_file(directory / name) != expected:
            raise RuntimeError("Locked Night-5A source artifact drift: %s" % (directory / name))
    return manifest


def source_record(config, dataset, alias, source, seed, mode):
    directory = night5a_path(config, dataset, source, seed)
    manifest = verify_source(directory)
    return {"dataset": dataset, "candidate_id": alias, "seed": int(seed), "status": "success",
            "record_path": str(directory / "run_manifest.json"), "record_sha256": sha256_file(directory / "run_manifest.json"),
            "source_candidate_id": source, "source_mode": mode, "source_embedding_sha256": sha256_file(directory / "embedding.npz"),
            "source_config_sha256": manifest["config_sha256"], "semantic_label_access": False}


def existing_valid(directory: Path, candidate_sha: str) -> bool:
    if not all((directory / name).is_file() for name in REQUIRED_NEW) or (directory / "failure.json").exists():
        return False
    manifest = json.loads((directory / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("config_sha256") != candidate_sha:
        return False
    return all(sha256_file(directory / name) == value for name, value in manifest.get("artifact_sha256", {}).items())


def run_training(config, contracts, night5a_contracts, prepared, artifacts, dataset, candidate_id, seed, ordinal, stage):
    assert_development_dataset(dataset)
    candidate = contracts[candidate_id]; directory = raw_path(config, dataset, candidate_id, seed)
    directory.mkdir(parents=True, exist_ok=True)
    if existing_valid(directory, candidate["config_sha256"]):
        return {"dataset": dataset, "candidate_id": candidate_id, "seed": seed, "status": "success",
                "record_path": str(directory / "run_manifest.json"), "record_sha256": sha256_file(directory / "run_manifest.json"),
                "source_mode": "existing_valid_resume", "semantic_label_access": False}
    if any(directory.iterdir()):
        raise RuntimeError("Partial/mismatched run cannot be overwritten: %s" % directory)
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(); started = time.perf_counter()
        cfg = training_cfg(config["datasets"][dataset])
        if candidate_id in SECONDLOOK_MAP:
            source = SECONDLOOK_MAP[candidate_id]
            trainer = Night5ATrainer(prepared.data, cfg, night5a_contracts[source], seed, torch.device("cuda:0"), artifacts, config["ige_epsilon"])
            source_mode = "night5a_exact_secondlook"
        else:
            trainer = Night5BTrainer(prepared.data, cfg, candidate, seed, torch.device("cuda:0"), artifacts, config["ige_epsilon"])
            source_mode = "night5b_clean_room"
        locked_input = input_sha256(prepared.data, prepared.obs_names, prepared.data["selected_gene_names"])
        result = trainer.train(); torch.cuda.synchronize(); train_seconds = time.perf_counter() - started
        attention_deviation = validate_attention(result.output, len(prepared.obs_names))
        predicted = cluster_exact(np.asarray(result.output["SpaLORA"], np.float32), config["datasets"][dataset]["n_clusters"], config["clustering"]["random_seed"])
        np.savez_compressed(directory / "embedding.npz", SpaLORA=np.asarray(result.output["SpaLORA"], np.float32))
        np.savez_compressed(directory / "attention.npz", alpha=np.asarray(result.output["alpha"], np.float32),
                            alpha_omics1=np.asarray(result.output["alpha_omics1"], np.float32), alpha_omics2=np.asarray(result.output["alpha_omics2"], np.float32))
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str)}).to_csv(directory / "observation_ids.csv", index=False)
        pd.DataFrame({"observation_id": prepared.obs_names.astype(str), "cluster": predicted}).to_csv(directory / "clusters.csv", index=False)
        write_csv(directory / "loss_trajectory.csv", list(result.logs))
        torch.save(result.model.state_dict(), directory / "model_state.pt")
        probe = result.probe or {}; auxiliary = getattr(result, "auxiliary", {})
        if "latent_reliability" in auxiliary:
            atomic_json(directory / "latent_reliability_audit.json", auxiliary["latent_reliability"])
        atomic_json(directory / "coefficient_probe.json", {"candidate_id": candidate_id, "raw_initial_losses": result.initial_losses,
                    "raw_rms_gradients": probe.get("gradients", {}), "frozen_coefficients": result.coefficients,
                    "initial_state_sha256": result.initial_state_sha256, "auxiliary": auxiliary, "semantic_label_access": False})
        artifact_sha = {name: sha256_file(directory / name) for name in REQUIRED_NEW if name != "run_manifest.json"}
        if (directory / "latent_reliability_audit.json").is_file(): artifact_sha["latent_reliability_audit.json"] = sha256_file(directory / "latent_reliability_audit.json")
        manifest = {"schema_version": 1, "dataset": dataset, "candidate_id": candidate_id, "seed": int(seed), "stage_first_executed": stage,
                    "run_order_ordinal_within_stage": int(ordinal), "config_sha256": candidate["config_sha256"], "source_mode": source_mode,
                    "locked_input_sha256": locked_input, "epochs": int(cfg["epochs"]), "initial_state_sha256": result.initial_state_sha256,
                    "final_state_sha256": result.final_state_sha256, "artifact_sha256": artifact_sha, "semantic_label_access": False,
                    "attention_max_row_sum_deviation": attention_deviation,
                    "timings": {"training_seconds": float(train_seconds), "total_seconds": float(time.perf_counter() - started)},
                    "resources": {"gpu_peak_allocated_mib": float(torch.cuda.max_memory_allocated()/1024**2),
                                  "gpu_peak_reserved_mib": float(torch.cuda.max_memory_reserved()/1024**2),
                                  "process_peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0)}}
        atomic_json(directory / "run_manifest.json", manifest)
        print("DONE %s %s seed=%d train=%.1fs" % (dataset, candidate_id, seed, train_seconds), flush=True)
        return {"dataset": dataset, "candidate_id": candidate_id, "seed": int(seed), "status": "success",
                "record_path": str(directory / "run_manifest.json"), "record_sha256": sha256_file(directory / "run_manifest.json"),
                "source_mode": source_mode, "semantic_label_access": False}
    except Exception as exc:
        atomic_json(directory / "failure.json", {"dataset": dataset, "candidate_id": candidate_id, "seed": seed, "stage": stage,
                    "error": repr(exc), "traceback": traceback.format_exc(), "retained": True})
        raise
    finally:
        gc.collect(); torch.cuda.empty_cache()


def run_diffusion(config, contracts, prepared, dataset, candidate_id, seed, ordinal):
    candidate = contracts[candidate_id]; source = DIFFUSION_MAP[candidate_id]; source_dir = night5a_path(config, dataset, source, seed)
    source_manifest = verify_source(source_dir); directory = raw_path(config, dataset, candidate_id, seed); directory.mkdir(parents=True, exist_ok=True)
    if existing_valid(directory, candidate["config_sha256"]):
        return {"dataset": dataset, "candidate_id": candidate_id, "seed": seed, "status": "success", "record_path": str(directory/"run_manifest.json"),
                "record_sha256": sha256_file(directory/"run_manifest.json"), "source_mode": "existing_valid_diffusion", "semantic_label_access": False}
    if any(directory.iterdir()): raise RuntimeError("Partial diffusion output cannot be overwritten")
    with np.load(source_dir / "embedding.npz") as ar: source_embedding = np.asarray(ar["SpaLORA"], np.float32)
    out = single_step_diffusion(source_embedding, prepared.data["adj_spatial_omics1"], float(candidate["diffusion_alpha"]))
    predicted = cluster_exact(out, config["datasets"][dataset]["n_clusters"], config["clustering"]["random_seed"])
    np.savez_compressed(directory / "embedding.npz", SpaLORA=out)
    import shutil
    shutil.copy2(source_dir / "attention.npz", directory / "attention.npz")
    shutil.copy2(source_dir / "observation_ids.csv", directory / "observation_ids.csv")
    pd.DataFrame({"observation_id": pd.read_csv(source_dir/"observation_ids.csv")["observation_id"].astype(str), "cluster": predicted}).to_csv(directory/"clusters.csv", index=False)
    write_csv(directory/"loss_trajectory.csv", [{"step": 0, "source_locked_embedding": True, "diffusion_alpha": candidate["diffusion_alpha"]}])
    atomic_json(directory/"coefficient_probe.json", {"candidate_id": candidate_id, "diffusion_alpha": candidate["diffusion_alpha"], "diffusion_steps": 1, "semantic_label_access": False})
    torch.save({"posthoc_diffusion": True, "source_final_state_sha256": source_manifest["final_state_sha256"]}, directory/"model_state.pt")
    artifact_sha={name:sha256_file(directory/name) for name in REQUIRED_NEW if name!="run_manifest.json"}
    manifest={"schema_version":1,"dataset":dataset,"candidate_id":candidate_id,"seed":seed,"stage_first_executed":"S1","run_order_ordinal_within_stage":ordinal,
              "config_sha256":candidate["config_sha256"],"source_mode":"locked_night5a_embedding_single_step_diffusion","source_candidate_id":source,
              "source_manifest_sha256":sha256_file(source_dir/"run_manifest.json"),"source_embedding_sha256":sha256_file(source_dir/"embedding.npz"),
              "source_config_sha256":source_manifest["config_sha256"],"artifact_sha256":artifact_sha,"semantic_label_access":False,
              "timings":{"training_seconds":0.0,"total_seconds":0.0},"resources":{"gpu_peak_allocated_mib":0.0,"gpu_peak_reserved_mib":0.0,"process_peak_rss_mib":0.0}}
    atomic_json(directory/"run_manifest.json",manifest)
    return {"dataset":dataset,"candidate_id":candidate_id,"seed":seed,"status":"success","record_path":str(directory/"run_manifest.json"),
            "record_sha256":sha256_file(directory/"run_manifest.json"),"source_mode":manifest["source_mode"],"semantic_label_access":False}


def s1_order():
    rows=[]; ordinal=0
    combo_ids = ("B09_SHRINK25_ANCHOR05", "B10_SHRINK25_ANCHOR10", "B11_SHRINK25_MNN005",
                 "B12_SHRINK25_MNN010", "B13_SHRINK25_DGI010", "B14_LATENT_RELIABILITY25",
                 "B15_LATENT_RELIABILITY50", "B16_SHRINK25_ANCHOR05_MNN005")
    laplacian_ids = ("B21_C09_LAPLACIAN005", "B22_C09_LAPLACIAN010",
                     "B23_C10_LAPLACIAN005", "B24_C10_LAPLACIAN010")
    for dataset in ("a1","placenta"):
        for cid in REFERENCE_MAP:
            for seed in range(5): ordinal+=1; rows.append({"ordinal":ordinal,"dataset":dataset,"candidate_id":cid,"seed":seed,"action":"reuse5"})
        for cid in SECONDLOOK_MAP:
            ordinal+=1; rows.append({"ordinal":ordinal,"dataset":dataset,"candidate_id":cid,"seed":0,"action":"reuse_seed0"})
            for seed in (1,2): ordinal+=1; rows.append({"ordinal":ordinal,"dataset":dataset,"candidate_id":cid,"seed":seed,"action":"train"})
        for cid in combo_ids:
            for seed in (0,1,2): ordinal+=1; rows.append({"ordinal":ordinal,"dataset":dataset,"candidate_id":cid,"seed":seed,"action":"train"})
        for cid in DIFFUSION_MAP:
            for seed in range(5): ordinal+=1; rows.append({"ordinal":ordinal,"dataset":dataset,"candidate_id":cid,"seed":seed,"action":"diffuse"})
        for cid in laplacian_ids:
            for seed in (0,1,2): ordinal+=1; rows.append({"ordinal":ordinal,"dataset":dataset,"candidate_id":cid,"seed":seed,"action":"train"})
    return rows


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("S1","S2"),required=True); args=parser.parse_args()
    config=json.loads(CONFIG_PATH.read_text(encoding="utf-8")); output=REPO/config["paths"]["output_root"]
    gate=json.loads((output/"night5b_gate_status.json").read_text(encoding="utf-8"))
    if not all(gate.get(k) for k in ("p0_git_pass","p0_protect_pass","p0_arch_pass","label_firewall_pass")):
        raise RuntimeError("Night-5B P0 did not authorize training")
    registry=load_registry(REPO/config["candidate_registry"]); contracts=registry_contracts(registry)
    night5a_contracts=json.loads((Path(config["paths"]["night5a_handoff"])/"candidate_contracts.json").read_text())["candidates"]
    index=json.loads(Path(config["paths"]["cache_manifest"]).read_text()); prepared={}; artifacts={}; forbidden=ground_truth_csv_paths(config)
    for dataset in ("a1","placenta"):
        row=index["datasets"][dataset]; item=load_cache(Path(config["paths"]["night3af_root"])/row["directory"],row["manifest_sha256"])
        assert_training_payload_label_free(item.data,training_cfg(config["datasets"][dataset]),forbidden); prepared[dataset]=item
        artifacts[dataset]=load_label_free_artifacts(Path(config["paths"]["label_free_artifacts"])/dataset)
    if args.stage=="S1": rows=s1_order()
    else:
        decision=json.loads((output/"s1_decision.json").read_text()); selected=decision["topup_candidates"]
        if len(selected)>6: raise RuntimeError("S2 candidate cap exceeded")
        rows=[]; ordinal=0
        for dataset in ("a1","placenta"):
            for cid in selected:
                for seed in (3,4): ordinal+=1; rows.append({"ordinal":ordinal,"dataset":dataset,"candidate_id":cid,"seed":seed,"action":"train"})
    training_count=sum(r["action"]=="train" for r in rows)
    expected=92 if args.stage=="S1" else len(json.loads((output/"s1_decision.json").read_text())["topup_candidates"])*4
    if training_count!=expected or (args.stage=="S2" and training_count>24): raise RuntimeError("Locked training budget mismatch")
    order={"schema_version":1,"stage":args.stage,"locked_before_training":True,"row_count":len(rows),"scientific_training_unit_count":training_count,"runs":rows}
    order["canonical_sha256"]=__import__("SpaLORA.night5a_rnd",fromlist=["canonical_sha256"]).canonical_sha256(order)
    order_path=output/(args.stage.lower()+"_run_order.json")
    if order_path.exists() and json.loads(order_path.read_text())!=order: raise RuntimeError("Locked run order drift")
    atomic_json(order_path,order)
    window=ScientificWindow(config,output,"training_"+args.stage.lower()).install(); records=[]
    try:
        for row in rows:
            try:
                if row["action"].startswith("reuse"):
                    source=REFERENCE_MAP.get(row["candidate_id"],SECONDLOOK_MAP.get(row["candidate_id"]))
                    record=source_record(config,row["dataset"],row["candidate_id"],source,row["seed"],row["action"])
                elif row["action"]=="diffuse": record=run_diffusion(config,contracts,prepared[row["dataset"]],row["dataset"],row["candidate_id"],row["seed"],row["ordinal"])
                else: record=run_training(config,contracts,night5a_contracts,prepared[row["dataset"]],artifacts[row["dataset"]],row["dataset"],row["candidate_id"],row["seed"],row["ordinal"],args.stage)
                records.append(record)
            except Exception as exc:
                directory=raw_path(config,row["dataset"],row["candidate_id"],row["seed"]); failure=directory/"failure.json"
                if not failure.exists(): atomic_json(failure,{"error":repr(exc),"traceback":traceback.format_exc(),"retained":True})
                records.append({"dataset":row["dataset"],"candidate_id":row["candidate_id"],"seed":row["seed"],"status":"failure","record_path":str(failure),"record_sha256":sha256_file(failure),"semantic_label_access":False})
                print("FAILED_RETAINED",row,repr(exc),flush=True)
        failures=sum(r["status"]=="failure" for r in records)
        lock=output/(args.stage.lower()+"_training_manifest.json")
        atomic_json(lock,{"schema_version":1,"stage":args.stage,"locked_before_semantic_label_access":True,"row_count":len(records),
                         "scientific_training_unit_count":training_count,"success_count":len(records)-failures,"failure_count":failures,
                         "run_order_sha256":sha256_file(order_path),"runs":records})
        firewall=window.close(passed=True)
        if not firewall["passed"]: raise RuntimeError("Label firewall failed")
        atomic_json(output/(args.stage.lower()+"_training_complete.json"),{"schema_version":1,"stage":args.stage,"row_count":len(records),
                    "scientific_training_unit_count":training_count,"failure_count":failures,"training_manifest_sha256":sha256_file(lock),"semantic_label_access":False})
        print(args.stage+"_TRAINING_LOCKED",len(records)-failures,failures,training_count)
    except Exception:
        try: window.close(passed=False)
        except Exception: pass
        raise


if __name__=="__main__": main()
