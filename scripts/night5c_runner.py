#!/usr/bin/env python3
"""Run only the locked Night-5C corrected or conditional Laplacian units."""
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
from SpaLORA.night5a_rnd import load_label_free_artifacts, sha256_file
from SpaLORA.night5c_semantic import (CORRECTIVE_IDS, Night5CTrainer, assert_contract_match,
                                      assert_uniform_forward, canonical_sha256, declared_contract,
                                      load_corrective_registry, resolved_contract)
from scripts.night3a_runner import cluster_exact, validate_attention

CONFIG_PATH = REPO / "configs/night5c_laplacian_correction.json"
REQUIRED = ("embedding.npz", "attention.npz", "clusters.csv", "observation_ids.csv",
            "loss_trajectory.csv", "coefficient_probe.json", "model_state.pt", "run_manifest.json")


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True); handle.flush(); os.fsync(handle.fileno())
    os.replace(str(temp), str(path))


def write_csv(path: Path, rows: list) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())


def training_cfg(dataset: dict) -> dict:
    return {key: dataset[key] for key in ("embedding_dim", "epochs", "loss_factors", "locked_m_bad_expected")}


def run_directory(config, stage, dataset, candidate, seed):
    root_key = "corrected_raw_runs" if stage == "S1_CORRECTED" else "conditional_raw_runs"
    return Path(config["paths"][root_key]) / dataset / candidate / ("seed_%d" % seed)


def old_invalid_directory(config, dataset, candidate, seed):
    return Path(config["paths"]["night5b_raw_runs"]) / dataset / candidate / ("seed_%d" % seed)


def verify_existing(directory: Path, config_sha: str) -> bool:
    if not all((directory / name).is_file() for name in REQUIRED) or (directory / "failure.json").exists():
        return False
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if manifest.get("config_sha256") != config_sha or not manifest.get("semantic_contract_match"):
        return False
    return all(sha256_file(directory / name) == value for name, value in manifest.get("artifact_sha256", {}).items())


def stage_rows(stage: str, output: Path):
    rows, ordinal = [], 0
    if stage == "S1_CORRECTED":
        candidates = list(CORRECTIVE_IDS); seeds = (0, 1, 2)
    else:
        decision = json.loads((output / "recomputed_s1_topup_candidates.json").read_text())
        candidates = list(decision["newly_selected_incomplete_laplacian_candidates"]); seeds = (3, 4)
        if len(candidates) > 4:
            raise RuntimeError("Conditional Laplacian candidate cap exceeded")
    for dataset in ("a1", "placenta"):
        for cid in candidates:
            for seed in seeds:
                ordinal += 1
                rows.append({"ordinal": ordinal, "dataset": dataset, "candidate_id": cid, "seed": seed})
    expected = 24 if stage == "S1_CORRECTED" else len(candidates) * 4
    if len(rows) != expected or len(rows) > (24 if stage == "S1_CORRECTED" else 16):
        raise RuntimeError("Locked Night-5C stage budget mismatch")
    return rows


def run_one(config, candidate, prepared, artifacts, row, stage):
    dataset, cid, seed = row["dataset"], row["candidate_id"], int(row["seed"])
    directory = run_directory(config, stage, dataset, cid, seed)
    directory.mkdir(parents=True, exist_ok=True)
    if verify_existing(directory, candidate["config_sha256"]):
        return {"dataset":dataset,"candidate_id":cid,"seed":seed,"status":"success","source_mode":"existing_valid_night5c_resume",
                "record_path":str(directory/"run_manifest.json"),"record_sha256":sha256_file(directory/"run_manifest.json"),"semantic_label_access":False}
    if any(directory.iterdir()):
        raise RuntimeError("Partial or mismatched Night-5C directory cannot be overwritten: %s" % directory)
    old_dir = old_invalid_directory(config, dataset, cid, seed)
    if stage == "S1_CORRECTED":
        if not (old_dir / "run_manifest.json").is_file():
            raise RuntimeError("Missing superseded invalid Night-5B manifest")
        supersedes = {"old_status":"permanently_invalid","old_run_manifest":str(old_dir/"run_manifest.json"),
                      "old_run_manifest_sha256":sha256_file(old_dir/"run_manifest.json"),"old_directory_read_only":True}
    else:
        supersedes = None
    try:
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        cfg = training_cfg(config["datasets"][dataset])
        trainer = Night5CTrainer(prepared.data, cfg, candidate, seed, torch.device("cuda:0"), artifacts, config["ige_epsilon"])
        probe_model = trainer.new_model()
        declared = declared_contract(candidate)
        resolved = resolved_contract(trainer, probe_model)
        assert_contract_match(declared, resolved)
        forward_deviation = assert_uniform_forward(trainer, probe_model)
        declared_sha = canonical_sha256(declared)
        resolved_sha = canonical_sha256({k:v for k,v in resolved.items() if k != "laplacian_frozen_coefficient"})
        del probe_model
        started = time.perf_counter()
        locked_input = input_sha256(prepared.data, prepared.obs_names, prepared.data["selected_gene_names"])
        result = trainer.train(); torch.cuda.synchronize(); train_seconds = time.perf_counter() - started
        attention_deviation = validate_attention(result.output, len(prepared.obs_names))
        for key in ("alpha", "alpha_omics1", "alpha_omics2"):
            if not np.array_equal(np.asarray(result.output[key]), np.full_like(np.asarray(result.output[key]), 0.5)):
                raise RuntimeError("Finished run lost exact uniform attention: %s" % key)
        predicted = cluster_exact(np.asarray(result.output["SpaLORA"], np.float32), config["datasets"][dataset]["n_clusters"], config["clustering"]["random_seed"])
        np.savez_compressed(directory/"embedding.npz", SpaLORA=np.asarray(result.output["SpaLORA"], np.float32))
        np.savez_compressed(directory/"attention.npz", alpha=np.asarray(result.output["alpha"],np.float32),
            alpha_omics1=np.asarray(result.output["alpha_omics1"],np.float32), alpha_omics2=np.asarray(result.output["alpha_omics2"],np.float32))
        pd.DataFrame({"observation_id":prepared.obs_names.astype(str)}).to_csv(directory/"observation_ids.csv",index=False)
        pd.DataFrame({"observation_id":prepared.obs_names.astype(str),"cluster":predicted}).to_csv(directory/"clusters.csv",index=False)
        write_csv(directory/"loss_trajectory.csv",list(result.logs))
        torch.save(result.model.state_dict(),directory/"model_state.pt")
        auxiliary=getattr(result,"auxiliary",{}); probe=result.probe or {}
        atomic_json(directory/"coefficient_probe.json",{"candidate_id":cid,"raw_initial_losses":result.initial_losses,
            "raw_rms_gradients":probe.get("gradients",{}),"frozen_coefficients":result.coefficients,
            "initial_state_sha256":result.initial_state_sha256,"auxiliary":auxiliary,
            "runtime_semantic_contract":{"declared":declared,"resolved":resolved,"declared_sha256":declared_sha,
                "resolved_sha256":resolved_sha,"semantic_contract_match":True},"semantic_label_access":False})
        artifacts_sha={name:sha256_file(directory/name) for name in REQUIRED if name!="run_manifest.json"}
        manifest={"schema_version":1,"dataset":dataset,"candidate_id":cid,"seed":seed,"stage_first_executed":stage,
            "run_order_ordinal_within_stage":int(row["ordinal"]),"config_sha256":candidate["config_sha256"],
            "source_mode":"night5c_corrected_uniform_all_clean_room","locked_input_sha256":locked_input,"epochs":int(cfg["epochs"]),
            "initial_state_sha256":result.initial_state_sha256,"final_state_sha256":result.final_state_sha256,
            "artifact_sha256":artifacts_sha,"semantic_label_access":False,"declared_runtime_contract":declared,
            "resolved_runtime_contract":resolved,"declared_runtime_contract_sha256":declared_sha,
            "resolved_runtime_contract_sha256":resolved_sha,"semantic_contract_match":True,
            "uniform_forward_max_deviation":forward_deviation,"attention_max_row_sum_deviation":attention_deviation,
            "supersedes_invalid_run":supersedes,"timings":{"training_seconds":float(train_seconds),"total_seconds":float(time.perf_counter()-started)},
            "resources":{"gpu_peak_allocated_mib":float(torch.cuda.max_memory_allocated()/1024**2),
                "gpu_peak_reserved_mib":float(torch.cuda.max_memory_reserved()/1024**2),
                "process_peak_rss_mib":float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024.0)}}
        atomic_json(directory/"run_manifest.json",manifest)
        print("DONE %s %s seed=%d train=%.1fs"%(dataset,cid,seed,train_seconds),flush=True)
        return {"dataset":dataset,"candidate_id":cid,"seed":seed,"status":"success","source_mode":manifest["source_mode"],
                "record_path":str(directory/"run_manifest.json"),"record_sha256":sha256_file(directory/"run_manifest.json"),"semantic_label_access":False}
    except Exception as exc:
        atomic_json(directory/"failure.json",{"dataset":dataset,"candidate_id":cid,"seed":seed,"stage":stage,
            "error":repr(exc),"traceback":traceback.format_exc(),"retained":True})
        raise
    finally:
        gc.collect(); torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=("S1_CORRECTED","S2_CONDITIONAL"),required=True); args=parser.parse_args()
    config=json.loads(CONFIG_PATH.read_text()); output=REPO/config["paths"]["output_root"]
    gate=json.loads((output/"night5c_gate_status.json").read_text())
    if not gate.get("p0_semantic_pass"):
        raise RuntimeError("P0-SEMANTIC did not authorize training")
    corrective=load_corrective_registry(REPO/config["paths"]["corrective_registry"])
    old=json.loads((Path(config["paths"]["night5b_handoff"])/"candidate_contracts.json").read_text())["candidates"]
    contracts={row["id"]:dict(old[row["id"]]) for row in corrective["candidates"]}
    rows=stage_rows(args.stage,output)
    order={"schema_version":1,"stage":args.stage,"locked_before_training":True,"row_count":len(rows),"scientific_training_unit_count":len(rows),"runs":rows}
    order["canonical_sha256"]=canonical_sha256(order)
    order_path=output/("corrected_run_order.json" if args.stage=="S1_CORRECTED" else "conditional_run_order.json")
    if order_path.exists() and json.loads(order_path.read_text())!=order:
        raise RuntimeError("Locked run order drift")
    atomic_json(order_path,order)
    index=json.loads(Path(config["paths"]["cache_manifest"]).read_text()); prepared={}; artifacts={}; forbidden=ground_truth_csv_paths(config)
    for dataset in ("a1","placenta"):
        row=index["datasets"][dataset]; item=load_cache(Path(config["paths"]["night3af_root"])/row["directory"],row["manifest_sha256"])
        assert_training_payload_label_free(item.data,training_cfg(config["datasets"][dataset]),forbidden); prepared[dataset]=item
        artifacts[dataset]=load_label_free_artifacts(Path(config["paths"]["label_free_artifacts"])/dataset)
    window=ScientificWindow(config,output,"night5c_"+args.stage.lower()).install(); records=[]
    try:
        for row in rows:
            try:
                records.append(run_one(config,contracts[row["candidate_id"]],prepared[row["dataset"]],artifacts[row["dataset"]],row,args.stage))
            except Exception as exc:
                directory=run_directory(config,args.stage,row["dataset"],row["candidate_id"],row["seed"]); failure=directory/"failure.json"
                records.append({"dataset":row["dataset"],"candidate_id":row["candidate_id"],"seed":row["seed"],"status":"failure",
                    "record_path":str(failure),"record_sha256":sha256_file(failure),"semantic_label_access":False})
                print("FAILED_RETAINED",row,repr(exc),flush=True)
        failures=sum(r["status"]=="failure" for r in records)
        lock=output/("corrected_training_manifest.json" if args.stage=="S1_CORRECTED" else "conditional_training_manifest.json")
        atomic_json(lock,{"schema_version":1,"stage":args.stage,"locked_before_semantic_label_access":True,"row_count":len(records),
            "scientific_training_unit_count":len(rows),"success_count":len(records)-failures,"failure_count":failures,
            "run_order_sha256":sha256_file(order_path),"runs":records})
        firewall=window.close(passed=failures==0)
        if failures or not firewall["passed"]:
            raise RuntimeError("Night-5C training stage failed or label firewall failed")
        complete=output/("corrected_training_complete.json" if args.stage=="S1_CORRECTED" else "conditional_training_complete.json")
        atomic_json(complete,{"schema_version":1,"stage":args.stage,"row_count":len(records),"scientific_training_unit_count":len(rows),
            "failure_count":failures,"training_manifest_sha256":sha256_file(lock),"semantic_label_access":False})
        print(args.stage+"_TRAINING_LOCKED",len(records),0,len(rows))
    except Exception:
        try: window.close(passed=False)
        except Exception: pass
        raise


if __name__=="__main__":
    main()
