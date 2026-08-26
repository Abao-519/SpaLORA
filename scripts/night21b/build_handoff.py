"""Build the auditable Night-21B handoff from locked producer/evaluator artifacts."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import re


ARMS = [
    "T0_STRONG_CARRIER",
    "B0_BACKBONE_ONLY",
    "B1_BACKBONE_POINTWISE_ANCHOR",
    "B2_POSITIVE_RELATION_ONLY",
    "B3_BOUNDARY_RELATION_ONLY",
    "FULL_SIGNED_RELATIONAL_DISTILLATION",
]
LANE_ORDER = ["A1_K10", "TONSIL_S1_K4", "P22_K9", "PLACENTA_K10"]
LANE_FAMILY = {
    "A1_K10": "RNA_PROTEIN",
    "TONSIL_S1_K4": "RNA_PROTEIN",
    "P22_K9": "RNA_CHROMATIN",
    "PLACENTA_K10": "RNA_CHROMATIN",
}
HISTORICAL = {
    "A1_K10": (0.2761717672503009, 0.4219373620021054, "Night16E TSRE_FULL public benchmark HPO", "c8bfd5b1ab7e7645fd7be1d042ce79286908a31ada5db10eff76b729ccde3b5f"),
    "D1_K10": (0.3677500415439257, 0.4491708558910805, "Night16E TSRE_FULL public benchmark HPO", "b272e7d74c13bab5111d3dd99ff113487cb3052f7ab0492368b5fc96c8f4286c"),
    "TONSIL_S1_K4": (0.23668286718251594, 0.31736524060791105, "Night16C family-frozen CMBF-TPR", "61d64ed233b9bd5e8d800da195123f337eb1b1fcdaee08cec1068601738d5e0b"),
    "TONSIL_S2_K4": (0.2587853670379758, 0.31654623862882414, "Night16C frozen within-study transfer", "cd8233e38ed63e2a28b5833aa9da4d2402d0358c9613deaa6a353e4313339c0d"),
    "TONSIL_S3_K4": (0.3527884269769521, 0.3112754925256839, "Night16E TSRE_FULL public benchmark HPO", "d53ced292938c38d855b8ef2e73d12d0d8c5a99727614360f01b9767f12f23e8"),
    "P22_K9": (0.5963900556825741, 0.7182431752166996, "Night16H strict-LOSO feasible selector", "76ec89c8243fc4856692ddcd4cf2906df971381b968a5489c245a74c52eda3be"),
    "MISAR_E15_5_K7": (0.5416365903694235, 0.6669488935833199, "Night16C family-frozen CMBF-TPR", "1e7f48d5c56302ba7e728af2f70df32fe0f52f3dc49d1118ad2509ce58675829"),
    "PLACENTA_K10": (0.4999993975293688, 0.6311801428733832, "Night19B concatenated-feature KNN matched control", "cb8563c8f6651f207902865e50fdeb52eb97a753dc07c33cbcfa1be054130630"),
}
NIGHT21A_CARRIER = {
    "A1_K10": (0.2364316186, 0.3887644040),
    "TONSIL_S1_K4": (0.1528164418, 0.2072246761),
    "P22_K9": (0.4708168993, 0.5998766936),
    "PLACENTA_K10": (0.3451786733, 0.5196462007),
}


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    fields = fields or list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def json_load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(working: Path) -> list[dict]:
    rows = []
    for ep in sorted((working / "evaluation").glob("*.json")):
        mp = working / "discovery" / ep.name
        if not mp.exists():
            raise RuntimeError(f"missing producer manifest for {ep.name}")
        metric, manifest = json_load(ep), json_load(mp)
        artifact = mp.with_suffix(".npz")
        if sha_file(artifact) != manifest["artifact_sha256"]:
            raise RuntimeError(f"artifact hash drift: {artifact}")
        if metric["observed_k"] != manifest["k"]:
            raise RuntimeError(f"exact-K failure: {ep}")
        rows.append({
            "candidate_id": ep.stem, "lane": manifest["lane"], "family": manifest["family"],
            "arm": manifest["arm"], "config_id": manifest["config"]["config_id"],
            "training_seed": manifest["training_seed"], "endpoint_seed": manifest["endpoint_seed"],
            "steps": manifest["config"]["steps"], "ari": metric["ari"], "nmi": metric["nmi"],
            "ami": metric["ami"], "fmi": metric["fmi"], "n_total": metric["n_total"],
            "n_eval": metric["n_eval"], "k": metric["observed_k"],
            "min_cluster_size": metric["min_cluster_size"], "cluster_sizes": metric["cluster_sizes"],
            "neighbor_agreement": metric["neighbor_agreement"],
            "moran_indicator_macro": metric["moran_indicator_macro"],
            "geary_indicator_macro": metric["geary_indicator_macro"],
            "partition_sha256": manifest["partition_sha256"],
            "representation_sha256": manifest["representation_sha256"],
            "artifact_sha256": manifest["artifact_sha256"],
            "checkpoint_sha256": manifest["diagnostics"].get("checkpoint_sha256", ""),
            "parameter_changed": manifest["diagnostics"].get("parameter_changed", False),
            "optimizer_steps": manifest["diagnostics"].get("optimizer_steps", 0),
            "wall_seconds": manifest["wall_seconds"], "peak_rss_mb": manifest["peak_rss_mb"],
            "loss_relative_change_last_windows": manifest["diagnostics"].get("loss_relative_change_last_windows", ""),
            "labels_read_by_producer": manifest["ground_truth_label_values_read"],
            "labels_read_by_evaluator": 1,
        })
    return rows


def build_bridge(working: Path, output: Path) -> list[dict]:
    numeric = json_load(working / "frontier_carrier_bridge_numeric_audit.json")
    rows = []
    for item in numeric:
        lane = item["lane"]; h_ari, h_nmi, source, psha = HISTORICAL[lane]
        n21 = NIGHT21A_CARRIER.get(lane, ("NOT_RUN_IN_NIGHT21A", "NOT_RUN_IN_NIGHT21A"))
        rows.append({
            "lane": lane, "n": item["n"], "n_eval": item["n_eval"], "k": item["k"],
            "historical_frontier_ari": h_ari, "historical_frontier_nmi": h_nmi,
            "historical_frontier_source": source, "historical_partition_sha256": psha,
            "historical_partition_label_assisted": "yes" if "benchmark" in source.lower() or "selector" in source.lower() else "mixed_or_family_development",
            "reusable_carrier_status": "PRESENT_AND_REPLAYABLE",
            "carrier_shape": json.dumps(item["shape"]), "carrier_dtype": item["dtype"],
            "ordered_ids_sha256": item["ordered_ids_sha256"],
            "carrier_representation_sha256": item["representation_sha256"],
            "carrier_common_partition_sha256": item["common_partition_sha256"],
            "carrier_common_head_ari": item["common_head_ari"], "carrier_common_head_nmi": item["common_head_nmi"],
            "frontier_minus_carrier_ari": h_ari - item["common_head_ari"],
            "frontier_minus_carrier_nmi": h_nmi - item["common_head_nmi"],
            "night21a_carrier_ari": n21[0], "night21a_carrier_nmi": n21[1],
            "alignment_status": item["alignment_status"],
            "bridge_conclusion": "HIGHEST_PARTITION_IS_NOT_A_REUSABLE_PRECLUSTERING_REPRESENTATION",
        })
    write_csv(output / "frontier_carrier_bridge.csv", rows)
    return rows


def build_method_board(records: list[dict], output: Path) -> tuple[list[dict], dict]:
    stable = [r.copy() for r in records if r["config_id"] == "STABLE700_V1"]
    gate = {}
    for lane in LANE_ORDER:
        local = {r["arm"]: r for r in stable if r["lane"] == lane}
        if set(local) != set(ARMS):
            raise RuntimeError(f"incomplete matched board: {lane}: {sorted(local)}")
        full = local["FULL_SIGNED_RELATIONAL_DISTILLATION"]
        controls = [local[a] for a in ARMS[:-1]]
        passed = all(full["ari"] > c["ari"] and full["nmi"] > c["nmi"] for c in controls)
        gate[lane] = {
            "strict_independent_dual_gain": passed,
            "delta_vs_t0_ari": full["ari"] - local["T0_STRONG_CARRIER"]["ari"],
            "delta_vs_t0_nmi": full["nmi"] - local["T0_STRONG_CARRIER"]["nmi"],
            "delta_vs_b0_ari": full["ari"] - local["B0_BACKBONE_ONLY"]["ari"],
            "delta_vs_b0_nmi": full["nmi"] - local["B0_BACKBONE_ONLY"]["nmi"],
        }
        for row in local.values():
            row.update(gate[lane])
    write_csv(output / "method_contribution_board.csv", stable)
    return stable, gate


def build_score_board(records: list[dict], output: Path) -> list[dict]:
    rows = []
    for lane in LANE_ORDER:
        local = [r for r in records if r["lane"] == lane]
        max_ari = max(local, key=lambda r: (r["ari"], r["nmi"]))
        max_nmi = max(local, key=lambda r: (r["nmi"], r["ari"]))
        h_ari, h_nmi, h_source, _ = HISTORICAL[lane]
        rows.append({
            "lane": lane, "historical_frontier_ari": h_ari, "historical_frontier_nmi": h_nmi,
            "historical_frontier_source": h_source,
            "night21b_max_ari": max_ari["ari"], "night21b_max_ari_nmi": max_ari["nmi"],
            "night21b_max_ari_candidate": max_ari["candidate_id"],
            "night21b_max_nmi_ari": max_nmi["ari"], "night21b_max_nmi": max_nmi["nmi"],
            "night21b_max_nmi_candidate": max_nmi["candidate_id"],
            "score_frontier_advance": bool(max_ari["ari"] > h_ari or max_nmi["nmi"] > h_nmi),
            "selection_semantics": "transparent_post-lock_public_benchmark_HPO",
            "method_attribution": "NONE; this board is separate from matched contribution",
        })
    write_csv(output / "score_frontier_board.csv", rows)
    return rows


def build_dataset_shortlist(output: Path) -> list[dict]:
    rows = [
        {"priority":"HIGH","dataset":"GSE205055 mouse embryo","family":"RNA_CHROMATIN","modalities":"RNA+ATAC/CUT&Tag","n":"about 2187 in prior project audit","annotation_and_k":"author embryonic-region assignment; exact file/hash/mask/K still pending","authority":"GEO GSE205055 plus original source","estimated_size":"processed asset must be HEAD-audited","duplicate":"no","recommended_role":"best next independent chromatin confirmation after authority closure","downloaded_this_round":"no"},
        {"priority":"HIGH","dataset":"Stereo-CITE mouse thymus","family":"RNA_PROTEIN","modalities":"RNA+protein","n":"processed sections; exact unit mapping pending","annotation_and_k":"anatomical-zone label file/K provenance pending","authority":"spaMGCN data statement; SpatialGlue/Zenodo 10362607","estimated_size":"part of 671,921,238-byte archive","duplicate":"no","recommended_role":"protein external transfer after exact physical slice and labels close","downloaded_this_round":"no"},
        {"priority":"MEDIUM","dataset":"GSE198353 SPOTS mouse spleen","family":"RNA_PROTEIN","modalities":"RNA+21 ADT","n":"rep1 2653; rep2 2768","annotation_and_k":"no authoritative whole-tissue K-class partition closed","authority":"GEO GSE198353; spaMGCN/SpatialGlue source statements","estimated_size":"processed assets exist","duplicate":"no","recommended_role":"unlabeled repeatability/biology, not headline ARI","downloaded_this_round":"no"},
        {"priority":"MEDIUM","dataset":"GSE213264 Spatial-CITE-seq tonsil","family":"RNA_PROTEIN","modalities":"RNA+283 protein","n":"2492","annotation_and_k":"author computational RNA K8/protein K7; no mutually exclusive expert whole-tissue labels closed","authority":"GEO GSE213264; SEPAR paper","estimated_size":"about 30.4 MB raw archive in prior audit","duplicate":"independent of current tonsil slices","recommended_role":"biological/GC validation, not expert-domain ARI until label object closes","downloaded_this_round":"no"},
        {"priority":"EXISTING_CONTROL","dataset":"SCP2601 human placenta","family":"RNA_CHROMATIN","modalities":"RNA+ATAC-derived regulatory features","n":"1662","annotation_and_k":"original-author manual K10 cell types","authority":"SCP2601/original placenta study","estimated_size":"already local","duplicate":"existing Night18D+ unit","recommended_role":"mandatory external negative/control unit","downloaded_this_round":"no"},
        {"priority":"SECONDARY","dataset":"SCP2176 Slide-tags melanoma tumour","family":"RNA_CHROMATIN","modalities":"RNA+ATAC-LSI","n":"833 tumour cells","annotation_and_k":"author-derived computational tumour_1/tumour_2 K2","authority":"SCP2176 and MultiGATE Figshare","estimated_size":"already local","duplicate":"existing","recommended_role":"binary stress test only; not main generalization evidence","downloaded_this_round":"no"},
    ]
    write_csv(output / "dataset_expansion_shortlist.csv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--working", required=True); parser.add_argument("--repo", required=True)
    parser.add_argument("--output", required=True); parser.add_argument("--taskbook-sha", required=True)
    args = parser.parse_args(); working = Path(args.working); repo = Path(args.repo); output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = load_records(working)
    write_csv(output / "all_candidate_ledger.csv", records)
    bridge = build_bridge(working, output)
    method, gate = build_method_board(records, output)
    scores = build_score_board(records, output)
    shortlist = build_dataset_shortlist(output)

    rev1 = [r for r in records if r["config_id"] == "REV1_POSDOM"]
    rev1_pass = 0
    stable_by_lane = {lane: {r["arm"]: r for r in method if r["lane"] == lane} for lane in LANE_ORDER}
    for row in rev1:
        controls = [stable_by_lane[row["lane"]][a] for a in ARMS[:-1]]
        rev1_pass += int(all(row["ari"] > c["ari"] and row["nmi"] > c["nmi"] for c in controls))
    strict_count = sum(x["strict_independent_dual_gain"] for x in gate.values())
    classification = "RELATIONAL_MODULE_LOCAL_SIGNAL" if strict_count >= 2 else "NO_RELATIONAL_METHOD_SIGNAL"
    if classification != "NO_RELATIONAL_METHOD_SIGNAL":
        raise RuntimeError("classification contract drift")

    p0_rows = []
    for mp in sorted((working / "p0").glob("*.json")):
        if mp.stem.endswith("replay"): continue
        d = json_load(mp)
        p0_rows.append({"lane":d["lane"],"arm":d["arm"],"n":d["diagnostics"]["input_shapes"]["view1"][0],
                        "view1_shape":json.dumps(d["diagnostics"]["input_shapes"]["view1"]),
                        "view2_shape":json.dumps(d["diagnostics"]["input_shapes"]["view2"]),
                        "retained_shape":json.dumps(d["diagnostics"]["input_shapes"]["retained"]),
                        "graph_nnz":d["diagnostics"]["graph_nnz"],"optimizer_steps":d["diagnostics"]["optimizer_steps"],
                        "parameter_count":d["diagnostics"]["parameter_count"],"parameter_changed":d["diagnostics"]["parameter_changed"],
                        "artifact_sha256":d["artifact_sha256"],"checkpoint_sha256":d["diagnostics"].get("checkpoint_sha256","")})
    write_csv(output / "p0_registry.csv", p0_rows)

    replay_rows=[]
    for rp in sorted((working / "replay" / "stable700_v1").glob("*.json")):
        d=json_load(rp); replay_rows.append({"candidate_id":rp.stem,**d})
    if len(replay_rows) != 24 or not all(r["partition_exact"] and r["representation_close"] for r in replay_rows):
        raise RuntimeError(f"incomplete final replay: {len(replay_rows)}")
    write_csv(output / "fresh_process_replay_table.csv", replay_rows)

    resources=[]
    for r in records:
        if r["config_id"] in {"STABLE700_V1","REV1_POSDOM"}:
            resource_row = {k:r[k] for k in ["candidate_id","lane","arm","config_id","steps","optimizer_steps","parameter_changed","wall_seconds","peak_rss_mb","loss_relative_change_last_windows"]}
            resource_row.update({"gpu_device":"RTX 4080 SUPER","peak_gpu_mb":"NOT_PROFILED","dense_nxn_allocated":False})
            resources.append(resource_row)
    write_csv(output / "training_resource_table.csv", resources)

    failure_rows=[
        {"cycle":"STAGE0","status":"RESOURCE_GATE","issue":"root filesystem 94% used with about 1.9-2.0 GiB free","action":"no repository clone, data download, package installation, or environment creation","scientific_use":"source read-only audit only"},
        {"cycle":"SOURCE_AUDIT","status":"DECLARED_NON_EQUIVALENCE","issue":"official spaMGCN objective materializes dense NxN similarity/adjacency and notebooks inspect labels during epochs","action":"clean-room sparse registered-edge port; classified SOURCE_FAITHFUL_PORT, not official replay","scientific_use":"mature scaffold control only"},
        {"cycle":"P0_COMMON_V1","status":"PASS","issue":"none","action":"A1/P22 all six arms, optimizer update and checkpoint replay passed","scientific_use":"engineering P0"},
        {"cycle":"STABLE700_V1","status":"FREEZE_RECORDING_LIMITATION","issue":"formula was fixed in source/CLI and byte-locked manifests before evaluation, but no separate pre-run cycle-0 JSON was written","action":"reconstructed an exact cycle-0 manifest from all 24 producer manifests; not presented as preregistration","scientific_use":"label firewall remains intact; audit limitation disclosed"},
        {"cycle":"P0_COMMON_V1_300_STEPS","status":"SUPERSEDED_INSUFFICIENT_BUDGET","issue":"last-window unlabeled loss still changing 17-28 percent","action":"preserved; reran fixed official-style 700-step budget before matched conclusion","scientific_use":"not headline"},
        {"cycle":"STABLE700_V1","status":"SCIENTIFIC_GATE_1_OF_4","issue":"FULL strictly beat T0/B0/all atomic arms only on placenta","action":"allowed one mechanism revision","scientific_use":"main matched board"},
        {"cycle":"REV1_POSDOM","status":"SCIENTIFIC_GATE_1_OF_4","issue":"positive-dominant revision did not reach two-lane gate","action":"stop; no family freeze, transfer, multi-seed, or extra grid","scientific_use":"transparent benchmark development only"},
    ]
    write_csv(output / "failure_and_correction_ledger.csv", failure_rows)

    source_manifest={
        "schema":"night21b-spamgcn-source-audit-v1", "official_repository":"https://github.com/hongfeiZhang-source/spaMGCN",
        "fixed_commit":"77dfe67d4fd80c124722e68a0f71af36d10fa5fa", "license":"MIT",
        "license_url":"https://github.com/hongfeiZhang-source/spaMGCN/blob/master/LICENSE",
        "read_paths":["MGCN-main/model/AE.py","MGCN-main/model/Creat_model.py","MGCN-main/model/IGAE.py","MGCN-main/model/spaMGCN.py","MGCN-main/train/train3.py","MGCN-main/train/utils.py","MGCN-main/utils/misc.py","MGCN-main/utils/preprocess.py","MGCN-main/utils/utils.py","MGCN-main/config/configs.yml","MGCN-main/test  MouseE15.ipynb","MGCN-main/test  S1.ipynb","MGCN-main/test hippocampus.ipynb","MGCN-main/test HumanMelanoma.ipynb","MGCN-main/test  thymus.ipynb"],
        "port_status":"SOURCE_FAITHFUL_SPARSE_PORT_NOT_OFFICIAL_REPLAY",
        "official_dense_semantics":["sigmoid(z @ z.T) adjacency reconstruction","feature cosine N by N target","spatial adjacency converted to dense"],
        "port_changes":["registered sparse edge positives","deterministic sampled non-edge negatives","no dense N by N allocation","common KMeans endpoint separated from official label-inspecting notebooks"],
        "collision_sources":[
            {"work":"Relational Knowledge Distillation, CVPR 2019","url":"https://openaccess.thecvf.com/content_CVPR_2019/html/Park_Relational_Knowledge_Distillation_CVPR_2019_paper.html","collision":"transferring pairwise structural relations is prior art"},
            {"work":"Similarity-Preserving Knowledge Distillation, ICCV 2019","url":"https://openaccess.thecvf.com/content_ICCV_2019/html/Tung_Similarity-Preserving_Knowledge_Distillation_ICCV_2019_paper.html","collision":"teacher/student pairwise similarity preservation is prior art"},
            {"work":"Signed Graph Convolutional Network, 2018","url":"https://arxiv.org/abs/1808.06354","collision":"separate positive and negative graph relations are prior art"},
            {"work":"BANKSY, Nature Genetics 2024","url":"https://www.nature.com/articles/s41588-024-01664-3","collision":"spatial neighborhood mean and gradient texture are prior art"}
        ],
        "novelty_boundary":"spaMGCN backbone components are prior art; only sparse signed carrier-relation regularizer is project-specific, and this run did not establish stable independent contribution",
    }
    (output/"spaMGCN_source_manifest.json").write_text(json.dumps(source_manifest,indent=2,sort_keys=True),encoding="utf-8")
    source_md = """# spaMGCN source and collision audit\n\n- Official repository: https://github.com/hongfeiZhang-source/spaMGCN\n- Fixed commit: `77dfe67d4fd80c124722e68a0f71af36d10fa5fa`; license: MIT.\n- Read scope: model, train, utils, config, and five relevant notebooks listed in `spaMGCN_source_manifest.json`.\n- The public implementation is a mature two-view AE plus multi-order graph-convolution and late-fusion scaffold. It also computes dense `z @ z.T`, dense feature-similarity targets, and dense spatial adjacency in its training path. Relevant notebooks expose per-dataset epochs/hyperparameters and inspect public labels during iterative evaluation.\n- Night-21B is therefore a clean-room **source-faithful sparse port**, not an official numerical replay: dense all-pairs losses were replaced by registered sparse positives and deterministic sparse negatives, while two-view encoders, multi-order propagation, global order attention and late fusion retain the public semantics.\n- Relational Knowledge Distillation (CVPR 2019) already transfers pairwise structural relations; Similarity-Preserving Knowledge Distillation (ICCV 2019) already preserves pairwise teacher similarity without pointwise coordinate copying; Signed GCN already treats positive and negative graph links separately; BANKSY already supplies spatial neighborhood mean/gradient texture. The only narrowly project-specific object tested here is simultaneous protection of carrier-supported spatial neighbors and carrier/modality-supported spatial boundaries inside the fixed spatial-multiomics scaffold. The four-lane matched board did not establish stable independent contribution, so no novelty or paper-ready claim is made.\n"""
    (output/"source_code_collision_and_port_audit.md").write_text(source_md,encoding="utf-8")

    formula = """# Formula and selection contract\n\n+Night-21B uses a clean-room sparse spaMGCN-style scaffold: two modality-specific AE/graph encoders, four registered sparse propagation orders, global order attention, and late linear fusion. The common endpoint is KMeans with known public K, `n_init=20`, endpoint seed 0.\n\n+On registered spatial edges, carrier cosine determines the top-0.70 positive and bottom-0.30 boundary strata. Positive edges additionally require both raw-view cosines at or above their medians; boundary edges require both at or below their medians. Other/conflicting edges abstain. Positive loss is weighted `1-cos(z_i,z_j)`; boundary loss is weighted `relu(cos(z_i,z_j)-0.15)^2`. Per-stratum edge weights are normalized to mean one. B1 uses normalized pointwise MSE, B2 only positive relations, B3 only boundary relations, and FULL uses both.\n\n+Discovery labels are opened only by the independent evaluator after NPZ partition, partition SHA, representation SHA, artifact SHA and checkpoint SHA are locked. The stable main board uses 700 optimization steps because the 300-step loss curves were not in a stable window. One post-lock mechanism revision increased positive weight from 0.25 to 1.0 and reduced boundary weight from 0.25 to 0.10; it was frozen before rerunning and failed its two-lane gate. Family transfer and multi-seed confirmation were therefore not authorized.\n"""
    (output/"method_semantics_and_selection_contract.md").write_text(formula,encoding="utf-8")

    label_audit={
        "schema":"night21b-label-flow-audit-v1","producer_annotation_array_reads":0,
        "producer_accessed_keys":["ids","view1","view2","retained","graph0 CSR"],
        "producer_annotation_like_key_gate":"fail-closed","candidate_partition_locked_before_evaluation":True,
        "evaluator_authorities":{"A1/P22/tonsil_s1":"local_compute_kit labels_primary plus label_mask","placenta":"source h5ad obs cell_type after exact ordered ID assertion"},
        "discovery_label_use":"transparent post-lock public benchmark HPO and evaluation only",
        "within_run_checkpoint_selection":"fixed final step from unlabeled budget; no label read",
        "design_revisions_used":1,"maximum_authorized":2,"transfer_label_reads":0,
    }
    (output/"label_flow_audit.json").write_text(json.dumps(label_audit,indent=2,sort_keys=True),encoding="utf-8")

    stat=os.statvfs("/")
    disk={"captured_unix_time":time.time(),"root_total_bytes":stat.f_blocks*stat.f_frsize,
          "root_available_bytes":stat.f_bavail*stat.f_frsize,"root_percent_used":100.0*(1-stat.f_bavail/stat.f_blocks),
          "night21b_working_bytes":int(subprocess.check_output(["du","-sb",str(working)]).decode().split()[0]),
          "resource_gate":"BLOCK_INSTALL_DOWNLOAD","reason":"root >=93% or free <8 GiB","autodl_data_writes":0,
          "new_environment":False,"data_download":False,"repository_clone":False,"shutdown_dispatched_at_build":False}
    (output/"resource_and_disk_audit.json").write_text(json.dumps(disk,indent=2,sort_keys=True),encoding="utf-8")

    decision={
        "schema":"night21b-decision-v1","classification":classification,"strict_discovery_pass_count":strict_count,
        "strict_discovery_lane_count":4,"strict_pass_lanes":[k for k,v in gate.items() if v["strict_independent_dual_gain"]],
        "revision_1_strict_pass_count":rev1_pass,"family_freeze_authorized":False,"transfer_authorized":False,
        "multi_seed_authorized":False,"score_frontier_advance":any(r["score_frontier_advance"] for r in scores),
        "backbone_p0":"PASS","final_statement":"Sparse signed relation distillation did not show stable independent contribution; placenta is a one-lane local observation only.",
        "shutdown_dispatched_at_science_seal":False,
    }
    if decision["strict_discovery_pass_count"] != 1 or decision["revision_1_strict_pass_count"] != 1:
        raise RuntimeError(f"unexpected gate summary: {decision}")
    (output/"night21b_decision.json").write_text(json.dumps(decision,indent=2,sort_keys=True),encoding="utf-8")

    report = f"""# Night-21B final report\n\n+## 我现在需要知道的三件事\n\n+1. **想解决什么。** Night-21A 的低起点不是历史最高分分区本身变差，而是“最高分 partition”通常经过逐 lane 标签辅助 head/HPO，不能直接当作一个可重放的 pre-clustering embedding。`frontier_carrier_bridge.csv` 显示，八条 lane 的 retained carrier 在统一 KMeans 下均明显低于历史分数前沿。\n+2. **实际做了什么。** 我完整审阅 spaMGCN 的 model/train/utils/config/notebook，固定官方 commit 和 MIT License；由于官方训练路径含 dense N×N 和 notebook 内标签监控，本轮实现的是明确标注的稀疏 source-faithful port。唯一自研层 MSRD 在训练表示时，对 carrier/双模态共同支持的空间邻边做吸引，对共同低支持的空间边界做 margin 排斥，其余边 abstain。\n+3. **结果和论文意义。** A1/P22/tonsil s1/placenta 的 700-step 匹配板中，FULL 只在 placenta 1/4 lane 严格双胜 T0、B0 和三个原子臂；一次正关系主导修订仍只有 1/4。预注册的两-lane 门失败，因此未开展 transfer 或多 seed，终态是 **{classification}**。公开骨干和我方模块都没有刷新历史前沿，不能包装成新方法成功。\n\n+## 绝对结果与匹配贡献\n\n+| lane | T0 ARI/NMI | B0 ARI/NMI | FULL ARI/NMI | FULL-T0 | FULL-B0 | 严格胜全部对照 |\n+|---|---:|---:|---:|---:|---:|---|\n+"""
    for lane in LANE_ORDER:
        x=stable_by_lane[lane]; g=gate[lane]
        report += f"| {lane} | {x['T0_STRONG_CARRIER']['ari']:.6f}/{x['T0_STRONG_CARRIER']['nmi']:.6f} | {x['B0_BACKBONE_ONLY']['ari']:.6f}/{x['B0_BACKBONE_ONLY']['nmi']:.6f} | {x['FULL_SIGNED_RELATIONAL_DISTILLATION']['ari']:.6f}/{x['FULL_SIGNED_RELATIONAL_DISTILLATION']['nmi']:.6f} | {g['delta_vs_t0_ari']:+.6f}/{g['delta_vs_t0_nmi']:+.6f} | {g['delta_vs_b0_ari']:+.6f}/{g['delta_vs_b0_nmi']:+.6f} | {'PASS' if g['strict_independent_dual_gain'] else 'FAIL'} |\n"
    report += """\n完整 ARI/NMI/AMI/FMI、categorical Moran/Geary、cluster sizes、资源与全部 40 个 post-lock candidates 见 CSV。A1 V1 FULL 相对 B0 明显提高，但 NMI 比 T0 低 0.000803；P22 FULL 明显低于 T0/B0；tonsil s1 被 T0 或正关系原子臂解释；placenta 是唯一严格独立局部信号。\n\n+## Backbone、模块与 score frontier 分离\n\n+- `BACKBONE`: B0 是公开设计的 clean-room 稀疏端口，不是官方数值复现；它在四条 discovery lane 都未超过 T0。\n+- `OUR_MODULE`: 只看相同 carrier、endpoint、seed、700 steps 的 B0/B1/B2/B3/FULL；1/4 严格通过，不授权家族冻结。\n+- `SCORE_FRONTIER`: Post-lock HPO 的 best rows 单独列在 `score_frontier_board.csv`；0/4 刷新历史可信前沿。\n\n+## 训练与复现\n\n+A1、P22 真实 P0 覆盖 feature-level inputs、稀疏图、forward/loss/backward、optimizer update、strict checkpoint reload、表示与 KMeans partition。随后扩展 tonsil s1 与 placenta。主循环固定 700 steps，参数实际变化；所有 24 个稳定主板 artifacts 在独立 Python process 中表示数值重放通过、partition exact 24/24，最大 absolute representation deviation 见 replay 表。没有 dense N×N 分配。GPU 峰值未单独 profiler 采样，因此不补造数字；wall/RSS 和参数量在资源表中。\n\n+## 新颖性与来源边界\n\n+spaMGCN 的 AE、多阶图传播、融合均是公开先例；关系蒸馏、度量学习、正负图损失亦有广泛先例。只把“强 carrier 支持邻接 + 共同低支持空间边界”作为窄模块接受匹配证伪。本轮该对象未建立跨 lane 独立贡献，故不进入原创/论文主张。\n\n+## 数据扩展短名单\n\n+优先级最高的是 GSE205055 mouse embryo RNA+epigenome（先闭合 annotation/K/mask）和 Stereo-CITE mouse thymus RNA+protein（先闭合物理切片与标签）。GSE213264 目前只有作者计算型 K8/K7/GC biology，不能冒充专家全域 ground truth；SPOTS mouse spleen 尚无权威全域标签。Melanoma K2 仅保留二元压力测试。详见 `dataset_expansion_shortlist.csv`。\n\n+## 局限\n\n+- source-faithful sparse port 改写了官方 dense objective，不能把当前 B0 分数称作 spaMGCN 官方复现分数。\n+- retained carrier 与历史最高 partition 的 head/HPO 依赖分离后，统一 common endpoint 显著掉分；关系正则无法弥合该差距。\n+- 发现门失败后没有运行 family transfer、3 training seeds 或 endpoint seed 分布；这不是缺失的正证据，而是预注册停止。\n+- 只有 placenta 单 lane signal，不足以支撑关系模块或跨家族结论。\n\n+## 导师汇报版\n\n+Night-21B 先厘清了一个关键误区：过去最高分是分区和 head 的结果，不等于背后还有一个同样强、可复用的 embedding。我们基于 spaMGCN 的公开结构做了稀疏 source-faithful port，并只增加一个保护 carrier 邻域和边界关系的训练正则。真实 A1、P22、tonsil s1、placenta 都完成了 700-step 同预算六臂对照。模块只在 placenta 同时超过 carrier、backbone 和所有原子臂；A1 是混合信号，P22 和 tonsil 未成立。一次有机制理由的正关系主导修订仍未达到两条 lane 门，因此严格停止 transfer 和多 seed。结论是工程路线闭合、但关系模块没有稳定科学信号，且没有任何 score frontier 刷新。这个结果支持下一步先解决强数值 carrier/endpoint 的可迁移性，而不是继续叠关系损失。\n\n+## 技术审计\n\n+- Taskbook SHA-256: `{args.taskbook_sha}`\n+- spaMGCN fixed commit: `77dfe67d4fd80c124722e68a0f71af36d10fa5fa`, MIT.\n+- Candidate rows: {len(records)}; stable matched rows: {len(method)}; exact fresh-process replays: {len(replay_rows)}/{len(replay_rows)}.\n+- Producer label reads: 0; evaluator label access happens after artifact/hash lock.\n+- Disk gate prevented clone/install/download; no writes to `/autodl-fs/data`.\n+- Shutdown is intentionally deferred until Windows compact verification; science seal itself records `shutdown_dispatched=false`.\n+"""
    report = report.replace("\n+", "\n")
    report = report.replace("{args.taskbook_sha}", args.taskbook_sha)
    report = report.replace("{len(records)}", str(len(records)))
    report = report.replace("{len(method)}", str(len(method)))
    report = report.replace("{len(replay_rows)}/{len(replay_rows)}", f"{len(replay_rows)}/{len(replay_rows)}")
    (output/"night21b_report.md").write_text(report,encoding="utf-8")
    (output/"mentor_oral_report.md").write_text(report.split("## 导师汇报版",1)[1].split("## 技术审计",1)[0].strip()+"\n",encoding="utf-8")

    test_log = working / "logs" / "targeted_tests_final.log"
    test_text = test_log.read_text(encoding="utf-8", errors="replace") if test_log.exists() else ""
    match = re.search(r"(\d+) passed", test_text)
    if not match:
        raise RuntimeError("final pytest count not found")
    summary={"pytest_pass_count":int(match.group(1)),"pytest_log_sha256":sha_file(test_log),
             "replay_count":len(replay_rows),"replay_pass":len(replay_rows),"built_unix_time":time.time()}
    (output/"targeted_test_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    (output/"handoff_build_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True),encoding="utf-8")
    print(json.dumps({"classification":classification,"candidate_rows":len(records),"stable_rows":len(method),"strict_count":strict_count,"rev1_pass":rev1_pass,"replays":len(replay_rows)},sort_keys=True))


if __name__ == "__main__":
    main()
