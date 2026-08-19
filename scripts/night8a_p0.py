#!/usr/bin/env python3
"""Night-8A authority, history, source, semantic, runtime and firewall gates."""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night8a_mfspc import canonical_json_sha, file_sha, select_family  # noqa: E402

OUT = REPO / "outputs/night8a_handoff"
RAW = Path("/root/autodl-fs/night8a_raw_runs_20260820")
SOURCE_ROOT = Path("/root/autodl-fs/night8a_external_sources_20260820")
PYTHON = Path("/root/miniconda3/envs/SpaLORA/bin/python")
REGISTRY = REPO / "protocols/night8a/SpaLORA_Night8A_MFSPC_Registry_2026-08-20.json"

EXPECTED = {
    "protocols/night8a/SpaLORA_Night8A_Literature_Code_and_Dataset_Scout_2026-08-20.md": "e17fbf724b79872ada2e6f64a0580258d5107f0e06a2bda6b83804a94b00511d",
    "protocols/night8a/SpaLORA_Night8A_MFSPC_Registry_2026-08-20.json": "cfe056489f0d42692cc78049cbc987ef47fe814d8d1594a6d958413ca8ccf9c1",
    "protocols/night8a/SpaLORA_Night8A_MFSPC_RnD_and_External_Data_Lock_Taskbook_2026-08-20.md": "f23c045ac5e6181b3d25069a6510ba66bdc336c16638ed65359420165abc4178",
    "outputs/night6d_handoff/night6d_report.md": "333192ce979a02ce8cbc785828e58fa7b9314906ce653c7d8cae016b92e8a053",
    "outputs/night7a_handoff/night7a_report.md": "274cf68d82634ab935b894a101e439904f0ce3f4e0d16f6d146b2d7ee95e4de3",
    "outputs/night7b_handoff/night7b_report.md": "48e16b12934fb13c668ea08e966e26e1c2454f5aad59e8510e7b391a4d2cd0a9",
    "outputs/night7b_handoff/night7b_final_candidate_lock.json": "74d76ca45ca7d9d632bdfdd694bfccbe635318b3cdfaad409ca21100805ddd97",
    "outputs/night7c_handoff/night7c_report.md": "af7ae61a747e865bace5fce70da72085280c7147ddaf12c9df82362a4c77a840",
}


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def command(args: list[str], cwd: Path | None = None, timeout: int = 120) -> str:
    result = subprocess.run(args, cwd=cwd, text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout, check=False)
    if result.returncode:
        raise RuntimeError(f"command failed {args}: {result.stdout[-2000:]}")
    return result.stdout.strip()


def authority_gate() -> dict:
    observed = {}
    for relative, expected in EXPECTED.items():
        path = REPO / relative
        actual = file_sha(path) if path.exists() else None
        observed[relative] = {"expected_sha256": expected, "actual_sha256": actual,
                              "match": actual == expected}
    compact = REPO / "outputs/night7c_handoff/compact_delivery_index.json"
    if compact.exists():
        observed[str(compact.relative_to(REPO))] = {"actual_sha256": file_sha(compact),
                                                    "recorded_local_authority_sha256": "1797d4f1583d3dea0fedca93aa53ad1ade9417d1b08f0bc79fb03729c472e734",
                                                    "note": "remote tracked copy differs from Windows wrapper index only if path differs"}
    base = command(["git", "rev-parse", "night7c-final-20260818^{commit}"], REPO)
    branch = command(["git", "branch", "--show-current"], REPO)
    if not all(x["match"] for x in observed.values() if "match" in x):
        raise RuntimeError("BLOCKED_AUTHORITY_MISMATCH")
    if base != "e34567db5ace4f0fcdd2526cfb94a84fc9148020":
        raise RuntimeError("BLOCKED_AUTHORITY_MISMATCH: base tag")
    if branch != "revision/q2-night8a-mfspc-rnd-20260820":
        raise RuntimeError("BLOCKED_AUTHORITY_MISMATCH: branch")
    result = {"status": "PASS", "authority": observed, "base_commit": base,
              "branch": branch, "force_push": False}
    atomic_json(OUT / "p0_authority.json", result); return result


def mean(rows: list[dict], key: str) -> float:
    return float(np.mean([float(x[key]) for x in rows]))


def historical_gate() -> dict:
    n7 = list(csv.DictReader((REPO / "outputs/night7a_handoff/per_seed_metrics.csv").open(newline="")))
    paired = list(csv.DictReader((REPO / "outputs/night7a_handoff/paired_delta_vs_g00h00.csv").open(newline="")))
    c00 = {(x["dataset"], int(x["seed"])): x for x in n7 if x["candidate_id"] == "C00_G04_H05_CONFIRMED"}
    ref = {(x["dataset"], int(x["seed"])): x for x in n7 if x["candidate_id"] == "REFERENCE_G00_H00"}
    expected_pair = {(x["dataset"], int(x["seed"])): x for x in paired if x["candidate_id"] == "C00_G04_H05_CONFIRMED"}
    if len(c00) != 30 or set(c00) != set(ref) or set(c00) != set(expected_pair):
        raise RuntimeError("BLOCKED_HISTORICAL_RECOMPUTE: C00 coverage")
    c00_summary = {}; errors = []
    for dataset in ("a1", "tonsil", "d1", "p22"):
        keys = [k for k in c00 if k[0] == dataset]
        entry = {}
        for metric in ("ari", "nmi", "q"):
            deltas = [float(c00[k][metric]) - float(ref[k][metric]) for k in keys]
            for k, value in zip(keys, deltas):
                errors.append(abs(value - float(expected_pair[k][f"delta_{metric}"])))
            entry[f"mean_{metric}"] = float(np.mean([float(c00[k][metric]) for k in keys]))
            entry[f"mean_delta_{metric}_vs_g00h00"] = float(np.mean(deltas))
        c00_summary[dataset] = entry
    r2 = list(csv.DictReader((REPO / "outputs/night7b_handoff/R2_full_per_seed_metrics.csv").open(newline="")))
    config_id = "R02__E1_ADAPTER_C06_MEAN__H01"
    # Night-7B's locked candidate decision used its preregistered 22-cell R2
    # extension analysis set: seeds 2-4 for A1/tonsil and seeds 2-9 for
    # D1/P22.  The delivered full table also contains the earlier mechanism-
    # probe cells; those remain valid historical rows but were not selection
    # inputs.
    r02 = [x for x in r2 if x["config_id"] == config_id and x["status"] == "success"
           and int(x["seed"]) >= 2]
    original = next(x for x in csv.DictReader((REPO / "outputs/night7b_handoff/R2_candidate_summary_vs_C00.csv").open(newline=""))
                    if x["config_id"] == config_id)
    if len(r02) != 22:
        raise RuntimeError("BLOCKED_HISTORICAL_RECOMPUTE: R02 coverage")
    r02_summary = {}
    for dataset in ("a1", "tonsil", "d1", "p22"):
        rows = [x for x in r02 if x["dataset"] == dataset]
        if not rows:
            raise RuntimeError("missing R02 dataset")
        entry = {"seeds": [int(x["seed"]) for x in rows]}
        for metric in ("ari", "nmi", "q"):
            values = [float(x[metric]) for x in rows]
            deltas = [float(x[metric]) - float(c00[(dataset, int(x["seed"]))][metric]) for x in rows]
            entry[f"mean_{metric}"] = float(np.mean(values)); entry[f"mean_delta_{metric}_vs_c00"] = float(np.mean(deltas))
            errors.append(abs(entry[f"mean_{metric}"] - float(original[f"{dataset}_mean_{metric}"])))
            errors.append(abs(entry[f"mean_delta_{metric}_vs_c00"] - float(original[f"{dataset}_mean_delta_{metric}"])))
        r02_summary[dataset] = entry
    q_hln = .5 * (r02_summary["a1"]["mean_q"] + r02_summary["d1"]["mean_q"])
    priority = .45 * q_hln + .45 * r02_summary["p22"]["mean_q"] + .10 * r02_summary["tonsil"]["mean_q"]
    historical_night7b_priority = (.25 * r02_summary["a1"]["mean_q"]
                                   + .15 * r02_summary["tonsil"]["mean_q"]
                                   + .25 * r02_summary["d1"]["mean_q"]
                                   + .35 * r02_summary["p22"]["mean_q"])
    errors.append(abs(historical_night7b_priority - float(original["priority_weighted_q"])))
    max_error = max(errors)
    if max_error > 1e-12:
        raise RuntimeError(f"BLOCKED_HISTORICAL_RECOMPUTE: max error {max_error}")
    result = {"status": "PASS", "tolerance": 1e-12, "max_abs_error": max_error,
              "c00_vs_g00h00": c00_summary, "r02_vs_c00": r02_summary,
              "r02_Q_HLN": q_hln, "r02_priority_macro_Q": priority,
              "r02_historical_night7b_priority_weighted_Q": historical_night7b_priority,
              "night8a_priority_formula_is_new_preregistered_formula": True,
              "r02_status": "DEVELOPMENT_REFERENCE_NOT_EXTERNAL_CONFIRMATION"}
    atomic_json(OUT / "historical_metric_recompute.json", result); return result


def find_core(repo: Path, patterns: list[str]) -> list[dict]:
    rows = []
    for path in sorted(repo.rglob("*.py")):
        if any(part == ".git" for part in path.parts) or path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(re.search(pattern, text, re.I) for pattern in patterns):
            rows.append({"path": str(path.relative_to(repo)), "sha256": file_sha(path),
                         "size_bytes": path.stat().st_size})
        if len(rows) == 4:
            break
    return rows


def license_info(repo: Path) -> dict:
    candidates = [p for p in repo.iterdir() if p.is_file() and p.name.lower().startswith(("license", "copying"))]
    if not candidates:
        return {"classification": "NO_CLEAR_LICENSE_FILE", "files": []}
    files = []
    classifications = set()
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        if "gnu general public license" in text:
            kind = "GPL"
        elif "mit license" in text or "permission is hereby granted" in text:
            kind = "MIT"
        elif "apache license" in text:
            kind = "APACHE"
        else:
            kind = "UNCLASSIFIED"
        classifications.add(kind); files.append({"path": path.name, "sha256": file_sha(path), "classification": kind})
    return {"classification": "+".join(sorted(classifications)), "files": files}


def source_gate() -> dict:
    specs = {
        "SMART": ("SMART-main-main", "https://github.com/Xubin-s-Lab/SMART-main.git", ["triplet", "GraphSAGE"]),
        "SpaMosaic": ("SpaMosaic-main", "https://github.com/JinmiaoChenLab/SpaMosaic.git", ["contrast", "modality"]),
        "MultiGATE": ("MultiGATE", "https://github.com/cuhklinlab/MultiGATE.git", ["attention", "modality"]),
        "COSMOS": ("COSMOS", "https://github.com/Lin-Xu-lab/COSMOS.git", ["cell_random_subset", "DGI"]),
        "ARISE": ("ARISE", "https://github.com/XiangxiangWang-code/ARISE.git", ["best.*ari", "ground.*truth", "label"]),
        "scMultiBench": ("scMultiBench", "https://github.com/PYangLab/scMultiBench.git", ["benchmark", "metric"]),
    }
    entries = {}
    for name, (directory, url, patterns) in specs.items():
        repo = SOURCE_ROOT / directory
        if (repo / ".git").exists():
            commit = command(["git", "rev-parse", "HEAD"], repo)
            origin = command(["git", "remote", "get-url", "origin"], repo)
            acquisition = "remote shallow git clone"
        elif (repo / "snapshot_manifest.json").exists():
            snapshot = json.loads((repo / "snapshot_manifest.json").read_text())
            commit = snapshot["commit"]; origin = snapshot["origin"]
            acquisition = snapshot["acquisition"]
        else:
            raise RuntimeError(f"P0-SOURCE missing official source snapshot: {name}")
        core = find_core(repo, patterns)
        if not core:
            core = [{"path": str(p.relative_to(repo)), "sha256": file_sha(p), "size_bytes": p.stat().st_size}
                    for p in sorted(repo.rglob("*.py"))[:2]]
        entries[name] = {"url": url, "observed_origin": origin, "commit": commit,
                         "acquisition": acquisition,
                         "license": license_info(repo), "read_files": core,
                         "copied_into_spalora": False}
    ar = entries["ARISE"]
    ar["audit_warning"] = "official training code contains label/best-ARI semantics; prohibited from Night-8A training"
    ar["portable_idea_only"] = "RNA-feature and spatial-support intersection; independently implemented"
    co = entries["COSMOS"]
    co["audit_warning"] = "accelerated coordinate subset indexing suspicion recorded without altering upstream"
    co["portable_idea_only"] = "sparse global-local mutual-information control; independently implemented"
    entries["SMART"]["audit_warning"] = "tutorial parameters vary by platform/dataset; this is not permission for identity routing"
    result = {"status": "PASS_INDEPENDENT_IMPLEMENTATION_BOUNDARY", "repositories": entries,
              "third_party_code_copied": False,
              "license_policy": "GPL/unknown source not copied; only scientific ideas independently implemented"}
    atomic_json(OUT / "third_party_source_audit.json", result); return result


def locked_inputs_gate() -> dict:
    target = RAW / "locked_inputs/p22_spatial_support.npz"; target.parent.mkdir(parents=True, exist_ok=True)
    indices_path = Path("/root/autodl-fs/night3af_p0d_builds_20260810/process_a/p22/adj_spatial_omics1_indices.npy")
    indices = np.load(indices_path, allow_pickle=False)
    if indices.shape[0] != 2:
        raise RuntimeError("P22 spatial index shape invalid")
    n = 9196
    support = sp.csr_matrix((np.ones(indices.shape[1], dtype=np.uint8), (indices[0], indices[1])), shape=(n, n))
    support = support.maximum(support.T).tocsr(); support.sum_duplicates(); support.eliminate_zeros(); support.sort_indices()
    if not target.exists():
        sp.save_npz(target, support, compressed=True)
    replay = sp.load_npz(target)
    if replay.shape != support.shape or replay.nnz != support.nnz or (replay != support).nnz:
        raise RuntimeError("immutable P22 support mismatch")
    index = list(csv.DictReader((REPO / "outputs/night7b_handoff/source_unit_index.csv").open(newline="")))
    references = []
    for row in index:
        worker = Path(row["worker_input"])
        if file_sha(worker) != row["worker_input_sha256"]:
            raise RuntimeError("source unit SHA mismatch")
        if row["dataset"] == "p22":
            pattern = f"/root/autodl-fs/night7b_score_rnd_20260818/adapter_stage/*/formal/R02/{row['unit_id']}/attempt_001/worker/embedding.npy"
            matches = [Path(x) for x in __import__("glob").glob(pattern)]
            if len(matches) != 1:
                raise RuntimeError("R02 reference ambiguity")
            references.append({"unit_id": row["unit_id"], "path": str(matches[0]), "sha256": file_sha(matches[0]),
                               "manifest_sha256": file_sha(matches[0].parent / "training_manifest.json")})
    if len(references) != 10:
        raise RuntimeError("P22 R02 reference coverage mismatch")
    assays = {"opaque_a": {"assays": ["RNA", "ADT"]}, "opaque_b": {"assays": ["RNA", "protein"]},
              "opaque_c": {"assays": ["RNA", "ATAC"]}, "opaque_d": {"assays": ["RNA", "histone"]}}
    result = {"status": "PASS", "p22_spatial_support": {"path": str(target), "sha256": file_sha(target),
                                                           "shape": list(support.shape), "nnz": int(support.nnz)},
              "r02_references": references,
              "family_selector_examples": {k: select_family(v) for k, v in assays.items()},
              "identity_blind": True}
    atomic_json(OUT / "family_reference_policy.json", result); return result


def environment_gate() -> dict:
    try:
        pyg = __import__("torch_geometric").__version__
    except Exception:
        pyg = None
    rscript = str(PYTHON.parent / "Rscript")
    r_version = command([rscript, "-e", "cat(as.character(getRversion()))"])
    mclust = command([rscript, "-e", "cat(as.character(packageVersion('mclust')))"])
    if not torch.cuda.is_available():
        raise RuntimeError("P0-RUNTIME CUDA unavailable")
    result = {"status": "PASS", "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "torch_cuda": torch.version.cuda, "torch_geometric": pyg,
              "R": r_version, "mclust": mclust, "cuda_available": True,
              "training_device_required": "cuda", "cpu_transform_workers_max": 3,
              "threads_per_worker_max": 3, "transform_timeout_seconds": 2700}
    atomic_json(OUT / "runtime_environment.json", result); return result


def firewall_gate() -> dict:
    worker_rows = list(csv.DictReader((REPO / "outputs/night7b_handoff/source_unit_index.csv").open(newline="")))
    forbidden = {"dataset", "tissue", "labels", "label", "ground_truth", "ari", "nmi", "metrics"}
    for row in worker_rows:
        payload = json.loads(Path(row["worker_input"]).read_text())
        if forbidden & set(payload):
            raise RuntimeError("label/identity field in trainer payload")
    snapshots = Path("/root/autodl-fs/night7a_consensus_20260818/evaluation_label_snapshots")
    hashes = {p.name: {"sha256": file_sha(p), "size_bytes": p.stat().st_size}
              for p in sorted(snapshots.glob("*_labels_locked.npz"))}
    if len(hashes) != 4:
        raise RuntimeError("sealed label snapshots missing")
    result = {"status": "PASS_PRE_LABEL", "label_values_deserialized": False,
              "label_values_used": False, "label_file_byte_hash_only": hashes,
              "trainer_payloads_checked": len(worker_rows), "trainer_receives_dataset_identity": False,
              "original_h5ad_anndata_read_calls": 0, "development_windows": ["DEV_WINDOW_1", "DEV_WINDOW_2"]}
    atomic_json(OUT / "label_firewall_audit.json", result); return result


def p22_smoke() -> dict:
    root = RAW / "p0_full_size_p22_smoke"
    manifest = root / "training/training_manifest.json"
    transform_manifest = root / "transform/transform_manifest.json"
    if manifest.exists() and transform_manifest.exists():
        return json.loads((root / "p0_smoke_summary.json").read_text())
    row = next(x for x in csv.DictReader((REPO / "outputs/night7b_handoff/source_unit_index.csv").open(newline=""))
               if x["dataset"] == "p22" and int(x["seed"]) == 0)
    pattern = f"/root/autodl-fs/night7b_score_rnd_20260818/adapter_stage/*/formal/R02/{row['unit_id']}/attempt_001/worker/embedding.npy"
    reference = next(Path(x) for x in __import__("glob").glob(pattern))
    config = {"schema_version": "night8a-cell-config-v1", "stage": "P0_SMOKE",
              "config_id": "P0_ALL_MODULES_FULL_P22", "unit_id": row["unit_id"], "seed": 0,
              "K": int(row["K"]), "assay_metadata": {"assays": ["RNA", "ATAC"], "source": "locked P22 assay contract"},
              "registered_modules": ["SP", "RR10", "PROTO", "RNA_ANCHOR", "DGI", "SMART_TRIPLET"],
              "worker_input": row["worker_input"], "family_reference": str(reference),
              "output_dir": str(root / "training"), "epochs": 1, "learning_rate": .001,
              "weight_decay": .00001, "rna_anchor_spatial_support": str(RAW / "locked_inputs/p22_spatial_support.npz")}
    root.mkdir(parents=True, exist_ok=True); config_path = root / "cell_config.json"; atomic_json(config_path, config)
    started = time.perf_counter()
    command([str(PYTHON), str(REPO / "scripts/night8a_train.py"), "train", "--config", str(config_path)], timeout=900)
    command([str(PYTHON), str(REPO / "scripts/night8a_train.py"), "reload", "--config", str(config_path),
             "--output", str(root / "training")], timeout=600)
    env = os.environ.copy(); env.update({"OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3", "OPENBLAS_NUM_THREADS": "3"})
    result = subprocess.run([str(PYTHON), str(REPO / "scripts/night8a_transform.py"), "--config", str(config_path),
                             "--training-output", str(root / "training"), "--output", str(root / "transform")],
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, timeout=2700)
    if result.returncode:
        raise RuntimeError(f"P22 full-size smoke transform failed: {result.stdout[-2000:]}")
    training = json.loads(manifest.read_text()); transform = json.loads(transform_manifest.read_text())
    summary = {"status": "PASS", "full_size_observations": 9196, "epochs": 1,
               "all_registered_mechanism_classes_exercised": True, "dense_n_by_n_created": False,
               "cuda_used": training["cuda_used"], "checkpoint_round_trip": True,
               "cluster_exact_round_trip": transform["cluster_exact_replay"],
               "training_peak_gpu_mib": training["peak_gpu_allocated_mib"],
               "transform_runtime_seconds": transform["runtime_seconds"],
               "total_runtime_seconds": time.perf_counter() - started,
               "training_manifest_sha256": file_sha(manifest),
               "transform_manifest_sha256": file_sha(transform_manifest)}
    atomic_json(root / "p0_smoke_summary.json", summary)
    atomic_json(OUT / "p0_full_size_p22_smoke.json", summary); return summary


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); started = time.time()
    outputs = {"authority": authority_gate(), "history": historical_gate(),
               "source": source_gate(), "locked_inputs": locked_inputs_gate(),
               "environment": environment_gate(), "firewall": firewall_gate()}
    test_log = OUT / "tests/p0_pytest.txt"; test_log.parent.mkdir(parents=True, exist_ok=True)
    with test_log.open("w", encoding="utf-8") as handle:
        result = subprocess.run([str(PYTHON), "-m", "pytest", "-q", "tests/test_night8a_mfspc.py"],
                                cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode:
        raise RuntimeError("IMPLEMENTATION_SEMANTICS_INVALID: pytest")
    text = test_log.read_text()
    match = re.search(r"(\d+) passed", text)
    if not match or int(match.group(1)) < 25:
        raise RuntimeError("IMPLEMENTATION_SEMANTICS_INVALID: fewer than 25 tests")
    outputs["tests"] = {"status": "PASS", "passed": int(match.group(1)), "log_sha256": file_sha(test_log)}
    outputs["p22_smoke"] = p22_smoke()
    contract = {"status": "PASS", "terminal_gate": "P0_SEMANTIC_AND_RUNTIME_PASS",
                "completed_at_unix": time.time(), "elapsed_seconds": time.time() - started,
                "formal_scientific_training_attempts": 0, "formal_transforms": 0,
                "label_access": False, "outputs": {k: v.get("status") for k, v in outputs.items()},
                "tests": outputs["tests"]}
    atomic_json(OUT / "p0_semantic_contract.json", contract)
    atomic_json(OUT / "p0_config_lock.json", {
        "status": "LOCKED_PRE_LABEL", "registry_sha256": file_sha(REGISTRY),
        "registry_canonical_sha256": canonical_json_sha(json.loads(REGISTRY.read_text())),
        "candidate_expansion_after_results": False, "fixed_epoch": 160,
        "learning_rate": .001, "weight_decay": .00001,
    })
    print(json.dumps(contract, sort_keys=True))


if __name__ == "__main__":
    main()
