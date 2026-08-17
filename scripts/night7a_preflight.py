#!/usr/bin/env python3
"""Source-code and metadata-only preflight; never runs a formal benchmark."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night7a_consensus import atomic_json, sha256_file  # noqa: E402

OUT = REPO / "outputs/night7a_handoff"
SOURCES = Path("/root/autodl-fs/night7a_external_sources_20260818")
METADATA = Path("/root/autodl-fs/night7a_dataset_metadata_20260818")
METHODS = (
    ("SpatialGlue", "https://github.com/JinmiaoChenLab/SpatialGlue", "classic_required"),
    ("Seurat_WNN", "https://github.com/satijalab/seurat", "classic_required"),
    ("COSMOS", "https://github.com/Lin-Xu-lab/COSMOS", "modern_high_priority"),
    ("SMART", "https://github.com/Xubin-s-Lab/SMART-main", "modern_high_priority"),
    ("PRESENT", "https://github.com/lizhen18THU/PRESENT", "modern_high_priority"),
    ("MultiGATE", "https://github.com/cuhklinlab/MultiGATE", "modern_high_priority"),
    ("SpatialCOC", "https://github.com/xjtu-omics/SpatialCOC", "modern_high_priority"),
    ("SpaMode", "https://github.com/bridge1924/SpaMode", "source_only_until_license_resolved"),
    ("SpaMCA", "https://github.com/wenwenmin/SpaMCA", "source_and_feasibility"),
    ("ARISE", "https://github.com/XiangxiangWang-code/ARISE", "fixed_endpoint_adapter_required"),
    ("GROVER", "https://github.com/Xubin-s-Lab/GROVER", "feasibility_only_private_weight_and_third_modality_risk"),
    ("MultiSP", "https://github.com/jinworks/MultiSP", "modern_frontier_source_and_feasibility"),
    ("SpatialEx", "https://github.com/KEAML-JLU/SpatialEx", "feasibility_only_histology_anchored"),
)
TEXT_EXTENSIONS = {".py", ".r", ".R", ".md", ".txt", ".yml", ".yaml", ".toml", ".json", ".ipynb", ".sh"}
LABEL_RE = re.compile(r"ground[_ .-]?truth|cell[_ .-]?type|annotation|\blabels?\b", re.I)
SELECTION_RE = re.compile(r"best[_ .-]?(ari|nmi|epoch|seed)|adjusted_rand|normalized_mutual|argmax\s*\(.*(ari|nmi)", re.I)
K_RE = re.compile(r"n_clusters|num[_ .-]?clusters?|cluster[_ .-]?num|resolution", re.I)
ENDPOINT_RE = re.compile(r"embedding|latent|leiden|louvain|mclust|spectral|cluster", re.I)
DATASET_PARAM_RE = re.compile(
    r"epochs?|n[_ .-]?epochs?|resolution|mask[_ .-]?rate|loss[_ .-]?weight|"
    r"n[_ .-]?clusters?|num[_ .-]?clusters?|dataset[_ .-]?name", re.I
)
ASSET_RE = re.compile(
    r"pretrain(ed)?|checkpoint|private|download.*weight|histology|H\s*&\s*E|"
    r"image[_ .-]?feature|third[_ .-]?modality", re.I
)


def run(command: list[str], cwd: Path | None = None, timeout: int = 600) -> tuple[int, str]:
    try:
        completed = subprocess.run(command, cwd=cwd, text=True,
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   timeout=timeout)
        return completed.returncode, completed.stdout[-20000:]
    except Exception as exc:
        return 255, f"{type(exc).__name__}: {exc}"


def clone(method: str, url: str) -> tuple[Path, list[dict]]:
    target = SOURCES / method
    failures = []
    if not target.exists():
        code, output = run(["git", "clone", "--depth", "1", "--filter=blob:none", url, str(target)], timeout=900)
        if code:
            failures.append({"operation": "git_clone", "exit_code": code, "output": output})
    if not (target / ".git").exists():
        return target, failures
    return target, failures


def git_value(target: Path, *args: str) -> str | None:
    code, output = run(["git", "-C", str(target), *args])
    return output.strip() if code == 0 else None


def find_files(target: Path, patterns: tuple[str, ...], max_depth: int = 4) -> list[str]:
    result = []
    for path in target.rglob("*"):
        if not path.is_file() or len(path.relative_to(target).parts) > max_depth:
            continue
        name = path.name.lower()
        if any(re.fullmatch(pattern, name) for pattern in patterns):
            result.append(path.relative_to(target).as_posix())
    return sorted(result)[:100]


def source_scan(target: Path) -> dict:
    findings = {"label_access": [], "metric_or_checkpoint_selection": [],
                "cluster_number": [], "dataset_specific_parameters": [],
                "private_weight_histology_or_third_modality": [],
                "joint_endpoint": []}
    patterns = ((LABEL_RE, "label_access"), (SELECTION_RE, "metric_or_checkpoint_selection"),
                (K_RE, "cluster_number"),
                (DATASET_PARAM_RE, "dataset_specific_parameters"),
                (ASSET_RE, "private_weight_histology_or_third_modality"),
                (ENDPOINT_RE, "joint_endpoint"))
    examined = 0
    for path in target.rglob("*"):
        if (not path.is_file() or ".git" in path.parts or path.suffix not in TEXT_EXTENSIONS or
                path.stat().st_size > 5_000_000):
            continue
        examined += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            compact = " ".join(line.strip().split())[:300]
            for pattern, key in patterns:
                if len(findings[key]) < 40 and pattern.search(line):
                    findings[key].append({"path": path.relative_to(target).as_posix(),
                                          "line": line_number, "snippet": compact})
    findings["files_examined"] = examined
    return findings


def license_audit(target: Path) -> dict:
    candidates = []
    for pattern in ("LICENSE*", "LICENCE*", "COPYING*"):
        candidates.extend(target.glob(pattern))
    candidates = sorted({path for path in candidates if path.is_file()})
    if not candidates:
        return {"status": "NO_LICENSE_FILE", "files": [],
                "recognized_open_source_license": False,
                "compatible_for_adapter_copy": False}
    rows, recognized, compatible = [], False, False
    for path in candidates:
        text = path.read_text(encoding="utf-8", errors="ignore")[:20000]
        kind = "UNRESOLVED"
        for token, name in (("MIT License", "MIT"), ("Apache License", "Apache"),
                            ("GNU GENERAL PUBLIC LICENSE", "GPL"),
                            ("GNU AFFERO GENERAL PUBLIC LICENSE", "AGPL"),
                            ("GNU LESSER GENERAL PUBLIC LICENSE", "LGPL"),
                            ("Mozilla Public License", "MPL"),
                            ("Redistribution and use", "BSD")):
            if token.lower() in text.lower():
                kind = name; recognized = True
                compatible = compatible or name in {"MIT", "Apache", "BSD"}
                break
        rows.append({"path": path.relative_to(target).as_posix(),
                     "sha256": sha256_file(path), "detected": kind})
    return {"status": "LICENSE_FILE_FOUND", "files": rows,
            "recognized_open_source_license": recognized,
            "compatible_for_adapter_copy": compatible}


def audit_method(method: str, url: str, tier: str) -> dict:
    target, failures = clone(method, url)
    if not (target / ".git").exists():
        return {"method": method, "repository": url, "tier": tier,
                "status": "BLOCKED_ENVIRONMENT", "failures": failures,
                "formal_benchmark_run": False}
    commit = git_value(target, "rev-parse", "HEAD")
    branch_ref = git_value(target, "symbolic-ref", "refs/remotes/origin/HEAD")
    branch = branch_ref.rsplit("/", 1)[-1] if branch_ref else git_value(
        target, "rev-parse", "--abbrev-ref", "HEAD"
    )
    submodules = git_value(target, "submodule", "status") or ""
    license_info = license_audit(target)
    scan = source_scan(target)
    environment = find_files(target, (r"requirements.*\.txt", r"environment.*\.ya?ml",
                                      r"setup\.py", r"pyproject\.toml", r"description", r"renv\.lock"))
    entrypoints = find_files(target, (r"(main|train|run|integration|demo|tutorial).*\.(py|r|sh|ipynb)",), 5)
    needs_adapter = bool(scan["metric_or_checkpoint_selection"] or scan["label_access"])
    asset_risk = bool(scan["private_weight_histology_or_third_modality"])
    if method in {"GROVER", "SpatialEx"} and asset_risk:
        status = "BLOCKED_PRIVATE_ASSET_OR_THIRD_MODALITY"
    elif not license_info["recognized_open_source_license"]:
        status = "SOURCE_ONLY_LICENSE_BLOCKED"
    elif needs_adapter:
        status = "READY_WITH_FIXED_ENDPOINT_ADAPTER"
    elif not scan["joint_endpoint"]:
        status = "BLOCKED_NO_JOINT_CLUSTER_ENDPOINT"
    else:
        status = "READY_COMMON_PROTOCOL"
    return {
        "method": method, "repository": url, "resolved_commit": commit,
        "default_branch_at_clone": branch, "tier": tier,
        "submodules": [line for line in submodules.splitlines() if line.strip()],
        "license": license_info, "environment_files": environment,
        "candidate_entrypoints": entrypoints, "source_scan": scan,
        "input_contract_preflight": {
            "expected_modalities": "derive from official entrypoint/tutorial before formal benchmark",
            "histology_or_private_asset_risk_detected": asset_risk,
            "source_evidence_paths": scan["private_weight_histology_or_third_modality"][:10],
        },
        "common_evaluator_endpoint": {
            "joint_representation_or_cluster_paths": scan["joint_endpoint"][:20],
            "official_tutorial_reaches_endpoint": bool(entrypoints and scan["joint_endpoint"]),
            "known_k_handling_paths": scan["cluster_number"][:20],
        },
        "official_behavior_disclosure": {
            "per_spot_label_paths_detected": len(scan["label_access"]),
            "best_metric_or_checkpoint_paths_detected": len(scan["metric_or_checkpoint_selection"]),
            "dataset_specific_parameter_paths_detected": len(scan["cluster_number"]),
            "broader_dataset_parameter_paths_detected": len(scan["dataset_specific_parameters"]),
        },
        "fixed_final_label_free_adapter": {
            "required": needs_adapter,
            "labels_passed": False, "checkpoint_policy": "fixed final",
            "K_source": "prelocked dataset contract", "seed_policy": "common fixed seeds",
            "metric_based_selection": False,
        },
        "expected_resources": "GPU and runtime to be measured only in a later common-protocol benchmark",
        "compatibility_correction": None,
        "status": status, "failures": failures,
        "formal_benchmark_run": False,
    }


def fetch_small(url: str, target: Path, maximum: int = 5_000_000) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "SpaLORA-Night7A-metadata-audit/1"})
    last_error = None
    for attempt, delay in enumerate((0, 5, 15), 1):
        if delay:
            time.sleep(delay)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                declared = response.headers.get("Content-Length")
                if declared and int(declared) > maximum:
                    raise RuntimeError(f"large-file refusal: {declared} > {maximum}: {url}")
                payload = response.read(maximum + 1)
                if len(payload) > maximum:
                    raise RuntimeError(f"large-file refusal after streaming: {url}")
                headers = dict(response.headers.items())
            break
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            if isinstance(exc, urllib.error.HTTPError) and exc.code not in {429, 500, 502, 503, 504}:
                raise
    else:
        raise RuntimeError(f"metadata fetch failed after 3 attempts: {url}: {last_error}")
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    os.replace(tmp, target)
    return {"url": url, "path": str(target), "size_bytes": len(payload),
            "sha256": sha256_file(target), "headers": headers}


def zenodo_contract() -> tuple[dict, dict]:
    fetch = fetch_small("https://zenodo.org/api/records/12654113", METADATA / "zenodo_12654113.json")
    record = json.loads(Path(fetch["path"]).read_text())
    files = [{"key": item["key"], "size_bytes": item["size"],
              "checksum": item.get("checksum"),
              "download_url": item.get("links", {}).get("self")}
             for item in record.get("files", [])]
    contract = {
        "id": "SPAMODE_HUMAN_TONSIL_THREE_SECTIONS",
        "study_and_accession": "Zenodo 12654113",
        "organism": "Homo sapiens", "tissue": "tonsil",
        "modalities": ["RNA", "protein"], "platform": "spatial multi-omics; verify in source metadata",
        "section_independence": "section 1 overlaps the Night-6C development tonsil; only sections 2/3 can be section-level fresh, and they are not study-independent",
        "spot_count": "not parsed from per-spot files in Night-7A",
        "pairing_key": "archive-level claim only; barcode pairing requires later controlled download",
        "coordinate_source": "archive-listed spatial coordinates; not opened in Night-7A",
        "annotation_provenance": "per-spot annotations were not opened; independent histology/manual provenance is not established by metadata alone",
        "known_k_without_labels": None,
        "license_or_terms": record.get("metadata", {}).get("license"),
        "files": files, "metadata_fetch": fetch,
        "large_archives_downloaded": False,
        "status": "NEEDS_MANUAL_PROVENANCE_REVIEW",
        "benchmark_feasibility": "RNA+protein methods feasible after controlled archive acquisition; section 1 excluded from fresh confirmation",
        "benchmark_feasibility_by_shortlist": "RNA+protein-capable methods require the same fixed-final, label-free endpoint; histology/private-asset methods remain subject to their source-audit status",
    }
    return contract, fetch


def geo_contract(accession: str, purpose: str) -> tuple[dict, dict]:
    url = f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={accession}&targ=self&form=text&view=full"
    fetch = fetch_small(url, METADATA / f"{accession}_metadata.txt")
    text = Path(fetch["path"]).read_text(encoding="utf-8", errors="ignore")
    supplements = sorted(set(re.findall(r"https?://\S+", text)))[:200]
    contract = {
        "id": accession, "study_and_accession": accession,
        "purpose": purpose, "official_metadata_url": url,
        "metadata_fetch": fetch, "archive_or_supplement_links": supplements,
        "per_spot_annotations_opened": False, "large_downloads": False,
    }
    return contract, fetch


def data_preflight() -> tuple[list[dict], list[dict]]:
    contracts, fetches = [], []
    try:
        zenodo, fetch = zenodo_contract(); fetches.append(fetch)
    except Exception as exc:
        zenodo = {
            "id": "SPAMODE_HUMAN_TONSIL_THREE_SECTIONS",
            "study_and_accession": "Zenodo 12654113", "organism": "Homo sapiens",
            "tissue": "tonsil", "modalities": ["RNA", "protein"],
            "metadata_source": "https://zenodo.org/records/12654113",
            "metadata_fetch_failure": f"{type(exc).__name__}: {exc}",
            "per_spot_annotations_opened": False, "large_archives_downloaded": False,
            "known_k_without_labels": None, "status": "NEEDS_MANUAL_PROVENANCE_REVIEW",
            "benchmark_feasibility": "not locked until official record metadata can be retrieved",
        }
    contracts.append(zenodo)
    try:
        gse205055, fetch = geo_contract("GSE205055", "fresh mouse RNA+ATAC candidate")
        fetches.append(fetch)
    except Exception as exc:
        gse205055 = {
            "id": "GSE205055", "study_and_accession": "GSE205055",
            "purpose": "fresh mouse RNA+ATAC candidate",
            "official_metadata_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE205055",
            "metadata_fetch_failure": f"{type(exc).__name__}: {exc}",
            "per_spot_annotations_opened": False, "large_downloads": False,
        }
    gse205055.update({
        "organism": "Mus musculus", "tissue": "embryo/brain sections",
        "modalities": ["RNA", "ATAC"], "platform": "spatial epigenome-transcriptome profiling",
        "section_independence": "study-level independent of all four development datasets",
        "spot_count": "not obtained without archive-level inspection",
        "pairing_key": "same-section RNA/ATAC pairing must be locked from archive metadata",
        "coordinate_source": "must be verified from official archive files",
        "paired_section_and_coordinates": "MISAR metadata candidate; exact section pairing requires archive-level verification",
        "annotation_provenance": "independent manual/histology annotation not established from accession metadata",
        "known_k_without_labels": None,
        "license_or_terms": "NCBI GEO public-access record; downstream file-specific terms require review",
        "file_size_checksum_status": "archive-level sizes/checksums not asserted without controlled listing audit",
        "status": "NEEDS_MANUAL_PROVENANCE_REVIEW",
        "benchmark_feasibility": "exploratory until pairing and annotation provenance are independently audited",
        "benchmark_feasibility_by_shortlist": "RNA+ATAC-capable common-protocol methods only; all others must be marked modality-incompatible rather than silently adapted",
    }); contracts.append(gse205055)
    try:
        oep = fetch_small("https://www.biosino.org/node/project/detail/OEP003285",
                          METADATA / "OEP003285_landing.html")
        fetches.append(oep); gse205055["oep003285_metadata_fetch"] = oep
    except Exception as exc:
        gse205055["oep003285_metadata_fetch_failure"] = f"{type(exc).__name__}: {exc}"
    try:
        gse198353, fetch = geo_contract("GSE198353", "label-free spleen replicate stability")
        fetches.append(fetch)
    except Exception as exc:
        gse198353 = {
            "id": "GSE198353", "study_and_accession": "GSE198353",
            "purpose": "label-free spleen replicate stability",
            "official_metadata_url": "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE198353",
            "metadata_fetch_failure": f"{type(exc).__name__}: {exc}",
            "per_spot_annotations_opened": False, "large_downloads": False,
        }
    gse198353.update({
        "organism": "Mus musculus", "tissue": "spleen",
        "modalities": ["RNA", "protein"], "replicates": 2,
        "platform": "SPOTS spatial RNA+protein",
        "section_independence": "study-level independent of all four development datasets",
        "spot_count": "not parsed from per-spot files in Night-7A",
        "pairing_key": "same-spot RNA/protein pairing requires controlled archive acquisition",
        "coordinate_source": "official supplementary spatial files; not opened in Night-7A",
        "annotation_provenance": "no official auditable manual domain labels established in metadata-only preflight",
        "known_k_without_labels": None,
        "license_or_terms": "NCBI GEO public-access record; downstream file-specific terms require review",
        "file_size_checksum_status": "archive-level sizes/checksums not asserted without controlled listing audit",
        "status": "READY_LABEL_FREE_REPLICATION",
        "benchmark_feasibility": "spatial continuity and replicate transfer only; not an accuracy ground-truth benchmark",
        "benchmark_feasibility_by_shortlist": "RNA+protein-capable common-protocol methods; accuracy methods requiring K/labels are not eligible for this label-free role",
    }); contracts.append(gse198353)
    if "metadata_fetch_failure" in gse198353:
        gse198353["status"] = "NEEDS_MANUAL_PROVENANCE_REVIEW"
        gse198353["benchmark_feasibility"] = (
            "not locked until official accession metadata can be retrieved"
        )
    contracts.append({
        "id": "SIMULATED_GROUND_TRUTH_PANEL", "study_and_accession": "preregistered simulation supplement",
        "mechanisms": ["boundary sharpness", "noise", "modality missingness", "graph disagreement"],
        "role": "controlled robustness and mechanism only; never replaces real external validation",
        "status": "NOT_SUITABLE", "per_spot_external_labels_opened": False,
    })
    for contract in contracts:
        atomic_json(OUT / "data_contracts" / f"{contract['id'].lower()}_contract.json", contract)
    return contracts, fetches


def toy_smoke() -> dict:
    synthetic_source = "labels = truth\nbest_ari = max(scores)\n"
    scanner_pass = bool(LABEL_RE.search(synthetic_source) and SELECTION_RE.search(synthetic_source))
    rng = np.random.RandomState(20260818)
    embedding = rng.normal(size=(30, 4))
    labels = KMeans(n_clusters=3, random_state=2020, n_init=20).fit_predict(embedding)
    return {
        "status": "PASS" if scanner_pass and len(labels) == 30 else "FAIL",
        "synthetic_label_selection_scanner": scanner_pass,
        "fixed_final_label_free_adapter": {"input_shape": [30, 4], "K": 3,
                                             "labels_supplied": False,
                                             "output_clusters": len(labels)},
        "formal_dataset_execution": False, "formal_benchmark_runs": 0,
        "development_dataset_inputs_used": False,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True); SOURCES.mkdir(parents=True, exist_ok=True); METADATA.mkdir(parents=True, exist_ok=True)
    audits = [audit_method(*row) for row in METHODS]
    rows = []
    for audit in audits:
        rows.append({
            "method": audit["method"], "repository": audit["repository"],
            "resolved_commit": audit.get("resolved_commit"),
            "default_branch": audit.get("default_branch_at_clone"),
            "tier": audit["tier"], "status": audit["status"],
            "license_status": audit.get("license", {}).get("status"),
            "license_detected": ",".join(
                item.get("detected", "") for item in audit.get("license", {}).get("files", [])
            ),
            "recognized_open_source_license": audit.get("license", {}).get(
                "recognized_open_source_license", False
            ),
            "license_compatible": audit.get("license", {}).get("compatible_for_adapter_copy", False),
            "label_access_hits": audit.get("official_behavior_disclosure", {}).get("per_spot_label_paths_detected"),
            "best_metric_selection_hits": audit.get("official_behavior_disclosure", {}).get("best_metric_or_checkpoint_paths_detected"),
            "formal_benchmark_run": False,
        })
    pd_path = OUT / "external_method_source_audit.csv"
    with pd_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    atomic_json(OUT / "external_method_readiness.json", {
        "status": "COMPLETE", "methods_audited": len(audits),
        "formal_benchmark_runs": 0, "audits": audits,
    })
    lines = ["# External method label-selection source audit", "",
             "This is a source-code preflight, not a benchmark. Evidence snippets identify code paths; no upstream no-license code is copied.", ""]
    for audit in audits:
        lines.extend([f"## {audit['method']}", "",
                      f"- Commit: `{audit.get('resolved_commit')}`",
                      f"- Readiness: `{audit['status']}`",
                      f"- Label-access hits: {audit.get('official_behavior_disclosure', {}).get('per_spot_label_paths_detected', 0)}",
                      f"- Metric/checkpoint-selection hits: {audit.get('official_behavior_disclosure', {}).get('best_metric_or_checkpoint_paths_detected', 0)}",
                      "- Formal adaptation rule: fixed final endpoint, common fixed seed, prelocked K, no labels or best-ARI/NMI selection.", ""])
    (OUT / "label_selection_code_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    smoke = toy_smoke(); atomic_json(OUT / "toy_smoke_test_manifest.json", smoke)
    contracts, fetches = data_preflight()
    atomic_json(OUT / "fresh_dataset_preflight.json", {
        "status": "COMPLETE", "fresh_external_per_spot_label_reads": 0,
        "large_archives_downloaded": False, "contracts": contracts,
        "metadata_fetches": fetches,
    })
    label_paths = {
        "a1": Path("/root/autodl-fs/Human lymph node/A1/A1_groundtruth.csv"),
        "tonsil": Path("/root/autodl-fs/datasets/human_tonsil_official/section1/s1_adata_rna.h5ad"),
        "d1": Path("/root/autodl-fs/Human lymph node/D1/D1_groundtruth.csv"),
        "p22": Path("/root/autodl-fs/P22 mouse brain coronal section/MouseBrain_groundtruth.csv"),
    }
    locked_outputs = {}
    lock_relatives = [
        "outputs/night7a_handoff/external_method_source_audit.csv",
        "outputs/night7a_handoff/external_method_readiness.json",
        "outputs/night7a_handoff/label_selection_code_audit.md",
        "outputs/night7a_handoff/toy_smoke_test_manifest.json",
        "outputs/night7a_handoff/fresh_dataset_preflight.json",
    ] + [
        f"outputs/night7a_handoff/data_contracts/{contract['id'].lower()}_contract.json"
        for contract in contracts
    ]
    for relative in lock_relatives:
        locked_outputs[relative] = sha256_file(REPO / relative)
    lock = {
        "status": "LOCKED_PRE_LABEL", "locked_outputs": locked_outputs,
        "development_label_byte_hashes": {key: sha256_file(path) for key, path in label_paths.items()},
        "development_label_byte_hash_only_no_parse": True,
        "fresh_external_label_reads": 0, "large_fresh_downloads": 0,
        "formal_benchmark_runs": 0, "scientific_training": 0,
        "gpu_use": 0,
    }
    lock_path = OUT / "benchmark_and_data_preflight_lock.json"
    atomic_json(lock_path, lock)
    os.chmod(lock_path, 0o444)
    print(json.dumps({"status": "LOCKED_PRE_LABEL", "methods": len(audits),
                      "data_contracts": len(contracts), "formal_benchmark_runs": 0}, sort_keys=True))


if __name__ == "__main__":
    main()
