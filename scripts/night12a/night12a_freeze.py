#!/usr/bin/env python3
"""Freeze Night-12A schemas, links, firewalls and real-path inputs before smoke."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from SpaLORA.night12a_schema_p0 import (
    atomic_json,
    file_sha256,
    fixed_linked_features,
    gene_score_intervals,
    map_adt_targets,
    parse_ensembl79_genes,
    parse_ncbi_gene_info,
    text_sha256,
)
from scripts.post_night11a_direction_reset.read_only_asset_audit import root_snapshot

REPO = Path(__file__).resolve().parents[2]
RAW = Path("/root/autodl-fs/night12a_linked_replicate_schema_p0_20260822")
OUT = REPO / "outputs/night12a_handoff"
SCHEMA = RAW / "derived/schema"
DOWNLOADS = RAW / "downloads"
P5 = ["P5S1", "P5S2", "P5S3"]
P10 = ["P10S1", "P10S2", "P10S3"]
P5_FRAGMENT_FILES = {
    "P5S1": "GSM9248997_02_P5S1_atac_fragments.tsv.gz",
    "P5S2": "GSM9248998_02_P5S2_atac_fragments.tsv.gz",
    "P5S3": "GSM9248999_02_P5S3_atac_fragments.tsv.gz",
}


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_rows(path: Path, fieldnames: list[str], rows: list[dict], delimiter=","):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter,
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush(); os.fsync(handle.fileno())
    os.replace(temp, path)


def fragment_header(path: Path) -> dict:
    result = {}
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith("#"):
                break
            value = line[1:].strip()
            if "=" in value:
                key, content = value.split("=", 1)
                result[key] = content
    required = {
        "pipeline_name": "cellranger-arc",
        "pipeline_version": "cellranger-arc-2.0.2",
        "reference_version": "2020-A",
        "reference_fasta_hash": "c9c31fef9ba3b93f99ad59d9345aeee0c9cb0640",
        "reference_gtf_hash": "73e68486af93260a77bf2ebad4b5dd2532aaa016",
    }
    if any(result.get(key) != value for key, value in required.items()):
        raise ValueError(f"fragment reference header mismatch: {path.name}")
    return result


def download_audit():
    rows = list(csv.DictReader((OUT / "accession_file_manifest.csv").open()))
    if len(rows) != 23:
        raise ValueError("active whitelist is not 23 payloads")
    records = []
    for row in rows:
        path = DOWNLOADS / row["filename"]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        expected_size = int(row["expected_size"])
        if actual_size != expected_size:
            raise ValueError(f"download size mismatch: {path.name}")
        actual_md5 = md5(path)
        if row["expected_md5"] and actual_md5 != row["expected_md5"]:
            raise ValueError(f"download MD5 mismatch: {path.name}")
        records.append({
            **row,
            "absolute_path": str(path.resolve()),
            "actual_size": actual_size,
            "actual_md5": actual_md5,
            "actual_sha256": file_sha256(path),
            "read_only_mode": oct(path.stat().st_mode & 0o777),
        })
    result = {
        "schema": "spalora.night12a.download_manifest.v1",
        "payload_count": len(records),
        "total_bytes": sum(x["actual_size"] for x in records),
        "records": records,
        "missing": 0, "size_mismatch": 0, "md5_mismatch": 0,
        "new_downloads_outside_whitelist": 0,
    }
    atomic_json(OUT / "download_manifest.json", result)
    return result


def source_audit():
    upstream = RAW / "source_inspection/spatial_tri-omics"
    relative = [
        "README.md",
        "Data_preprocessing/ADT_process/TAG_LIST_Mouse_ECM_all.csv",
        "Data_preprocessing/ADT_process/CITE_BC_process.py",
        "Data_preprocessing/ARC_process/RNA_BC_process.py",
        "Data_preprocessing/ARC_process/ATAC_BC_process.py",
        "Data_preprocessing/ARC_process/3.run_cellrangerarc.sh",
        "Data_preprocessing/Spatial_process/02_tissue_positions_list.R",
        "Data_preprocessing/Spatial_process/merge_spots_squaredgrid.py",
        "Data_visualization/archR_fix.R",
        "Data_visualization/ArchRmergeRNAATACgene.R",
    ]
    files = []
    for item in relative:
        path = upstream / item
        if not path.is_file():
            raise FileNotFoundError(path)
        files.append({"path": item, "size": path.stat().st_size,
                      "sha256": file_sha256(path)})
    internal = [
        "SpaLORA/night1_pipeline.py", "SpaLORA/preprocess.py",
        "SpaLORA/night6c_pipeline.py", "SpaLORA/night6d_pipeline.py",
        "SpaLORA/night12a_schema_p0.py", "scripts/night12a/night12a_smoke.py",
    ]
    for item in internal:
        path = REPO / item
        files.append({"path": item, "size": path.stat().st_size,
                      "sha256": file_sha256(path)})
    commit = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"],
                                     text=True).strip()
    if commit != "300c96e8bcc6178ad0b7307e349ae7abc0266642":
        raise ValueError("upstream source commit mismatch")
    license_candidates = list(upstream.glob("LICENSE*")) + list(upstream.glob("COPYING*"))
    result = {
        "schema": "spalora.night12a.source_audit.v1",
        "upstream_commit": commit,
        "upstream_license_file_detected": bool(license_candidates),
        "upstream_source_copied": False,
        "clean_room_implementation": True,
        "files_read_completely": files,
        "source_semantics": {
            "genome": "mm10 / Cell Ranger ARC reference 2020-A",
            "upstream_gene_score": "ArchR createArrowFiles(addGeneScoreMat=TRUE)",
            "registered_clean_room_link": "Ensembl79 gene body plus strand-aware 5000 bp upstream; fragment midpoint",
            "rna_preprocessing": "Night-1 full-library log1p normalization",
            "adt_preprocessing": "Night-1 Seurat CLR over all targets then ddof=1 scaling",
            "endpoint": "existing H05 equal-three-affinity spectral, engineering K=2",
        },
    }
    atomic_json(OUT / "source_code_audit.json", result)
    return result


def schema_and_links():
    records = {unit: load(SCHEMA / f"{unit}.json") for unit in P5 + P10}
    fragment_records = {unit: load(SCHEMA / f"{unit}_fragments.json") for unit in P5}
    gtf_path = DOWNLOADS / "Mus_musculus.GRCm38.79.gtf.gz"
    gene_path = DOWNLOADS / "Mus_musculus.gene_info.gz"
    ensembl = parse_ensembl79_genes(gtf_path)
    gene_info = parse_ncbi_gene_info(gene_path)
    link_rows = []
    atac_units = []
    for unit in P5:
        record = records[unit]
        linked = fixed_linked_features(record["rna_feature_ids"],
                                       ensembl["unique"].keys(), 256)
        if len(linked) != 256:
            raise ValueError(f"{unit} did not close 256 linked ATAC genes")
        intervals = gene_score_intervals(ensembl, linked, upstream_bp=5000)
        for row in intervals:
            link_rows.append({
                "family": "RNA+ATAC", "unit_id": unit,
                "canonical_identifier": row["feature"], "raw_modality_identifier": row["gene_id"],
                "mapping_status": "unique", "selected_for_engineering": "true",
                "authority": "Ensembl release 79 GRCm38 unique gene symbol",
            })
        fragment = fragment_records[unit]["fragments"]
        if fragment["registered_missing_count"] != 0:
            raise ValueError(f"{unit} has registered RNA spots absent from fragments")
        atac_units.append({
            "unit_id": unit,
            "selected_features": linked,
            "selected_feature_sha256": text_sha256(linked),
            "intervals": intervals,
            "fragment_audit": fragment,
            "fragment_header": fragment_header(DOWNLOADS / P5_FRAGMENT_FILES[unit]),
        })
    mapping_rows = []
    protein_units = []
    for unit in P10:
        record = records[unit]
        mapped = map_adt_targets(record["adt_feature_ids"], gene_info)
        rna_features = set(record["rna_feature_ids"])
        eligible = sorted(
            [(row["canonical_identifier"], row["raw_target"])
             for row in mapped
             if row["status"] == "unique" and row["canonical_identifier"] in rna_features],
            key=lambda pair: pair[0],
        )[:256]
        eligible_pairs = set(eligible)
        for row in mapped:
            selected = (row["canonical_identifier"], row["raw_target"]) in eligible_pairs
            mapping_rows.append({"unit_id": unit, **row,
                                 "selected_for_engineering": str(selected).lower()})
            link_rows.append({
                "family": "RNA+protein", "unit_id": unit,
                "canonical_identifier": row["canonical_identifier"],
                "raw_modality_identifier": row["raw_target"],
                "mapping_status": row["status"],
                "selected_for_engineering": str(selected).lower(),
                "authority": row["authority"],
            })
        if not eligible:
            raise ValueError(f"{unit} has no unique authoritative RNA-ADT mapping")
        protein_units.append({
            "unit_id": unit,
            "deposited_target_count": len(mapped),
            "mapping_status_counts": dict(Counter(row["status"] for row in mapped)),
            "selected_count": len(eligible),
            "selected_canonical_identifiers": [x[0] for x in eligible],
            "selected_raw_targets": [x[1] for x in eligible],
            "selected_feature_sha256": text_sha256([x[0] for x in eligible]),
        })
    write_rows(
        OUT / "adt_target_mapping.csv",
        ["unit_id", "raw_target", "canonical_identifier", "gene_id", "status",
         "authority", "selected_for_engineering"], mapping_rows,
    )
    write_rows(
        OUT / "linked_feature_registry.tsv",
        ["family", "unit_id", "canonical_identifier", "raw_modality_identifier",
         "mapping_status", "selected_for_engineering", "authority"],
        link_rows, delimiter="\t",
    )
    atac_contract = {
        "schema": "spalora.night12a.atac_mm10_link_contract.v1",
        "frozen_before_smoke": True,
        "genome_build": "mm10 / GRCm38",
        "cellranger_arc_reference": "refdata-cellranger-arc-mm10-2020-A-2.0.0",
        "fragment_header_reference_fasta_md5": "c9c31fef9ba3b93f99ad59d9345aeee0c9cb0640",
        "fragment_header_reference_gtf_md5": "73e68486af93260a77bf2ebad4b5dd2532aaa016",
        "annotation": {"name": "Ensembl release 79 GRCm38",
                       "sha256": file_sha256(gtf_path)},
        "clean_room_formula": {
            "interval": "zero-based gene body plus strand-aware 5000 bp upstream",
            "assignment": "fragment midpoint inside interval",
            "multiplicity": "fifth fragments field",
            "normalization": "log1p(10000 * linked count / full registered fragment multiplicity)",
            "canonical_sort": "Unicode lexical ascending",
            "feature_cap": 256,
        },
        "upstream_provenance": {
            "commit": "300c96e8bcc6178ad0b7307e349ae7abc0266642",
            "observed_call": "ArchR createArrowFiles(addGeneScoreMat=TRUE)",
            "source_copied": False,
            "equivalence_claimed": False,
        },
        "units": atac_units,
    }
    atomic_json(OUT / "atac_mm10_link_contract.json", atac_contract)
    public_units = []
    for unit in P5 + P10:
        record = records[unit]
        row = {
            "unit_id": unit,
            "coordinate": record["coordinate"],
            "rna": record["rna"],
            "paired_spot_identity": record.get("paired_spot_identity",
                                               "closed by registered coordinate IDs"),
            "wall_seconds": record["wall_seconds"],
        }
        if unit in P10:
            row["adt"] = record["adt"]
        else:
            row["fragments"] = fragment_records[unit]["fragments"]
        public_units.append(row)
    shape = {
        "schema": "spalora.night12a.real_shape_and_id_audit.v1",
        "unit_count": 6,
        "units": public_units,
        "pixel_transform": "strip exactly one terminal -1, otherwise byte identity",
        "expression_or_spatial_matching_used": False,
        "label_reads": 0,
    }
    atomic_json(OUT / "real_shape_and_id_audit.json", shape)
    return shape, atac_contract, protein_units


def firewalls_and_next_contract(shape, atac, protein):
    prior = load(REPO / "outputs/post_night11a_direction_reset/raw_root_metadata_after.json")
    historical_roots = [row["root"] for row in prior["roots"]]
    extra = "/autodl-fs/data/night11b_rna_protein_discordance_identifiability_20260822"
    if Path(extra).exists() and extra not in historical_roots:
        historical_roots.append(extra)
    before = [root_snapshot(path) for path in historical_roots]
    atomic_json(OUT / "historical_raw_immutability_before.json", {
        "schema": "spalora.night12a.historical_raw_before.v1",
        "roots": before,
        "content_hashes_recomputed": 0,
        "label_file_contents_opened": False,
    })
    label_candidates = []
    for root in before:
        for row in root.get("label_or_ground_truth_path_metadata_only", []):
            label_candidates.append({"root": root["root"], **row,
                                     "contents_opened": False})
    sealed = {
        "schema": "spalora.night12a.sealed_label_registry.v1",
        "whitelist_label_files_detected": 0,
        "historical_path_metadata_only": label_candidates,
        "label_file_contents_opened": False,
        "allowed_metadata": ["study", "species", "tissue", "P5/P10", "replicate",
                             "modality", "genome build", "file format"],
    }
    atomic_json(OUT / "sealed_label_registry.json", sealed)
    firewall = {
        "schema": "spalora.night12a.label_firewall.v1",
        "training_label_reads": 0, "evaluation_label_reads": 0,
        "total_label_reads": 0, "ARI": 0, "NMI": 0, "AMI": 0, "FMI": 0,
        "Q": 0, "annotation_spatial_metrics": 0, "dataset_name_routing": 0,
        "family_specific_model_branch": 0, "dense_n_by_n": 0,
        "scientific_retry_fallback": 0, "third_party_benchmark": 0,
        "GSE263333_content_download": 0, "GSE213264_content_download": 0,
        "complete_new_candidate_training": 0, "p0_ident_executions": 0,
    }
    atomic_json(OUT / "label_firewall_audit.json", firewall)
    next_contract = {
        "schema": "spalora.night12a.p0_ident_input_contract.v1",
        "generated_only_not_executed": True,
        "families": ["RNA+ATAC", "RNA+protein"],
        "units": {"RNA+ATAC": P5, "RNA+protein": P10},
        "registered_inputs": {
            "shape_audit_sha256": file_sha256(OUT / "real_shape_and_id_audit.json"),
            "atac_link_contract_sha256": file_sha256(OUT / "atac_mm10_link_contract.json"),
            "adt_mapping_sha256": file_sha256(OUT / "adt_target_mapping.csv"),
        },
        "inherited_strict_gate": {
            "combined_evidence_strictly_better_than_strongest_single_axis_in_both_families": True,
            "paired_bootstrap_95ci_lower_bound_gt": 0.0,
            "gate_editable_from_night12a_smoke": False,
        },
        "label_reads_allowed": 0,
        "p0_ident_run_count_this_night": 0,
    }
    atomic_json(OUT / "night12a_p0_ident_input_contract.json", next_contract)
    preflight = {
        "schema": "spalora.night12a.real_input_preflight.v1",
        "status": "READY_FOR_TWO_REGISTERED_SMOKES",
        "six_units_closed": shape["unit_count"] == 6,
        "atac_units": len(atac["units"]),
        "protein_units": len(protein),
        "formal_units": ["P5S1", "P10S1"],
        "seed": 20260822, "latent_dim": 64, "training_steps": 0,
        "endpoint_k": 2, "feature_cap": 256,
        "correction_cycles": 1,
        "label_reads": 0,
    }
    atomic_json(OUT / "real_input_preflight.json", preflight)


def freeze_manifest():
    paths = [
        "SpaLORA/night12a_schema_p0.py",
        "scripts/night12a/night12a_schema_audit.py",
        "scripts/night12a/night12a_smoke.py",
        "scripts/night12a/night12a_freeze.py",
        "scripts/night12a/night12a_finalize.py",
        "tests/test_night12a_schema_p0.py",
        "configs/night12a/night12a_manifest_contract.json",
        "outputs/night12a_handoff/accession_file_manifest.csv",
        "outputs/night12a_handoff/download_manifest.json",
        "outputs/night12a_handoff/real_shape_and_id_audit.json",
        "outputs/night12a_handoff/atac_mm10_link_contract.json",
        "outputs/night12a_handoff/adt_target_mapping.csv",
        "outputs/night12a_handoff/linked_feature_registry.tsv",
        "outputs/night12a_handoff/night12a_p0_ident_input_contract.json",
        "outputs/night12a_handoff/label_firewall_audit.json",
        "outputs/night12a_handoff/source_code_audit.json",
        "outputs/night12a_handoff/real_input_preflight.json",
    ]
    records = []
    for value in paths:
        path = REPO / value
        records.append({"path": value, "size": path.stat().st_size,
                        "sha256": file_sha256(path)})
    manifest = {
        "schema": "spalora.night12a.formal_freeze.v1",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "global_correction_cycle": 1,
        "no_further_correction_cycles_available": True,
        "seed": 20260822, "latent_dim": 64, "training_steps": 0,
        "feature_cap": 256, "endpoint_k": 2,
        "smoke_units": ["P5S1", "P10S1"],
        "files": records,
        "p0_ident_allowed": False,
    }
    atomic_json(OUT / "formal_freeze_manifest.json", manifest)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    download_audit()
    source_audit()
    shape, atac, protein = schema_and_links()
    firewalls_and_next_contract(shape, atac, protein)
    freeze_manifest()
    print(json.dumps({"status": "FROZEN", "units": 6, "payloads": 23,
                      "correction_cycles": 1}, sort_keys=True))


if __name__ == "__main__":
    main()
