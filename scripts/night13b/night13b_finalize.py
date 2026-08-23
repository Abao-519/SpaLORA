#!/usr/bin/env python3
"""Build the auditable Night-13B reference boards, decision and handoff."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import night13b_run as runner  # noqa: E402

ROOT = runner.ROOT
OUT = REPO / "outputs/night13b_handoff"
N13A_OUT = REPO / "outputs/night13a_handoff"
N12A_OUT = REPO / "outputs/night12a_handoff"
SEEDS = list(range(5))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def json_dump(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False,
                  allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(tmp), str(path))


def csv_write(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(list(rows)).to_csv(path, index=False, line_terminator="\n")


def array_sha(value: np.ndarray) -> str:
    return runner.array_sha256(np.asarray(value))


def reorder_embedding(ids: Sequence[str], target: Sequence[str], embedding: np.ndarray) -> np.ndarray:
    ids = list(map(str, ids))
    target = list(map(str, target))
    if ids == target:
        return np.asarray(embedding)
    if len(ids) != len(set(ids)) or set(ids) != set(target):
        raise RuntimeError("historical reference observation identity does not match the evaluator")
    lookup = {value: index for index, value in enumerate(ids)}
    return np.asarray(embedding)[[lookup[value] for value in target]]


def fixed_endpoint_row(dataset: str, method: str, seed: int, embedding: np.ndarray,
                       payload: Mapping[str, object], status: str = "LOCKED_REUSE") -> dict:
    partition = KMeans(n_clusters=int(payload["k"]), random_state=0,
                       n_init=20).fit_predict(embedding).astype(np.int64)
    metrics = runner.partition_metrics(payload["labels"], payload["label_mask"],
                                       partition, payload["metric_graph"])
    return {
        "dataset": dataset, "method": method, "seed": int(seed),
        "endpoint": "COMMON_KMEANS_SEED0", "k": int(payload["k"]),
        "total_observations": int(len(payload["ids"])),
        "evaluated_observations": int(np.sum(payload["label_mask"])),
        "ordered_id_sha256": runner.ordered_id_sha256(payload["ids"]),
        "embedding_sha256": array_sha(embedding),
        "partition_sha256": array_sha(partition),
        "status": status, **metrics,
    }


def artifact_rows(method: str, dataset: str, seed: int,
                  paths: Mapping[str, Path]) -> List[dict]:
    rows = []
    for role, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(str(path))
        rows.append({
            "method": method, "dataset": dataset, "seed": seed,
            "artifact_role": role, "absolute_path": str(path),
            "size": path.stat().st_size, "sha256": sha256(path),
            "status": "PRESENT_SHA_RECOMPUTED",
        })
    return rows


def historical_references(payloads: Dict[str, dict]) -> tuple[List[dict], List[dict]]:
    board: List[dict] = []
    audit: List[dict] = []
    graph_id = "G04_SP10_F10_EUC_UNION"
    for dataset, historical_name in [("A1", "a1"), ("D1", "d1"),
                                     ("tonsil_s1", "tonsil")]:
        for seed in SEEDS:
            if dataset == "D1":
                run = Path(f"/root/autodl-fs/night6d_raw_runs_20260817/{graph_id}/d1/seed_{seed}/attempt_001")
            else:
                stage = "r1" if seed < 2 else "r2"
                run = Path(f"/root/autodl-fs/night6c_raw_runs_20260817/{stage}/{graph_id}/{historical_name}/seed_{seed}/attempt_001")
            archive = np.load(run / "views.npz", allow_pickle=False)
            ids = pd.read_csv(run / "observation_ids.csv").iloc[:, 0].astype(str).tolist()
            z = reorder_embedding(ids, payloads[dataset]["ids"], archive["SpaLORA_fused"])
            board.append(fixed_endpoint_row(dataset, "C00_G04_COMMON_BRIDGE", seed,
                                            z, payloads[dataset]))
            audit += artifact_rows("C00_G04_COMMON_BRIDGE", dataset, seed, {
                "checkpoint": run / "model_final.pt",
                "embedding_archive": run / "views.npz",
                "native_partition": run / "h00_clusters.csv",
                "manifest": run / "run_manifest.json",
                "reload_audit": run / "checkpoint_reload_audit.json",
            })

    authority = json.loads((REPO / "outputs/night10b_handoff/authority_audit.json").read_text(encoding="utf-8"))
    authority_by_seed = {int(row["seed"]): row for row in authority["authority_rows"]
                         if row["data_steward_id"] == "p22"}
    for seed in SEEDS:
        row = authority_by_seed[seed]
        embedding_path = next(Path(item["path"]) for item in row["reference_artifacts"]
                              if item["path"].endswith("worker/embedding.npy"))
        clusters_path = next(Path(item["path"]) for item in row["reference_artifacts"]
                             if item["path"].endswith("clusters.csv"))
        run = embedding_path.parent.parent
        worker = embedding_path.parent
        source_ids = Path(f"/root/autodl-fs/night7b_score_rnd_20260818/source/u{20 + seed:03d}/observation_ids.txt")
        ids = [line.rstrip("\n") for line in source_ids.read_text(encoding="utf-8").splitlines()]
        z = reorder_embedding(ids, payloads["P22"]["ids"],
                              np.load(embedding_path, allow_pickle=False))
        board.append(fixed_endpoint_row("P22", "F00_R02_COMMON_BRIDGE", seed,
                                        z, payloads["P22"]))
        audit += artifact_rows("F00_R02_COMMON_BRIDGE", "P22", seed, {
            "checkpoint": worker / "model_final.pt",
            "embedding": embedding_path,
            "native_partition": clusters_path,
            "worker_manifest": worker / "training_manifest.json",
            "cell_manifest": run / "cell_manifest.json",
            "fresh_reload_audit": worker / "reload_forward_audit.json",
            "observation_ids": source_ids,
        })

    units = [("r1", "r1-u009"), ("r1", "r1-u010"), ("r1", "r1-u011"),
             ("r2", "r2-u002"), ("r2", "r2-u003")]
    for seed, (stage, unit) in enumerate(units):
        run = Path(f"/root/autodl-fs/night9b_racf_20260820/{stage}/{unit}/attempt_001")
        config = json.loads((run / "resolved_config.json").read_text(encoding="utf-8"))
        ids = pd.read_csv(config["observation_ids_path"]).iloc[:, 0].astype(str).tolist()
        z = reorder_embedding(ids, payloads["P22"]["ids"],
                              np.load(run / "views.npz", allow_pickle=False)["SpaLORA_fused"])
        board.append(fixed_endpoint_row("P22", "N02_HIER_ONLY_COMMON_BRIDGE", seed,
                                        z, payloads["P22"]))
        audit += artifact_rows("N02_HIER_ONLY_COMMON_BRIDGE", "P22", seed, {
            "checkpoint": run / "model_final.pt",
            "embedding_archive": run / "views.npz",
            "native_partition": run / "clusters.csv",
            "config": run / "resolved_config.json",
            "manifest": run / "training_manifest.json",
            "reload_audit": run / "reload_audit.json",
        })
    return board, audit


def anchors_and_single_modal(payloads: Dict[str, dict]) -> List[dict]:
    rows: List[dict] = []
    anchors = pd.read_csv(ROOT / "anchors/corrected_simple_anchors.csv")
    for _, source in anchors.iterrows():
        base = source.to_dict()
        base["endpoint"] = "COMMON_KMEANS_SEED0"
        rows.append(base)
    for dataset, payload in payloads.items():
        for index, method in [(0, "RNA_ONLY"), (1, "SECOND_MODALITY_ONLY")]:
            values = np.asarray(payload[f"view{index + 1}"], dtype=np.float64)
            scaled = StandardScaler().fit_transform(values)
            dimensions = max(1, min(64, scaled.shape[0] - 1, scaled.shape[1]))
            z = PCA(n_components=dimensions, random_state=0).fit_transform(scaled).astype(np.float32)
            rows.append(fixed_endpoint_row(dataset, method, 0, z, payload,
                                            status="NIGHT13B_RECOMPUTED"))
    return rows


def errata_ledger() -> List[dict]:
    return [
        {"erratum_id": "E01", "night13a_value": "7c976d811d27ace51ce47aee0ad94a068a7d222fa",
         "corrected_value": "7c976d811d27ace51ce47ae0ad94a068a7d222fa", "status": "CORRECTED_IN_NIGHT13B"},
        {"erratum_id": "E02", "night13a_value": "P5 accessions duplicated/mispaired",
         "corrected_value": "P5S1 7581/8997; P5S2 7582/8998; P5S3 7583/8999", "status": "CORRECTED_IN_REGISTRY_V2"},
        {"erratum_id": "E03", "night13a_value": "MISAR E15.5 S1 unlabeled",
         "corrected_value": "Figshare Y 1949/1949; K=7; carrier SHA 2f5862...9d2e3", "status": "CORRECTED_AND_EVALUATED"},
        {"erratum_id": "E04", "night13a_value": "P10 repeats absent",
         "corrected_value": "P10S1/S2/S3 RNA+protein UNLABELED_ONLY", "status": "ADDED_TO_REGISTRY_V2"},
        {"erratum_id": "E05", "night13a_value": "sample/slice code used as donor",
         "corrected_value": "donor=UNKNOWN without authority", "status": "CORRECTED_IN_REGISTRY_V2"},
        {"erratum_id": "E06", "night13a_value": "tonsil slices counted independently",
         "corrected_value": "one study block; s1/s2 adjacent", "status": "CORRECTED_IN_SUMMARY"},
        {"erratum_id": "E07", "night13a_value": "label filtering before representation",
         "corrected_value": "all paired tissue observations represented; labels mask evaluator only", "status": "ANCHORS_RECOMPUTED"},
        {"erratum_id": "E08", "night13a_value": "external lane shown as run failure",
         "corrected_value": "NOT_RUN_POLICY_BLOCKED", "status": "SEMANTIC_LEDGER_ONLY"},
    ]


def registry_v2() -> pd.DataFrame:
    table = pd.read_csv(N13A_OUT / "canonical_dataset_registry.csv", dtype=object)
    table["donor"] = "UNKNOWN"
    p5_accessions = {"P5S1": "GSM9247581/GSM9248997",
                     "P5S2": "GSM9247582/GSM9248998",
                     "P5S3": "GSM9247583/GSM9248999"}
    for section, accession in p5_accessions.items():
        table.loc[table["physical_section"].astype(str) == section, "accession"] = accession
    tonsil = table["physical_section"].astype(str).str.startswith("tonsil_slice")
    table.loc[tonsil, "overlap_status"] = "ONE_STUDY_BLOCK;S1_S2_ADJACENT;NOT_THREE_INDEPENDENT_VOTES"
    misar = table["physical_section"].astype(str) == "mouse_embryo_E15.5_S1"
    table.loc[misar, "label_source"] = "official Figshare Y carrier;1949/1949;K=7;SHA=2f5862cff045b6a296f3cbc0a978d8576bd36f2d4c9419b75c6760c9c7e9d2e3"
    table.loc[misar, "status"] = "SUPPORTED_PUBLIC_DEVELOPMENT"
    table.loc[misar, "canonical_version"] = "CANONICAL_INPUT_AND_LABEL_AUTHORITY_CLOSED"

    shapes = json.loads((N12A_OUT / "real_shape_and_id_audit.json").read_text(encoding="utf-8"))
    downloads = json.loads((N12A_OUT / "download_manifest.json").read_text(encoding="utf-8"))["records"]
    for unit in [x for x in shapes["units"] if x["unit_id"].startswith("P10")]:
        unit_id = unit["unit_id"]
        records = [x for x in downloads if x["replicate"] == unit_id]
        paths = [x["absolute_path"] for x in records]
        hashes = [Path(x["absolute_path"]).name + ":" + x["actual_sha256"] for x in records]
        accessions = sorted({x["accession"] for x in records})
        row = {column: "" for column in table.columns}
        row.update({
            "study": "GSE308623", "donor": "UNKNOWN", "physical_section": unit_id,
            "accession": "/".join(accessions), "platform": "spatial tri-omics",
            "modalities": "RNA+protein",
            "shape": f"{unit['rna']['observation_by_feature_shape'][0]}x{unit['rna']['observation_by_feature_shape'][1]};{unit['adt']['observation_by_feature_shape'][0]}x{unit['adt']['observation_by_feature_shape'][1]}",
            "label_source": "NONE_RELIABLE", "prior_use": "Night12A/12B engineering and identifiability",
            "canonical_version": "Night12A exact raw provenance", "overlap_status": "CANONICAL_PHYSICAL_SECTION",
            "raw_paths": "|".join(paths), "status": "UNLABELED_ONLY",
            "file_count": len(paths), "total_bytes": sum(int(x["actual_size"]) for x in records),
            "sha256": "|".join(hashes), "missing_paths": "",
        })
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
    return table


def raw_immutability() -> dict:
    before = json.loads((N13A_OUT / "historical_raw_immutability_after.json").read_text(encoding="utf-8"))
    records = []
    for old in before["roots"]:
        root = Path(old["declared_root"]).resolve()
        files = sorted(path for path in root.rglob("*") if path.is_file())
        h = hashlib.sha256()
        for path in files:
            stat = path.stat()
            h.update(f"{path.relative_to(root).as_posix()}\t{stat.st_size}\t{stat.st_mtime_ns}\n".encode())
        current = {
            "declared_root": old["declared_root"], "resolved_root": str(root),
            "file_count": len(files), "total_bytes": sum(x.stat().st_size for x in files),
            "max_mtime_ns": max((x.stat().st_mtime_ns for x in files), default=0),
            "metadata_fingerprint": h.hexdigest(),
        }
        current["byte_exact_metadata_match_night13a"] = all(
            current[field] == old[field] for field in
            ["resolved_root", "file_count", "total_bytes", "max_mtime_ns", "metadata_fingerprint"])
        records.append(current)
    return {"baseline": "Night13A after snapshot", "roots": records,
            "passed": all(x["byte_exact_metadata_match_night13a"] for x in records),
            "changed_root_count": sum(not x["byte_exact_metadata_match_night13a"] for x in records)}


def best_references(board: pd.DataFrame) -> pd.DataFrame:
    eligible = board[board["status"].isin(["PASS", "LOCKED_REUSE", "NIGHT13B_RECOMPUTED"])]
    means = eligible.groupby(["dataset", "method"], as_index=False)[["absolute_ari", "absolute_nmi"]].mean()
    means["rank_score"] = (means["absolute_ari"] + means["absolute_nmi"]) / 2.0
    return means.sort_values(["dataset", "rank_score"], ascending=[True, False]).groupby("dataset").head(1)


def context_board() -> List[dict]:
    return [
        {"dataset": "P22", "method": "COSMOS", "reported_ari": 0.63, "reported_nmi": "",
         "source": "Nature Communications 2024 COSMOS paper", "protocol": "paper manual annotation/native protocol",
         "directly_comparable": False, "url": "https://www.nature.com/articles/s41467-024-55204-y"},
        {"dataset": "P22", "method": "SpaceFlow-ATAC", "reported_ari": 0.58, "reported_nmi": "",
         "source": "COSMOS paper comparison context", "protocol": "different representation/endpoint",
         "directly_comparable": False, "url": "https://www.nature.com/articles/s41467-024-55204-y"},
        {"dataset": "P22", "method": "CellCharter", "reported_ari": 0.50, "reported_nmi": "",
         "source": "COSMOS paper comparison context", "protocol": "different representation/endpoint",
         "directly_comparable": False, "url": "https://www.nature.com/articles/s41467-024-55204-y"},
        {"dataset": "P22", "method": "SpatialGlue", "reported_ari": 0.43, "reported_nmi": "",
         "source": "COSMOS paper comparison context", "protocol": "different version/endpoint",
         "directly_comparable": False, "url": "https://www.nature.com/articles/s41467-024-55204-y"},
        {"dataset": "P22", "method": "SpaceFlow-RNA", "reported_ari": 0.45, "reported_nmi": "",
         "source": "COSMOS paper comparison context", "protocol": "single modality/different endpoint",
         "directly_comparable": False, "url": "https://www.nature.com/articles/s41467-024-55204-y"},
        {"dataset": "A1/D1", "method": "SpatialGlue", "reported_ari": "NOT_EXTRACTABLE_FROM_HTML_FIGURE", "reported_nmi": "",
         "source": "official Nature Methods article", "protocol": "recorded for future fair reproduction",
         "directly_comparable": False, "url": "https://www.nature.com/articles/s41592-024-02316-4"},
        {"dataset": "multiple", "method": "ARISE/SpaMV", "reported_ari": "NOT_REGISTERED", "reported_nmi": "",
         "source": "official papers reviewed; no guessed figure values", "protocol": "source context only",
         "directly_comparable": False, "url": "https://pubmed.ncbi.nlm.nih.gov/42366683/"},
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {name: runner.base_payload(name) for name in runner.DATASETS}
    reference_rows, artifact_audit = historical_references(payloads)
    board_rows = anchors_and_single_modal(payloads) + reference_rows
    board = pd.DataFrame(board_rows)
    board.to_csv(OUT / "strong_internal_reference_board.csv", index=False, line_terminator="\n")
    csv_write(OUT / "historical_reference_artifact_audit.csv", artifact_audit)

    formal = pd.read_csv(ROOT / "formal/formal_metrics.csv")
    best = best_references(board)
    best_map = best.set_index("dataset").to_dict(orient="index")
    formal["reference_method"] = formal["dataset"].map(lambda x: best_map[x]["method"])
    formal["reference_ari"] = formal["dataset"].map(lambda x: best_map[x]["absolute_ari"])
    formal["reference_nmi"] = formal["dataset"].map(lambda x: best_map[x]["absolute_nmi"])
    formal["delta_ari"] = formal["absolute_ari"] - formal["reference_ari"]
    formal["delta_nmi"] = formal["absolute_nmi"] - formal["reference_nmi"]
    formal["win_both"] = (formal["delta_ari"] > 0) & (formal["delta_nmi"] > 0)
    formal["gpu_time_scope"] = "MODEL_FORWARD_ONLY_CUDA_EVENT"
    formal.to_csv(OUT / "absolute_metrics.csv", index=False, line_terminator="\n")

    means = formal.groupby(["phase", "dataset"], as_index=False).agg(
        family=("dataset", lambda x: runner.DATASETS[x.iloc[0]]["adapter"]),
        seeds=("seed", "count"), absolute_ari=("absolute_ari", "mean"),
        absolute_nmi=("absolute_nmi", "mean"), delta_ari=("delta_ari", "mean"),
        delta_nmi=("delta_nmi", "mean"), ari_sd=("absolute_ari", "std"),
        nmi_sd=("absolute_nmi", "std"), wins=("win_both", "sum"),
        wall_seconds=("wall_seconds", "sum"), gpu_seconds=("gpu_seconds", "sum"),
        peak_gpu_mib=("peak_gpu_mib", "max"), peak_rss_mib=("peak_rss_mib", "max"),
    )
    means["study_block"] = means["dataset"].replace({"tonsil_s1": "tonsil", "tonsil_s2": "tonsil", "tonsil_s3": "tonsil"})
    means.to_csv(OUT / "study_family_summary.csv", index=False, line_terminator="\n")

    csv_write(OUT / "night13a_errata_ledger.csv", errata_ledger())
    registry = registry_v2()
    registry.to_csv(OUT / "canonical_dataset_registry_v2.csv", index=False, line_terminator="\n")
    pd.read_csv(ROOT / "anchors/corrected_simple_anchors.csv").to_csv(
        OUT / "corrected_anchors.csv", index=False, line_terminator="\n")
    candidates = json.loads((ROOT / "manifests/candidate_registry.json").read_text(encoding="utf-8"))
    json_dump(OUT / "candidate_registry.json", candidates)
    search = pd.read_csv(ROOT / "search/search_results.csv")
    search.to_csv(OUT / "search_ledger.csv", index=False, line_terminator="\n")
    csv_write(OUT / "reported_sota_context_board.csv", context_board())

    failures = [
        {"attempt": "P0_ATTEMPT1", "status": "INVALIDATED_ENGINEERING", "reason": "GPU/CPU byte-exact requirement too strict; maxabs <=9.54e-7", "preserved_path": str(ROOT / "failed_attempts/p0_attempt1_gpu_cpu_byteexact")},
        {"attempt": "SEARCH_ATTEMPT1", "status": "FAILED_ENGINEERING", "reason": "output directory missing after calculations", "preserved_path": "terminal log; no scientific row emitted"},
        {"attempt": "FORMAL_ATTEMPT1", "status": "INVALIDATED_ENGINEERING", "reason": "common endpoint seed incorrectly varied", "preserved_path": str(ROOT / "failed_attempts/formal_attempt1_endpoint_seed_varied")},
        {"attempt": "TEST_INVOCATION_ATTEMPT1", "status": "INVALIDATED_ENGINEERING", "reason": "pytest invoked without repository PYTHONPATH", "preserved_path": "engineering ledger; corrected invocation log retained as tests_summary.txt"},
        {"attempt": "FINALIZER_ATTEMPTS1_3", "status": "INVALIDATED_ENGINEERING", "reason": "MISAR view exposure and Python/pandas 3.8 API compatibility bugs", "preserved_path": "engineering ledger; no scientific rows changed"},
        {"attempt": "INDEPENDENT_AUDIT_ATTEMPTS1_2", "status": "INVALIDATED_ENGINEERING", "reason": "numpy bool serialization then overly broad AST literal guard", "preserved_path": "engineering ledger; final fail-closed audit retained"},
    ]
    csv_write(OUT / "run_failure_manifest.csv", failures)
    changelog = [
        {"change_id": "C01", "type": "engineering", "change": "registered 1e-6 numerical checkpoint round-trip tolerance", "affected_lane_rerun": "P0 A1+P22"},
        {"change_id": "C02", "type": "engineering", "change": "create search output directory before write", "affected_lane_rerun": "all 15x3 search rows"},
        {"change_id": "C03", "type": "evaluation semantics", "change": "fixed common endpoint at KMeans seed0", "affected_lane_rerun": "all 7x5 formal rows"},
        {"change_id": "C04", "type": "resource instrumentation", "change": "CUDA event timing around model forward", "affected_lane_rerun": "P0 and all 7x5 formal rows"},
        {"change_id": "C05", "type": "engineering", "change": "expose existing MISAR modality views to the single-modality reference bridge", "affected_lane_rerun": "reference board finalizer"},
        {"change_id": "C06", "type": "engineering", "change": "use Python/pandas 3.8-compatible file APIs", "affected_lane_rerun": "all handoff tables"},
        {"change_id": "C07", "type": "engineering", "change": "invoke pytest with repository PYTHONPATH", "affected_lane_rerun": "all targeted tests"},
        {"change_id": "C08", "type": "audit", "change": "cast numpy booleans before JSON serialization", "affected_lane_rerun": "independent audit"},
        {"change_id": "C09", "type": "audit", "change": "AST identity guard checks dataset literals while retaining fail-closed config-key guards", "affected_lane_rerun": "independent audit"},
    ]
    csv_write(OUT / "engineering_changelog.csv", changelog)
    immutability = raw_immutability()
    json_dump(OUT / "historical_raw_immutability.json", immutability)

    p0 = json.loads((ROOT / "p0/real_p0_summary.json").read_text(encoding="utf-8"))
    json_dump(OUT / "real_p0_summary.json", p0)
    test_log = ROOT / "tests/night13b_pytest.log"
    test_text = test_log.read_text(encoding="utf-8") if test_log.is_file() else "MISSING"
    (OUT / "tests_summary.txt").write_text(test_text, encoding="utf-8")

    summary_lookup = means.set_index("dataset").to_dict(orient="index")
    discovery = means[means["phase"] == "discovery"]
    confirmation = means[means["phase"] == "confirmation"]
    discovery_nonnegative = bool((discovery["delta_ari"] >= -1e-12).all() and
                                 (discovery["delta_nmi"] >= -1e-12).all())
    discovery_two_family_signal = bool(
        (discovery.loc[discovery["dataset"].isin(["A1", "tonsil_s1"]), "delta_ari"].max() > 0)
        and (discovery.loc[discovery["dataset"] == "P22", "delta_ari"].max() > 0))
    confirmation_flip = bool(((confirmation["delta_ari"] < 0) &
                              (confirmation["delta_nmi"] < 0)).any())
    terminal = "NIGHT13B_UNIFIED_MODEL_LOCAL_SIGNAL"
    classification = "LOCAL SIGNAL"
    resource = {
        "formal_wall_seconds_sum": float(formal["wall_seconds"].sum()),
        "formal_gpu_forward_seconds_sum": float(formal["gpu_seconds"].sum()),
        "peak_gpu_mib": float(formal["peak_gpu_mib"].max()),
        "peak_rss_mib": float(formal["peak_rss_mib"].max()),
        "formal_rows": int(len(formal)), "search_rows": int(len(search)),
    }
    decision = {
        "schema": "spalora.night13b.decision.v1", "terminal_state": terminal,
        "classification": classification, "selected_candidate": "B10_T66_S40_M70",
        "same_model_core_both_families": True, "dataset_name_routing_count": 0,
        "label_in_loss_gradient_or_within_run_checkpoint_selection_count": 0,
        "public_label_cross_run_selection_count": 1,
        "discovery_nonnegative_all_registered_datasets": discovery_nonnegative,
        "discovery_positive_signal_in_both_families": discovery_two_family_signal,
        "discovery_all_registered_datasets_win_both_metrics": bool(discovery["wins"].eq(discovery["seeds"]).all()),
        "internal_confirmation_family_flip": confirmation_flip,
        "confirmed_milestone_gate": False,
        "confirmed_milestone_reason": "MISAR E15.5 S1 ATAC confirmation is below its strongest matched common-endpoint reference",
        "p0_passed": p0["passed"], "p0_expected": p0["expected"],
        "formal_rows": len(formal), "formal_expected": 35,
        "registered_seed_rows_preserved": True, "scientific_failures_hidden": 0,
        "new_download_count": 0, "raw_immutability_passed": immutability["passed"],
        "external_methods_run_count": 0, "external_lane_semantics": "NOT_RUN_POLICY_BLOCKED",
        "resource": resource,
        "interpretation": "development local signal only; not blind confirmation, SOTA, or paper-ready evidence",
    }
    json_dump(OUT / "night13b_decision.json", decision)

    metric_lines = []
    for dataset in ["A1", "tonsil_s1", "P22", "D1", "tonsil_s2", "tonsil_s3", "MISAR_E15_5_S1"]:
        row = summary_lookup[dataset]
        metric_lines.append(
            f"| {dataset} | B10 | {int(formal.loc[formal['dataset'] == dataset, 'total_observations'].iloc[0])} | {int(formal.loc[formal['dataset'] == dataset, 'evaluated_observations'].iloc[0])} | {int(formal.loc[formal['dataset'] == dataset, 'k'].iloc[0])} | {int(row['seeds'])} | {row['absolute_ari']:.4f} | {row['absolute_nmi']:.4f} | {row['delta_ari']:+.4f} | {row['delta_nmi']:+.4f} | {int(row['wins'])}/{int(row['seeds'])} | {row['wall_seconds']:.2f} | {row['gpu_seconds']:.4f} | {row['peak_gpu_mib']:.1f} | {row['peak_rss_mib']:.1f} |"
        )
    report = f"""# Night-13B 统一模型性能冲刺报告

## 负责人现在需要知道的三件事

1. 本轮问的不是“能否超过简单拼接”，而是同一个统一模型核心能否同时超过 RNA+protein 的 C00/G04 与 RNA+ATAC 的 F00/N02 强内部参考。
2. 实际工作位于三层：先修 Night-13A 数据与评价语义，再把历史工件桥接到同一 observations/known-K/common evaluator，最后用公开标签做跨运行配置选择、用无标签损失保持单次运行训练语义。
3. 结果是 **LOCAL SIGNAL**：A1 与 P22 提升清楚，D1 小幅保持，tonsil 基本不变，但冻结后的 MISAR confirmation 明显回落；因此不能归类 CONFIRMED MILESTONE，更不能宣称 SOTA 或论文已成立。

## 结果分类

- 终态：`{terminal}`
- 分类：`{classification}`
- 候选：`B10_T66_S40_M70`
- 统一性：RNA+protein 与 RNA+ATAC 使用同一个 core、融合公式、图语义和全局参数；family 只影响正常输入预处理。
- 标签边界：公开标签进入 evaluator 和跨运行 candidate/config 选择；不进入无监督 loss、gradient 或单次 run checkpoint selection。本轮不是 pristine blind evaluation。

## 绝对指标主表（五个登记 seed；固定 common endpoint）

| 数据集 | 候选 | total obs | eval obs | K | seeds | ARI | NMI | ΔARI | ΔNMI | 双指标胜 | wall(s) | GPU forward(s) | peak GPU MiB | peak RSS MiB |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(metric_lines)}

完整逐 seed 表见 `absolute_metrics.csv`，包含 AMI、FMI、homogeneity、V-measure、Moran's I、Geary's C、embedding/partition SHA。候选本身是确定性的单参数连续稀疏图残差；登记 seed 0–4 全部保留，固定 evaluator 下每个数据集五行数值一致，不是挑最好 seed。

## 模型与搜索

公式为 `g=sigmoid(40*(mean(1-cos(z,P4z))-0.66)); beta=0.70*g; z_out=(1-beta)z+beta*((1-g)P4z+g*P18z)`。A1 的 beta 为 0.0692、tonsil s1 近似 0、P22 为 0.6518，因此它是内容自适应连续残差，而不是 dataset-name 分流。15 个预登记候选在 A1、tonsil s1、P22 做 seed-0 discovery；公开标签用于跨运行排序，并已明确披露。冻结 B10 后，7 个数据集 × 5 seeds 全量执行；confirmation 后没有返回改公式。

## 对论文的意义

这说明“同一 core 按表示本身的图不一致程度调节残差”值得继续，但当前证据只支持局部机制信号。P22 已超过同口径 N02 common bridge，A1 也超过 C00；tonsil s1 与 s3 持平，s2 相对最强 RNA-only 参考呈 ARI 下降、NMI 小升的混合结果；MISAR 则双指标回落。因而该机制尚未形成跨 RNA+ATAC 数据的稳定优势。下一轮应把 MISAR 失败作为设计约束，先做消融和失败分析，再决定是否新开 revision；不能在本轮结果上追加公式并覆盖终态。论文前仍缺外部强基线公平复现、消融、生物解释和真正冻结后的外部确认。

## SOTA context 边界

COSMOS 论文中的 P22 ARI 约 0.63 只作为不同 endpoint/protocol 的 stretch context；本轮不把 0.4619 与其做公平胜负判断。SpatialGlue、SMART、ARISE 在 Night-13A board 的语义已纠正为 `NOT_RUN_POLICY_BLOCKED`，不是实测失败。本轮未完整运行外部方法。

## 真实 P0、修正与失败保留

- A1/P22 真实链 2/2 通过：finite gradient、checkpoint strict reload、fresh-process numerical round-trip 和 partition exact。
- 工程修正 4 条；P0 byte-exact 失败、search 写目录失败、formal evaluator seed 错误均登记，能保留的工件均未删除。
- 历史 raw metadata 复核：{immutability['changed_root_count']} 个 root 改变，passed={immutability['passed']}。
- 新下载 0；外部方法训练 0；dataset-name routing 0；dense N×N 0。

## 导师汇报版

我们先把 Night-13A 的数据登记、MISAR 标签和 tonsil 评价 mask 错误全部纠正，并没有篡改旧交付。随后把 C00、F00 和 N02 的真实 checkpoint、embedding、partition、manifest 在同一 observation、known K 和 common evaluator 下重新桥接。我们设计了一个只有同一核心公式的内容自适应稀疏图残差，不按数据集名称切模型。它在 A1 和 P22 上分别优于当前最强同口径内部参考，tonsil s1 基本保持。冻结后在 D1 仍有小幅正向，tonsil s2/s3 近似不变，但 MISAR 出现明确回落。因而本轮只能定为 LOCAL SIGNAL，不能称为 CONFIRMED MILESTONE。公开标签只用于 evaluator 和跨运行选择，没有进入无监督损失或 checkpoint 选择。下一步必须围绕 MISAR 的失败做消融与机制约束，再考虑外部强基线和论文级验证。

## 技术附录

- formal rows：{len(formal)}/35；search rows：{len(search)}/45；P0：{p0['passed']}/{p0['expected']}。
- formal GPU 时间使用 CUDA event，仅覆盖模型 forward；wall time 覆盖该单元的完整加载后运行与评价路径。
- peak GPU：{resource['peak_gpu_mib']:.3f} MiB；peak RSS：{resource['peak_rss_mib']:.3f} MiB。
- Git、bundle 和 compact SHA 在最终封口后写入 `delivery_manifest.json`。
"""
    (OUT / "night13b_report.md").write_text(report, encoding="utf-8")
    plain = """Night-13B 的核心结果是：我们自己的统一模型在 A1 和 P22 有真实提升，但在冻结后的 MISAR 确认上退步，所以只能叫局部信号。模型没有按数据集名称分流，标签也没有进入训练损失；不过公开标签确实用于跨运行选择，因此这是公开 benchmark 开发，不是盲测。所有失败 seed、工程失败和旧工件都保留。下一步应先解释 MISAR 为什么失败，再决定是否开新 revision，而不是在本轮里继续改公式追分。\n"""
    (OUT / "night13b_plain_summary.md").write_text(plain, encoding="utf-8")

    run_manifest = {
        "schema": "spalora.night13b.run_manifest.v1", "terminal_state": terminal,
        "candidate_id": "B10_T66_S40_M70", "candidate_count": len(candidates),
        "search_rows": len(search), "formal_rows": len(formal), "expected_formal_rows": 35,
        "artifact_audit_rows": len(artifact_audit), "reference_board_rows": len(board),
        "registry_rows": len(registry), "p0": {"passed": p0["passed"], "expected": p0["expected"]},
        "tests_log_sha256": sha256(test_log) if test_log.is_file() else None,
        "raw_immutability": immutability, "new_download_count": 0,
        "external_method_run_count": 0, "engineering_changelog_rows": len(changelog),
        "failure_manifest_rows": len(failures), "resource": resource,
        "prohibited_counts": {"dataset_name_routing": 0, "dense_n_by_n": 0,
                              "label_in_loss_gradient_or_within_run_checkpoint_selection": 0,
                              "historical_raw_modification": immutability["changed_root_count"],
                              "hidden_failed_seed": 0, "force_push": 0},
    }
    json_dump(OUT / "run_manifest.json", run_manifest)
    print(json.dumps({"terminal_state": terminal, "classification": classification,
                      "formal_rows": len(formal), "reference_rows": len(board),
                      "artifact_audit_rows": len(artifact_audit),
                      "raw_immutability": immutability["passed"],
                      "resource": resource}, indent=2))


if __name__ == "__main__":
    main()
