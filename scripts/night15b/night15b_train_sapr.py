#!/usr/bin/env python3
"""Night-15B SAPR GPU screen, finalist expansion and replay audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/night15b"))

from SpaLORA.night15b_sapr import (  # noqa: E402
    SAPRCore,
    build_stability_anchors,
    sapr_loss,
    scipy_csr_to_torch,
)
from night15b_local_runner import (  # noqa: E402
    csr_from_archive,
    metrics,
    refine_partition,
    row_stochastic,
    sha256_array,
)


DEFAULT_KIT = Path("/root/autodl-fs/night15b_stability_anchored_prototype_score_sprint_20260824/local_compute_kit")
DEFAULT_HEAD = Path("/root/autodl-fs/night15b_stability_anchored_prototype_score_sprint_20260824/head_hpo")
DEFAULT_OUTPUT = Path("/root/autodl-fs/night15b_stability_anchored_prototype_score_sprint_20260824/sapr")


def canonical_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def candidate_registry() -> List[dict]:
    rows = [
        ("S10_STABILITY_TRUST", "STABILITY_TRUST", 64, 100, {"anchor": 1.0, "trust": 1.0}),
        ("S11_STABILITY_TRUST_STRONG", "STABILITY_TRUST", 64, 140, {"anchor": 1.0, "trust": 2.0}),
        ("S12_STABILITY_TRUST_WEAK", "STABILITY_TRUST", 64, 100, {"anchor": 1.0, "trust": 0.35}),
        ("S20_CROSS_VIEW", "CROSS_VIEW", 64, 120, {"anchor": 1.0, "trust": 0.7, "cross_view": 0.5}),
        ("S21_CROSS_VIEW_STRONG", "CROSS_VIEW", 64, 140, {"anchor": 1.0, "trust": 0.7, "cross_view": 1.2}),
        ("S22_CROSS_VIEW_BALANCED", "CROSS_VIEW", 48, 140, {"anchor": 1.0, "trust": 0.8, "cross_view": 0.6, "balance": 0.08}),
        ("S30_BOUNDARY_GRAPH", "BOUNDARY_GRAPH", 64, 120, {"anchor": 1.0, "trust": 0.8, "boundary": 0.4}),
        ("S31_BOUNDARY_GRAPH_STRONG", "BOUNDARY_GRAPH", 64, 160, {"anchor": 1.0, "trust": 0.6, "boundary": 1.0}),
        ("S32_BOUNDARY_BALANCED", "BOUNDARY_GRAPH", 48, 140, {"anchor": 1.0, "trust": 0.8, "boundary": 0.5, "balance": 0.08}),
        ("S40_COMBINED_MINIMAL", "COMBINED", 64, 160, {"anchor": 1.0, "trust": 0.8, "cross_view": 0.35, "boundary": 0.35}),
        ("S41_COMBINED_BALANCED", "COMBINED", 64, 180, {"anchor": 1.0, "trust": 0.8, "cross_view": 0.5, "boundary": 0.5, "balance": 0.05}),
        ("S42_COMBINED_COMPACT", "COMBINED", 32, 160, {"anchor": 1.0, "trust": 0.9, "cross_view": 0.4, "boundary": 0.4, "balance": 0.05}),
    ]
    result = []
    for identifier, mechanism, latent, steps, loss in rows:
        config = {
            "candidate_id": identifier,
            "mechanism_family": mechanism,
            "latent_dim": latent,
            "hidden_dim": 128,
            "steps": steps,
            "learning_rate": 1e-3,
            "weight_decay": 1e-4,
            "dropout": 0.05,
            "loss": {"anchor": 0.0, "trust": 0.0, "cross_view": 0.0, "boundary": 0.0, "balance": 0.0, **loss},
            "ground_truth_in_input_or_loss": False,
            "dataset_name_routing": False,
        }
        config["config_sha256"] = canonical_sha(config)
        result.append(config)
    return result


def standardize(value: np.ndarray) -> np.ndarray:
    return StandardScaler(copy=True).fit_transform(np.asarray(value, dtype=np.float32)).astype(np.float32)


def lane_dataset(lane: str) -> str:
    if lane.startswith("P22"):
        return "P22"
    if lane.startswith("MISAR_E15_5_S1"):
        return "MISAR_E15_5_S1"
    return lane


def load_lane(kit: Path, head: Path, lane: str) -> dict:
    dataset = lane_dataset(lane)
    archive = np.load(kit / f"{dataset}.npz", allow_pickle=False)
    bank = np.load(head / f"{dataset}_selected_partition_bank.npz", allow_pickle=False)
    retained = np.asarray(bank[f"{lane}__retained_embedding"], dtype=np.float32)
    retained_id = str(bank[f"{lane}__retained_embedding_id"][0])
    partitions = np.asarray(bank[f"{lane}__teacher_partitions"], dtype=np.int32)
    graph = row_stochastic(csr_from_archive(archive, "graph"))
    k = int(archive["k_primary"][0])
    labels = archive["labels_primary"].astype(str)
    if lane.endswith("_K12"):
        k = 12
    elif lane == "P22_3DOT_K18":
        k = 18
        labels = archive["labels_k18_author_assignment"].astype(str)
    anchors = build_stability_anchors(partitions, graph, k)
    return {
        "dataset": dataset,
        "lane": lane,
        "k": k,
        # The model receives a standardized adapter view, while the retained
        # teacher control must remain byte-for-byte the representation on which
        # the frozen endpoint was selected.  Re-standardizing that control
        # changes the comparison rather than disabling the new residual core.
        "retained": retained,
        "base": standardize(retained),
        "retained_id": retained_id,
        "view1": standardize(archive["view1"]),
        "view2": standardize(archive["view2"]),
        "labels": labels,
        "mask": archive["label_mask"].astype(bool),
        "graph": graph,
        "anchors": anchors,
        "ids": archive["ids"].astype(str),
    }


def endpoint_spec(ledger: pd.DataFrame, lane: str, retained_id: str) -> dict:
    rows = ledger[(ledger.lane == lane) & (ledger.status == "PASS") & (ledger.embedding_id == retained_id)].copy()
    if rows.empty:
        raise RuntimeError(f"no endpoint row for {lane}/{retained_id}")
    rows["rank_score"] = rows.absolute_ari + 0.35 * rows.absolute_nmi
    row = rows.sort_values("rank_score", ascending=False).iloc[0]
    return {"head": str(row["head"]), "endpoint_seed": int(row["endpoint_seed"]), "source_phase": str(row["phase"])}


def run_endpoint(embedding: np.ndarray, k: int, graph: sp.spmatrix, spec: Mapping[str, object]) -> np.ndarray:
    head = str(spec["head"])
    seed = int(spec["endpoint_seed"])
    if head.startswith("GMM_DIAG"):
        partition = GaussianMixture(k, covariance_type="diag", random_state=seed, n_init=1, max_iter=120, reg_covar=1e-5).fit_predict(embedding)
    elif head.startswith("GMM_TIED"):
        partition = GaussianMixture(k, covariance_type="tied", random_state=seed, n_init=1, max_iter=120, reg_covar=1e-5).fit_predict(embedding)
    else:
        partition = KMeans(k, random_state=seed, n_init=30).fit_predict(embedding)
    match = re.search(r"REFINE_T([0-9.]+)_S(\d+)", head)
    if match:
        partition = refine_partition(partition, graph, k, float(match.group(1)), int(match.group(2)))
    return np.asarray(partition, dtype=np.int32)


def model_forward_numpy(model: SAPRCore, tensors: Mapping[str, torch.Tensor], residual_enabled: bool) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        output = model(
            tensors["base"], tensors["view1"], tensors["view2"], tensors["graph"],
            tensors["confidence"], tensors["boundary"], residual_enabled=residual_enabled,
        )["embedding"]
    return output.detach().cpu().numpy().astype(np.float32)


def prepare_tensors(payload: Mapping[str, object], device: torch.device) -> dict:
    anchors = payload["anchors"]
    return {
        "base": torch.as_tensor(payload["base"], dtype=torch.float32, device=device),
        "view1": torch.as_tensor(payload["view1"], dtype=torch.float32, device=device),
        "view2": torch.as_tensor(payload["view2"], dtype=torch.float32, device=device),
        "graph": scipy_csr_to_torch(payload["graph"], device),
        "consensus": torch.as_tensor(anchors.consensus, dtype=torch.long, device=device),
        "confidence": torch.as_tensor(anchors.confidence, dtype=torch.float32, device=device),
        "boundary": torch.as_tensor(anchors.boundary, dtype=torch.float32, device=device),
        "interior": torch.as_tensor(anchors.interior_weight, dtype=torch.float32, device=device),
    }


def train_one(payload: Mapping[str, object], config: Mapping[str, object], seed: int, save: Path = None) -> Tuple[dict, dict]:
    started = time.perf_counter()
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensors = prepare_tensors(payload, device)
    model = SAPRCore(
        payload["base"].shape[1], payload["view1"].shape[1], payload["view2"].shape[1],
        int(config["latent_dim"]), int(payload["k"]), int(config["hidden_dim"]), float(config["dropout"]),
    ).to(device)
    model.initialize_prototypes(tensors["base"], tensors["consensus"], tensors["interior"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    losses = []
    for step in range(int(config["steps"])):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(
            tensors["base"], tensors["view1"], tensors["view2"], tensors["graph"],
            tensors["confidence"], tensors["boundary"], residual_enabled=True,
        )
        observed = sapr_loss(output, tensors["consensus"], tensors["confidence"], tensors["boundary"], tensors["graph"], config["loss"])
        if not bool(torch.isfinite(observed["total"])):
            raise FloatingPointError("non-finite SAPR loss")
        observed["total"].backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not bool(torch.isfinite(gradient_norm)):
            raise FloatingPointError("non-finite SAPR gradient")
        optimizer.step()
        if step in (0, int(config["steps"]) - 1):
            losses.append({key: float(value.detach().cpu()) for key, value in observed.items()})
    full = model_forward_numpy(model, tensors, True)
    disabled = model_forward_numpy(model, tensors, False)
    state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
    if save is not None:
        save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": state,
            "config": dict(config),
            "seed": int(seed),
            "lane": payload["lane"],
            "base_dim": payload["base"].shape[1],
            "view1_dim": payload["view1"].shape[1],
            "view2_dim": payload["view2"].shape[1],
            "k": payload["k"],
            "expected_full_sha256": sha256_array(full),
            "expected_disabled_sha256": sha256_array(disabled),
        }, save)
    audit = {
        "device": str(device),
        "optimizer_steps": int(config["steps"]),
        "effective_parameter_count": int(sum(item.numel() for item in model.parameters() if item.requires_grad)),
        "initial_and_final_loss": losses,
        "full_embedding_sha256": sha256_array(full),
        "disabled_embedding_sha256": sha256_array(disabled),
        "mean_gate": float((0.5 * (1.0 - payload["anchors"].confidence) + 0.5 * payload["anchors"].boundary).mean()),
        "mean_pairwise_teacher_ari": payload["anchors"].mean_pairwise_ari,
        "teacher_consensus_cardinality_repair_count": payload["anchors"].cardinality_repair_count,
        "input_shapes": {
            "retained_embedding": list(payload["base"].shape),
            "view1": list(payload["view1"].shape),
            "view2": list(payload["view2"].shape),
            "sparse_graph": list(payload["graph"].shape),
            "sparse_graph_nnz": int(payload["graph"].nnz),
        },
        "ordered_id_sha256": hashlib.sha256("\n".join(map(str, payload["ids"])).encode()).hexdigest(),
        "wall_seconds": time.perf_counter() - started,
        "peak_gpu_mib": float(torch.cuda.max_memory_allocated() / 1048576.0) if device.type == "cuda" else 0.0,
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
    }
    return {"full": full, "disabled": disabled}, audit


def evaluate_modes(payload: Mapping[str, object], embeddings: Mapping[str, np.ndarray], spec: Mapping[str, object]) -> List[dict]:
    rows = []
    for mode, embedding in embeddings.items():
        partition = run_endpoint(embedding, int(payload["k"]), payload["graph"], spec)
        rows.append({"mode": mode, "partition_sha256": sha256_array(partition), **metrics(payload["labels"], payload["mask"], partition, payload["graph"])})
    return rows


def run_candidate(payload: Mapping[str, object], config: Mapping[str, object], seed: int, spec: Mapping[str, object], save: Path = None) -> Tuple[List[dict], dict]:
    embeddings, audit = train_one(payload, config, seed, save)
    embeddings = {"RETAINED_TEACHER": payload["retained"], "SAPR_FULL": embeddings["full"], "SAPR_RESIDUAL_DISABLED": embeddings["disabled"]}
    rows = evaluate_modes(payload, embeddings, spec)
    reference = next(item for item in rows if item["mode"] == "RETAINED_TEACHER")
    for row in rows:
        row.update({
            "dataset": payload["dataset"], "lane": payload["lane"],
            "candidate_id": config["candidate_id"], "mechanism_family": config["mechanism_family"],
            "config_sha256": config["config_sha256"], "training_seed": int(seed),
            "endpoint_seed": spec["endpoint_seed"], "head": spec["head"], "k": payload["k"],
            "total_observations": len(payload["ids"]), "evaluated_observations": int(payload["mask"].sum()),
            "delta_ari_vs_retained": row["absolute_ari"] - reference["absolute_ari"],
            "delta_nmi_vs_retained": row["absolute_nmi"] - reference["absolute_nmi"],
            "status": "PASS", "wall_seconds": audit["wall_seconds"],
            "peak_gpu_mib": audit["peak_gpu_mib"], "peak_rss_mib": audit["peak_rss_mib"],
        })
    return rows, audit


def select_configs(rows: pd.DataFrame, top_n: int = 3) -> List[str]:
    full = rows[(rows["mode"] == "SAPR_FULL") & (rows["status"] == "PASS")].copy()
    full["family"] = np.where(full.dataset.isin(["P22", "MISAR_E15_5_S1"]), "RNA+ATAC", "RNA+protein")
    family = full.groupby(["candidate_id", "family"], as_index=False).agg(delta_ari=("delta_ari_vs_retained", "mean"), delta_nmi=("delta_nmi_vs_retained", "mean"))
    pivot = family.pivot(index="candidate_id", columns="family", values=["delta_ari", "delta_nmi"]).fillna(-1.0)
    scored = []
    for candidate in pivot.index:
        protein = float(pivot.loc[candidate, ("delta_ari", "RNA+protein")] + 0.35 * pivot.loc[candidate, ("delta_nmi", "RNA+protein")])
        atac = float(pivot.loc[candidate, ("delta_ari", "RNA+ATAC")] + 0.35 * pivot.loc[candidate, ("delta_nmi", "RNA+ATAC")])
        scored.append((min(protein, atac) + 0.25 * (protein + atac), candidate))
    return [candidate for _, candidate in sorted(scored, reverse=True)[:top_n]]


def reload_checkpoint(checkpoint: Path, kit: Path, head: Path) -> dict:
    saved = torch.load(checkpoint, map_location="cpu")
    payload = load_lane(kit, head, saved["lane"])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    tensors = prepare_tensors(payload, device)
    config = saved["config"]
    model = SAPRCore(saved["base_dim"], saved["view1_dim"], saved["view2_dim"], int(config["latent_dim"]), saved["k"], int(config["hidden_dim"]), float(config["dropout"])).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    full = model_forward_numpy(model, tensors, True)
    disabled = model_forward_numpy(model, tensors, False)
    result = {
        "checkpoint": str(checkpoint),
        "strict_load": True,
        "full_sha256": sha256_array(full),
        "disabled_sha256": sha256_array(disabled),
        "full_match": sha256_array(full) == saved["expected_full_sha256"],
        "disabled_match": sha256_array(disabled) == saved["expected_disabled_sha256"],
    }
    if not result["full_match"] or not result["disabled_match"]:
        raise RuntimeError("fresh-process checkpoint numerical replay mismatch")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path, default=DEFAULT_KIT)
    parser.add_argument("--head", type=Path, default=DEFAULT_HEAD)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase", choices=("p0", "screen", "finalists", "confirmation", "all", "reload"), default="all")
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    if args.phase == "reload":
        print(json.dumps(reload_checkpoint(args.checkpoint, args.kit, args.head), sort_keys=True))
        return
    args.output.mkdir(parents=True, exist_ok=True)
    registry = candidate_registry()
    (args.output / "candidate_registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger_path = args.output / "sapr_run_ledger.csv"
    rows: List[dict] = []
    if ledger_path.is_file():
        try:
            rows = pd.read_csv(ledger_path).to_dict("records")
        except pd.errors.EmptyDataError:
            rows = []
    audit_path = args.output / "resource_and_training_audit.json"
    audits: List[dict] = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.is_file() else []
    head_ledger = pd.read_csv(args.head / "all_head_run_ledger.csv")
    discovery = ["A1", "tonsil_s1", "P22", "MISAR_E15_5_S1", "MISAR_E15_5_S1_K12"]

    if args.phase in {"p0", "all"}:
        p0_config = next(item for item in registry if item["candidate_id"] == "S40_COMBINED_MINIMAL")
        p0_rows = []
        replay = []
        for lane in ("A1", "P22"):
            payload = load_lane(args.kit, args.head, lane)
            spec = endpoint_spec(head_ledger, lane, payload["retained_id"])
            checkpoint = args.output / "checkpoints" / f"P0_{lane}.pt"
            observed, audit = run_candidate(payload, p0_config, 20260824, spec, checkpoint)
            for row in observed:
                row["experiment_phase"] = "P0"
            p0_rows.extend(observed)
            audits.append({"phase": "P0", "lane": lane, **audit})
            command = [sys.executable, str(Path(__file__).resolve()), "--phase", "reload", "--kit", str(args.kit), "--head", str(args.head), "--checkpoint", str(checkpoint)]
            replay.append(json.loads(subprocess.check_output(command, text=True)))
        p0 = {
            "status": "PASS" if all(item["full_match"] and item["disabled_match"] for item in replay) else "FAILED",
            "family_count": 2, "fresh_process_replays": replay, "rows": p0_rows,
            "labels_in_model_input_or_loss": 0, "finite_gradient_and_loss": True,
        }
        (args.output / "real_p0_and_roundtrip.json").write_text(json.dumps(p0, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        if p0["status"] != "PASS":
            raise RuntimeError("SAPR real P0 failed")

    if args.phase in {"screen", "all"}:
        for config in registry:
            for lane in discovery:
                payload = load_lane(args.kit, args.head, lane)
                spec = endpoint_spec(head_ledger, lane, payload["retained_id"])
                try:
                    observed, audit = run_candidate(payload, config, 0, spec)
                    for row in observed:
                        row["experiment_phase"] = "SCREEN"
                    rows.extend(observed)
                    audits.append({"phase": "SCREEN", "lane": lane, "candidate_id": config["candidate_id"], "seed": 0, **audit})
                except Exception as error:
                    rows.append({"dataset": payload["dataset"], "lane": lane, "candidate_id": config["candidate_id"], "training_seed": 0, "mode": "SAPR_FULL", "experiment_phase": "SCREEN", "status": "FAILED", "failure": repr(error)})
                pd.DataFrame(rows).to_csv(ledger_path, index=False)
        selected = select_configs(pd.DataFrame(rows), 3)
        (args.output / "screen_selection.json").write_text(json.dumps({"selected": selected}, indent=2) + "\n", encoding="utf-8")

    if args.phase in {"finalists", "all"}:
        selection = json.loads((args.output / "screen_selection.json").read_text(encoding="utf-8"))["selected"]
        configs = {item["candidate_id"]: item for item in registry}
        replay = []
        for candidate_id in selection:
            for seed in (1, 2):
                for lane in discovery:
                    payload = load_lane(args.kit, args.head, lane)
                    spec = endpoint_spec(head_ledger, lane, payload["retained_id"])
                    checkpoint = args.output / "checkpoints" / f"FINAL_{candidate_id}_{lane}_seed{seed}.pt"
                    observed, audit = run_candidate(payload, configs[candidate_id], seed, spec, checkpoint)
                    for row in observed:
                        row["experiment_phase"] = "FINALIST"
                    rows.extend(observed)
                    audits.append({"phase": "FINALIST", "lane": lane, "candidate_id": candidate_id, "seed": seed, **audit})
                    command = [sys.executable, str(Path(__file__).resolve()), "--phase", "reload", "--kit", str(args.kit), "--head", str(args.head), "--checkpoint", str(checkpoint)]
                    replay.append(json.loads(subprocess.check_output(command, text=True)))
                    pd.DataFrame(rows).to_csv(ledger_path, index=False)
        (args.output / "finalist_roundtrip_audit.json").write_text(json.dumps(replay, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    if args.phase in {"confirmation", "all"}:
        frame = pd.DataFrame(rows)
        finalists = json.loads((args.output / "screen_selection.json").read_text(encoding="utf-8"))["selected"]
        finalist_rows = frame[(frame.candidate_id.isin(finalists)) & (frame["mode"] == "SAPR_FULL") & (frame.status == "PASS")]
        ranking = finalist_rows.groupby("candidate_id").agg(delta_ari=("delta_ari_vs_retained", "mean"), delta_nmi=("delta_nmi_vs_retained", "mean"))
        ranking["score"] = ranking.delta_ari + 0.35 * ranking.delta_nmi
        winner = str(ranking.score.idxmax())
        config = next(item for item in registry if item["candidate_id"] == winner)
        replay = []
        for lane in ("D1", "tonsil_s2", "tonsil_s3"):
            payload = load_lane(args.kit, args.head, lane)
            spec = endpoint_spec(head_ledger, lane, payload["retained_id"])
            for seed in (0, 1, 2):
                checkpoint = args.output / "checkpoints" / f"CONFIRM_{winner}_{lane}_seed{seed}.pt"
                observed, audit = run_candidate(payload, config, seed, spec, checkpoint)
                for row in observed:
                    row["experiment_phase"] = "CONFIRMATION"
                rows.extend(observed)
                audits.append({"phase": "CONFIRMATION", "lane": lane, "candidate_id": winner, "seed": seed, **audit})
                command = [sys.executable, str(Path(__file__).resolve()), "--phase", "reload", "--kit", str(args.kit), "--head", str(args.head), "--checkpoint", str(checkpoint)]
                replay.append(json.loads(subprocess.check_output(command, text=True)))
                pd.DataFrame(rows).to_csv(ledger_path, index=False)
        (args.output / "confirmation_roundtrip_audit.json").write_text(json.dumps(replay, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (args.output / "frozen_finalist.json").write_text(json.dumps({"candidate_id": winner, "config": config}, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if rows:
        pd.DataFrame(rows).to_csv(ledger_path, index=False)
    audit_path.write_text(json.dumps(audits, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(rows), "phase": args.phase}))


if __name__ == "__main__":
    main()
