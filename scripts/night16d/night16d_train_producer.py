#!/usr/bin/env python3
"""Label-free producer for Night-16D CMBF-RL training and replay."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import resource
import time

import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans
import torch

from SpaLORA.night16d_cmbf_rl import (
    CMBFRLConfig,
    CMBFResidualLearner,
    array_sha256,
    build_tri_state_field,
    config_sha256,
    encode_partition,
    graph_from_csr_arrays,
    operation_loss,
    retained_teacher_representation,
    robust_standardize,
    state_dict_sha256,
    teacher_representation,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_numeric_input(kit_root: Path, start_root: Path, lane: str, retained_root: Path | None = None) -> dict[str, object]:
    with np.load(kit_root / f"{lane}.npz", allow_pickle=False) as z:
        # The producer intentionally never requests labels or masks.
        value = {
            "ids": z["ids"].copy(),
            "view1": z["view1"].astype(np.float32),
            "view2": z["view2"].astype(np.float32),
            "k": int(z["k_primary"][0]),
            "start_bank": z["teacher_partitions"].astype(np.int32),
            "graph": graph_from_csr_arrays(
                z["graph__data"], z["graph__indices"], z["graph__indptr"], z["graph__shape"]
            ),
        }
    initial = encode_partition(np.load(start_root / f"{lane}.npy", allow_pickle=False))
    if not (len(initial) == len(value["ids"]) == len(value["view1"]) == len(value["view2"])):
        raise ValueError(f"{lane}: numeric input observation mismatch")
    if len(np.unique(initial)) != int(value["k"]):
        raise ValueError(f"{lane}: teacher partition K mismatch")
    value["initial"] = initial
    if retained_root is not None:
        bank_path = retained_root / f"{lane}_selected_partition_bank.npz"
        with np.load(bank_path, allow_pickle=False) as bank:
            key = f"{lane}__retained_embedding"
            identifier_key = f"{lane}__retained_embedding_id"
            retained = bank[key].astype(np.float32)
            identifier = str(bank[identifier_key][0])
        if len(retained) != len(initial):
            raise ValueError(f"{lane}: retained embedding observation mismatch")
        value["retained"] = retained
        value["retained_id"] = identifier
        value["retained_file_sha256"] = file_sha256(bank_path)
    return value


def _node_statistics(field) -> np.ndarray:
    return np.stack(
        [
            field.node_support,
            field.node_boundary,
            field.node_conflict,
            field.rejected_mass,
            field.start_stability,
            field.prototype_confidence,
        ],
        axis=1,
    ).astype(np.float32)


def _tensor_field(field, device: torch.device) -> dict[str, torch.Tensor]:
    cast = lambda x, dtype=torch.float32: torch.as_tensor(x, dtype=dtype, device=device)
    return {
        "row": cast(field.row, torch.long),
        "col": cast(field.col, torch.long),
        "base": cast(field.base_weight),
        "support": cast(field.support),
        "boundary": cast(field.boundary),
        "conflict": cast(field.conflict),
        "rank1": cast(field.rank1),
        "rank2": cast(field.rank2),
        "trust": cast(field.trust),
        "node_statistics": cast(_node_statistics(field)),
    }


def _generic_residual(teacher: np.ndarray, graph: sp.csr_matrix, strength: float) -> np.ndarray:
    row_sum = np.asarray(graph.sum(axis=1)).reshape(-1)
    inverse = np.divide(1.0, row_sum, out=np.zeros_like(row_sum), where=row_sum > 0)
    mean = sp.diags(inverse).dot(graph).dot(teacher)
    return (teacher + float(strength) * (mean - teacher)).astype(np.float32)


def prototype_endpoint(
    embedding: np.ndarray,
    initial: np.ndarray,
    trust: np.ndarray,
    iterations: int,
    core_quantile: float,
    trust_threshold: float,
    move_margin: float,
    seed: int,
) -> np.ndarray:
    embedding = np.asarray(embedding, dtype=np.float64)
    initial = encode_partition(initial)
    k = int(initial.max()) + 1
    cutoff = float(np.quantile(trust, float(core_quantile)))
    centers = []
    for group in range(k):
        mask = (initial == group) & (trust >= cutoff)
        if int(mask.sum()) < 3:
            mask = initial == group
        centers.append(np.median(embedding[mask], axis=0))
    centers = np.stack(centers)
    del seed  # kept in the public API to separate training and endpoint seeds
    partition = initial.copy()
    for _ in range(max(1, int(iterations))):
        distance = np.mean((embedding[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        proposal = np.argmin(distance, axis=1).astype(np.int32)
        current = distance[np.arange(len(initial)), initial]
        proposed = distance[np.arange(len(initial)), proposal]
        scale = max(float(np.median(distance)), 1e-12)
        advantage = (current - proposed) / scale
        movable = (trust < float(trust_threshold)) & (advantage > float(move_margin))
        partition = np.where(movable, proposal, initial).astype(np.int32)
        for group in range(k):
            mask = (partition == group) & (trust >= cutoff)
            if int(mask.sum()) < 3:
                mask = partition == group
            if np.any(mask):
                centers[group] = np.median(embedding[mask], axis=0)
    return encode_partition(partition)


def _forward_cpu(
    model: CMBFResidualLearner,
    x1: np.ndarray,
    x2: np.ndarray,
    teacher: np.ndarray,
    field,
) -> tuple[np.ndarray, dict[str, object]]:
    model = model.cpu().eval()
    tensor = _tensor_field(field, torch.device("cpu"))
    with torch.no_grad():
        output = model(
            torch.from_numpy(x1), torch.from_numpy(x2), torch.from_numpy(teacher),
            tensor["node_statistics"], tensor["trust"], tensor["row"], tensor["col"],
            tensor["base"], tensor["support"], tensor["boundary"], tensor["conflict"],
        )
    return output["z"].numpy().astype(np.float32), {
        "gate_mean": [float(x) for x in output["gates"].mean(0).numpy()],
        "alpha_mean": float(output["alpha"].mean()),
        "alpha_max": float(output["alpha"].max()),
    }


def train_one(
    data: dict[str, object],
    config: CMBFRLConfig,
    training_seed: int,
    endpoint_seed: int,
    checkpoint_path: Path | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    config.validate()
    started = time.perf_counter()
    np.random.seed(int(training_seed))
    torch.manual_seed(int(training_seed))
    torch.cuda.manual_seed_all(int(training_seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    initial = data["initial"]
    x1 = robust_standardize(data["view1"])
    x2 = robust_standardize(data["view2"])
    if config.teacher_source == "retained":
        if "retained" not in data:
            raise ValueError("retained teacher source requested without a registered retained embedding")
        teacher = retained_teacher_representation(
            data["retained"], initial, config.latent_dim, config.teacher_scale, config.content_scale
        )
    else:
        teacher = teacher_representation(
            x1, x2, initial, config.latent_dim, config.teacher_scale, config.content_scale
        )
    field = build_tri_state_field(
        x1,
        x2,
        data["graph"],
        initial,
        data["start_bank"],
        config.rank_mode,
        shuffle_seed=(int(training_seed) + 991) if config.shuffle_edge_states else None,
    )
    mode = config.operation_mode
    training_rows: list[dict[str, float]] = []
    parameter_before = "NOT_APPLICABLE"
    parameter_after = "NOT_APPLICABLE"
    parameter_delta = 0.0
    optimizer_steps = 0
    gradient_finite = True
    peak_gpu = 0.0
    model: CMBFResidualLearner | None = None
    forward_detail: dict[str, object] = {"gate_mean": [1.0, 0.0, 0.0, 0.0], "alpha_mean": 0.0, "alpha_max": 0.0}

    if mode in {"teacher", "teacher_head"}:
        z = teacher.copy()
    elif mode == "generic":
        z = _generic_residual(teacher, field.graph, config.residual_scale)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model = CMBFResidualLearner(x1.shape[1], x2.shape[1], config).to(device)
        parameter_before = state_dict_sha256(model.state_dict())
        before_vector = torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()])
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        tensor = _tensor_field(field, device)
        tx1, tx2 = torch.from_numpy(x1).to(device), torch.from_numpy(x2).to(device)
        tteacher = torch.from_numpy(teacher).to(device)
        for step in range(int(config.training_steps)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            output = model(
                tx1, tx2, tteacher, tensor["node_statistics"], tensor["trust"],
                tensor["row"], tensor["col"], tensor["base"], tensor["support"],
                tensor["boundary"], tensor["conflict"],
            )
            loss, detail = operation_loss(
                output, tx1, tx2, tteacher, tensor["trust"], tensor["row"], tensor["col"],
                tensor["base"], tensor["support"], tensor["boundary"], tensor["conflict"],
                tensor["rank1"], tensor["rank2"], config,
            )
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite training loss")
            loss.backward()
            gradient_finite = bool(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            if not gradient_finite:
                raise RuntimeError("non-finite gradient")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            optimizer_steps += 1
            if step in {0, int(config.training_steps) - 1}:
                training_rows.append({"step": float(step + 1), **detail})
        if device.type == "cuda":
            peak_gpu = float(torch.cuda.max_memory_allocated() / 1024**2)
        parameter_after = state_dict_sha256(model.state_dict())
        after_vector = torch.cat([p.detach().cpu().reshape(-1) for p in model.parameters()])
        parameter_delta = float(torch.linalg.vector_norm(after_vector - before_vector))
        z, forward_detail = _forward_cpu(model, x1, x2, teacher, field)

    same_head_teacher = prototype_endpoint(
        teacher, initial, field.trust, config.endpoint_iterations, config.endpoint_core_quantile,
        config.endpoint_trust_threshold, config.endpoint_move_margin, endpoint_seed
    )
    partition = prototype_endpoint(
        z, initial, field.trust, config.endpoint_iterations, config.endpoint_core_quantile,
        config.endpoint_trust_threshold, config.endpoint_move_margin, endpoint_seed
    )
    # INPUT_STRONG_START is the byte-exact no-op control.  TEACHER_HEAD keeps
    # the same representation endpoint used by learned candidates and is
    # audited separately so endpoint drift cannot masquerade as method gain.
    if mode == "teacher":
        partition = initial.copy()
    k = int(data["k"])
    sizes = np.bincount(partition, minlength=k)
    threshold = max(5, int(np.ceil(0.01 * len(partition) / k)))
    status = "PASS"
    failure = ""
    if len(np.unique(partition)) != k or int(sizes.min()) < threshold:
        status = "INVALID_EXACT_K_OR_MICROCLUSTER"
        failure = f"cluster_sizes={sizes.tolist()}, threshold={threshold}"
    detail = {
        "status": status,
        "failure": failure,
        "config": asdict(config),
        "config_sha256": config_sha256(config),
        "training_seed": int(training_seed),
        "endpoint_seed": int(endpoint_seed),
        "optimizer_steps": int(optimizer_steps),
        "parameter_sha256_before": parameter_before,
        "parameter_sha256_after": parameter_after,
        "parameter_delta_l2": parameter_delta,
        "gradient_finite": gradient_finite,
        "embedding_sha256": array_sha256(z),
        "teacher_sha256": array_sha256(teacher),
        "teacher_source": config.teacher_source,
        "retained_embedding_sha256": array_sha256(data["retained"]) if "retained" in data else "NOT_APPLICABLE",
        "retained_embedding_id": data.get("retained_id", "NOT_APPLICABLE"),
        "embedding_delta_l2": float(np.linalg.norm(z.astype(np.float64) - teacher.astype(np.float64))),
        "partition_sha256": array_sha256(partition),
        "teacher_partition_sha256": array_sha256(initial),
        "same_head_teacher_partition_sha256": array_sha256(same_head_teacher),
        "same_head_teacher_changed_from_input": int(np.sum(same_head_teacher != initial)),
        "changed_from_teacher": int(np.sum(partition != initial)),
        "cluster_sizes": [int(x) for x in sizes],
        "min_cluster_size": int(sizes.min()),
        "min_cluster_threshold": int(threshold),
        "training_loss_rows": training_rows,
        "field": {
            "rank_mode": config.rank_mode,
            "support_mean": float(np.mean(field.support)),
            "boundary_mean": float(np.mean(field.boundary)),
            "conflict_mean": float(np.mean(field.conflict)),
            "rejected_mass_mean": float(np.mean(field.rejected_mass)),
            "trust_mean": float(np.mean(field.trust)),
            "edge_count": int(len(field.row)),
        },
        "forward": forward_detail,
        "wall_seconds": float(time.perf_counter() - started),
        "peak_gpu_mib": peak_gpu,
        "peak_rss_mib": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0),
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
    }
    if checkpoint_path is not None and model is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": asdict(config),
                "input_dim1": int(x1.shape[1]),
                "input_dim2": int(x2.shape[1]),
                "training_seed": int(training_seed),
                "endpoint_seed": int(endpoint_seed),
                "expected_embedding_sha256": detail["embedding_sha256"],
                "expected_partition_sha256": detail["partition_sha256"],
                "expected_ids_sha256": array_sha256(data["ids"]),
                "expected_retained_sha256": array_sha256(data["retained"]) if "retained" in data else "NOT_APPLICABLE",
            },
            checkpoint_path,
        )
        detail["checkpoint_sha256"] = file_sha256(checkpoint_path)
        detail["checkpoint_path"] = str(checkpoint_path)
    return z, partition, detail


def candidate_id(config: CMBFRLConfig, training_seed: int, endpoint_seed: int) -> str:
    return f"RL_{config_sha256(config)[:12]}_t{int(training_seed)}_e{int(endpoint_seed)}"


def run_screen(args: argparse.Namespace) -> None:
    registry = json.loads(Path(args.config_registry).read_text(encoding="utf-8"))
    configs = [CMBFRLConfig(**item) for item in registry["configs"]]
    seeds = [int(x) for x in registry.get("training_seeds", [0])]
    endpoint_seeds = [int(x) for x in registry.get("endpoint_seeds", [0])]
    lanes = [x for x in args.lanes.split(",") if x]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for lane in lanes:
        data = load_numeric_input(Path(args.kit_root), Path(args.start_root), lane, Path(args.retained_root) if args.retained_root else None)
        partitions, identifiers = [], []
        for config in configs:
            for seed in seeds:
                for endpoint_seed in endpoint_seeds:
                    identifier = candidate_id(config, seed, endpoint_seed)
                    try:
                        _, partition, detail = train_one(data, config, seed, endpoint_seed, None)
                    except Exception as exc:
                        partition = data["initial"].copy()
                        detail = {
                            "status": "FAILED",
                            "failure": f"{type(exc).__name__}: {exc}",
                            "config": asdict(config),
                            "config_sha256": config_sha256(config),
                            "training_seed": seed,
                            "endpoint_seed": endpoint_seed,
                            "producer_label_reads": 0,
                            "dense_n_by_n_count": 0,
                        }
                    partitions.append(partition)
                    identifiers.append(identifier)
                    rows.append(
                        {
                            "lane": lane,
                            "candidate_id": identifier,
                            "config_json": json.dumps(asdict(config), sort_keys=True, separators=(",", ":")),
                            **{k: v if not isinstance(v, (dict, list)) else json.dumps(v, separators=(",", ":")) for k, v in detail.items()},
                        }
                    )
        np.savez_compressed(
            out / f"{lane}_locked_partitions.npz",
            candidate_ids=np.asarray(identifiers),
            partitions=np.stack(partitions).astype(np.int32),
            ordered_ids=data["ids"],
        )
    write_csv(out / "producer_ledger.csv", rows)
    write_json(
        out / "producer_manifest.json",
        {
            "lanes": lanes,
            "config_registry_sha256": file_sha256(Path(args.config_registry)),
            "candidate_partitions_locked_before_evaluation": True,
            "rows": len(rows),
            "producer_label_reads": 0,
            "dense_n_by_n_count": 0,
        },
    )


def run_p0(args: argparse.Namespace) -> None:
    config = CMBFRLConfig(**json.loads(Path(args.config).read_text(encoding="utf-8")))
    data = load_numeric_input(Path(args.kit_root), Path(args.start_root), args.lane, Path(args.retained_root) if args.retained_root else None)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = out / "checkpoint.pt"
    z, partition, detail = train_one(data, config, args.training_seed, args.endpoint_seed, checkpoint)
    np.save(out / "embedding.npy", z, allow_pickle=False)
    np.save(out / "partition.npy", partition, allow_pickle=False)
    detail.update(
        {
            "lane": args.lane,
            "input_shapes": {
                "view1": [int(x) for x in data["view1"].shape],
                "view2": [int(x) for x in data["view2"].shape],
                "teacher": [int(len(partition)), int(config.latent_dim)],
                "graph": [int(x) for x in data["graph"].shape],
            },
            "view1_dtype": str(data["view1"].dtype),
            "view2_dtype": str(data["view2"].dtype),
            "graph_nnz": int(data["graph"].nnz),
            "ordered_ids_sha256": array_sha256(data["ids"]),
            "checkpoint_strict_reload": False,
        }
    )
    write_json(out / "producer_record.json", detail)
    if detail["status"] != "PASS":
        raise RuntimeError(detail["failure"])


def replay(args: argparse.Namespace) -> None:
    source = Path(args.input)
    record = json.loads((source / "producer_record.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(source / "checkpoint.pt", map_location="cpu")
    config = CMBFRLConfig(**checkpoint["config"])
    data = load_numeric_input(Path(args.kit_root), Path(args.start_root), args.lane, Path(args.retained_root) if args.retained_root else None)
    x1, x2 = robust_standardize(data["view1"]), robust_standardize(data["view2"])
    if config.teacher_source == "retained":
        teacher = retained_teacher_representation(data["retained"], data["initial"], config.latent_dim, config.teacher_scale, config.content_scale)
    else:
        teacher = teacher_representation(x1, x2, data["initial"], config.latent_dim, config.teacher_scale, config.content_scale)
    field = build_tri_state_field(
        x1, x2, data["graph"], data["initial"], data["start_bank"], config.rank_mode,
        shuffle_seed=(int(checkpoint["training_seed"]) + 991) if config.shuffle_edge_states else None,
    )
    model = CMBFResidualLearner(checkpoint["input_dim1"], checkpoint["input_dim2"], config)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    z, _ = _forward_cpu(model, x1, x2, teacher, field)
    partition = prototype_endpoint(
        z, data["initial"], field.trust, config.endpoint_iterations, config.endpoint_core_quantile,
        config.endpoint_trust_threshold, config.endpoint_move_margin,
        int(checkpoint["endpoint_seed"]),
    )
    result = {
        "lane": args.lane,
        "checkpoint_strict_reload": True,
        "embedding_sha256": array_sha256(z),
        "partition_sha256": array_sha256(partition),
        "embedding_exact": array_sha256(z) == checkpoint["expected_embedding_sha256"],
        "partition_exact": array_sha256(partition) == checkpoint["expected_partition_sha256"],
        "ordered_ids_exact": array_sha256(data["ids"]) == checkpoint["expected_ids_sha256"],
        "retained_embedding_exact": (
            array_sha256(data["retained"]) == checkpoint["expected_retained_sha256"]
            if checkpoint["expected_retained_sha256"] != "NOT_APPLICABLE"
            else True
        ),
        "producer_label_reads": 0,
        "dense_n_by_n_count": 0,
    }
    result["status"] = "PASS" if all(result[x] for x in ("embedding_exact", "partition_exact", "ordered_ids_exact", "retained_embedding_exact")) else "FAIL"
    write_json(source / "fresh_process_replay.json", result)
    if result["status"] != "PASS":
        raise RuntimeError("fresh-process replay mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("screen")
    p.add_argument("--kit-root", required=True); p.add_argument("--start-root", required=True)
    p.add_argument("--config-registry", required=True); p.add_argument("--lanes", required=True); p.add_argument("--output", required=True); p.add_argument("--retained-root")
    p.set_defaults(func=run_screen)
    p = sub.add_parser("p0")
    p.add_argument("--kit-root", required=True); p.add_argument("--start-root", required=True); p.add_argument("--lane", required=True)
    p.add_argument("--config", required=True); p.add_argument("--training-seed", type=int, default=0); p.add_argument("--endpoint-seed", type=int, default=0); p.add_argument("--output", required=True); p.add_argument("--retained-root")
    p.set_defaults(func=run_p0)
    p = sub.add_parser("replay")
    p.add_argument("--kit-root", required=True); p.add_argument("--start-root", required=True); p.add_argument("--lane", required=True); p.add_argument("--input", required=True); p.add_argument("--retained-root")
    p.set_defaults(func=replay)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
