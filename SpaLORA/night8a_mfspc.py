"""Night-8A MF-SPC primitives.

This file is an independent implementation guided by the preregistered
Night-8A contracts.  It contains no third-party source-code fragments and no
dataset-name, tissue-name, label, or evaluation-metric routing.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy.sparse as sp
import torch
from sklearn.neighbors import NearestNeighbors
from torch import nn
from torch.nn import functional as F


RNA_PROTEIN = "RNA_PROTEIN"
RNA_EPIGENOME = "RNA_EPIGENOME"
ALLOWED_MODULES = {
    "SP", "RR10", "RR30", "PROTO", "RNA_ANCHOR", "DGI", "SMART_TRIPLET"
}


def canonical_json_sha(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def array_sha(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(value)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode())
    h.update(json.dumps(list(arr.shape), separators=(",", ":")).encode())
    h.update(arr.tobytes(order="C"))
    return h.hexdigest()


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def normalized_assays(metadata: Mapping[str, object]) -> tuple[str, ...]:
    if "assays" not in metadata:
        raise ValueError("assay metadata missing; identity-based inference is forbidden")
    raw = metadata["assays"]
    if not isinstance(raw, (list, tuple)) or len(raw) != 2:
        raise ValueError("assays must contain exactly two modalities")
    result = []
    for value in raw:
        token = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "gene_expression": "rna", "transcriptome": "rna", "transcriptomics": "rna",
            "antibody_derived_tag": "adt", "antibody_derived_tags": "adt",
            "protein_abundance": "protein", "proteomics": "protein",
            "chromatin_accessibility": "atac", "accessibility": "atac",
            "histone_modification": "histone", "epigenomics": "epigenome",
        }
        result.append(aliases.get(token, token))
    return tuple(sorted(result))


def select_family(metadata: Mapping[str, object], forbidden_sentinel: object | None = None) -> str:
    """Resolve the modality family from assay metadata only.

    ``forbidden_sentinel`` exists only for a negative test.  It is deliberately
    never inspected, indexed, converted, or serialized.
    """
    assays = set(normalized_assays(metadata))
    if assays in ({"rna", "adt"}, {"rna", "protein"}):
        return RNA_PROTEIN
    if assays in ({"rna", "atac"}, {"rna", "histone"}, {"rna", "epigenome"}):
        return RNA_EPIGENOME
    raise ValueError(f"unknown assay combination: {sorted(assays)}")


def resolve_modules(registered: Sequence[str], family: str) -> tuple[str, ...]:
    modules = tuple(str(x) for x in registered)
    if len(set(modules)) != len(modules) or not set(modules) <= ALLOWED_MODULES:
        raise ValueError("invalid or duplicate module registry")
    if "RR10" in modules and "RR30" in modules:
        raise ValueError("RR10 and RR30 are mutually exclusive")
    if family not in {RNA_PROTEIN, RNA_EPIGENOME}:
        raise ValueError("invalid family")
    return tuple(x for x in modules if not (x == "RNA_ANCHOR" and family == RNA_PROTEIN))


def row_l2_np(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float32)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def row_l2(value: torch.Tensor) -> torch.Tensor:
    return value / torch.clamp(torch.linalg.vector_norm(value, dim=1, keepdim=True), min=1e-12)


def canonical_sparse(value: sp.spmatrix) -> sp.csr_matrix:
    out = value.tocsr().astype(np.float64)
    out.sum_duplicates(); out.eliminate_zeros(); out.sort_indices()
    return out


def sparse_row_normalize(value: sp.spmatrix, self_loops: bool = True) -> sp.csr_matrix:
    out = canonical_sparse(value)
    if self_loops:
        out = canonical_sparse(out + sp.eye(out.shape[0], format="csr"))
    degree = np.asarray(out.sum(axis=1)).ravel()
    return canonical_sparse(sp.diags(1.0 / np.maximum(degree, 1e-12)) @ out)


def rna_anchor_support(spatial: sp.spmatrix, rna: np.ndarray,
                       observation_ids: Sequence[str], k: int = 10) -> sp.csr_matrix:
    """Sparse intersection of spatial support and RNA feature-neighbour support."""
    n = len(rna)
    if spatial.shape != (n, n) or len(observation_ids) != n:
        raise ValueError("RNA-anchor input shape mismatch")
    model = NearestNeighbors(n_neighbors=min(k + 1, n), metric="euclidean", n_jobs=1)
    model.fit(row_l2_np(rna))
    dist, ind = model.kneighbors(return_distance=True)
    rows, cols = [], []
    names = np.asarray(observation_ids, dtype=str)
    for i in range(n):
        pairs = [(float(d), str(names[j]), int(j)) for d, j in zip(dist[i], ind[i]) if int(j) != i]
        for _, _, j in sorted(pairs)[:k]:
            rows.append(i); cols.append(j)
    feature = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    feature = canonical_sparse(feature.maximum(feature.T))
    spatial_bin = canonical_sparse((spatial != 0).astype(np.float64))
    shared = canonical_sparse(spatial_bin.multiply(feature))
    return sparse_row_normalize(shared, self_loops=True)


def centered_cross_covariance_loss(shared: torch.Tensor, private: torch.Tensor) -> torch.Tensor:
    if private.shape[1] == 0:
        return shared.new_zeros(())
    a = shared - shared.mean(0, keepdim=True)
    b = private - private.mean(0, keepdim=True)
    denom = max(1, shared.shape[0] - 1)
    cross = a.T @ b / denom
    return cross.square().mean()


def vicreg_loss(a: torch.Tensor, b: torch.Tensor) -> tuple[torch.Tensor, Mapping[str, torch.Tensor]]:
    invariance = F.mse_loss(a, b)
    std_a = torch.sqrt(a.var(dim=0, unbiased=False) + 1e-4)
    std_b = torch.sqrt(b.var(dim=0, unbiased=False) + 1e-4)
    variance = 0.5 * (F.relu(1.0 - std_a).mean() + F.relu(1.0 - std_b).mean())
    ac = a - a.mean(0, keepdim=True); bc = b - b.mean(0, keepdim=True)
    denom = max(1, a.shape[0] - 1)
    ca = ac.T @ ac / denom; cb = bc.T @ bc / denom
    mask = ~torch.eye(ca.shape[0], dtype=torch.bool, device=ca.device)
    covariance = 0.5 * (ca[mask].square().mean() + cb[mask].square().mean())
    total = 25.0 * invariance + 25.0 * variance + covariance
    return total, {"rr_invariance": invariance, "rr_variance": variance,
                   "rr_covariance": covariance}


def fixed_triplets(a: np.ndarray, b: np.ndarray, observation_ids: Sequence[str],
                   seed: int, k: int = 3, farthest_fraction: float = .60,
                   block: int = 256) -> tuple[np.ndarray, np.ndarray, dict]:
    """Deterministic cross-modal MNN positives and farthest-fraction negatives.

    Pairwise distances are materialized only in row blocks; no resident N x N
    matrix is created.
    """
    a, b = row_l2_np(a), row_l2_np(b)
    n = len(a); names = np.asarray(observation_ids, dtype=str)
    kk = min(int(k), n)
    ab = np.empty((n, kk), dtype=np.int64)
    ba = np.empty((n, kk), dtype=np.int64)
    for start in range(0, n, block):
        stop = min(n, start + block)
        dist = 2.0 - 2.0 * (a[start:stop] @ b.T)
        for off, row in enumerate(range(start, stop)):
            ab[row] = np.lexsort((names, dist[off]))[:kk]
    for start in range(0, n, block):
        stop = min(n, start + block)
        dist = 2.0 - 2.0 * (b[start:stop] @ a.T)
        for off, row in enumerate(range(start, stop)):
            ba[row] = np.lexsort((names, dist[off]))[:kk]
    positives = np.empty(n, dtype=np.int64)
    negatives = np.empty(n, dtype=np.int64)
    cutoff = max(1, int(math.floor(float(farthest_fraction) * n)))
    fallback = 0
    for i in range(n):
        mutual = [int(j) for j in ab[i] if i in set(ba[int(j)])]
        if mutual:
            positives[i] = min(mutual, key=lambda j: names[j])
        else:
            positives[i] = int(ab[i, 0]); fallback += 1
        similarity = b @ a[i]
        far = np.lexsort((names, similarity))[:cutoff]
        token = f"night8a|{int(seed)}|{i}|negative".encode()
        pick = int.from_bytes(hashlib.sha256(token).digest()[:8], "little") % len(far)
        negatives[i] = int(far[pick])
    return positives, negatives, {
        "k": kk, "farthest_fraction": float(farthest_fraction),
        "positive_sha256": array_sha(positives),
        "negative_sha256": array_sha(negatives),
        "fallback_nearest_nonmutual_count": fallback,
        "resident_dense_n_by_n": False,
    }


class SharedPrivateEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, use_private: bool):
        super().__init__()
        self.input_dim = int(input_dim); self.latent_dim = int(latent_dim)
        self.shared_dim = int(round(latent_dim * .75)) if use_private else latent_dim
        self.private_dim = latent_dim - self.shared_dim if use_private else 0
        self.shared = nn.Linear(input_dim, self.shared_dim)
        self.private = nn.Linear(input_dim, self.private_dim) if self.private_dim else None
        self.decoder = nn.Linear(self.shared_dim + self.private_dim, input_dim)

    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = row_l2(self.shared(value))
        private = torch.tanh(self.private(value)) if self.private is not None else value.new_zeros((len(value), 0))
        joined = torch.cat((shared, private), dim=1)
        return shared, private, self.decoder(joined)


class MFSPCModel(nn.Module):
    """Compact shared/private adapter over immutable family-reference views."""
    def __init__(self, dim: int, modules: Sequence[str], prototype_count: int,
                 fused_dim: int | None = None):
        super().__init__()
        self.modules = tuple(modules)
        fused_dim = int(dim if fused_dim is None else fused_dim)
        use_private = "SP" in self.modules
        self.encoder1 = SharedPrivateEncoder(dim, fused_dim, use_private)
        self.encoder2 = SharedPrivateEncoder(dim, fused_dim, use_private)
        shared_dim = self.encoder1.shared_dim
        self.to_fused = nn.Linear(shared_dim, fused_dim, bias=False)
        self.prototypes = nn.Parameter(torch.empty(prototype_count, shared_dim)) if "PROTO" in self.modules else None
        if self.prototypes is not None:
            nn.init.normal_(self.prototypes, std=.02)
            self.register_buffer("teacher_prototypes", self.prototypes.detach().clone())
            self.register_buffer("ema_occupancy", torch.full((prototype_count,), 1.0 / prototype_count))

    def forward(self, x1: torch.Tensor, x2: torch.Tensor,
                family_reference: torch.Tensor) -> Mapping[str, torch.Tensor]:
        s1, p1, r1 = self.encoder1(x1)
        s2, p2, r2 = self.encoder2(x2)
        candidate = row_l2(self.to_fused((s1 + s2) * .5))
        if not self.modules:
            fused = family_reference
        else:
            fused = row_l2(.75 * family_reference + .25 * candidate)
        return {"shared1": s1, "shared2": s2, "private1": p1, "private2": p2,
                "recon1": r1, "recon2": r2, "candidate": candidate, "fused": fused}

    @torch.no_grad()
    def update_teacher(self, student_occupancy: torch.Tensor, momentum: float = .99) -> None:
        if self.prototypes is None:
            return
        self.teacher_prototypes.mul_(momentum).add_(self.prototypes.detach(), alpha=1.0 - momentum)
        self.ema_occupancy.mul_(momentum).add_(student_occupancy.detach(), alpha=1.0 - momentum)


def prototype_loss(model: MFSPCModel, output: Mapping[str, torch.Tensor],
                   temperature: float = .10) -> tuple[torch.Tensor, Mapping[str, torch.Tensor], torch.Tensor]:
    if model.prototypes is None:
        raise RuntimeError("prototype module inactive")
    proto = row_l2(model.prototypes)
    teacher = row_l2(model.teacher_prototypes.detach())
    assignments = [torch.softmax(x @ proto.T / temperature, dim=1)
                   for x in (output["shared1"], output["shared2"])]
    fused_shared = row_l2((output["shared1"] + output["shared2"]) * .5)
    teacher_q = torch.softmax(fused_shared.detach() @ teacher.T / temperature, dim=1)
    consensus = torch.stack([-(teacher_q * torch.log(q + 1e-12)).sum(1).mean() for q in assignments]).mean()
    occupancy = torch.stack(assignments).mean((0, 1))
    target = torch.clamp(model.ema_occupancy.detach(), min=1e-6)
    occupancy_kl = torch.sum(occupancy * (torch.log(occupancy + 1e-12) - torch.log(target)))
    total = consensus + .01 * occupancy_kl
    return total, {"prototype_consensus": consensus, "prototype_occupancy_kl": occupancy_kl}, occupancy


def dgi_loss(local: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    summary = torch.sigmoid(local.mean(0))
    positive = torch.sum(local * summary, dim=1)
    negative = torch.sum(local[permutation] * summary, dim=1)
    return F.softplus(-positive).mean() + F.softplus(negative).mean()


@dataclass
class EMAScaler:
    beta: float = .99
    warmup_epochs: int = 10
    base_ema: float | None = None
    aux_ema: dict[str, float] | None = None

    def __post_init__(self) -> None:
        if self.aux_ema is None:
            self.aux_ema = {}

    def factor(self, name: str, base: torch.Tensor, auxiliary: torch.Tensor, epoch: int) -> float:
        b = float(base.detach().abs().cpu())
        a = float(auxiliary.detach().abs().cpu())
        self.base_ema = b if self.base_ema is None else self.beta * self.base_ema + (1 - self.beta) * b
        old = self.aux_ema.get(name)
        self.aux_ema[name] = a if old is None else self.beta * old + (1 - self.beta) * a
        if epoch < self.warmup_epochs:
            return 0.0
        return float(np.clip(self.base_ema / (self.aux_ema[name] + 1e-8), .1, 10.0))


def tensor_state_sha(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key in sorted(state):
        value = state[key].detach().cpu().contiguous()
        h.update(key.encode()); h.update(str(value.dtype).encode())
        h.update(json.dumps(list(value.shape), separators=(",", ":")).encode())
        h.update(value.numpy().tobytes(order="C"))
    return h.hexdigest()


def parameter_grad_audit(model: nn.Module) -> dict:
    result = {}
    for name, parameter in model.named_parameters():
        grad = parameter.grad
        result[name] = {
            "requires_grad": bool(parameter.requires_grad),
            "grad_present": grad is not None,
            "grad_finite": bool(grad is not None and torch.isfinite(grad).all()),
            "grad_norm": None if grad is None else float(torch.linalg.vector_norm(grad.detach()).cpu()),
        }
    return result
