"""Night-6A clean-room structural rescue modules; no semantic-label reader."""
from __future__ import annotations

import hashlib
import itertools
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from .night5a_rnd import Night5AModel, _support

LOSS_GROUPS = ("L_rna_recon_raw", "L_mod2_recon_raw", "L_corr1_raw")

def sparse_sha256(graph: torch.Tensor) -> str:
    graph = graph.coalesce().cpu()
    h = hashlib.sha256()
    h.update(graph.indices().numpy().astype("<i8", copy=False).tobytes())
    h.update(graph.values().numpy().astype("<f4", copy=False).tobytes())
    h.update(np.asarray(graph.shape, dtype="<i8").tobytes())
    return h.hexdigest()

def _normalize(raw: sp.csr_matrix) -> torch.Tensor:
    raw = raw.maximum(raw.T).tocsr()
    degree = np.asarray(raw.sum(axis=1)).ravel()
    inv = np.power(np.maximum(degree, 1e-12), -0.5)
    normalized = (sp.diags(inv) @ raw @ sp.diags(inv)).tocoo()
    graph = torch.sparse_coo_tensor(
        torch.as_tensor(np.vstack((normalized.row, normalized.col)), dtype=torch.long),
        torch.as_tensor(normalized.data, dtype=torch.float32), size=normalized.shape,
    ).coalesce()
    return graph

def pruned_graph(spatial: torch.Tensor, rna: torch.Tensor, mode: str,
                 epsilon: Optional[float] = None, coordinates: Optional[np.ndarray] = None,
                 barcodes: Optional[Sequence[str]] = None) -> Tuple[torch.Tensor, dict]:
    a_spatial, a_rna = _support(spatial), _support(rna)
    shared = a_spatial.multiply(a_rna).tocsr()
    zero_before, rescued = [], []
    if mode == "soft":
        if epsilon not in (0.10, 0.25, 0.50):
            raise ValueError("registered soft epsilon required")
        raw = float(epsilon) * a_spatial + (1.0 - float(epsilon)) * shared
    elif mode == "hard":
        if coordinates is None or barcodes is None:
            raise ValueError("hard prune requires coordinates/barcodes")
        raw = shared.copy().tolil()
        zero_before = np.flatnonzero(np.asarray(raw.sum(axis=1)).ravel() == 0).tolist()
        spatial_csr = a_spatial.tocsr(); coords = np.asarray(coordinates); names = np.asarray(barcodes).astype(str)
        for node in zero_before:
            neighbors = spatial_csr.indices[spatial_csr.indptr[node]:spatial_csr.indptr[node + 1]]
            if not len(neighbors):
                raise RuntimeError("original spatial isolate cannot be rescued")
            distances = np.linalg.norm(coords[neighbors] - coords[node], axis=1)
            minimum = distances.min(); tied = neighbors[np.isclose(distances, minimum, rtol=0, atol=1e-12)]
            chosen = int(tied[np.argmin(names[tied])])
            raw[node, chosen] = 1.0; raw[chosen, node] = 1.0
            rescued.append({"node": int(node), "barcode": names[node], "neighbor": chosen,
                            "neighbor_barcode": names[chosen], "distance": float(minimum)})
        raw = raw.tocsr()
    else:
        raise ValueError("mode must be hard or soft")
    raw = raw.maximum(raw.T).tocsr(); raw.setdiag(1.0); raw.eliminate_zeros()
    graph = _normalize(raw)
    stats = {"mode": mode, "epsilon": epsilon, "spatial_edges": int(a_spatial.nnz // 2),
             "shared_edges": int(shared.nnz // 2), "shared_edge_fraction": float(shared.nnz / max(a_spatial.nnz, 1)),
             "zero_degree_before_rescue": len(zero_before), "rescued_edges": rescued,
             "normalized_adjacency_sha256": sparse_sha256(graph)}
    return graph, stats

def deterministic_pcgrad(vectors: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    projected = [v.clone() for v in vectors]
    for i in range(len(projected)):
        for j in range(i):
            dot = torch.dot(projected[i].reshape(-1), projected[j].reshape(-1))
            denom = torch.dot(projected[j].reshape(-1), projected[j].reshape(-1))
            if dot < 0 and denom > 0:
                projected[i] = projected[i] - dot / denom * projected[j]
    return projected

def minnorm_weights(vectors: Sequence[torch.Tensor], total: float, grid: int = 1001) -> np.ndarray:
    matrix = np.asarray([[float(torch.dot(a.reshape(-1), b.reshape(-1))) for b in vectors] for a in vectors])
    if len(vectors) == 2:
        denom = matrix[0,0] + matrix[1,1] - 2*matrix[0,1]
        first = 0.5 if denom <= 1e-20 else np.clip((matrix[1,1]-matrix[0,1])/denom, 0, 1)
        return float(total) * np.asarray([first, 1-first])
    best, best_value = None, np.inf
    for i in range(grid):
        a = i/(grid-1)
        for j in range(grid-i):
            b = j/(grid-1); c = 1-a-b
            w=np.asarray([a,b,c]); value=float(w@matrix@w)
            if value < best_value: best,best_value=w,value
    return float(total)*best

class AlignmentModel(Night5AModel):
    def __init__(self, *args, projection_dim: int = 32, **kwargs):
        super().__init__(*args, **kwargs)
        embedding_dim = int(args[1])
        self.project1 = nn.Linear(embedding_dim, projection_dim, bias=False)
        self.project2 = nn.Linear(embedding_dim, projection_dim, bias=False)

def barlow_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    if z1.shape != z2.shape or z1.ndim != 2:
        raise ValueError("paired equal-dimensional embeddings required")
    first=(z1-z1.mean(0))/torch.clamp(z1.std(0,unbiased=False),min=1e-8)
    second=(z2-z2.mean(0))/torch.clamp(z2.std(0,unbiased=False),min=1e-8)
    c=first.T@second/len(first); diag=torch.diagonal(c)-1
    off=c-torch.diag(torch.diagonal(c))
    return torch.sum(diag**2)+torch.sum(off**2)/c.shape[0]

def neighbor_positive_sets(shared: sp.csr_matrix, barcodes: Sequence[str]) -> Sequence[Tuple[int, ...]]:
    shared=shared.tocsr(); names=np.asarray(barcodes).astype(str); result=[]
    for i in range(shared.shape[0]):
        neighbors=shared.indices[shared.indptr[i]:shared.indptr[i+1]]
        chosen=int(neighbors[np.argmin(names[neighbors])]) if len(neighbors) else None
        result.append((i,) if chosen is None else (i,chosen))
    return result

def neighbor_infonce(z1: torch.Tensor, z2: torch.Tensor, positives: Sequence[Tuple[int,...]],
                      temperature: float=0.2, chunk_size: Optional[int]=None) -> torch.Tensor:
    first=F.normalize(z1.double(),dim=1); second=F.normalize(z2.double(),dim=1); n=len(first); chunk=chunk_size or n
    forward_logits=first@second.T/temperature
    losses=[]
    for start in range(0,n,chunk):
        logits=forward_logits[start:start+chunk]
        for local,i in enumerate(range(start,min(start+chunk,n))):
            pos=torch.as_tensor(positives[i],device=logits.device)
            losses.append(-(torch.logsumexp(logits[local,pos],0)-torch.logsumexp(logits[local],0)))
    reverse=[[] for _ in range(n)]
    for i,rows in enumerate(positives):
        for j in rows: reverse[j].append(i)
    reverse=[tuple(sorted(set(x))) or (i,) for i,x in enumerate(reverse)]
    reverse_logits=forward_logits.T
    for start in range(0,n,chunk):
        logits=reverse_logits[start:start+chunk]
        for local,i in enumerate(range(start,min(start+chunk,n))):
            pos=torch.as_tensor(reverse[i],device=logits.device)
            losses.append(-(torch.logsumexp(logits[local,pos],0)-torch.logsumexp(logits[local],0)))
    return torch.stack(losses).mean()

def calibrate_alignment_weight(alignment: torch.Tensor, legacy_losses: Mapping[str, torch.Tensor],
                               shared_parameters: Iterable[torch.nn.Parameter], forbidden=None) -> float:
    if forbidden is not None:
        raise ValueError("semantic/downstream input forbidden in alignment calibration")
    params=list(shared_parameters)
    norms=[]
    for name in LOSS_GROUPS:
        grads=torch.autograd.grad(legacy_losses[name],params,retain_graph=True,allow_unused=True)
        norms.append(np.sqrt(sum(float(torch.sum(g.detach().double()**2)) for g in grads if g is not None)))
    grads=torch.autograd.grad(alignment,params,retain_graph=True,allow_unused=True)
    align=np.sqrt(sum(float(torch.sum(g.detach().double()**2)) for g in grads if g is not None))
    return float(min(0.25,0.10*float(np.median(norms))/max(align,1e-12)))

__all__=["LOSS_GROUPS","AlignmentModel","barlow_loss","calibrate_alignment_weight",
         "deterministic_pcgrad","minnorm_weights","neighbor_infonce","neighbor_positive_sets",
         "pruned_graph","sparse_sha256"]
