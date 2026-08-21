from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night10a_qcrd import (
    QCRDAdapter, deterministic_mask, frozen_quality, qcrd_forward,
    qcrd_loss_components, row_normalize,
)
from scripts.night10a.night10a_rev1_run import torch_loss


def test_optimized_runner_loss_is_exact_canonical_formula():
    rng=np.random.default_rng(17); n,d=13,8
    first=row_normalize(rng.normal(size=(n,d)).astype(np.float32)); second=row_normalize(rng.normal(size=(n,d)).astype(np.float32)); fused=row_normalize((first+second)/2)
    rows=np.arange(n-1); graph=sp.coo_matrix((np.ones(2*(n-1)),(np.r_[rows,rows+1],np.r_[rows+1,rows])),shape=(n,n)).tocsr()
    p1=np.arange(n)%3; p2=(np.arange(n)+1)%3; pf=np.arange(n)%4
    quality=frozen_quality(first,second,fused,p1,p2,graph,4,mnn_k=3)
    first_t=torch.tensor(first); second_t=torch.tensor(second); fused_t=torch.tensor(fused)
    upper=sp.triu(graph,k=1).tocoo(); keep=pf[upper.row]!=pf[upper.col]; br=torch.tensor(upper.row[keep]); bc=torch.tensor(upper.col[keep])
    for candidate in ("Q02_SPOT_QUALITY_BLEND","Q04_SPOT_GATED_MASKED_RESIDUAL","Q07_CONFIDENCE_MNN_RESIDUAL"):
        torch.manual_seed(3); model=QCRDAdapter(d)
        with torch.no_grad(): model.up.weight.normal_(0,.02)
        mask=None if candidate=="Q02_SPOT_QUALITY_BLEND" else torch.tensor(deterministic_mask(candidate,"f"*64,2,7,n,d))
        output=qcrd_forward(model,first_t,second_t,fused_t,quality,candidate,None,mask)
        expected=qcrd_loss_components(output,first_t,second_t,fused_t,quality,candidate,graph,pf,mask)
        observed=torch_loss(output,first_t,second_t,fused_t,quality,candidate,mask,br,bc)
        for key in expected:
            assert torch.allclose(expected[key],observed[key],rtol=0,atol=1e-7), (candidate,key,expected[key],observed[key])


def test_formal_matrix_cardinality_and_no_retry_contract():
    from scripts.night10a.night10a_rev1_run import CANDIDATES, DATA
    assert len(CANDIDATES)==7
    assert sum(len(list(v["r1"]))*len(CANDIDATES) for v in DATA.values())==84
    assert sum((len(list(v["r2"]))-len(list(v["r1"])))*3 for v in DATA.values())==54
