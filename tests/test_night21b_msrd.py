import numpy as np
import scipy.sparse as sp
import torch

from SpaLORA.night21b_msrd import (
    MSRDConfig, SparseMGCNPort, common_kmeans_endpoint, prepare_graph,
    relation_evidence, reload_msrd, train_msrd,
)


def tiny(seed=0):
    rng=np.random.RandomState(seed); n=36
    labels=np.repeat(np.arange(3),12)
    view1=rng.normal(size=(n,8))+labels[:,None]*0.7
    view2=rng.normal(size=(n,6))+labels[:,None]*0.6
    retained=np.c_[view1[:,:4],view2[:,:4]]
    rows=[]; cols=[]
    for i in range(n):
        for delta in (1,2):
            j=(i+delta)%n; rows.extend([i,j]); cols.extend([j,i])
    graph=sp.csr_matrix((np.ones(len(rows)),(rows,cols)),shape=(n,n))
    return view1.astype('f4'),view2.astype('f4'),retained.astype('f4'),graph


def test_graph_sparse_and_row_stochastic():
    *_,g=tiny(); p=prepare_graph(g)
    assert sp.issparse(p) and np.allclose(np.asarray(p.sum(axis=1)).ravel(),1)


def test_relations_are_nonempty_disjoint_and_finite():
    v1,v2,r,g=tiny(); e=relation_evidence(r,v1,v2,g)
    pos=set(zip(e.positive_rows,e.positive_cols)); neg=set(zip(e.boundary_rows,e.boundary_cols))
    assert pos and neg and pos.isdisjoint(neg)
    assert np.isfinite(e.positive_weights).all() and np.isfinite(e.boundary_weights).all()


def test_arm_changes_gradients_and_checkpoint_reloads():
    v1,v2,r,g=tiny(); c=MSRDConfig('TEST','RNA_CHROMATIN',steps=3,hidden_dim=16,graph_order=2)
    rep,state,d=train_msrd(v1,v2,r,g,c,'FULL_SIGNED_RELATIONAL_DISTILLATION',0,'cpu')
    reread=reload_msrd(v1,v2,r,g,c,0,state,'cpu')
    assert d['parameter_changed'] and d['max_gradient_norm']>0
    assert np.array_equal(rep,reread)
    assert len(np.unique(common_kmeans_endpoint(rep,3)))==3


def test_full_and_backbone_losses_are_distinct():
    v1,v2,r,g=tiny(); c=MSRDConfig('TEST','RNA_CHROMATIN',steps=2,hidden_dim=16,graph_order=2)
    rep0,_,d0=train_msrd(v1,v2,r,g,c,'B0_BACKBONE_ONLY',0,'cpu')
    repf,_,df=train_msrd(v1,v2,r,g,c,'FULL_SIGNED_RELATIONAL_DISTILLATION',0,'cpu')
    assert not np.array_equal(rep0,repf)
    assert df['positive_relation_count']>0 and df['boundary_relation_count']>0

