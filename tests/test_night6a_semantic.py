import json
from pathlib import Path
import numpy as np
import pytest
import scipy.sparse as sp
import torch

from SpaLORA.night6a_structural import *

def sparse(matrix):
    r,c=np.nonzero(matrix); return torch.sparse_coo_tensor(torch.tensor([r,c]),torch.tensor(matrix[r,c],dtype=torch.float32),matrix.shape).coalesce()

def test_soft_prune_exact_and_unique():
    s=np.array([[0,1,1],[1,0,1],[1,1,0.]],float); r=np.array([[0,1,0],[1,0,0],[0,0,0.]],float)
    hashes=[]
    for eps in (.1,.25,.5):
        g,a=pruned_graph(sparse(s),sparse(r),'soft',eps); hashes.append(a['normalized_adjacency_sha256']); assert a['spatial_edges']==3 and a['shared_edges']==1
    assert len(set(hashes))==3

def test_hard_rescue_lexical_and_no_new_support():
    s=np.array([[0,1,1],[1,0,0],[1,0,0.]],float); r=np.zeros_like(s); coords=np.array([[0,0],[1,0],[-1,0.]])
    _,a=pruned_graph(sparse(s),sparse(r),'hard',coordinates=coords,barcodes=['z','b','a'])
    assert a['zero_degree_before_rescue']==3 and a['rescued_edges'][0]['neighbor_barcode']=='a'

def test_pcgrad_analytic_and_bit_exact():
    orth=[torch.tensor([1.,0]),torch.tensor([0.,1])]; assert all(torch.equal(a,b) for a,b in zip(orth,deterministic_pcgrad(orth)))
    same=[torch.tensor([1.,0]),torch.tensor([2.,0])]; assert all(torch.equal(a,b) for a,b in zip(same,deterministic_pcgrad(same)))
    opposite=[torch.tensor([1.,0]),torch.tensor([-1.,0])]; first=deterministic_pcgrad(opposite); second=deterministic_pcgrad(opposite); assert first[1].abs().max()==0 and all(torch.equal(a,b) for a,b in zip(first,second))

def test_minnorm_two_three_vectors():
    w=minnorm_weights([torch.tensor([1.,0]),torch.tensor([0.,1])],4); assert w==pytest.approx([2,2]); assert (w>=0).all() and w.sum()==pytest.approx(4)
    w3=minnorm_weights([torch.tensor([1.,0]),torch.tensor([0.,1]),torch.tensor([-1.,0])],3,101); assert (w3>=0).all() and w3.sum()==pytest.approx(3)

def test_barlow_pairing_and_permutation():
    raw=torch.tensor(np.random.default_rng(7).normal(size=(20,4)),dtype=torch.float32)
    centered=raw-raw.mean(0); q,_=torch.linalg.qr(centered); x=q[:,:4]
    assert barlow_loss(x,x)<1e-5
    assert barlow_loss(x,x[:,[1,0,2,3]])>barlow_loss(x,x)
    order=torch.arange(len(x)); order[0],order[2]=order[2].clone(),order[0].clone()
    assert barlow_loss(x,x[order])>barlow_loss(x,x)

def test_neighbor_exact_chunk_and_sets():
    graph=sp.csr_matrix(np.array([[0,1,1],[1,0,0],[1,0,0]])); sets=neighbor_positive_sets(graph,['z','b','a']); assert sets[0]==(0,2)
    z1=torch.randn(3,5);z2=torch.randn(3,5); assert neighbor_infonce(z1,z2,sets).item()==pytest.approx(neighbor_infonce(z1,z2,sets,chunk_size=1).item(),abs=1e-10)

def test_calibration_rejects_label_argument():
    p=torch.nn.Parameter(torch.tensor([1.,2.])); losses={k:(p**2).sum() for k in LOSS_GROUPS}
    with pytest.raises(ValueError): calibrate_alignment_weight((p.sum())**2,losses,[p],forbidden=['label'])

def test_lower_is_better_directions_and_path_guards():
    directions={'geary':'lower','boundary_disagreement':'lower','ari':'higher','nmi':'higher'}
    assert directions['geary']==directions['boundary_disagreement']=='lower'
    forbidden=['p22','d1']; assert all(x in forbidden for x in ['p22','d1'])
