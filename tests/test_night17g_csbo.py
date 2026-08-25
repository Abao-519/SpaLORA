import numpy as np
import scipy.sparse as sp
from SpaLORA.night17g_csbo import CSBOConfig,_edge_weights,_ordinal_rank,build_edge_states,endpoint_partition,permute_edge_states,reload_core,train_core

def _toy():
 ids=np.array(["a","b","c","d","e","f"]); rows=np.array([0,1,2,3,4,0,1]); cols=np.array([1,2,3,4,5,5,4]); data=np.linspace(.2,1,len(rows)); g=sp.coo_matrix((data,(rows,cols)),shape=(6,6)); g=(g+g.T).tocsr()
 x1=np.array([[0,0],[.1,0],[.2,0],[4,4],[4.1,4],[4.2,4]],dtype=np.float32); x2=x1.copy(); x2[1]=[4,4]
 return ids,g,x1,x2
def test_states_finite_nonnegative_and_no_self():
 ids,g,x1,x2=_toy(); s=build_edge_states(x1,x2,g,ids)
 assert np.all(s.rows!=s.cols); assert np.all(s.rows>=0); assert np.all(s.cols>=0)
 for x in (s.attraction,s.boundary,s.conflict): assert np.all(np.isfinite(x)) and np.all(x>=0)
 assert np.allclose(s.attraction+s.boundary+s.conflict,1.0,rtol=0,atol=2e-7)
 assert min(s.attraction.sum(),s.boundary.sum(),s.conflict.sum())>0
def test_permutation_deterministic_nonidentity_mass_preserved():
 ids,g,x1,x2=_toy(); s=build_edge_states(x1,x2,g,ids); p1=permute_edge_states(s,ids); p2=permute_edge_states(s,ids)
 assert p1.state_sha256==p2.state_sha256!=s.state_sha256
 for key in ("attraction","boundary","conflict"):
  assert np.isclose(np.sum(s.base_weight*getattr(s,key)),np.sum(p1.base_weight*getattr(p1,key)),rtol=1e-7,atol=1e-8)
def test_edge_order_invariant_graph_storage():
 ids,g,x1,x2=_toy(); s1=build_edge_states(x1,x2,g,ids); coo=g.tocoo(); order=np.arange(coo.nnz)[::-1]; g2=sp.coo_matrix((coo.data[order],(coo.row[order],coo.col[order])),shape=g.shape).tocsr(); s2=build_edge_states(x1,x2,g2,ids)
 assert s1.state_sha256==s2.state_sha256
def test_endpoint_exact_k():
 rng=np.random.RandomState(0); x=np.r_[rng.normal(-2,.1,(10,3)),rng.normal(2,.1,(10,3))].astype(np.float32); init=np.r_[np.zeros(10),np.ones(10)].astype(int); p=endpoint_partition(x,init,2); assert np.unique(p).size==2 and np.bincount(p).min()>0
def test_midrank_ties_are_equal():
 r=_ordinal_rank(np.array([0.1,0.5,0.5,0.9])); assert r[1]==r[2]
def test_arm_weights_are_distinct():
 ids,g,x1,x2=_toy(); s=build_edge_states(x1,x2,g,ids)
 full=_edge_weights(s,"FULL_CSBO"); back=_edge_weights(s,"BACKBONE_NO_CSBO"); unsigned=_edge_weights(s,"UNSIGNED_ONLY")
 assert not np.array_equal(full[0],back[0]); assert not np.array_equal(full[1],unsigned[1])
def test_small_train_reload_exact_and_parameter_change():
 ids,g,x1,x2=_toy(); retained=np.c_[x1,x2].astype(np.float32); initial=np.array([0,0,0,1,1,1]); states=build_edge_states(x1,x2,g,ids)
 cfg=CSBOConfig("T",8,.05,1e-3,2,.2,2,.1,.2,.2)
 rep,part,state,diag=train_core(x1,x2,retained,initial,states,cfg,"FULL_CSBO",0,"cpu")
 replay=reload_core(state,x1,x2,retained,initial,cfg,"cpu")
 assert diag["parameter_change_l2"]>0 and np.array_equal(rep,replay) and np.array_equal(part,endpoint_partition(replay,initial,2))
