import json
from pathlib import Path
from SpaLORA.night5b_rnd import single_step_diffusion
import numpy as np, torch
def test_locked_matrix():
 c=json.loads(Path('configs/night5d_locked_p22_confirmation.json').read_text());assert c['seeds']==list(range(10));assert c['statistics']['signflip_permutations']==1024;assert c['statistics']['bootstrap_replicates']==100000
def test_diffusion_alpha_zero_and_sparse():
 z=np.eye(3,dtype=np.float32);i=torch.tensor([[0,1,2],[0,1,2]]);a=torch.sparse_coo_tensor(i,torch.ones(3),(3,3)).coalesce();assert np.array_equal(single_step_diffusion(z,a,0),z);assert single_step_diffusion(z,a,.1).shape==z.shape
def test_no_forbidden_dataset_in_config():
 text=Path('configs/night5d_locked_p22_confirmation.json').read_text().lower();assert 'd1_lymph' not in text and 'gse198353' not in text and 'night4b' not in text
