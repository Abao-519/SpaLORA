"""Night-6A runtime trainer built on the verified Night-5 C04 implementation."""
from __future__ import annotations
import copy, math, time
from typing import Mapping
import numpy as np
import scipy.sparse as sp
import torch

from .night3a_ige import (_clone_state, calibrated_total,
    model_state_sha256, raw_losses, required_record_steps, run_initial_probe, state_dict_sha256)
from .night5a_rnd import Night5ATrainer, Night5ATrainingResult, _Forward
from .night3b_ablation import active_ige_coefficients
from .night6a_structural import (AlignmentModel, LOSS_GROUPS, barlow_loss,
    calibrate_alignment_weight, deterministic_pcgrad, minnorm_weights,
    neighbor_infonce, neighbor_positive_sets, pruned_graph, sparse_sha256)
from .preprocess import fix_seed

BASE={"id":"C04_SHRINK25","attention":"shrink_to_uniform","learned_fraction_rho":0.25,
      "corr2":False,"loss_calibration":"active_set_IGE"}

def resolved_changes(candidate):
    changes=candidate.get("changes",[]); result={"graph":None,"optimization":None,"alignment":None,"staged":False}
    for item in changes:
        if isinstance(item,dict): result["graph"]=("soft",float(item["soft_prune_epsilon"]))
        elif item=="hard_prune": result["graph"]=("hard",None)
        elif item in ("pcgrad","minnorm"): result["optimization"]=item
        elif item in ("barlow_alignment","neighbor_alignment"): result["alignment"]=item
        elif item=="staging": result["staged"]=True
        else: raise ValueError("unregistered change %r"%item)
    return result

class Night6ATrainer:
    def __init__(self,data,cfg,candidate,seed,device,obs_names,coordinates,eps=1e-12):
        self.data=dict(data);self.cfg=dict(cfg);self.candidate=dict(candidate);self.seed=int(seed);self.device=device
        self.obs_names=np.asarray(obs_names).astype(str);self.coordinates=np.asarray(coordinates);self.eps=float(eps)
        self.changes=resolved_changes(candidate);self.graph_stats=None;self.positives=None
        if candidate["id"]=="N00":
            self.delegate=Night5ATrainer(data,cfg,BASE,seed,device,{},eps);return
        self.delegate=None
        if self.changes["graph"]:
            mode,value=self.changes["graph"]
            graph,stats=pruned_graph(data["adj_spatial_omics1"],data["adj_feature_omics1"],mode,value,self.coordinates,self.obs_names)
            self.data["adj_spatial_omics1"]=graph;self.data["adj_spatial_omics2"]=graph;self.graph_stats=stats
        self.features1=torch.as_tensor(self.data["features_omics1"],dtype=torch.float32,device=device)
        self.features2=torch.as_tensor(self.data["features_omics2"],dtype=torch.float32,device=device)
        self.adjacencies=tuple(self.data[k].to(device) for k in ("adj_spatial_omics1","adj_feature_omics1","adj_spatial_omics2","adj_feature_omics2"))
        if self.changes["alignment"]=="neighbor_alignment":
            shared=sp.csr_matrix((len(self.obs_names),len(self.obs_names)))
            from .night5a_rnd import _support
            shared=_support(self.data["adj_spatial_omics1"]).multiply(_support(self.data["adj_feature_omics1"]))
            self.positives=neighbor_positive_sets(shared,self.obs_names)

    def new_model(self):
        fix_seed(self.seed); alignment=self.changes["alignment"] is not None
        cls=AlignmentModel if alignment else __import__("SpaLORA.night5a_rnd",fromlist=["Night5AModel"]).Night5AModel
        return cls(self.features1.shape[1],int(self.cfg["embedding_dim"]),self.features2.shape[1],int(self.cfg["embedding_dim"]),
                   attention_policy="shrink_to_uniform",learned_fraction=.25).to(self.device)

    def _alignment(self,model,result):
        kind=self.changes["alignment"]
        if not kind:return result["emb_latent_combined"].sum()*0
        z1=model.project1(result["emb_latent_omics1"]);z2=model.project2(result["emb_latent_omics2"])
        return barlow_loss(z1,z2) if kind=="barlow_alignment" else neighbor_infonce(z1,z2,self.positives,.2,512)

    @staticmethod
    def _shared(model):
        return [(n,p) for n,p in model.named_parameters() if p.requires_grad and not n.startswith(("decoder1","decoder2"))]

    def train(self):
        if self.delegate:
            result=self.delegate.train();result.auxiliary.update({"night6a_candidate":"N00","actual_mechanisms":[],"graph_sha256":sparse_sha256(self.data["adj_spatial_omics1"])});return result
        model=self.new_model();forward=_Forward(self.features1,self.features2,self.adjacencies)
        initial=_clone_state(model);initial_hash=state_dict_sha256(initial);probe=run_initial_probe(model,forward,self.eps)
        model.load_state_dict(initial);coeff=active_ige_coefficients(probe["gradients"],{k:k!="L_corr2_raw" for k in probe["gradients"]},self.eps)
        initial_result=forward(model);initial_raw=raw_losses(initial_result,self.features1,self.features2)
        initial_losses={k:float(v.detach().cpu()) for k,v in initial_raw.items()}
        alignment_weight=0.0
        if self.changes["alignment"]:
            alignment_weight=calibrate_alignment_weight(self._alignment(model,initial_result),initial_raw,[p for _,p in self._shared(model)])
        model.load_state_dict(initial);optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=0)
        record_steps=set(required_record_steps(int(self.cfg["epochs"])));logs=[];started=time.perf_counter();nonfinite=0;conflicts=[];weights=[]
        for step in range(1,int(self.cfg["epochs"])+1):
            result=forward(model);losses=raw_losses(result,self.features1,self.features2)
            enabled=(not self.changes["staged"]) or step>int(.25*int(self.cfg["epochs"]))
            align=self._alignment(model,result)*(alignment_weight if enabled else 0.0)
            total=calibrated_total(losses,coeff)+align;optimizer.zero_grad()
            shared=self._shared(model);params=[p for _,p in shared]; opt=self.changes["optimization"]
            if opt:
                group_grads=[]
                for key in LOSS_GROUPS:
                    g=torch.autograd.grad(losses[key],params,retain_graph=True,allow_unused=True)
                    group_grads.append([torch.zeros_like(p) if x is None else x for p,x in zip(params,g)])
                raw_flat=[torch.cat([x.reshape(-1) for x in group]) for group in group_grads]
                flat=[float(coeff[key])*value for key,value in zip(LOSS_GROUPS,raw_flat)] if opt=="pcgrad" else raw_flat
                cos=[]
                for i,j in ((0,1),(0,2),(1,2)):
                    cos.append(float(torch.dot(flat[i],flat[j])/(torch.norm(flat[i])*torch.norm(flat[j])+1e-12)))
                conflicts.append(cos);total.backward(retain_graph=False)
                align_grads=[torch.zeros_like(p) for p in params]
                if align.requires_grad and float(align.detach())!=0:
                    # already included in total grads; preserve by subtracting legacy ordinary later is unnecessary;
                    # capture total shared grads then replace only legacy component approximately below.
                    pass
                ordinary=[p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
                legacy_ordinary=[]
                for pi,p in enumerate(params):legacy_ordinary.append(sum(coeff[k]*group_grads[gi][pi] for gi,k in enumerate(LOSS_GROUPS)))
                extra=[o-l for o,l in zip(ordinary,legacy_ordinary)]
                if opt=="pcgrad":
                    projected=deterministic_pcgrad(flat);offsets=np.cumsum([0]+[p.numel() for p in params])
                    coordinated=[sum(v[offsets[i]:offsets[i+1]].reshape_as(p) for v in projected) for i,p in enumerate(params)]
                    weights.append([1.,1.,1.])
                else:
                    w=minnorm_weights(flat,sum(coeff[k] for k in LOSS_GROUPS),101);weights.append(w.tolist())
                    coordinated=[sum(float(w[g])*group_grads[g][i] for g in range(3)) for i,p in enumerate(params)]
                for p,c,e in zip(params,coordinated,extra):p.grad=c+e
            else:total.backward()
            finite=all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());nonfinite+=0 if finite else 1
            optimizer.step()
            if step in record_steps:
                logs.append({"step":step,"total_loss":float(total.detach().cpu()),"alignment_raw":float(self._alignment(model,forward(model)).detach().cpu()),
                             "alignment_weight":alignment_weight,"alignment_enabled":enabled,"nonfinite_gradient_count":nonfinite})
        model.eval()
        with torch.no_grad():result=forward(model)
        output={"emb_latent_omics1":torch.nn.functional.normalize(result["emb_latent_omics1"],dim=1).cpu().numpy(),
                "emb_latent_omics2":torch.nn.functional.normalize(result["emb_latent_omics2"],dim=1).cpu().numpy(),
                "SpaLORA":torch.nn.functional.normalize(result["emb_latent_combined"],dim=1).cpu().numpy(),
                "alpha_omics1":result["alpha_omics1"].cpu().numpy(),"alpha_omics2":result["alpha_omics2"].cpu().numpy(),"alpha":result["alpha"].cpu().numpy()}
        aux={"night6a_candidate":self.candidate["id"],"actual_mechanisms":self.changes,"graph_stats":self.graph_stats,
             "graph_sha256":sparse_sha256(self.adjacencies[0]),"alignment_weight":alignment_weight,"conflict_cosines":conflicts,
             "coordination_weights":weights,"nonfinite_gradient_count":nonfinite,"parameter_count":sum(p.numel() for p in model.parameters())}
        return Night5ATrainingResult(output,logs,model,probe,coeff,initial_losses,initial_hash,model_state_sha256(model),aux)
