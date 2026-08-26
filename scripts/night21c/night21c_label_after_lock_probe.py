"""Diagnostic supervised probes; never used by a Night-21C producer or selector."""
from __future__ import annotations
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,f1_score
from sklearn.model_selection import StratifiedKFold,cross_val_predict
from sklearn.neighbors import KNeighborsClassifier,NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from night21c_endpoint_evaluator import load_truth


def main():
    p=argparse.ArgumentParser(); p.add_argument("--embedding",required=True); p.add_argument("--embedding-key",default="representation"); p.add_argument("--authority",required=True); p.add_argument("--reference-h5ad"); p.add_argument("--lane",required=True); p.add_argument("--representation-source",required=True); p.add_argument("--output",required=True); a=p.parse_args()
    path=Path(a.embedding)
    with np.load(path,allow_pickle=False) as z: ids=np.asarray(z["ids"]); x=np.asarray(z[a.embedding_key],dtype=np.float64)
    labels,mask,_=load_truth(a.authority,ids,a.reference_h5ad); x=x[mask]; y=labels[mask]
    _,counts=np.unique(y,return_counts=True); folds=min(5,int(counts.min()))
    if folds<2: raise RuntimeError("not enough examples for stratified probe")
    cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=0)
    models={
        "LINEAR_LOGISTIC":make_pipeline(StandardScaler(),LogisticRegression(max_iter=3000,class_weight="balanced",random_state=0,n_jobs=1)),
        "KNN_15":make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=min(15,len(y)-1),weights="distance",n_jobs=1)),
        "NEAREST_CENTROID":make_pipeline(StandardScaler(),NearestCentroid()),
    }
    rows=[]
    for name,model in models.items():
        pred=cross_val_predict(model,x,y,cv=cv,n_jobs=1,method="predict")
        rows.append({"lane":a.lane,"representation_source":a.representation_source,"probe":name,"folds":folds,"accuracy":accuracy_score(y,pred),"balanced_accuracy":balanced_accuracy_score(y,pred),"macro_f1":f1_score(y,pred,average="macro"),"n_eval":len(y),"label_flow":"after embedding lock; diagnostic only"})
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    with out.open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out.with_suffix(".json").write_text(json.dumps({"schema":"night21c-label-after-lock-probe-v1","embedding_file_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"labels_used_for_training_probe_only":True,"probe_influenced_unsupervised_result":False},indent=2,sort_keys=True),encoding="utf-8")


if __name__=="__main__": main()
