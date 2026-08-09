"""Night-2C interface to the exact tested Night-2B parity-locked trainer."""

from __future__ import annotations

import hashlib
import json
import os
import platform
from pathlib import Path

import anndata
import numpy as np
import rpy2
import scanpy
import scipy
import sklearn
import torch

from .night2b_loss_audit import (
    LOSS_LOG_FIELDS,
    ParityLockedTrainer,
    RnaLoss,
    VARIANTS,
    compute_locked_asr_weights,
    compute_loss_components,
    legacy_bug_weight_vector,
    required_checkpoint_epochs,
    rna_loss_dispatch,
    selected_legacy_gene_names,
    validate_name_weight_alignment,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(str(temporary), str(path))


def environment_payload() -> dict:
    os.environ.setdefault("R_HOME", "/opt/R/4.0.3/lib/R")
    import rpy2.robjects as ro

    payload = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "scanpy": scanpy.__version__,
        "anndata": anndata.__version__,
        "rpy2": rpy2.__version__,
        "r": str(ro.r("R.version.string")[0]),
        "mclust": str(ro.r("as.character(packageVersion('mclust'))")[0]),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    payload["fingerprint"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload


def critical_hashes(repo: Path, config: dict) -> dict:
    repo = Path(repo)
    relative = (
        "configs/night2c_numerical_equivalence_factorial.json",
        "SpaLORA/night2c_loss_audit.py",
        "SpaLORA/night2b_loss_audit.py",
        "scripts/night2c_p0c_gate.py",
        "scripts/night2c_factorial.py",
        "SpaLORA/model.py",
        "SpaLORA/preprocess.py",
        "SpaLORA/SpaLORA_pyG.py",
    )
    hashes = {name: sha256_file(repo / name) for name in relative}
    for dataset, cfg in config["datasets"].items():
        hashes["input:%s:rna" % dataset] = sha256_file(Path(cfg["rna"]))
        hashes["input:%s:modality2" % dataset] = sha256_file(Path(cfg["modality2"]))
        if not str(cfg["ground_truth"]).startswith("obs["):
            hashes["input:%s:ground_truth" % dataset] = sha256_file(Path(cfg["ground_truth"]))
    return hashes


__all__ = [
    "LOSS_LOG_FIELDS", "ParityLockedTrainer", "RnaLoss", "VARIANTS",
    "compute_locked_asr_weights", "compute_loss_components", "legacy_bug_weight_vector",
    "required_checkpoint_epochs", "rna_loss_dispatch", "selected_legacy_gene_names",
    "validate_name_weight_alignment", "sha256_file", "atomic_json", "environment_payload",
    "critical_hashes",
]
