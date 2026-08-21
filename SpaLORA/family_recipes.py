"""Content-addressed frozen assay-family recipes for Night-10B."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


@dataclass(frozen=True)
class FrozenRecipe:
    family: str
    recipe_id: str
    primary_assay: str
    auxiliary_assay: str
    backbones: tuple[str, ...]
    endpoint: str
    preprocessing_contract_id: str
    adapter: Mapping[str, Any] | None = None
    sparse_graph_required: bool = True
    dense_n_by_n_allowed: bool = False

    def as_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "adapter": dict(self.adapter) if self.adapter is not None else None,
            "auxiliary_assay": self.auxiliary_assay,
            "backbones": list(self.backbones),
            "dense_n_by_n_allowed": self.dense_n_by_n_allowed,
            "endpoint": self.endpoint,
            "family": self.family,
            "preprocessing_contract_id": self.preprocessing_contract_id,
            "primary_assay": self.primary_assay,
            "recipe_id": self.recipe_id,
            "sparse_graph_required": self.sparse_graph_required,
        }
        return value

    @property
    def config_sha256(self) -> str:
        return canonical_json_sha256(self.as_dict())


RNA_PROTEIN_RECIPE = FrozenRecipe(
    family="RNA_PROTEIN",
    recipe_id="C00_G04_H05_CONFIRMED",
    primary_assay="RNA",
    auxiliary_assay="PROTEIN",
    backbones=("G04_SP10_F10_EUC_UNION",),
    endpoint="H05_EQUAL3_AFFINITY_SPECTRAL",
    preprocessing_contract_id="frozen_RNA_PROTEIN_A1_tonsil_D1_semantics",
)

RNA_EPIGENOME_RECIPE = FrozenRecipe(
    family="RNA_EPIGENOME",
    recipe_id="F00_R02_FULL",
    primary_assay="RNA",
    auxiliary_assay="ATAC",
    backbones=("G00_SP18_F20_CORR_UNION", "G04_SP10_F10_EUC_UNION"),
    endpoint="E1_ADAPTER_C06_MEAN__H01",
    preprocessing_contract_id="frozen_sparse_RNA_ATAC_LSI_semantics",
    adapter={
        "best_epoch_selection": False,
        "early_stopping": False,
        "epochs": 160,
        "fusion": "equal",
        "id": "R02",
        "losses": ["RECON", "MNN"],
    },
)

FROZEN_RECIPES = {
    ("RNA", "PROTEIN"): RNA_PROTEIN_RECIPE,
    ("RNA", "ATAC"): RNA_EPIGENOME_RECIPE,
}


__all__ = [
    "FROZEN_RECIPES", "FrozenRecipe", "RNA_EPIGENOME_RECIPE",
    "RNA_PROTEIN_RECIPE", "canonical_json_bytes", "canonical_json_sha256",
]
