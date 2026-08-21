from __future__ import annotations

import ast
import inspect
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from SpaLORA.family_policy import UnsupportedAssayPair, resolve_family_policy
from SpaLORA.family_recipes import canonical_json_sha256
from SpaLORA.family_runtime import ensure_sparse_square, reject_dense_n_by_n


REPO = Path(__file__).resolve().parents[2]
EXPECTED_CONFIG_SHA = {
    "C00_G04_H05_CONFIRMED": "f987a7a4e3771fa9f1e4f4a1181f2fa9665f184a7c384c26714ac083ecd56a1a",
    "F00_R02_FULL": "0fdfd2a254be77034049ba14f91275986176903ef9aa98a2b51e1b663f8d8b7a",
}


def test_two_legal_pairs_map_exactly() -> None:
    protein = resolve_family_policy(" RNA ", " protein ").as_dict()
    epigenome = resolve_family_policy("rna", "atac").as_dict()
    assert (protein["family"], protein["recipe_id"]) == (
        "RNA_PROTEIN", "C00_G04_H05_CONFIRMED"
    )
    assert (epigenome["family"], epigenome["recipe_id"]) == (
        "RNA_EPIGENOME", "F00_R02_FULL"
    )


def test_resolver_signature_is_assay_only() -> None:
    assert list(inspect.signature(resolve_family_policy).parameters) == [
        "primary_assay", "auxiliary_assay"
    ]


def test_resolver_ast_is_identity_and_label_blind() -> None:
    path = REPO / "SpaLORA/family_policy.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    forbidden = {
        "dataset", "dataset_id", "dataset_name", "tissue", "species", "stage",
        "shape", "file_path", "annotation", "labels", "ari", "nmi", "metric",
    }
    used = {
        node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    assert not (used & forbidden)
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add((node.module or "").lower())
    assert not any("evaluat" in value or "annot" in value for value in imports)


@pytest.mark.parametrize(
    "primary,auxiliary",
    [
        ("PROTEIN", "RNA"), ("ATAC", "RNA"), ("RNA", "ADT"),
        ("RNA", ""), ("", "PROTEIN"), ("RNA+ATAC", "PROTEIN"),
        ("RNA", "ATAC,PROTEIN"), (None, "ATAC"),
    ],
)
def test_unknown_reversed_missing_or_multivalue_pairs_fail_closed(primary, auxiliary) -> None:
    with pytest.raises(UnsupportedAssayPair, match="FAIL_CLOSED"):
        resolve_family_policy(primary, auxiliary)


def test_runtime_identity_blind_for_opaque_steward_ids() -> None:
    first = resolve_family_policy("RNA", "PROTEIN").as_dict()
    second = resolve_family_policy("RNA", "PROTEIN").as_dict()
    assert first == second
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )


def test_frozen_recipe_semantics_and_config_sha() -> None:
    pairs = [
        ("RNA", "PROTEIN", REPO / "configs/family_policy/rna_protein_c00.json"),
        ("RNA", "ATAC", REPO / "configs/family_policy/rna_epigenome_f00.json"),
    ]
    for primary, auxiliary, path in pairs:
        recipe = resolve_family_policy(primary, auxiliary).recipe
        on_disk = json.loads(path.read_text(encoding="utf-8"))
        assert on_disk == recipe.as_dict()
        assert recipe.config_sha256 == canonical_json_sha256(on_disk)
        assert recipe.config_sha256 == EXPECTED_CONFIG_SHA[recipe.recipe_id]
    protein = resolve_family_policy("RNA", "PROTEIN").recipe
    assert protein.backbones == ("G04_SP10_F10_EUC_UNION",)
    assert protein.endpoint == "H05_EQUAL3_AFFINITY_SPECTRAL"
    atac = resolve_family_policy("RNA", "ATAC").recipe
    assert atac.backbones == (
        "G00_SP18_F20_CORR_UNION", "G04_SP10_F10_EUC_UNION"
    )
    assert atac.adapter == {
        "best_epoch_selection": False, "early_stopping": False, "epochs": 160,
        "fusion": "equal", "id": "R02", "losses": ["RECON", "MNN"],
    }
    assert atac.endpoint == "E1_ADAPTER_C06_MEAN__H01"


def test_sparse_boundary_and_dense_square_guard() -> None:
    sparse = sp.eye(7, format="csr")
    assert ensure_sparse_square(sparse, 7, "probe").shape == (7, 7)
    with pytest.raises(RuntimeError, match="dense N-by-N"):
        reject_dense_n_by_n(np.zeros((7, 7)), 7, "probe")
    with pytest.raises(RuntimeError, match="must be sparse"):
        ensure_sparse_square(np.eye(7), 7, "probe")


def test_cli_one_config_produces_one_atomic_row(tmp_path: Path) -> None:
    output = tmp_path / "resolved.json"
    subprocess.run(
        [
            sys.executable, str(REPO / "scripts/night10b/run_family_policy.py"),
            "resolve", "--primary-assay", "RNA", "--auxiliary-assay", "ATAC",
            "--output", str(output),
        ],
        cwd=REPO, check=True,
    )
    row = json.loads(output.read_text(encoding="utf-8"))
    assert row["recipe_id"] == "F00_R02_FULL"
    assert not output.with_name(output.name + ".tmp").exists()


def test_cli_failure_never_writes_success(tmp_path: Path) -> None:
    output = tmp_path / "should_not_exist.json"
    completed = subprocess.run(
        [
            sys.executable, str(REPO / "scripts/night10b/run_family_policy.py"),
            "resolve", "--primary-assay", "ATAC", "--auxiliary-assay", "RNA",
            "--output", str(output),
        ],
        cwd=REPO,
    )
    assert completed.returncode != 0
    assert not output.exists()
