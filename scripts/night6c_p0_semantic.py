#!/usr/bin/env python3
"""Run and lock the Night-6C pre-science semantic test gate."""
from __future__ import annotations
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from SpaLORA.night3af_cache import sha256_file
from SpaLORA.night6c_pipeline import atomic_json


def main() -> None:
    out = REPO / "outputs/night6c_handoff"
    graph_index = json.loads((out / "graph_cache_manifest_index.json").read_text())
    baseline = json.loads((out / "p0_baseline_spec.json").read_text())
    if graph_index["entry_count"] != 18 or graph_index["status"] != "LOCKED":
        raise RuntimeError("18/18 graph caches not locked")
    if baseline["status"] != "PASS":
        raise RuntimeError("baseline specification not locked")
    branch = subprocess.check_output(["git", "branch", "--show-current"], cwd=REPO, text=True).strip()
    if branch != "revision/q2-night6c-clean-baseline-graph-rescue-20260817":
        raise RuntimeError("Night-6C branch mismatch")
    import anndata, h5py, numpy, scipy, sklearn, torch
    environment = {
        "python": platform.python_version(), "python_executable": sys.executable,
        "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "numpy": numpy.__version__, "scipy": scipy.__version__,
        "sklearn": sklearn.__version__, "anndata": anndata.__version__,
        "h5py": h5py.__version__,
        "git": subprocess.check_output(["git", "--version"], text=True).strip(),
        "R": subprocess.check_output(["R", "--version"], text=True).splitlines()[0],
        "mclust": subprocess.check_output(["Rscript", "-e", "cat(as.character(packageVersion('mclust')))"], text=True).strip(),
        "branch": branch,
    }
    atomic_json(out / "environment_versions.json", environment)
    protect_path = out / "p0_protect_audit.json"
    protect = json.loads(protect_path.read_text())
    protect["environment_versions_file"] = str(out / "environment_versions.json")
    protect["environment_versions_sha256"] = sha256_file(out / "environment_versions.json")
    protect["current_branch_asserted"] = branch
    atomic_json(protect_path, protect)
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q",
                           "tests/test_night6c_semantic.py"], cwd=REPO,
                          text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT)
    log = out / "tests/p0_semantic_pytest.txt"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"P0 semantic tests failed; see {log}")
    payload = {
        "status": "P0_SEMANTIC_PASS", "gate_message": "P0-SEMANTIC PASS; BEGIN R1 FRESH PAIRED TRAINING",
        "pytest_exit_code": proc.returncode, "pytest_log": str(log),
        "pytest_log_sha256": sha256_file(log),
        "graph_registry": "9/9 parsed unique", "head_registry": "12/12 parsed unique",
        "graph_caches": "18/18 locked", "toy_checkpoint_round_trip": "fresh-process PASS",
        "formal_training_started_before_gate": False,
        "covered": ["kNN lexical tie-break", "union/mutual", "feature metrics",
                    "per-modality intersection", "isolates", "H00-H11 execution",
                    "self-tuning sparse affinity", "coassociation", "MRF posterior",
                    "protected paths", "label payload", "intermediate metric rejection"],
        "silent_fallback": False, "label_values_deserialized": False,
    }
    atomic_json(out / "p0_semantic_contract.json", payload)
    atomic_json(out / "tests_and_invariance_audit.json", {
        "status": "P0_PASS_FORMAL_RESULTS_PENDING", "p0_semantic": payload,
        "fixed_seeds": [0,1,2,3,4], "fixed_run_order": True,
        "parameter_tuning": False, "seed_search": False,
        "label_checkpoint_selection": False, "protected_dataset_access": False,
    })
    print(json.dumps({"status": payload["status"], "pytest_log_sha256": payload["pytest_log_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
