from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from SpaLORA.night3af_cache import verify_cache


REPO = Path("/root/autodl-fs/SpaLORA-night3b")
OUT = REPO / "outputs/night3b_handoff/protection_preflight"
BASE = "384e66587149a687b3eac4a6d1918d8d4972dc06"
CACHE_REPO = Path("/root/autodl-fs/SpaLORA-night3af")


def run(*args: str, cwd: Path) -> str:
    return subprocess.check_output(args, cwd=str(cwd), text=True).strip()


def verify_sha_manifest(name: str, manifest: Path, cwd: Path, expected: int) -> dict:
    result = subprocess.run(
        ["sha256sum", "-c", str(manifest)], cwd=str(cwd), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    log_path = OUT / f"{name}_sha256_check.log"
    log_path.write_text(result.stdout, encoding="utf-8")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    ok = sum(line.endswith(": OK") for line in result.stdout.splitlines())
    failed = [line for line in result.stdout.splitlines() if "FAILED" in line]
    if result.returncode != 0 or ok != expected or failed:
        raise RuntimeError(
            f"{name} protection failed: returncode={result.returncode}, "
            f"ok={ok}/{expected}, failed={len(failed)}"
        )
    return {"manifest": str(manifest), "cwd": str(cwd), "ok": ok, "expected": expected}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head = run("git", "rev-parse", "HEAD", cwd=REPO)
    tag = run("git", "rev-parse", "night3af-final-20260810^{commit}", cwd=REPO)
    baseline = run("git", "rev-parse", "baseline/pre-night3b-20260810^{commit}", cwd=REPO)
    status_before = run("git", "status", "--porcelain", cwd=REPO)
    if head != BASE or tag != BASE or baseline != BASE:
        raise RuntimeError(f"baseline mismatch: head={head}, tag={tag}, protection={baseline}")
    # At this point only the preflight output directory itself may be untracked.
    expected_new = (
        "outputs/night3b_handoff",
        "docs/SpaLORA_Night3B_Architecture_Ablation_and_Interpretability_Taskbook_2026-08-10.md",
        "scripts/night3b_preflight.py",
    )
    unknown = [
        line for line in status_before.splitlines()
        if not any(path in line for path in expected_new)
    ]
    if unknown:
        raise RuntimeError(f"unknown worktree modifications: {unknown}")

    protections = {
        "night3af": verify_sha_manifest(
            "night3af",
            REPO / "outputs/night3af_handoff/SHA256SUMS",
            REPO / "outputs/night3af_handoff",
            709,
        ),
        "night3ar": verify_sha_manifest(
            "night3ar",
            Path("/root/autodl-fs/night3af_preexisting_20260810/night3ar_final_tracked_before.sha256"),
            Path("/root/autodl-fs/SpaLORA-night3ar"),
            211,
        ),
        "night3a": verify_sha_manifest(
            "night3a",
            Path("/root/autodl-fs/night3ar_preexisting_20260810/night3a_final_files_before.sha256"),
            Path("/root/autodl-fs/SpaLORA-night3a"),
            198,
        ),
        "night2c": verify_sha_manifest(
            "night2c",
            Path("/root/autodl-fs/night3a_preexisting_20260810/night2c_files_before.sha256"),
            Path("/root/autodl-fs/SpaLORA-night2c"),
            913,
        ),
    }

    top_manifest_path = CACHE_REPO / "outputs/night3af_handoff/preprocessing_cache_manifest.json"
    top_manifest = json.loads(top_manifest_path.read_text(encoding="utf-8"))
    if top_manifest.get("immutable") is not True:
        raise RuntimeError("Night-3AF cache manifest is not marked immutable")
    caches = {}
    for dataset in ("a1", "placenta", "p22"):
        row = top_manifest["datasets"][dataset]
        directory = CACHE_REPO / row["directory"]
        manifest = verify_cache(directory, row["manifest_sha256"])
        writable = [str(path) for path in directory.iterdir() if path.is_file() and os.access(path, os.W_OK)]
        # Root's os.access ignores mode bits, so explicitly check write bits.
        mode_writable = [str(path) for path in directory.iterdir() if path.is_file() and path.stat().st_mode & 0o222]
        if mode_writable:
            raise RuntimeError(f"published cache is not immutable: {mode_writable}")
        if manifest["canonical_cache_content_sha256"] != row["canonical_cache_content_sha256"]:
            raise RuntimeError(f"cache content hash mismatch: {dataset}")
        caches[dataset] = {
            "directory": str(directory),
            "manifest_sha256": sha256(directory / "manifest.json"),
            "canonical_cache_content_sha256": manifest["canonical_cache_content_sha256"],
            "canonical_model_input_sha256": manifest["canonical_model_input_sha256"],
            "mode_writable_files": len(mode_writable),
            "root_os_access_writable_files_ignored": len(writable),
        }

    payload = {
        "schema_version": 1,
        "status": "PASS",
        "baseline_commit": BASE,
        "baseline_tag": "night3af-final-20260810",
        "protection_tag": "baseline/pre-night3b-20260810",
        "protections": protections,
        "immutable_caches": caches,
        "semantic_label_values_read": False,
        "cache_reprocessing_performed": False,
    }
    path = OUT / "protection_preflight.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with path.open("a", encoding="utf-8") as handle:
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"status": "PASS", "protections": {k: v["ok"] for k, v in protections.items()}, "caches": 3}))


if __name__ == "__main__":
    main()
