#!/usr/bin/env python3
"""Reproducible parallel HTTP Range downloader for large public evidence files."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import time
import urllib.request
from pathlib import Path


def remote_size(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "SpaLORA-Night4A/1"})
    with urllib.request.urlopen(request, timeout=60) as response:
        if response.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError("server does not advertise byte ranges")
        return int(response.headers["Content-Length"])


def download_one(url: str, directory: Path, start: int, end: int, retries: int = 5) -> Path:
    expected = end - start + 1
    final = directory / f"{start:012d}-{end:012d}.chunk"
    if final.exists() and final.stat().st_size == expected:
        return final
    partial = final.with_suffix(".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(
                url,
                headers={"Range": f"bytes={start}-{end}", "User-Agent": "SpaLORA-Night4A/1"},
            )
            with urllib.request.urlopen(request, timeout=180) as response, partial.open("wb") as output:
                if response.status != 206:
                    raise RuntimeError(f"expected HTTP 206, got {response.status}")
                shutil.copyfileobj(response, output, length=1024 * 1024)
            if partial.stat().st_size != expected:
                raise RuntimeError(f"short range {partial.stat().st_size} != {expected}")
            os.replace(partial, final)
            return final
        except Exception:
            if attempt == retries:
                raise
            time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--chunk-mib", type=int, default=16)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    size = remote_size(args.url)
    if args.output.exists() and args.output.stat().st_size == size:
        print(json.dumps({"status": "already_complete", "bytes": size, "sha256": sha256(args.output)}))
        return
    chunk_bytes = args.chunk_mib * 1024 * 1024
    ranges = [(start, min(start + chunk_bytes - 1, size - 1)) for start in range(0, size, chunk_bytes)]
    chunk_dir = args.output.with_name(args.output.name + ".chunks")
    chunk_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, args.url, chunk_dir, start, end): (start, end) for start, end in ranges}
        for future in concurrent.futures.as_completed(futures):
            future.result(); completed += 1
            if completed % 8 == 0 or completed == len(ranges):
                print(f"ranges={completed}/{len(ranges)}", flush=True)
    assembling = args.output.with_name(args.output.name + ".assembling")
    with assembling.open("wb") as target:
        for start, end in ranges:
            with (chunk_dir / f"{start:012d}-{end:012d}.chunk").open("rb") as source:
                shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        target.flush(); os.fsync(target.fileno())
    if assembling.stat().st_size != size:
        raise RuntimeError("assembled size mismatch")
    os.replace(assembling, args.output)
    result = {"status": "complete", "url": args.url, "bytes": size, "sha256": sha256(args.output), "seconds": time.time() - started}
    args.output.with_name(args.output.name + ".download.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
