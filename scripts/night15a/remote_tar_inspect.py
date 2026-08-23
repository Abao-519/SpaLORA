#!/usr/bin/env python3
"""Index an uncompressed remote TAR with HTTP range requests."""
from __future__ import annotations

import argparse
import json
import os
import re
import urllib.request
from pathlib import Path
from time import sleep


def get_range(url: str, start: int, end: int, retries: int = 5) -> bytes:
    error = None
    for attempt in range(retries):
        try:
            request = urllib.request.Request(
                url, headers={"Range": f"bytes={int(start)}-{int(end)}"}
            )
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
                if response.status != 206:
                    raise RuntimeError("server did not honor HTTP range")
            if len(data) != int(end) - int(start) + 1:
                raise RuntimeError("range length mismatch")
            return data
        except Exception as caught:
            error = caught
            sleep(2**attempt)
    raise error


def content_length(url: str) -> int:
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        return int(response.headers["Content-Length"])


def tar_index(url: str) -> tuple[int, list[dict]]:
    total = content_length(url)
    cursor = 0
    members = []
    while cursor + 512 <= total:
        header = get_range(url, cursor, cursor + 511)
        if header == b"\0" * 512:
            break
        name = header[0:100].split(b"\0", 1)[0].decode("utf-8")
        prefix = header[345:500].split(b"\0", 1)[0].decode("utf-8")
        if prefix:
            name = prefix + "/" + name
        raw_size = header[124:136].split(b"\0", 1)[0].strip() or b"0"
        size = int(raw_size, 8)
        typeflag = header[156:157].decode("ascii", errors="replace")
        data_start = cursor + 512
        members.append(
            {
                "name": name,
                "size": size,
                "typeflag": typeflag,
                "header_offset": cursor,
                "data_start": data_start,
                "data_end": data_start + size - 1,
            }
        )
        cursor = data_start + ((size + 511) // 512) * 512
    return total, members


def extract(url: str, member: dict, output: Path, chunk_size: int = 1024 * 1024) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    cursor = int(member["data_start"])
    end = int(member["data_end"])
    with temporary.open("wb") as handle:
        while cursor <= end:
            block_end = min(end, cursor + int(chunk_size) - 1)
            handle.write(get_range(url, cursor, block_end))
            cursor = block_end + 1
        handle.flush()
        os.fsync(handle.fileno())
    if temporary.stat().st_size != int(member["size"]):
        raise RuntimeError("extracted member size mismatch")
    os.replace(str(temporary), str(output))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--extract-regex")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    total, members = tar_index(args.url)
    extracted = []
    if args.extract_regex:
        if not args.output_dir:
            raise ValueError("--output-dir is required with --extract-regex")
        pattern = re.compile(args.extract_regex)
        for member in members:
            if pattern.search(member["name"]):
                output = Path(args.output_dir) / Path(member["name"]).name
                extract(args.url, member, output)
                extracted.append(str(output))
    audit = {
        "url": args.url,
        "content_length": total,
        "members": members,
        "extracted": extracted,
    }
    path = Path(args.audit)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(audit, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


if __name__ == "__main__":
    main()
