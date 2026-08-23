#!/usr/bin/env python3
"""Extract one registered member from a large remote ZIP via HTTP ranges."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import urllib.request
import zlib
from pathlib import Path
from time import sleep


def get_range(url: str, start: int, end: int) -> bytes:
    request = urllib.request.Request(
        url, headers={"Range": "bytes=%d-%d" % (int(start), int(end))}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        data = response.read()
        if response.status != 206:
            raise RuntimeError("server did not honor HTTP range")
    expected = int(end) - int(start) + 1
    if len(data) != expected:
        raise RuntimeError("range length mismatch")
    return data


def get_range_chunked(url: str, start: int, end: int, chunk_size: int = 4 * 1024 * 1024) -> bytes:
    blocks = []
    cursor = int(start)
    while cursor <= int(end):
        block_end = min(int(end), cursor + int(chunk_size) - 1)
        error = None
        for attempt in range(5):
            try:
                blocks.append(get_range(url, cursor, block_end))
                error = None
                break
            except Exception as caught:  # network retry, exact byte range unchanged
                error = caught
                sleep(2 ** attempt)
        if error is not None:
            raise error
        cursor = block_end + 1
    return b"".join(blocks)


def central_directory(url: str):
    request = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(request, timeout=60) as response:
        headers = dict(response.headers.items())
        size = int(response.headers["Content-Length"])
    tail_start = max(0, size - 131072)
    tail = get_range(url, tail_start, size - 1)
    offset = tail.rfind(b"PK\x05\x06")
    if offset < 0:
        raise RuntimeError("ZIP end-of-central-directory record not found")
    record = struct.unpack_from("<4s4H2LH", tail, offset)
    entries, directory_size, directory_offset = record[4], record[5], record[6]
    directory = get_range(
        url, int(directory_offset), int(directory_offset + directory_size - 1)
    )
    result = {}
    cursor = 0
    while cursor + 46 <= len(directory):
        if directory[cursor : cursor + 4] != b"PK\x01\x02":
            raise RuntimeError("invalid central-directory signature")
        values = struct.unpack_from("<4s6H3L5H2L", directory, cursor)
        method = values[4]
        crc32 = values[7]
        compressed_size = values[8]
        uncompressed_size = values[9]
        name_length, extra_length, comment_length = values[10:13]
        local_offset = values[16]
        name = directory[
            cursor + 46 : cursor + 46 + name_length
        ].decode("utf-8")
        result[name] = {
            "compression_method": int(method),
            "crc32": int(crc32),
            "compressed_size": int(compressed_size),
            "uncompressed_size": int(uncompressed_size),
            "local_header_offset": int(local_offset),
        }
        cursor += 46 + name_length + extra_length + comment_length
    if len(result) != int(entries):
        raise RuntimeError("central-directory entry count mismatch")
    return result, {
        "content_length": size,
        "response_headers": headers,
        "central_directory_entries": int(entries),
        "central_directory_size": int(directory_size),
        "central_directory_offset": int(directory_offset),
    }


def extract(url: str, member: str, output: Path, audit: Path) -> None:
    entries, root_audit = central_directory(url)
    if member not in entries:
        raise KeyError(member)
    item = entries[member]
    local_offset = int(item["local_header_offset"])
    header = get_range(url, local_offset, local_offset + 29)
    if header[:4] != b"PK\x03\x04":
        raise RuntimeError("invalid local-header signature")
    values = struct.unpack("<4s5H3L2H", header)
    name_length, extra_length = values[9], values[10]
    data_start = local_offset + 30 + name_length + extra_length
    compressed = get_range_chunked(
        url, data_start, data_start + int(item["compressed_size"]) - 1
    )
    method = int(item["compression_method"])
    if method == 0:
        value = compressed
    elif method == 8:
        value = zlib.decompress(compressed, -15)
    else:
        raise RuntimeError("unsupported ZIP compression method")
    if len(value) != int(item["uncompressed_size"]):
        raise RuntimeError("uncompressed size mismatch")
    actual_crc = zlib.crc32(value) & 0xFFFFFFFF
    if actual_crc != int(item["crc32"]):
        raise RuntimeError("member CRC mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".part")
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(output))
    result = {
        **root_audit,
        "url": url,
        "member": member,
        "member_metadata": item,
        "downloaded_range": [
            int(data_start),
            int(data_start + int(item["compressed_size"]) - 1),
        ],
        "output_path": str(output),
        "output_size": output.stat().st_size,
        "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "output_crc32": actual_crc,
    }
    audit.parent.mkdir(parents=True, exist_ok=True)
    temporary_audit = audit.with_name(audit.name + ".tmp")
    with temporary_audit.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary_audit), str(audit))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--member", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit", required=True)
    args = parser.parse_args()
    extract(args.url, args.member, Path(args.output), Path(args.audit))


if __name__ == "__main__":
    main()
