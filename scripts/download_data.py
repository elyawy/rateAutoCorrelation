#!/usr/bin/env python3
"""Download and verify the files listed in data/MANIFEST.csv."""

from __future__ import annotations

import csv
import hashlib
import os
import tempfile
from pathlib import Path
from urllib.request import urlopen

import config


def digest(path: Path) -> str:
    sha = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", dir=destination.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with urlopen(url) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    manifest_path = Path(config.MANIFEST_PATH)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    if not config.RECORD_ID:
        raise RuntimeError("Set config.RECORD_ID before downloading Zenodo data")

    with manifest_path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    for row in rows:
        relative_path = Path(row["path"])
        destination = Path(config.DATA_DIR) / relative_path
        url = row.get("source_url") or (
            f"https://zenodo.org/records/{config.RECORD_ID}/files/{relative_path.name}"
        )
        expected_hash = row["sha256"]
        if destination.exists() and digest(destination) == expected_hash:
            print(f"verified {destination}")
            continue
        print(f"downloading {url} -> {destination}")
        download(url, destination)
        actual_hash = digest(destination)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {destination}: expected {expected_hash}, got {actual_hash}"
            )
        if row.get("size_bytes") and destination.stat().st_size != int(row["size_bytes"]):
            raise RuntimeError(f"Size mismatch for {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
