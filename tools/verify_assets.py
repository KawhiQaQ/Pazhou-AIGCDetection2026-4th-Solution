#!/usr/bin/env python3
"""Verify recovered model bytes without loading pickle files or requiring CUDA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct

from restore_assets import ROOT, digest


def verify(root):
    report = json.loads((root / "asset_inventory.json").read_text())
    total = 0
    for rel, expected in report["files"].items():
        path = root / rel
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError("Unsafe manifest path")
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise RuntimeError(f"Missing/truncated asset: {path}")
        if digest(path) != expected["sha256"]:
            raise RuntimeError(f"Asset hash mismatch: {path}")
        if path.suffix == ".safetensors":
            with path.open("rb") as f:
                size = struct.unpack("<Q", f.read(8))[0]
                if size > path.stat().st_size - 8:
                    raise RuntimeError(f"Invalid safetensors header: {path}")
                header = json.loads(f.read(size))
            data_size = path.stat().st_size - size - 8
            for key, value in header.items():
                if key == "__metadata__":
                    continue
                lo, hi = value["data_offsets"]
                if not 0 <= lo <= hi <= data_size:
                    raise RuntimeError(f"Invalid tensor offsets: {path} {key}")
        total += expected["bytes"]
    # The image backbone's original independently recorded per-file hashes.
    original = json.loads((root / "manifest.json").read_text())
    for record in original.get("files", []):
        actual = report["files"][record["path"]]
        if actual["sha256"] != record["sha256"] or actual["bytes"] != record["bytes"]:
            raise RuntimeError(f"Original asset manifest mismatch: {record['path']}")
    return {"track": report["track"], "files": len(report["files"]), "bytes": total, "hashes_verified": True, "cuda_inference_tested": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, default=ROOT / ".local/checkpoints")
    args = parser.parse_args()
    for track in ("image", "text"):
        print(json.dumps(verify(args.checkpoints / track)))


if __name__ == "__main__":
    main()
