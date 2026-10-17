#!/usr/bin/env python3
"""Restore competition training data from user-supplied official ZIPs.

Only known training filenames are extracted. Test data are opt-in. UTF-8/GBK
archive wrapper names are ignored; paths inside the destination are fixed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def restore(archive, kind, out, include_test=False):
    count = 0
    with zipfile.ZipFile(archive) as z:
        for member in z.infolist():
            if member.is_dir():
                continue
            parts = PurePosixPath(member.filename).parts
            if ".." in parts or PurePosixPath(member.filename).is_absolute():
                raise ValueError("Unsafe ZIP path")
            name = parts[-1]
            rel = None
            allowed = {f"{kind}_sample_data.csv"}
            if include_test:
                allowed |= {f"{kind}_test.csv", f"{kind}_submission_example.csv"}
            if name in allowed:
                rel = Path(name)
            elif kind == "image" and "image_sample_data" in parts:
                rel = Path("data/image_sample_data") / name
            elif kind == "image" and include_test and "image_test" in parts:
                rel = Path("data/image_test") / name
            if rel is None:
                continue
            target = out / kind / rel
            if not target.resolve().is_relative_to(out.resolve()):
                raise ValueError("Destination escapes root")
            data = z.read(member)  # zipfile verifies CRC before writing.
            if target.exists():
                if target.read_bytes() != data:
                    raise RuntimeError(f"Refusing to replace different file: {target}")
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(data)
            count += 1
    return {"archive": archive.name, "kind": kind, "restored_files": count}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archives", type=Path, default=ROOT / ".local/legacy")
    p.add_argument("--output", type=Path, default=ROOT / ".local/legacy/data/competition")
    p.add_argument("--include-test", action="store_true")
    a = p.parse_args()
    for filename, kind in (("初赛图片检测数据集.zip", "image"), ("初赛文本检测数据集.zip", "text")):
        print(json.dumps(restore(a.archives / filename, kind, a.output, a.include_test), ensure_ascii=False))


if __name__ == "__main__":
    main()
