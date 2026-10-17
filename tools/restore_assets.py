#!/usr/bin/env python3
"""Recover only model assets from the two verified docker-save archives.

No Docker daemon, network, shell extraction, pickle loading or GPU is needed.
Never extract an arbitrary tar path or overwrite a different local asset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVES = {
    "image": ("pazhou_aigc_image_v696_linux_amd64.tar", "d9a95261a5c9d3f7344ecf7b2b2099c3a9c888c3db32d0a8bfb5166c86bac35b", "opt/v178/assets/"),
    "text": ("pazhou_aigc_text_tv388_linux_amd64.tar", "39eb6a79e9610ee4401b58c250bc24bc380e6529bc54543ffabc21e718a5afff", "opt/tv388/assets/"),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_relative(name, prefix):
    name = name.removeprefix("./")
    if not name.startswith(prefix):
        return None
    rel = PurePosixPath(name[len(prefix):])
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        raise ValueError(f"Unsafe asset name: {name}")
    return Path(*rel.parts)


def restore(track, archives, destination):
    filename, expected, prefix = ARCHIVES[track]
    archive = archives / filename
    print(f"Verifying {archive.name}", flush=True)
    if digest(archive) != expected:
        raise RuntimeError(f"Archive SHA-256 mismatch: {archive}")
    out = destination / track
    records = {}
    with tarfile.open(archive, "r:") as outer:
        manifests = json.load(outer.extractfile("manifest.json"))
        if len(manifests) != 1:
            raise RuntimeError("Expected one image in the Docker archive")
        for layer in manifests[0]["Layers"]:
            with tarfile.open(fileobj=outer.extractfile(layer), mode="r:") as inner:
                for member in inner:
                    rel = safe_relative(member.name, prefix)
                    if rel is None or member.isdir():
                        continue
                    if not member.isfile():
                        raise RuntimeError(f"Non-regular asset: {member.name}")
                    target = out / rel
                    if not target.resolve().is_relative_to(out.resolve()):
                        raise RuntimeError("Asset escapes destination")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    temp = target.with_name(target.name + ".partial")
                    h = hashlib.sha256()
                    written = 0
                    with inner.extractfile(member) as src, temp.open("wb") as dst:
                        for block in iter(lambda: src.read(8 * 1024 * 1024), b""):
                            h.update(block)
                            dst.write(block)
                            written += len(block)
                    if written != member.size:
                        temp.unlink()
                        raise RuntimeError(f"Short read: {member.name}")
                    if target.exists():
                        if digest(target) != h.hexdigest():
                            temp.unlink()
                            raise RuntimeError(f"Refusing to replace different asset: {target}")
                        temp.unlink()
                    else:
                        os.replace(temp, target)
                    records[rel.as_posix()] = {"bytes": written, "sha256": h.hexdigest()}
                    print(f"  restored {track}/{rel} ({written:,} bytes)", flush=True)
    if not records:
        raise RuntimeError(f"No model assets found for {track}")
    report = {"track": track, "archive": filename, "archive_sha256": expected, "files": records}
    (out / "asset_inventory.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--track", choices=["all", *ARCHIVES], default="all")
    parser.add_argument("--archives", type=Path, default=ROOT / "outputs/docker")
    parser.add_argument("--destination", type=Path, default=ROOT / ".local/checkpoints")
    args = parser.parse_args()
    tracks = ARCHIVES if args.track == "all" else [args.track]
    for track in tracks:
        restore(track, args.archives, args.destination)


if __name__ == "__main__":
    main()
