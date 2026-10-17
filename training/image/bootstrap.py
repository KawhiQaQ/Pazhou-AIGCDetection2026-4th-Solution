#!/usr/bin/env python3
"""Prepare portable input links and recover exact ancestor manifests without training.

Never overwrites different files. Relative links remain portable when the entire
repository (including .local private assets) is copied to a new server.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def write_csv(path, fields, rows):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    value = stream.getvalue().encode()
    if path.exists():
        with path.open(newline="") as existing:
            old = list(csv.DictReader(existing))
        if old != rows:
            raise RuntimeError(f"Refusing to overwrite different manifest: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def recover(source, target):
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    memberships = {
        "community_forensics_transfer_v1": 20944, "aigi_v2_modern": 2500,
        "aigi_v2_paired_real": 2322, "aigi_v2_paired_extension": 1800,
        "aigi_v2_breadth_fake": 6000, "aigi_v2_breadth_real": 6000,
    }
    groups = {key: [r for r in rows if r["relative_path"].split("/")[0] == key] for key in memberships}
    if any(len(groups[k]) != n for k, n in memberships.items()) or sum(memberships.values()) != len(rows):
        raise ValueError("V6 manifest ancestry counts differ from the preserved historical selection")
    cf = [dict(r, relative_path=r["relative_path"].split("/", 1)[1]) for r in groups["community_forensics_transfer_v1"]]
    write_csv(target / "community_forensics_transfer_v1/manifest.csv", fields, cf)
    allowed = {"community_forensics_transfer_v1", "aigi_v2_modern"}
    for version, extra in [(2, None), (3, "aigi_v2_paired_real"), (4, "aigi_v2_paired_extension")]:
        if extra:
            allowed.add(extra)
        selected = [r for r in rows if r["relative_path"].split("/")[0] in allowed]
        write_csv(target / f"external_transfer_v{version}_manifest.csv", fields, selected)
    for key in ["aigi_v2_modern", "aigi_v2_paired_real", "aigi_v2_paired_extension"]:
        write_csv(target / f"{key}_manifest.csv", fields, groups[key])
    return {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(), "rows": len(rows),
            "counts": memberships, "method": "preserved ordered row membership; no resampling", "historical_byte_identity_claimed": False}


def link(source, target):
    if not source.exists():
        return False
    if target.exists() or target.is_symlink():
        if target.resolve() != source.resolve():
            if target.is_file() and source.is_file() and hashlib.sha256(target.read_bytes()).digest() == hashlib.sha256(source.read_bytes()).digest():
                return True
            raise RuntimeError(f"Refusing to replace existing input: {target}")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(os.path.relpath(source.resolve(), target.parent.resolve()), target_is_directory=source.is_dir())
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=ROOT / ".local/legacy/data")
    p.add_argument("--backbones-root", type=Path, default=ROOT / ".local/checkpoints/image/backbones")
    p.add_argument("--work-dir", type=Path, default=ROOT / "runs/image_training")
    p.add_argument("--selection-root", type=Path, default=ROOT / "research/image/manifests")
    a = p.parse_args()
    work, data, weights = a.work_dir.resolve(), a.data_root.resolve(), a.backbones_root.resolve()
    if work == ROOT or any(work == q or work.is_relative_to(q) for q in [ROOT / ".local", ROOT / "research", ROOT / "outputs", data, weights]):
        raise ValueError("Use a separate training work directory, never a data/checkpoint/source directory")
    paths = {"pe_core_g14.safetensors": "pe_core_g14/model.safetensors", "pe_spatial_g14.safetensors": "pe_spatial_g14/model.safetensors",
             "clip/ViT-L-14.pt": "clip/ViT-L-14.pt", "clip/ViT-L-14-336px.pt": "clip/ViT-L-14-336px.pt",
             "rine.pth": "rine/model_ldm_trainable.pth", "npr.pth": "npr/model_epoch_last_3090.pth", "deepfake_vit": "deepfake_vit"}
    linked = [destination for source, destination in paths.items() if link(weights / source, work / ".cache" / destination)]
    link(data / "competition/image", work / "data/competition/image")
    link(data / "folds/image_folds_v1.csv", work / "data/folds/image_folds_v1.csv")
    external = work / "data/external"
    selection = a.selection_root.resolve()
    report = recover(selection / "external_transfer_v6_manifest.csv", external)
    for name in ["aigi_v2_breadth_pairs_manifest.csv", "openfake_frontier_api_manifest.csv", "external_transfer_v5_manifest.csv", "external_transfer_v6_manifest.csv", "external_transfer_v7_manifest.csv"]:
        source = selection / name if (selection / name).is_file() else data / "external" / name
        link(source, external / name)
    # External download targets must be real directories, not links into the
    # preserved archive. Remove only identical links created by an earlier
    # bootstrap version; never remove their source directories or contents.
    for folder in ((data / "external").iterdir() if (data / "external").is_dir() else []):
        if not folder.is_dir():
            continue
        target = external / folder.name
        if target.is_symlink():
            if target.resolve() != folder.resolve():
                raise RuntimeError(f"Unexpected external-data link: {target}")
            target.unlink()
        target.mkdir(parents=True, exist_ok=True)
    report["linked_backbones"] = linked
    report["additional_download_required"] = ["missing manifest-referenced external image bytes", ".cache/bombek_dual/pytorch_model.pt (full public model)"]
    work.mkdir(parents=True, exist_ok=True)
    if not (work / "data/folds/image_folds_v1.csv").is_file() and (work / "data/competition/image/image_sample_data.csv").is_file():
        subprocess.run([sys.executable, str(ROOT / "research/image/scripts/build_folds.py"), "--project-root", str(work), "--task", "image"], check=True)
    (work / "input-bootstrap.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
