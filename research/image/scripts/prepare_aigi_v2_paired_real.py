#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path

import pandas as pd
from PIL import Image
import pyarrow.parquet as pq

from prepare_aigi_v2_transfer import REPOSITORY, download, fetch_tree


def extract_targets(
    shard: Path,
    source_name: str,
    output_root: Path,
    targets: set[str],
    seen_ids: set[str],
    competition_hashes: set[str],
    rows: list[dict[str, object]],
) -> int:
    table = pq.read_table(
        shard,
        columns=["image_id", "source_real_id", "label", "generator", "image", "sha256"],
    )
    values = table.to_pydict()
    excluded = 0
    for index in range(table.num_rows):
        image_id = str(values["image_id"][index])
        if image_id not in targets or image_id in seen_ids:
            continue
        if int(values["label"][index]) != 0 or str(values["generator"][index]) != "real":
            raise RuntimeError(f"invalid paired-real row in {source_name}:{index}")
        image_bytes = values["image"][index]
        expected_sha256 = str(values["sha256"][index])
        actual_sha256 = hashlib.sha256(image_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"image SHA-256 mismatch in {source_name}:{index}")
        if actual_sha256 in competition_hashes:
            excluded += 1
            continue
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
            image_format = str(image.format or "JPEG").lower()
        suffix = ".jpg" if image_format in {"jpeg", "jpg"} else f".{image_format}"
        relative_path = Path("aigi_v2_paired_real") / f"{image_id}{suffix}"
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)
        rows.append(
            {
                "sample_key": f"aigi_v2_paired_{image_id}",
                "relative_path": relative_path.as_posix(),
                "label": 0,
                "model_name": "AIGIv2_paired_real",
                "architecture": "real",
                "subset": "modern_paired_real",
                "source_shard": source_name,
                "source_row": index,
                "source_sha256": actual_sha256,
                "source_real_id": str(values["source_real_id"][index]),
            }
        )
        seen_ids.add(image_id)
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    args = parser.parse_args()

    root = args.project_root.resolve()
    external_root = root / "data" / "external"
    modern_manifest_path = external_root / "aigi_v2_modern_manifest.csv"
    previous_manifest_path = external_root / "external_transfer_v2_manifest.csv"
    folds_path = root / "data" / "folds" / "image_folds_v1.csv"
    cache_dir = root / ".cache" / "aigi_v2_shards"
    if not all(path.is_file() for path in (modern_manifest_path, previous_manifest_path, folds_path)):
        raise FileNotFoundError("modern fake manifest, previous combined manifest, or folds are missing")

    modern = pd.read_csv(modern_manifest_path)
    targets = set(modern.source_real_id.astype(str))
    competition_hashes = set(pd.read_csv(folds_path, usecols=["sha256"]).sha256.astype(str))
    seen_ids: set[str] = set()
    rows: list[dict[str, object]] = []
    excluded = 0
    processed_names: set[str] = set()

    cache_dir.mkdir(parents=True, exist_ok=True)
    for shard in sorted(cache_dir.glob("real-*.parquet")):
        excluded += extract_targets(
            shard,
            shard.name,
            external_root,
            targets,
            seen_ids,
            competition_hashes,
            rows,
        )
        processed_names.add(shard.name)
        shard.unlink()

    tree_url = (
        f"{args.mirror}/api/datasets/{REPOSITORY}/tree/main/real"
        "?recursive=false&expand=true&limit=100"
    )
    entries = fetch_tree(tree_url, args.mirror)
    candidates = sorted(
        str(entry["path"])
        for entry in entries
        if str(entry.get("path", "")).endswith(".parquet")
    )
    for source_path in candidates:
        source_name = Path(source_path).name
        if source_name in processed_names:
            continue
        active_shard = cache_dir / "active_real.parquet"
        url = f"{args.mirror}/datasets/{REPOSITORY}/resolve/main/{source_path}"
        download(url, active_shard)
        try:
            excluded += extract_targets(
                active_shard,
                source_name,
                external_root,
                targets,
                seen_ids,
                competition_hashes,
                rows,
            )
        finally:
            active_shard.unlink(missing_ok=True)
        print(f"paired real: {len(seen_ids)}/{len(targets)}", flush=True)

    missing = sorted(targets - seen_ids)
    if missing:
        raise RuntimeError(f"missing {len(missing)} paired real images: {missing[:5]}")

    paired_manifest = pd.DataFrame(rows).sort_values("sample_key")
    paired_manifest_path = external_root / "aigi_v2_paired_real_manifest.csv"
    paired_manifest.to_csv(paired_manifest_path, index=False)
    previous = pd.read_csv(previous_manifest_path)
    combined = pd.concat([previous, paired_manifest], ignore_index=True, sort=False)
    combined_path = external_root / "external_transfer_v3_manifest.csv"
    combined.to_csv(combined_path, index=False)
    audit = {
        "source": REPOSITORY,
        "modern_fake_rows": int(len(modern)),
        "unique_requested_source_real_ids": len(targets),
        "paired_real_rows": int(len(paired_manifest)),
        "combined_rows": int(len(combined)),
        "competition_rows_used_for_training": 0,
        "competition_hashes_checked_for_exact_deduplication": len(competition_hashes),
        "exact_competition_duplicates_excluded": excluded,
        "unmatched_source_real_ids": 0,
    }
    (external_root / "external_transfer_v3_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
