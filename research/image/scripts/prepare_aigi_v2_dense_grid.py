#!/usr/bin/env python3
"""Complete the locally paired AIGI-v2 real/fake generator grid.

Only fake images whose exact ``source_real_id`` already has a local real image are
eligible.  Existing V4 rows are never redownloaded, competition images are used
only for exact-hash exclusion, and downloaded parquet shards are deleted as soon
as they have been inspected.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from PIL import Image
import pyarrow.parquet as pq

from prepare_aigi_v2_paired_extension import (
    GENERATORS,
    count_between,
    parquet_source_range,
)
from prepare_aigi_v2_transfer import REPOSITORY, download, fetch_tree


def extract_dense_shard(
    shard: Path,
    source_name: str,
    generator: str,
    output_root: Path,
    requested_targets: set[str],
    competition_hashes: set[str],
    existing_hashes: set[str],
    rows: list[dict[str, object]],
) -> tuple[int, int]:
    table = pq.read_table(
        shard,
        columns=["image_id", "source_real_id", "label", "generator", "image", "sha256"],
    )
    values = table.to_pydict()
    added = 0
    duplicate_targets = 0
    for index in range(table.num_rows):
        source_real_id = str(values["source_real_id"][index])
        if source_real_id not in requested_targets:
            continue
        if int(values["label"][index]) != 1 or str(values["generator"][index]) != generator:
            raise RuntimeError(f"invalid fake row in {source_name}:{index}")
        image_bytes = values["image"][index]
        expected_sha256 = str(values["sha256"][index])
        actual_sha256 = hashlib.sha256(image_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"image SHA-256 mismatch in {source_name}:{index}")

        # A duplicate contributes no new evidence.  Mark its target covered so a
        # full-grid run does not fail merely because two generators emitted the
        # exact same bytes.
        if actual_sha256 in competition_hashes or actual_sha256 in existing_hashes:
            requested_targets.remove(source_real_id)
            duplicate_targets += 1
            continue

        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
            image_format = str(image.format or "PNG").lower()
        suffix = ".jpg" if image_format in {"jpeg", "jpg"} else f".{image_format}"
        image_id = str(values["image_id"][index])
        relative_path = Path("aigi_v2_dense_grid") / generator / f"{image_id}{suffix}"
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)
        rows.append(
            {
                "sample_key": f"aigi_v2_dense_{generator}_{image_id}",
                "relative_path": relative_path.as_posix(),
                "label": 1,
                "model_name": f"AIGIv2_{generator}",
                "architecture": generator,
                "subset": "paired_generator_dense_grid",
                "source_shard": source_name,
                "source_row": index,
                "source_sha256": actual_sha256,
                "source_real_id": source_real_id,
            }
        )
        requested_targets.remove(source_real_id)
        existing_hashes.add(actual_sha256)
        added += 1
    return added, duplicate_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--max-new-per-generator",
        type=int,
        default=0,
        help="0 completes every missing pair; a positive value caps each generator.",
    )
    parser.add_argument("--metadata-workers", type=int, default=16)
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    args = parser.parse_args()

    root = args.project_root.resolve()
    external_root = root / "data" / "external"
    paired_path = external_root / "aigi_v2_paired_real_manifest.csv"
    previous_path = external_root / "external_transfer_v4_manifest.csv"
    folds_path = root / "data" / "folds" / "image_folds_v1.csv"
    required = (paired_path, previous_path, folds_path)
    if not all(path.is_file() for path in required):
        raise FileNotFoundError("V4 external manifest, paired reals, or fixed folds are missing")

    paired = pd.read_csv(paired_path)
    previous = pd.read_csv(previous_path)
    paired_targets = set(paired.source_real_id.astype(str))
    competition_hashes = set(pd.read_csv(folds_path, usecols=["sha256"]).sha256.astype(str))
    existing_hashes = set(previous.source_sha256.dropna().astype(str))
    cache_dir = root / ".cache" / "aigi_v2_shards"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    audit_generators: dict[str, object] = {}
    total_duplicate_targets = 0
    for generator in GENERATORS:
        model_name = f"AIGIv2_{generator}"
        used_targets = set(
            previous.loc[previous.model_name == model_name, "source_real_id"]
            .dropna()
            .astype(str)
        )
        missing_targets = sorted(paired_targets - used_targets)
        if args.max_new_per_generator > 0:
            missing_targets = missing_targets[: args.max_new_per_generator]
        requested_targets = set(missing_targets)
        requested_count = len(requested_targets)
        if requested_count == 0:
            audit_generators[generator] = {
                "requested_targets": 0,
                "rows": 0,
                "exact_duplicate_targets": 0,
                "preexisting_targets": len(used_targets),
                "downloaded_shards": 0,
                "downloaded_bytes": 0,
            }
            continue

        quoted_path = urllib.parse.quote(f"data/{generator}", safe="/")
        tree_url = (
            f"{args.mirror}/api/datasets/{REPOSITORY}/tree/main/{quoted_path}"
            "?recursive=false&expand=true&limit=100"
        )
        entries = fetch_tree(tree_url, args.mirror)
        source_paths = sorted(
            str(entry["path"])
            for entry in entries
            if str(entry.get("path", "")).endswith(".parquet")
        )
        sorted_targets = sorted(requested_targets)

        def inspect_source(source_path: str) -> tuple[int, str]:
            encoded = urllib.parse.quote(source_path, safe="/")
            url = f"{args.mirror}/datasets/{REPOSITORY}/resolve/main/{encoded}"
            low, high, _ = parquet_source_range(url)
            return count_between(sorted_targets, low, high), source_path

        with ThreadPoolExecutor(max_workers=args.metadata_workers) as executor:
            inspected = list(executor.map(inspect_source, source_paths))
        candidates = [item for item in inspected if item[0] > 0]
        candidates.sort(key=lambda item: (-item[0], item[1]))

        added = 0
        duplicate_targets = 0
        downloaded_shards = 0
        downloaded_bytes = 0
        for _, source_path in candidates:
            if not requested_targets:
                break
            active_shard = cache_dir / f"active_dense_{generator}.parquet"
            encoded = urllib.parse.quote(source_path, safe="/")
            url = f"{args.mirror}/datasets/{REPOSITORY}/resolve/main/{encoded}"
            download(url, active_shard)
            downloaded_shards += 1
            downloaded_bytes += active_shard.stat().st_size
            try:
                shard_added, shard_duplicates = extract_dense_shard(
                    active_shard,
                    Path(source_path).name,
                    generator,
                    external_root,
                    requested_targets,
                    competition_hashes,
                    existing_hashes,
                    rows,
                )
                added += shard_added
                duplicate_targets += shard_duplicates
            finally:
                active_shard.unlink(missing_ok=True)
            print(
                f"{generator}: covered {added + duplicate_targets}/{requested_count} "
                f"({added} retained) from {downloaded_shards} shards",
                flush=True,
            )
        if requested_targets:
            examples = sorted(requested_targets)[:5]
            raise RuntimeError(
                f"missing {len(requested_targets)} requested targets for {generator}; "
                f"examples={examples}"
            )
        total_duplicate_targets += duplicate_targets
        audit_generators[generator] = {
            "requested_targets": requested_count,
            "rows": added,
            "exact_duplicate_targets": duplicate_targets,
            "preexisting_targets": len(used_targets),
            "metadata_candidates": len(candidates),
            "downloaded_shards": downloaded_shards,
            "downloaded_bytes": downloaded_bytes,
        }

    dense = pd.DataFrame(rows).sort_values(["model_name", "sample_key"])
    dense_path = external_root / "aigi_v2_dense_grid_manifest.csv"
    dense.to_csv(dense_path, index=False)
    combined = pd.concat([previous, dense], ignore_index=True, sort=False)
    combined_path = external_root / "external_transfer_v5_manifest.csv"
    combined.to_csv(combined_path, index=False)
    audit = {
        "source": REPOSITORY,
        "policy": "complete missing fake-generator partners only for exact local paired reals",
        "max_new_per_generator": args.max_new_per_generator,
        "paired_real_pool": len(paired_targets),
        "generators": audit_generators,
        "new_rows": int(len(dense)),
        "combined_rows": int(len(combined)),
        "competition_rows_used_for_training": 0,
        "competition_hashes_checked_for_exact_deduplication": len(competition_hashes),
        "exact_duplicate_targets_excluded": total_duplicate_targets,
        "new_real_rows": 0,
        "active_shards_remaining": len(list(cache_dir.glob("active_dense_*.parquet"))),
    }
    (external_root / "external_transfer_v5_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
