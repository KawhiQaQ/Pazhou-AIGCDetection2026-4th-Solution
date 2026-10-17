#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import hashlib
import json
from pathlib import Path
import urllib.parse

import fsspec
import pandas as pd
from PIL import Image
import pyarrow.parquet as pq

from prepare_aigi_v2_transfer import REPOSITORY, download, fetch_tree


GENERATORS = (
    "sd15",
    "sdxl",
    "flux_schnell",
    "kandinsky22",
    "pixart_sigma",
    "wuerstchen",
)


def parquet_source_range(url: str) -> tuple[str, str, int]:
    with fsspec.open(url, "rb", block_size=64 * 1024) as handle:
        parquet = pq.ParquetFile(handle)
        column_index = parquet.schema.names.index("source_real_id")
        minimum: str | None = None
        maximum: str | None = None
        rows = 0
        for index in range(parquet.metadata.num_row_groups):
            group = parquet.metadata.row_group(index)
            stats = group.column(column_index).statistics
            if stats is None or not stats.has_min_max:
                raise RuntimeError(f"missing source_real_id statistics: {url}")
            low = stats.min.decode() if isinstance(stats.min, bytes) else str(stats.min)
            high = stats.max.decode() if isinstance(stats.max, bytes) else str(stats.max)
            minimum = low if minimum is None else min(minimum, low)
            maximum = high if maximum is None else max(maximum, high)
            rows += group.num_rows
    if minimum is None or maximum is None:
        raise RuntimeError(f"empty parquet metadata: {url}")
    return minimum, maximum, rows


def count_between(sorted_values: list[str], low: str, high: str) -> int:
    return bisect.bisect_right(sorted_values, high) - bisect.bisect_left(
        sorted_values, low
    )


def extract_shard(
    shard: Path,
    source_name: str,
    generator: str,
    output_root: Path,
    available_targets: set[str],
    competition_hashes: set[str],
    existing_hashes: set[str],
    rows: list[dict[str, object]],
    remaining: int,
) -> tuple[int, int]:
    table = pq.read_table(
        shard,
        columns=["image_id", "source_real_id", "label", "generator", "image", "sha256"],
    )
    values = table.to_pydict()
    excluded = 0
    added = 0
    for index in range(table.num_rows):
        if added >= remaining:
            break
        source_real_id = str(values["source_real_id"][index])
        if source_real_id not in available_targets:
            continue
        if int(values["label"][index]) != 1 or str(values["generator"][index]) != generator:
            raise RuntimeError(f"invalid fake row in {source_name}:{index}")
        image_bytes = values["image"][index]
        expected_sha256 = str(values["sha256"][index])
        actual_sha256 = hashlib.sha256(image_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"image SHA-256 mismatch in {source_name}:{index}")
        if actual_sha256 in competition_hashes or actual_sha256 in existing_hashes:
            excluded += 1
            continue
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
            image_format = str(image.format or "PNG").lower()
        suffix = ".jpg" if image_format in {"jpeg", "jpg"} else f".{image_format}"
        image_id = str(values["image_id"][index])
        relative_path = Path("aigi_v2_paired_extension") / generator / f"{image_id}{suffix}"
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)
        rows.append(
            {
                "sample_key": f"aigi_v2_extension_{generator}_{image_id}",
                "relative_path": relative_path.as_posix(),
                "label": 1,
                "model_name": f"AIGIv2_{generator}",
                "architecture": generator,
                "subset": "paired_generator_extension",
                "source_shard": source_name,
                "source_row": index,
                "source_sha256": actual_sha256,
                "source_real_id": source_real_id,
            }
        )
        available_targets.remove(source_real_id)
        existing_hashes.add(actual_sha256)
        added += 1
    return added, excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target-per-generator", type=int, default=300)
    parser.add_argument("--metadata-workers", type=int, default=16)
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    args = parser.parse_args()

    root = args.project_root.resolve()
    external_root = root / "data" / "external"
    modern_path = external_root / "aigi_v2_modern_manifest.csv"
    paired_path = external_root / "aigi_v2_paired_real_manifest.csv"
    previous_path = external_root / "external_transfer_v3_manifest.csv"
    folds_path = root / "data" / "folds" / "image_folds_v1.csv"
    if not all(path.is_file() for path in (modern_path, paired_path, previous_path, folds_path)):
        raise FileNotFoundError("V3 external manifests or fixed competition folds are missing")

    modern = pd.read_csv(modern_path)
    paired = pd.read_csv(paired_path)
    previous = pd.read_csv(previous_path)
    paired_targets = set(paired.source_real_id.astype(str))
    competition_hashes = set(pd.read_csv(folds_path, usecols=["sha256"]).sha256.astype(str))
    existing_hashes = set(previous.source_sha256.dropna().astype(str))
    cache_dir = root / ".cache" / "aigi_v2_shards"
    cache_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    audit_generators: dict[str, object] = {}
    excluded = 0
    for generator in GENERATORS:
        model_name = f"AIGIv2_{generator}"
        used_targets = set(
            modern.loc[modern.model_name == model_name, "source_real_id"].astype(str)
        )
        available_targets = paired_targets - used_targets
        if len(available_targets) < args.target_per_generator:
            raise RuntimeError(
                f"insufficient paired targets for {generator}: {len(available_targets)}"
            )
        sorted_targets = sorted(available_targets)
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
        def inspect_source(source_path: str) -> tuple[int, str, int]:
            encoded = urllib.parse.quote(source_path, safe="/")
            url = f"{args.mirror}/datasets/{REPOSITORY}/resolve/main/{encoded}"
            low, high, shard_rows = parquet_source_range(url)
            matches = count_between(sorted_targets, low, high)
            return matches, source_path, shard_rows

        with ThreadPoolExecutor(max_workers=args.metadata_workers) as executor:
            inspected = list(executor.map(inspect_source, source_paths))
        candidates = [item for item in inspected if item[0] > 0]
        candidates.sort(key=lambda item: (-item[0], item[1]))

        added = 0
        downloaded_shards = 0
        downloaded_bytes = 0
        for _, source_path, _ in candidates:
            if added >= args.target_per_generator:
                break
            active_shard = cache_dir / "active_extension.parquet"
            encoded = urllib.parse.quote(source_path, safe="/")
            url = f"{args.mirror}/datasets/{REPOSITORY}/resolve/main/{encoded}"
            download(url, active_shard)
            downloaded_shards += 1
            downloaded_bytes += active_shard.stat().st_size
            try:
                count, shard_excluded = extract_shard(
                    active_shard,
                    Path(source_path).name,
                    generator,
                    external_root,
                    available_targets,
                    competition_hashes,
                    existing_hashes,
                    rows,
                    args.target_per_generator - added,
                )
                added += count
                excluded += shard_excluded
            finally:
                active_shard.unlink(missing_ok=True)
            print(
                f"{generator}: {added}/{args.target_per_generator} "
                f"from {downloaded_shards} shards",
                flush=True,
            )
        if added != args.target_per_generator:
            raise RuntimeError(
                f"insufficient paired extension rows for {generator}: "
                f"{added} != {args.target_per_generator}"
            )
        audit_generators[generator] = {
            "rows": added,
            "metadata_candidates": len(candidates),
            "downloaded_shards": downloaded_shards,
            "downloaded_bytes": downloaded_bytes,
            "preexisting_generator_targets_excluded": len(used_targets),
        }

    extension = pd.DataFrame(rows).sort_values(["model_name", "sample_key"])
    extension_path = external_root / "aigi_v2_paired_extension_manifest.csv"
    extension.to_csv(extension_path, index=False)
    combined = pd.concat([previous, extension], ignore_index=True, sort=False)
    combined_path = external_root / "external_transfer_v4_manifest.csv"
    combined.to_csv(combined_path, index=False)
    audit = {
        "source": REPOSITORY,
        "policy": "new fake images only when their exact paired real image already exists locally",
        "target_per_generator": args.target_per_generator,
        "generators": audit_generators,
        "new_rows": int(len(extension)),
        "combined_rows": int(len(combined)),
        "competition_rows_used_for_training": 0,
        "competition_hashes_checked_for_exact_deduplication": len(competition_hashes),
        "exact_existing_or_competition_duplicates_excluded": excluded,
        "new_real_rows": 0,
    }
    (external_root / "external_transfer_v4_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
