#!/usr/bin/env python3
"""Build source-diverse AIGI-v2 real/fake pairs for V43.

Unlike V38's dense generator grid over 2,322 existing real sources, this script
selects each new source_real_id exactly once across all generators and retrieves
its exact real counterpart.  This makes source/content breadth the only intended
data change relative to V36's external manifest.
"""

from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import random
import urllib.parse

import pandas as pd
from PIL import Image
import pyarrow.parquet as pq

from prepare_aigi_v2_transfer import download, fetch_tree, REPOSITORY


GENERATORS = (
    "sd15",
    "sdxl",
    "flux_schnell",
    "kandinsky22",
    "pixart_sigma",
    "wuerstchen",
)


def verified_image(image_bytes: bytes, expected_sha256: str) -> tuple[str, str]:
    actual_sha256 = hashlib.sha256(image_bytes).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"image SHA-256 mismatch: {actual_sha256} != {expected_sha256}")
    with Image.open(BytesIO(image_bytes)) as image:
        image.verify()
        image_format = str(image.format or "PNG").lower()
    suffix = ".jpg" if image_format in {"jpeg", "jpg"} else f".{image_format}"
    return actual_sha256, suffix


def tree_files(mirror: str, directory: str) -> list[dict[str, object]]:
    quoted = urllib.parse.quote(directory, safe="/")
    url = (
        f"{mirror}/api/datasets/{REPOSITORY}/tree/main/{quoted}"
        "?recursive=false&expand=true&limit=100"
    )
    return [
        entry
        for entry in fetch_tree(url, mirror)
        if str(entry.get("path", "")).endswith(".parquet")
    ]


def extract_fake_shard(
    shard: Path,
    *,
    generator: str,
    target_count: int,
    output_root: Path,
    existing_source_ids: set[str],
    selected_source_ids: set[str],
    seen_hashes: set[str],
    excluded_hashes: set[str],
    rows: list[dict[str, object]],
) -> tuple[int, int]:
    table = pq.read_table(
        shard,
        columns=["image_id", "source_real_id", "label", "generator", "image", "sha256"],
    )
    values = table.to_pydict()
    added = 0
    excluded = 0
    for index in range(table.num_rows):
        if added >= target_count:
            break
        if int(values["label"][index]) != 1 or str(values["generator"][index]) != generator:
            raise RuntimeError(f"invalid fake row in {shard.name}:{index}")
        source_real_id = str(values["source_real_id"][index])
        if source_real_id in existing_source_ids or source_real_id in selected_source_ids:
            continue
        image_bytes = values["image"][index]
        actual_sha256, suffix = verified_image(image_bytes, str(values["sha256"][index]))
        if actual_sha256 in excluded_hashes or actual_sha256 in seen_hashes:
            excluded += 1
            continue
        image_id = str(values["image_id"][index])
        relative_path = Path("aigi_v2_breadth_fake") / generator / f"{image_id}{suffix}"
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)
        rows.append(
            {
                "sample_key": f"aigi_v2_breadth_fake_{generator}_{image_id}",
                "relative_path": relative_path.as_posix(),
                "label": 1,
                "model_name": f"AIGIv2_{generator}",
                "architecture": generator,
                "subset": "source_diverse_paired_fake",
                "source_shard": shard.name,
                "source_row": index,
                "source_sha256": actual_sha256,
                "source_real_id": source_real_id,
            }
        )
        selected_source_ids.add(source_real_id)
        seen_hashes.add(actual_sha256)
        added += 1
    return added, excluded


def extract_real_shard(
    shard: Path,
    *,
    output_root: Path,
    targets: set[str],
    matched: set[str],
    seen_hashes: set[str],
    excluded_hashes: set[str],
    rows: list[dict[str, object]],
) -> tuple[int, set[str]]:
    table = pq.read_table(
        shard,
        columns=["image_id", "source_real_id", "label", "generator", "image", "sha256"],
    )
    values = table.to_pydict()
    excluded = 0
    rejected: set[str] = set()
    for index in range(table.num_rows):
        image_id = str(values["image_id"][index])
        if image_id not in targets or image_id in matched:
            continue
        if int(values["label"][index]) != 0 or str(values["generator"][index]) != "real":
            raise RuntimeError(f"invalid real row in {shard.name}:{index}")
        image_bytes = values["image"][index]
        actual_sha256, suffix = verified_image(image_bytes, str(values["sha256"][index]))
        if actual_sha256 in excluded_hashes or actual_sha256 in seen_hashes:
            excluded += 1
            rejected.add(image_id)
            matched.add(image_id)
            continue
        relative_path = Path("aigi_v2_breadth_real") / f"{image_id}{suffix}"
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)
        rows.append(
            {
                "sample_key": f"aigi_v2_breadth_real_{image_id}",
                "relative_path": relative_path.as_posix(),
                "label": 0,
                "model_name": "AIGIv2_breadth_real",
                "architecture": "real",
                "subset": "source_diverse_paired_real",
                "source_shard": shard.name,
                "source_row": index,
                "source_sha256": actual_sha256,
                "source_real_id": image_id,
            }
        )
        seen_hashes.add(actual_sha256)
        matched.add(image_id)
    return excluded, rejected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target-per-generator", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    args = parser.parse_args()

    root = args.project_root.resolve()
    external_root = root / "data" / "external"
    previous_path = external_root / "external_transfer_v4_manifest.csv"
    folds_path = root / "data" / "folds" / "image_folds_v1.csv"
    if not previous_path.is_file() or not folds_path.is_file():
        raise FileNotFoundError("V36 external manifest or fixed competition folds are missing")

    previous = pd.read_csv(previous_path)
    existing_source_ids = set(previous.source_real_id.dropna().astype(str))
    existing_hashes = set(previous.source_sha256.dropna().astype(str))
    competition_hashes = set(pd.read_csv(folds_path, usecols=["sha256"]).sha256.astype(str))
    excluded_hashes = existing_hashes | competition_hashes
    seen_hashes: set[str] = set()
    selected_source_ids: set[str] = set()
    fake_rows: list[dict[str, object]] = []
    real_rows: list[dict[str, object]] = []
    fake_counts = {generator: 0 for generator in GENERATORS}
    exact_duplicates_excluded = 0
    downloaded_bytes = 0
    cache_dir = root / ".cache" / "aigi_v2_breadth"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fake_progress_path = cache_dir / "fake_progress.csv"
    real_progress_path = cache_dir / "real_progress.csv"
    rejected_progress_path = cache_dir / "rejected_real_ids.json"

    if fake_progress_path.is_file():
        loaded_rows = pd.read_csv(fake_progress_path).to_dict("records")
        fake_rows = []
        for generator in GENERATORS:
            generator_rows = [
                row for row in loaded_rows if str(row["architecture"]) == generator
            ]
            fake_rows.extend(generator_rows[: args.target_per_generator])
            for row in generator_rows[args.target_per_generator :]:
                (external_root / str(row["relative_path"])).unlink(missing_ok=True)
        pd.DataFrame(fake_rows).to_csv(fake_progress_path, index=False)
        selected_source_ids.update(str(row["source_real_id"]) for row in fake_rows)
        seen_hashes.update(str(row["source_sha256"]) for row in fake_rows)
        for generator in GENERATORS:
            fake_counts[generator] = sum(
                str(row["architecture"]) == generator for row in fake_rows
            )
        print(f"resumed {len(fake_rows)} fake rows", flush=True)

    for generator in GENERATORS:
        candidates = tree_files(args.mirror, f"data/{generator}")
        generator_seed = int(hashlib.sha256(generator.encode()).hexdigest()[:8], 16)
        random.Random(args.seed ^ generator_seed).shuffle(candidates)
        for entry in candidates:
            remaining = args.target_per_generator - fake_counts[generator]
            if remaining <= 0:
                break
            source_path = str(entry["path"])
            active = cache_dir / f"active_fake_{generator}.parquet"
            url = f"{args.mirror}/datasets/{REPOSITORY}/resolve/main/{urllib.parse.quote(source_path, safe='/')}"
            download(url, active)
            downloaded_bytes += int(entry.get("size", active.stat().st_size))
            try:
                added, excluded = extract_fake_shard(
                    active,
                    generator=generator,
                    target_count=remaining,
                    output_root=external_root,
                    existing_source_ids=existing_source_ids,
                    selected_source_ids=selected_source_ids,
                    seen_hashes=seen_hashes,
                    excluded_hashes=excluded_hashes,
                    rows=fake_rows,
                )
                fake_counts[generator] += added
                exact_duplicates_excluded += excluded
                pd.DataFrame(fake_rows).to_csv(fake_progress_path, index=False)
            finally:
                active.unlink(missing_ok=True)
            print(
                f"fake {generator}: {fake_counts[generator]}/{args.target_per_generator}",
                flush=True,
            )
        if fake_counts[generator] != args.target_per_generator:
            raise RuntimeError(f"insufficient new source IDs for {generator}: {fake_counts[generator]}")

    targets = set(selected_source_ids)
    matched: set[str] = set()
    rejected: set[str] = set()
    if real_progress_path.is_file():
        real_rows = pd.read_csv(real_progress_path).to_dict("records")
        matched.update(str(row["source_real_id"]) for row in real_rows)
        seen_hashes.update(str(row["source_sha256"]) for row in real_rows)
    if rejected_progress_path.is_file():
        rejected.update(json.loads(rejected_progress_path.read_text(encoding="utf-8")))
        matched.update(rejected)
    if matched:
        print(f"resumed {len(matched)}/{len(targets)} real matches", flush=True)
    for entry in tree_files(args.mirror, "real"):
        if len(matched) >= len(targets):
            break
        source_path = str(entry["path"])
        active = cache_dir / "active_real.parquet"
        url = f"{args.mirror}/datasets/{REPOSITORY}/resolve/main/{urllib.parse.quote(source_path, safe='/')}"
        download(url, active)
        downloaded_bytes += int(entry.get("size", active.stat().st_size))
        try:
            excluded, shard_rejected = extract_real_shard(
                active,
                output_root=external_root,
                targets=targets,
                matched=matched,
                seen_hashes=seen_hashes,
                excluded_hashes=excluded_hashes,
                rows=real_rows,
            )
            exact_duplicates_excluded += excluded
            rejected.update(shard_rejected)
            pd.DataFrame(real_rows).to_csv(real_progress_path, index=False)
            rejected_progress_path.write_text(
                json.dumps(sorted(rejected)) + "\n", encoding="utf-8"
            )
        finally:
            active.unlink(missing_ok=True)
        print(f"real pairs: {len(matched)}/{len(targets)}", flush=True)

    missing = targets - matched
    if missing:
        raise RuntimeError(f"missing {len(missing)} real counterparts: {sorted(missing)[:5]}")

    if rejected:
        retained_fake_rows = []
        for row in fake_rows:
            if str(row["source_real_id"]) in rejected:
                (external_root / str(row["relative_path"])).unlink(missing_ok=True)
                seen_hashes.discard(str(row["source_sha256"]))
            else:
                retained_fake_rows.append(row)
        fake_rows = retained_fake_rows
    retained_targets = targets - rejected
    if len(fake_rows) != len(real_rows) or len(fake_rows) != len(retained_targets):
        raise RuntimeError("breadth data is not one-to-one paired after deduplication")

    new_manifest = pd.DataFrame(fake_rows + real_rows).sort_values(
        ["source_real_id", "label", "sample_key"]
    )
    pair_counts = new_manifest.groupby("source_real_id").label.agg(["count", "sum"])
    if not ((pair_counts["count"] == 2) & (pair_counts["sum"] == 1)).all():
        raise RuntimeError("each retained source_real_id must have one real and one fake")
    if new_manifest.source_sha256.duplicated().any():
        raise RuntimeError("new breadth manifest contains duplicate image hashes")
    if set(new_manifest.source_sha256) & excluded_hashes:
        raise RuntimeError("new breadth manifest overlaps existing or competition hashes")

    new_path = external_root / "aigi_v2_breadth_pairs_manifest.csv"
    new_manifest.to_csv(new_path, index=False)
    combined = pd.concat([previous, new_manifest], ignore_index=True, sort=False)
    combined_path = external_root / "external_transfer_v6_manifest.csv"
    combined.to_csv(combined_path, index=False)
    audit = {
        "source": REPOSITORY,
        "design": "globally unique new source_real_id with exactly one fake generator and its exact real",
        "seed": args.seed,
        "requested_per_generator": args.target_per_generator,
        "fake_counts_before_real_dedup": fake_counts,
        "rejected_real_hash_collisions": len(rejected),
        "retained_pairs": len(retained_targets),
        "new_rows": len(new_manifest),
        "combined_rows": len(combined),
        "existing_source_ids_excluded": len(existing_source_ids),
        "competition_hashes_checked": len(competition_hashes),
        "exact_hash_duplicates_excluded": exact_duplicates_excluded,
        "competition_rows_used_for_training": 0,
        "downloaded_bytes": downloaded_bytes,
        "active_shards_remaining": len(list(cache_dir.glob("active_*.parquet"))),
    }
    audit_path = external_root / "external_transfer_v6_audit.json"
    audit_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
