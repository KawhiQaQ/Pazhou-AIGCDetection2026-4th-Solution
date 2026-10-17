#!/usr/bin/env python3
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import urllib.parse
import urllib.request
import urllib.error
import time

import pandas as pd
from PIL import Image
import pyarrow.parquet as pq


REPOSITORY = "Shanmuk4622/ai-detection-dataset-v2"
GENERATORS = ("flux_schnell", "sdxl", "pixart_sigma", "kandinsky22", "wuerstchen")


def fetch_json_page(url: str) -> tuple[object, str | None]:
    request = urllib.request.Request(
        url, headers={"User-Agent": "pazhou-aigc-transfer/1.0"}
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        payload = json.load(response)
        link = response.headers.get("Link", "")
    match = re.search(r'<([^>]+)>;\s*rel="next"', link)
    return payload, match.group(1) if match else None


def fetch_tree(url: str, mirror: str) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    while url:
        payload, next_url = fetch_json_page(url)
        if not isinstance(payload, list):
            raise TypeError("invalid repository tree response")
        entries.extend(item for item in payload if isinstance(item, dict))
        url = next_url.replace("https://huggingface.co", mirror, 1) if next_url else ""
    return entries


def download(
    url: str,
    destination: Path,
    max_attempts: int = 8,
    expected_size: int | None = None,
) -> None:
    """Download atomically, retrying transient mirror throttling/network errors."""
    temporary = destination.with_name(destination.name + ".part")
    if (
        expected_size is not None
        and temporary.is_file()
        and temporary.stat().st_size == expected_size
    ):
        temporary.replace(destination)
        return
    for attempt in range(1, max_attempts + 1):
        offset = temporary.stat().st_size if temporary.is_file() else 0
        headers = {"User-Agent": "pazhou-aigc-transfer/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(
            url, headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                append = offset > 0 and getattr(response, "status", None) == 206
                with temporary.open("ab" if append else "wb") as handle:
                    shutil.copyfileobj(response, handle, length=8 * 1024 * 1024)
            if expected_size is not None and temporary.stat().st_size != expected_size:
                raise RuntimeError(
                    f"downloaded size mismatch: {temporary.stat().st_size} != {expected_size}"
                )
            temporary.replace(destination)
            return
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, RuntimeError):
            if attempt == max_attempts:
                raise
            time.sleep(min(5 * 2 ** (attempt - 1), 60))


def extract_shard(
    shard: Path,
    output_root: Path,
    competition_hashes: set[str],
    rows: list[dict[str, object]],
    seen_ids: set[str],
    counts: dict[str, int],
    target_per_generator: int,
) -> int:
    table = pq.read_table(
        shard,
        columns=["image_id", "label", "generator", "image", "sha256", "source_real_id"],
    )
    values = table.to_pydict()
    excluded = 0
    for index in range(table.num_rows):
        generator = str(values["generator"][index])
        if generator not in counts or counts[generator] >= target_per_generator:
            continue
        if int(values["label"][index]) != 1:
            raise RuntimeError(f"non-fake row in selected generator shard: {shard}:{index}")
        image_id = str(values["image_id"][index])
        if image_id in seen_ids:
            continue
        image_bytes = values["image"][index]
        expected_sha256 = str(values["sha256"][index])
        actual_sha256 = hashlib.sha256(image_bytes).hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"image SHA-256 mismatch in {shard}:{index}")
        if actual_sha256 in competition_hashes:
            excluded += 1
            continue
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
            image_format = str(image.format or "PNG").lower()
        suffix = ".jpg" if image_format in {"jpeg", "jpg"} else f".{image_format}"
        filename = f"{image_id}{suffix}"
        relative_path = Path("aigi_v2_modern") / generator / filename
        destination = output_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(image_bytes)
        rows.append(
            {
                "sample_key": f"aigi_v2_{generator}_{image_id}",
                "relative_path": relative_path.as_posix(),
                "label": 1,
                "model_name": f"AIGIv2_{generator}",
                "architecture": generator,
                "subset": "modern_paired_fake",
                "source_shard": shard.name,
                "source_row": index,
                "source_sha256": actual_sha256,
                "source_real_id": str(values["source_real_id"][index]),
            }
        )
        seen_ids.add(image_id)
        counts[generator] += 1
    return excluded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target-per-generator", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--mirror", default="https://hf-mirror.com")
    args = parser.parse_args()

    root = args.project_root.resolve()
    external_root = root / "data" / "external"
    old_root = external_root / "community_forensics_transfer_v1"
    old_manifest_path = old_root / "manifest.csv"
    folds_path = root / "data" / "folds" / "image_folds_v1.csv"
    cache_dir = root / ".cache" / "aigi_v2_shards"
    if not old_manifest_path.is_file() or not folds_path.is_file():
        raise FileNotFoundError("V1 external manifest or fixed competition folds are missing")

    competition_hashes = set(pd.read_csv(folds_path, usecols=["sha256"]).sha256.astype(str))
    rows: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    counts = {generator: 0 for generator in GENERATORS}
    excluded = 0

    # Consume already downloaded probe shards first, then remove them after verified extraction.
    for shard in sorted(cache_dir.glob("*.parquet")):
        excluded += extract_shard(
            shard,
            external_root,
            competition_hashes,
            rows,
            seen_ids,
            counts,
            args.target_per_generator,
        )
        shard.unlink()

    cache_dir.mkdir(parents=True, exist_ok=True)
    for generator in GENERATORS:
        quoted_path = urllib.parse.quote(f"data/{generator}", safe="/")
        tree_url = (
            f"{args.mirror}/api/datasets/{REPOSITORY}/tree/main/{quoted_path}"
            "?recursive=false&expand=true&limit=100"
        )
        entries = fetch_tree(tree_url, args.mirror)
        candidates = sorted(
            str(entry["path"])
            for entry in entries
            if isinstance(entry, dict) and str(entry.get("path", "")).endswith(".parquet")
        )
        generator_seed = int(hashlib.sha256(generator.encode()).hexdigest()[:8], 16)
        random.Random(args.seed ^ generator_seed).shuffle(candidates)
        print(f"{generator}: discovered {len(candidates)} shards", flush=True)
        for source_path in candidates:
            if counts[generator] >= args.target_per_generator:
                break
            active_shard = cache_dir / "active.parquet"
            encoded_path = urllib.parse.quote(source_path, safe="/")
            url = f"{args.mirror}/datasets/{REPOSITORY}/resolve/main/{encoded_path}"
            download(url, active_shard)
            try:
                excluded += extract_shard(
                    active_shard,
                    external_root,
                    competition_hashes,
                    rows,
                    seen_ids,
                    counts,
                    args.target_per_generator,
                )
            finally:
                active_shard.unlink(missing_ok=True)
            print(
                f"{generator}: {counts[generator]}/{args.target_per_generator}",
                flush=True,
            )
        if counts[generator] < args.target_per_generator:
            raise RuntimeError(
                f"insufficient rows for {generator}: {counts[generator]} < {args.target_per_generator}"
            )

    new_manifest = pd.DataFrame(rows).sort_values(["model_name", "sample_key"])
    new_manifest_path = external_root / "aigi_v2_modern_manifest.csv"
    new_manifest.to_csv(new_manifest_path, index=False)

    old_manifest = pd.read_csv(old_manifest_path)
    old_manifest["relative_path"] = old_manifest.relative_path.map(
        lambda value: (Path("community_forensics_transfer_v1") / str(value)).as_posix()
    )
    combined = pd.concat([old_manifest, new_manifest], ignore_index=True, sort=False)
    combined_path = external_root / "external_transfer_v2_manifest.csv"
    combined.to_csv(combined_path, index=False)

    audit = {
        "source": REPOSITORY,
        "selected_generators": list(GENERATORS),
        "new_counts": counts,
        "new_rows": int(len(new_manifest)),
        "combined_rows": int(len(combined)),
        "competition_rows_used_for_training": 0,
        "competition_hashes_checked_for_exact_deduplication": len(competition_hashes),
        "exact_competition_duplicates_excluded": excluded,
        "real_pool": "reused CommunityForensics-Small real rows; no additional real images",
    }
    (external_root / "external_transfer_v2_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
