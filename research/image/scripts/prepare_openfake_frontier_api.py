#!/usr/bin/env python3
"""Deterministically sample OpenFake through the official rows API.

This is a bandwidth-efficient alternative to downloading a multi-gigabyte
Parquet shard. It stages selected files locally before syncing them to the GPU
server; selection is independent of competition validation outcomes.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import hashlib
import json
from pathlib import Path
import random
import threading
import time
import urllib.parse
import urllib.request

import pandas as pd
from PIL import Image


REPOSITORY = "ComplexDataLab/OpenFake"
TOTAL_TRAIN_ROWS = 2_306_584


def fetch_json(url: str, attempts: int = 10) -> dict[str, object]:
    for attempt in range(1, attempts + 1):
        try:
            request = urllib.request.Request(
                url, headers={"User-Agent": "pazhou-openfake-transfer/1.0"}
            )
            with urllib.request.urlopen(request, timeout=180) as response:
                payload = json.load(response)
            if not isinstance(payload, dict):
                raise TypeError("unexpected JSON payload")
            return payload
        except Exception:
            if attempt == attempts:
                raise
            time.sleep(min(3 * 2 ** (attempt - 1), 30))
    raise AssertionError("unreachable")


def select_round_robin(rows: list[dict[str, object]], target: int, seed: int) -> list[dict[str, object]]:
    groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[str(row["model"])].append(row)
    rng = random.Random(seed)
    for values in groups.values():
        rng.shuffle(values)
    keys = sorted(groups)
    selected: list[dict[str, object]] = []
    while len(selected) < target:
        progressed = False
        rng.shuffle(keys)
        for key in keys:
            if groups[key] and len(selected) < target:
                selected.append(groups[key].pop())
                progressed = True
        if not progressed:
            break
    if len(selected) != target:
        raise RuntimeError(f"only selected {len(selected)}/{target} rows")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--target-per-label", type=int, default=1500)
    parser.add_argument("--metadata-blocks", type=int, default=64)
    parser.add_argument("--metadata-workers", type=int, default=2)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260808)
    args = parser.parse_args()

    root = args.project_root.resolve()
    external_root = root / "data" / "external"
    output_root = external_root / "openfake_frontier_api"
    output_root.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(external_root / "external_transfer_v7_manifest.csv")
    competition = pd.read_csv(root / "data" / "folds" / "image_folds_v1.csv")
    excluded_hashes = set(previous.source_sha256.dropna().astype(str)) | set(
        competition.sha256.astype(str)
    )

    rng = random.Random(args.seed)
    offsets = rng.sample(range(0, TOTAL_TRAIN_ROWS - 100, 100), args.metadata_blocks)
    metadata_cache_dir = root / ".cache" / "openfake_frontier_api"
    metadata_cache_dir.mkdir(parents=True, exist_ok=True)
    metadata_cache_path = metadata_cache_dir / "metadata_blocks.json"
    cached_blocks: dict[str, list[dict[str, object]]] = {}
    if metadata_cache_path.is_file():
        cached_blocks = json.loads(metadata_cache_path.read_text(encoding="utf-8"))

    def fetch_block(offset: int) -> list[dict[str, object]]:
        url = "https://datasets-server.huggingface.co/rows?" + urllib.parse.urlencode(
            {
                "dataset": REPOSITORY,
                "config": "core",
                "split": "train",
                "offset": offset,
                "length": 100,
            }
        )
        payload = fetch_json(url)
        result: list[dict[str, object]] = []
        for item in payload["rows"]:  # type: ignore[index]
            row = dict(item["row"])
            row["row_idx"] = int(item["row_idx"])
            result.append(row)
        return result

    metadata_rows: list[dict[str, object]] = []
    for offset in offsets:
        metadata_rows.extend(cached_blocks.get(str(offset), []))
    pending_offsets = [offset for offset in offsets if str(offset) not in cached_blocks]
    if metadata_rows:
        print(f"resumed metadata rows: {len(metadata_rows)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.metadata_workers) as executor:
        futures = {executor.submit(fetch_block, offset): offset for offset in pending_offsets}
        for future in as_completed(futures):
            offset = futures[future]
            block_rows = future.result()
            cached_blocks[str(offset)] = block_rows
            metadata_rows.extend(block_rows)
            metadata_cache_path.write_text(
                json.dumps(cached_blocks, ensure_ascii=False), encoding="utf-8"
            )
            print(f"metadata rows: {len(metadata_rows)}/{args.metadata_blocks * 100}", flush=True)
    unique_rows = {int(row["row_idx"]): row for row in metadata_rows}
    real_candidates = sorted(
        [row for row in unique_rows.values() if row["label"] == "real"],
        key=lambda row: int(row["row_idx"]),
    )
    fake_candidates = sorted(
        [
            row
            for row in unique_rows.values()
            if row["label"] == "fake"
            and str(row.get("release_date", ""))[:4].isdigit()
            and int(str(row["release_date"])[:4]) >= 2024
        ],
        key=lambda row: int(row["row_idx"]),
    )
    selected = select_round_robin(real_candidates, args.target_per_label, args.seed ^ 0x0EA1)
    selected += select_round_robin(fake_candidates, args.target_per_label, args.seed ^ 0xFA4E)

    progress_lock = threading.Lock()
    progress = 0

    def download_row(row: dict[str, object]) -> dict[str, object]:
        nonlocal progress
        row_idx = int(row["row_idx"])
        matches = list(output_root.glob(f"*_{row_idx}.*"))
        if matches:
            destination = matches[0]
            image_bytes = destination.read_bytes()
        else:
            image_info = row["image"]
            if not isinstance(image_info, dict) or "src" not in image_info:
                raise RuntimeError(f"missing image URL at row {row_idx}")
            image_url = str(image_info["src"])
            image_bytes = b""
            for attempt in range(1, 7):
                try:
                    request = urllib.request.Request(
                        image_url, headers={"User-Agent": "pazhou-openfake-transfer/1.0"}
                    )
                    with urllib.request.urlopen(request, timeout=180) as response:
                        image_bytes = response.read()
                    break
                except Exception:
                    if attempt == 6:
                        raise
                    time.sleep(min(3 * 2 ** (attempt - 1), 30))
            with Image.open(BytesIO(image_bytes)) as image:
                image.verify()
                image_format = str(image.format or "JPEG").lower()
            suffix = ".jpg" if image_format in {"jpg", "jpeg"} else f".{image_format}"
            destination = output_root / f"{row['label']}_{row_idx}{suffix}"
            destination.write_bytes(image_bytes)
        with Image.open(BytesIO(image_bytes)) as image:
            image.verify()
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        if image_hash in excluded_hashes:
            raise RuntimeError(f"exact previous/competition collision at row {row_idx}")
        with progress_lock:
            progress += 1
            if progress % 100 == 0:
                print(f"images: {progress}/{len(selected)}", flush=True)
        label_name = str(row["label"])
        source_type = str(row["type"])
        model_name = str(row["model"])
        return {
            "sample_key": f"openfake_api_{row_idx}",
            "relative_path": destination.relative_to(external_root).as_posix(),
            "label": 1 if label_name == "fake" else 0,
            "model_name": model_name,
            "architecture": model_name,
            "subset": "core_train_frontier_2024plus" if label_name == "fake" else "core_train_real",
            "source_shard": "official_rows_api_deterministic_blocks",
            "source_row": row_idx,
            "source_sha256": image_hash,
            "source_real_id": "",
            "source_type": source_type,
            "release_date": str(row["release_date"]),
        }

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_row, row) for row in selected]
        for future in as_completed(futures):
            records.append(future.result())

    manifest = pd.DataFrame(records).sort_values(["label", "architecture", "sample_key"])
    if manifest.source_sha256.duplicated().any():
        raise RuntimeError("selected OpenFake images contain duplicate hashes")
    counts = manifest.label.value_counts().to_dict()
    if counts != {0: args.target_per_label, 1: args.target_per_label}:
        raise RuntimeError(f"unexpected label counts: {counts}")
    retained_paths = {
        (external_root / relative_path).resolve() for relative_path in manifest.relative_path
    }
    orphan_files_removed = 0
    for path in output_root.iterdir():
        if path.is_file() and path.resolve() not in retained_paths:
            path.unlink()
            orphan_files_removed += 1
    manifest_path = external_root / "openfake_frontier_api_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    audit = {
        "source": REPOSITORY,
        "config": "core",
        "split": "train",
        "selection": (
            f"{args.metadata_blocks} deterministic random 100-row blocks; "
            "round-robin by model; fake release year >= 2024"
        ),
        "seed": args.seed,
        "metadata_rows": len(unique_rows),
        "offsets_sha256": hashlib.sha256(json.dumps(offsets).encode()).hexdigest(),
        "label_counts": {str(key): int(value) for key, value in counts.items()},
        "fake_generator_count": int(manifest[manifest.label == 1].architecture.nunique()),
        "fake_generator_counts": {
            str(key): int(value)
            for key, value in manifest[manifest.label == 1].architecture.value_counts().items()
        },
        "real_source_counts": {
            str(key): int(value)
            for key, value in manifest[manifest.label == 0].architecture.value_counts().items()
        },
        "exact_overlap_previous_or_competition": 0,
        "competition_rows_used_for_training": 0,
        "retained_bytes": sum((external_root / path).stat().st_size for path in manifest.relative_path),
        "orphan_staging_files_removed": orphan_files_removed,
    }
    (external_root / "openfake_frontier_api_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
