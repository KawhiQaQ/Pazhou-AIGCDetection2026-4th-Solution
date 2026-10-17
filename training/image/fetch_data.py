#!/usr/bin/env python3
"""Restore exact manifest-selected image bytes from primary dataset repositories.

No resampling: every restored image must match its recorded SHA-256. AIGI may
require scanning many large Parquet shards; use --plan before downloading.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
REPOS = {"community": "OwensLab/CommunityForensics-Small", "aigi": "Shanmuk4622/ai-detection-dataset-v2", "openfake": "ComplexDataLab/OpenFake"}


def request(url):
    for attempt in range(5):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "DualTrace-reproduction/1.0"}), timeout=180) as response:
                return response.read()
        except (OSError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 16))


def destination(root, row):
    path = (root / row["relative_path"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Manifest path escapes output root")
    return path


def save(root, row, value):
    digest = hashlib.sha256(value).hexdigest()
    if digest != row["source_sha256"]:
        raise ValueError(f"Source bytes changed: {row['sample_key']} expected {row['source_sha256']} got {digest}")
    path = destination(root, row)
    if path.exists() and path.read_bytes() != value:
        raise RuntimeError(f"Refusing to overwrite different bytes: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(value)


def image_bytes(value):
    if isinstance(value, bytes):
        return value
    if isinstance(value, dict) and isinstance(value.get("bytes"), bytes):
        return value["bytes"]
    raise ValueError("Dataset image has no embedded bytes; cannot recreate original byte-identical file")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work-dir", type=Path, default=ROOT / "runs/image_training")
    p.add_argument("--dataset", choices=["all", *REPOS], default="all")
    p.add_argument("--plan", action="store_true")
    p.add_argument("--openfake-mode", choices=["rows", "parquet"], default="rows", help="Hash-scan Parquet if upstream rows-API indices changed; substantially more bandwidth")
    a = p.parse_args()
    external = a.work_dir.resolve() / "data/external"
    selections = {}
    with (external / "external_transfer_v6_manifest.csv").open(newline="") as stream:
        v6 = list(csv.DictReader(stream))
    with (external / "openfake_frontier_api_manifest.csv").open(newline="") as stream:
        openfake = list(csv.DictReader(stream))
    selected = {"community": [r for r in v6 if r["relative_path"].startswith("community_forensics_transfer_v1/")],
                "aigi": [r for r in v6 if r["relative_path"].startswith("aigi_")], "openfake": openfake}
    for name, rows in selected.items():
        if a.dataset not in ["all", name]:
            continue
        missing = []
        for row in rows:
            path = destination(external, row)
            if not path.is_file():
                missing.append(row)
            elif hashlib.sha256(path.read_bytes()).hexdigest() != row["source_sha256"]:
                raise ValueError(f"Existing image hash mismatch: {path}")
        selections[name] = missing
    print(json.dumps({name: {"repository": REPOS[name], "missing_images": len(rows)} for name, rows in selections.items()}, indent=2), flush=True)
    if a.plan:
        return
    for name, rows in selections.items():
        if not rows:
            continue
        if name == "openfake" and a.openfake_mode == "rows":
            for index, row in enumerate(rows, 1):
                query = urllib.parse.urlencode({"dataset": REPOS[name], "config": "core", "split": "train", "offset": int(row["source_row"]), "length": 1})
                result = json.loads(request("https://datasets-server.huggingface.co/rows?" + query))
                item = result["rows"][0]
                if int(item["row_idx"]) != int(row["source_row"]):
                    raise ValueError("OpenFake row index changed")
                save(external, row, request(item["row"]["image"]["src"]))
                if index % 100 == 0:
                    print(f"openfake: {index}/{len(rows)}", flush=True)
            continue
        from huggingface_hub import HfApi, hf_hub_download
        import pyarrow.parquet as pq
        files = sorted(f for f in HfApi().list_repo_files(REPOS[name], repo_type="dataset") if f.endswith(".parquet"))
        if name == "openfake":
            files = [f for f in files if f.startswith("core/train")]
        if name == "community":
            wanted_shards = {row["source_shard"] for row in rows}
            files = [f for f in files if Path(f).name in wanted_shards]
            if {Path(f).name for f in files} != wanted_shards:
                raise ValueError("CommunityForensics source shard names changed")
        pending = {row["source_sha256"]: row for row in rows}
        for filename in files:
            if not pending:
                break
            with tempfile.TemporaryDirectory(prefix="dualtrace-image-shard-") as temporary:
                parquet = hf_hub_download(REPOS[name], filename, repo_type="dataset", local_dir=temporary)
                reader = pq.ParquetFile(parquet)
                if "image" not in reader.schema_arrow.names:
                    raise ValueError(f"No image column in {filename}")
                hash_column = next((c for c in ["sha256", "image_sha256"] if c in reader.schema_arrow.names), None)
                columns = ["image"] + ([hash_column] if hash_column else [])
                for batch in reader.iter_batches(batch_size=32, columns=columns):
                    records = batch.to_pylist()
                    for record in records:
                        if hash_column and record[hash_column] not in pending:
                            continue
                        value = image_bytes(record["image"])
                        digest = hashlib.sha256(value).hexdigest()
                        if digest in pending:
                            save(external, pending.pop(digest), value)
            print(f"{name}: restored {len(rows)-len(pending)}/{len(rows)} after {filename}", flush=True)
        if pending:
            raise RuntimeError(f"{name}: {len(pending)} selected hashes not found; do not replace with new samples")


if __name__ == "__main__":
    main()
