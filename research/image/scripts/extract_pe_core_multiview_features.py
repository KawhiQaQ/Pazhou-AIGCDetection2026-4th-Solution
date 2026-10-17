#!/usr/bin/env python3
"""Extract deterministic global/native PE-Core embeddings for a multi-view kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from aigc_detection.image_v1 import GlobalLocalValidationCrops  # noqa: E402
from train_pe_core_linear import ImageRows, load_encoder, resolve, seed_everything  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/V50_pe_core_multiview_fold0.yaml")
    )
    parser.add_argument("--mode", choices=["competition", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grayscale", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load(resolve(root, str(args.config)).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    model_config = config["model"]
    data_config = config["data"]

    if args.mode == "competition":
        frame = pd.read_csv(resolve(root, str(data_config["labels_csv"])), dtype={"id": str})
        image_dir = resolve(root, str(data_config["image_dir"]))
    else:
        frame = pd.read_csv(
            root / "data/competition/image/image_test.csv", dtype={"id": str}
        )
        frame["label"] = -1
        image_dir = root / "data/competition/image/data/image_test"
    if args.smoke_only:
        frame = frame.iloc[:1].copy()
    rows = [
        {"id": str(row.id), "label": int(row.label), "path": image_dir / str(row.id)}
        for row in frame.itertuples(index=False)
    ]
    if any(not Path(str(row["path"])).is_file() for row in rows):
        raise RuntimeError("multi-view manifest contains missing images")

    multiview_transform = GlobalLocalValidationCrops(
        int(data_config["input_size"]),
        int(data_config["resize_size"]),
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5),
    )
    transform = (
        transforms.Compose(
            [transforms.Grayscale(num_output_channels=3), multiview_transform]
        )
        if args.grayscale
        else multiview_transform
    )
    workers = 0 if args.smoke_only else int(data_config["workers"])
    loader = DataLoader(
        ImageRows(rows, transform),
        batch_size=1 if args.smoke_only or args.mode == "competition" else 4,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, checkpoint, checkpoint_hash = load_encoder(config, root, device)
    expected_hash = str(model_config["checkpoint_sha256"])
    if checkpoint_hash != expected_hash or sha256(checkpoint) != expected_hash:
        raise RuntimeError("PE-Core checkpoint identity mismatch")
    encoder.eval()

    feature_parts: list[np.ndarray] = []
    labels: list[float] = []
    ids: list[str] = []
    # Test extraction is inference-only and the 24 GB production GPU can safely
    # process four 448px crops at once.  This changes throughput only; each view
    # and its deterministic transform remain identical to CV extraction.
    view_chunk = max(int(data_config["view_chunk_size"]), 16) if args.mode == "test" else int(data_config["view_chunk_size"])
    started = time.time()
    for batch_index, (views, batch_labels, batch_ids) in enumerate(loader, start=1):
        if views.shape[1] != 10:
            raise RuntimeError(f"unexpected multi-view tensor: {tuple(views.shape)}")
        batch_images = views.shape[0]
        views = views.flatten(0, 1)
        outputs: list[torch.Tensor] = []
        for start in range(0, len(views), view_chunk):
            images = views[start : start + view_chunk].to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                embeddings = encoder(images)
            outputs.append(F.normalize(embeddings.float(), dim=1).cpu())
        stacked = torch.cat(outputs, dim=0)
        feature_parts.append(
            stacked.reshape(batch_images, 10, -1).numpy().astype(np.float16)
        )
        labels.extend(batch_labels.numpy().tolist())
        ids.extend(str(value) for value in batch_ids)
        if batch_index % 50 == 0 or batch_index == len(loader):
            print(
                json.dumps(
                    {
                        "multiview": f"{len(ids)}/{len(rows)}",
                        "elapsed_seconds": round(time.time() - started, 2),
                    }
                ),
                flush=True,
            )
    features = np.concatenate(feature_parts)
    if features.shape[:2] != (len(rows), 10) or not np.isfinite(features).all():
        raise RuntimeError(f"invalid multi-view feature cache: {features.shape}")
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=features,
        labels=np.asarray(labels, dtype=np.float32),
        ids=np.asarray(ids),
        checkpoint_sha256=np.asarray(checkpoint_hash),
        protocol=np.asarray(
            "five_resized_global_plus_five_native_grayscale_PE-Core-G14-448"
            if args.grayscale
            else "five_resized_global_plus_five_native_PE-Core-G14-448"
        ),
    )
    print(
        json.dumps(
            {
                "output": str(output),
                "shape": list(features.shape),
                "elapsed_seconds": round(time.time() - started, 2),
            }
        )
    )


if __name__ == "__main__":
    main()
