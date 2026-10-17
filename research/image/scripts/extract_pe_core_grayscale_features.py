#!/usr/bin/env python3
"""Extract deterministic grayscale center-crop PE-Core embeddings."""

from __future__ import annotations

import argparse
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
from torchvision.transforms import InterpolationMode
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_pe_core_linear import (  # noqa: E402
    ImageRows,
    load_encoder,
    resolve,
    seed_everything,
    sha256_file,
)


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/V50_pe_core_multiview_fold0.yaml")
    )
    parser.add_argument("--mode", choices=["competition", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
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
        frame = pd.read_csv(root / "data/competition/image/image_test.csv", dtype={"id": str})
        frame["label"] = -1
        image_dir = root / "data/competition/image/data/image_test"
    rows = [
        {"id": str(row.id), "label": int(row.label), "path": image_dir / str(row.id)}
        for row in frame.itertuples(index=False)
    ]
    if any(not Path(str(row["path"])).is_file() for row in rows):
        raise RuntimeError("grayscale manifest contains missing images")

    transform = transforms.Compose(
        [
            transforms.Resize(
                int(data_config["resize_size"]), interpolation=InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(int(data_config["input_size"])),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    workers = int(data_config["workers"])
    loader = DataLoader(
        ImageRows(rows, transform),
        batch_size=2,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, checkpoint, checkpoint_hash = load_encoder(config, root, device)
    expected_hash = str(model_config["checkpoint_sha256"])
    if checkpoint_hash != expected_hash or sha256_file(checkpoint) != expected_hash:
        raise RuntimeError("PE-Core checkpoint identity mismatch")

    feature_parts: list[np.ndarray] = []
    labels: list[float] = []
    ids: list[str] = []
    started = time.time()
    for images, batch_labels, batch_ids in loader:
        with torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ):
            embeddings = encoder(images.to(device, non_blocking=True))
        embeddings = F.normalize(embeddings.float(), dim=1)
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("PE-Core produced non-finite grayscale embeddings")
        feature_parts.append(embeddings.cpu().numpy().astype(np.float16))
        labels.extend(batch_labels.numpy().tolist())
        ids.extend(str(value) for value in batch_ids)
        if len(ids) % 100 == 0 or len(ids) == len(rows):
            print(
                json.dumps(
                    {
                        "features": f"{len(ids)}/{len(rows)}",
                        "elapsed_seconds": round(time.time() - started, 2),
                    }
                ),
                flush=True,
            )

    features = np.concatenate(feature_parts)
    if features.shape != (len(rows), 1280) or not np.isfinite(features).all():
        raise RuntimeError(f"invalid grayscale cache: {features.shape}")
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=features,
        labels=np.asarray(labels, dtype=np.float32),
        ids=np.asarray(ids),
        checkpoint_sha256=np.asarray(checkpoint_hash),
        protocol=np.asarray("resize512-center448-grayscale_PE-Core-G14"),
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
