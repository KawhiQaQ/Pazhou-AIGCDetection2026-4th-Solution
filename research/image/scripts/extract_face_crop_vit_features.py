#!/usr/bin/env python3
"""Extract deterministic largest-face CLS features from a frozen deepfake ViT."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import ViTForImageClassification


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA256 = "b38e7d3de2be44b224cb4cffe881a0c4b75f7283c5d4a2017fbadd58bd41fdef"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FaceCropRows(Dataset):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.detector: cv2.CascadeClassifier | None = None
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        if self.detector is None:
            cv2.setNumThreads(1)
            self.detector = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            )
        row = self.rows[index]
        path = Path(str(row["path"]))
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"cannot decode: {path}")
        detections = self.detector.detectMultiScale(
            cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY), 1.1, 4, minSize=(24, 24)
        )
        image = Image.open(path).convert("RGB")
        detected = len(detections) > 0
        if detected:
            x, y, width, height = max(detections, key=lambda box: box[2] * box[3])
            padding = max(width, height) // 2
            image = image.crop(
                (
                    max(0, x - padding),
                    max(0, y - padding),
                    min(image.width, x + width + padding),
                    min(image.height, y + height + padding),
                )
            )
        return self.transform(image), float(row["label"]), str(row["id"]), float(detected)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--mode", choices=["competition", "test"], required=True)
    parser.add_argument("--checkpoint-dir", type=Path, default=Path(".cache/deepfake_vit"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    root = args.project_root.resolve()
    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir.is_absolute() else root / args.checkpoint_dir
    checkpoint = checkpoint_dir / "model.safetensors"
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("face ViT checkpoint hash mismatch")
    if args.mode == "competition":
        frame = pd.read_csv(root / "data/competition/image/image_sample_data.csv")
        image_dir = root / "data/competition/image/data/image_sample_data"
    else:
        frame = pd.read_csv(root / "data/competition/image/image_test.csv")
        image_dir = root / "data/competition/image/data/image_test"
        frame["label"] = -1
    rows = [
        {"id": row.id, "label": row.label, "path": image_dir / row.id}
        for row in frame.itertuples(index=False)
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ViTForImageClassification.from_pretrained(
        checkpoint_dir, local_files_only=True, use_safetensors=True
    ).to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    loader = DataLoader(
        FaceCropRows(rows), batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, persistent_workers=args.workers > 0,
    )
    features: list[np.ndarray] = []
    labels: list[float] = []
    ids: list[str] = []
    detected: list[float] = []
    with torch.inference_mode():
        for index, (pixels, batch_labels, batch_ids, batch_detected) in enumerate(loader, start=1):
            output = model.vit(pixel_values=pixels.to(device, non_blocking=True))
            features.append(output.last_hidden_state[:, 0].float().cpu().numpy().astype(np.float16))
            labels.extend(batch_labels.tolist())
            ids.extend(batch_ids)
            detected.extend(batch_detected.tolist())
            if index % 50 == 0 or len(ids) == len(rows):
                print({"features": f"{len(ids)}/{len(rows)}", "faces": int(sum(detected))}, flush=True)
    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.concatenate(features), labels=np.asarray(labels, dtype=np.float32),
        ids=np.asarray(ids), face_detected=np.asarray(detected, dtype=np.uint8),
        checkpoint_sha256=np.asarray(CHECKPOINT_SHA256),
        crop_protocol=np.asarray("largest_haar_face_plus_50pct_context_or_full_image"),
    )
    print({"output": str(output), "rows": len(ids), "faces": int(sum(detected))})


if __name__ == "__main__":
    main()
