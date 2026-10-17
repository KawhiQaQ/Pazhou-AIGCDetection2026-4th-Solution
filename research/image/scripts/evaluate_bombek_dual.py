#!/usr/bin/env python3
"""Safely evaluate the public OpenFake SigLIP2+DINOv2 detector."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import timm
from transformers import SiglipVisionConfig, SiglipVisionModel
from peft import LoraConfig, get_peft_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from aigc_detection.metrics import binary_metrics  # noqa: E402


class LoRALinear(nn.Module):
    def __init__(self, original: nn.Linear, rank: int = 32, alpha: float = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.original = original
        self.scaling = alpha / rank
        for parameter in original.parameters():
            parameter.requires_grad = False
        self.lora_A = nn.Linear(original.in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, original.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.original(inputs) + self.lora_B(self.lora_A(self.dropout(inputs))) * self.scaling


class ClassificationHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(2176),
            nn.Linear(2176, 512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(inputs).squeeze(-1)


class SiglipVisionWrapper(nn.Module):
    """Reproduce the prefix layout saved by the author's transformers version."""

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionModel(config)

    def forward(self, pixel_values: torch.Tensor):
        return self.vision_model(pixel_values=pixel_values)


class DualDetector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        vision_config = SiglipVisionConfig(
            hidden_size=1152,
            image_size=384,
            intermediate_size=4304,
            num_attention_heads=16,
            num_hidden_layers=27,
            patch_size=14,
        )
        self.siglip = SiglipVisionWrapper(vision_config)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.1,
            bias="none",
        )
        self.siglip = get_peft_model(self.siglip, lora_config)
        self.dinov2 = timm.create_model(
            "vit_large_patch14_dinov2.lvd142m", pretrained=False, num_classes=0, img_size=392
        )
        for module in self.dinov2.modules():
            if hasattr(module, "qkv") and isinstance(module.qkv, nn.Linear):
                module.qkv = LoRALinear(module.qkv)
        self.classifier = ClassificationHead()

    def forward(self, siglip_pixels: torch.Tensor, dinov2_pixels: torch.Tensor):
        siglip_features = self.siglip(pixel_values=siglip_pixels).pooler_output
        dinov2_features = self.dinov2(dinov2_pixels)
        combined = torch.cat([siglip_features.float(), dinov2_features.float()], dim=1)
        return self.classifier(combined), combined


class Images(Dataset):
    def __init__(self, rows: list[dict[str, object]]) -> None:
        self.rows = rows
        self.siglip_transform = transforms.Compose(
            [
                transforms.Resize((384, 384), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),
            ]
        )
        self.dinov2_transform = transforms.Compose(
            [
                transforms.Resize((392, 392), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(str(row["path"])) as image:
            image = image.convert("RGB")
            return (
                self.siglip_transform(image),
                self.dinov2_transform(image),
                float(row["label"]),
                str(row["id"]),
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=Path(".cache/bombek_dual/pytorch_model.pt"))
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--dataset", choices=["competition", "external"], default="competition")
    parser.add_argument("--all-rows", action="store_true")
    parser.add_argument("--batch-size", type=int, default=6)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/V98_bombek_dual_zero_shot/fold0"))
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    checkpoint_path = args.checkpoint if args.checkpoint.is_absolute() else root / args.checkpoint
    # The checkpoint config stores a NumPy scalar; allow only that inert type while
    # keeping PyTorch's restricted weights-only unpickler enabled.
    with torch.serialization.safe_globals(
        [np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("unexpected public checkpoint structure")
    model = DualDetector()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    for parameter in model.parameters():
        parameter.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)

    if args.dataset == "competition":
        labels = pd.read_csv(root / "data/competition/image/image_sample_data.csv")
        folds = pd.read_csv(root / "data/folds/image_folds_v1.csv")
        frame = labels.merge(folds[["id", "fold"]], on="id", validate="one_to_one")
        if not args.all_rows:
            frame = frame.loc[frame.fold.eq(args.fold)]
        frame = frame.reset_index(drop=True)
        rows = [
            {
                "id": row.id,
                "label": row.label,
                "path": root / "data/competition/image/data/image_sample_data" / row.id,
            }
            for row in frame.itertuples(index=False)
        ]
    else:
        frame = pd.read_csv(root / "data/external/aigi_v2_breadth_pairs_manifest.csv")
        rows = [
            {
                "id": row.sample_key,
                "label": row.label,
                "path": root / "data/external" / row.relative_path,
            }
            for row in frame.itertuples(index=False)
        ]
    if args.smoke_only:
        rows = rows[:2]
    loader = DataLoader(
        Images(rows), batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
        pin_memory=True, persistent_workers=args.workers > 0,
    )
    scores: list[float] = []
    targets: list[float] = []
    ids: list[str] = []
    feature_parts: list[np.ndarray] = []
    with torch.inference_mode():
        for siglip_pixels, dinov2_pixels, labels_batch, ids_batch in loader:
            siglip_pixels = siglip_pixels.to(device, non_blocking=True)
            dinov2_pixels = dinov2_pixels.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                logits, combined = model(siglip_pixels, dinov2_pixels)
            if not torch.isfinite(logits).all() or combined.shape[1] != 2176:
                raise RuntimeError("invalid public detector output")
            scores.extend(logits.float().cpu().tolist())
            if args.feature_cache is not None:
                feature_parts.append(combined.to(torch.float16).cpu().numpy())
            targets.extend(labels_batch.tolist())
            ids.extend(ids_batch)
    if args.smoke_only:
        print(json.dumps({"smoke": "passed", "rows": len(scores), "finite": True}))
        return
    if args.feature_cache is not None:
        cache_path = args.feature_cache if args.feature_cache.is_absolute() else root / args.feature_cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=np.concatenate(feature_parts),
            parent_scores=np.asarray(scores, dtype=np.float32),
            labels=np.asarray(targets, dtype=np.float32),
            ids=np.asarray(ids),
        )
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "label": targets, "score": scores}).to_csv(
        output_dir / "valid_predictions.csv", index=False
    )
    report = {
        "experiment": "V98_bombek_dual_zero_shot",
        "fold": args.fold,
        "metrics": binary_metrics(targets, scores),
        "audit": {
            "checkpoint_loaded_with_weights_only": True,
            "trainable_parameters": 0,
            "competition_training_rows_accessed": 0,
            "competition_test_rows_accessed": 0,
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
