from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import timm
import torch
from torch import nn
from torch.utils.data import Dataset

from .image_v1 import CLIPMixtureOfLayersDetector


class DistillationTransferDataset(
    Dataset[tuple[torch.Tensor, dict[str, torch.Tensor], str]]
):
    """External-only transfer images paired with cached teacher outputs."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        root: Path,
        transform: Callable[[Image.Image], torch.Tensor],
        teacher_caches: dict[str, dict[str, np.ndarray]],
    ) -> None:
        self.manifest = manifest.reset_index(drop=True)
        self.root = root
        self.transform = transform
        self.teacher_caches = teacher_caches
        expected = self.manifest["sample_key"].astype(str).to_numpy()
        for name, cache in teacher_caches.items():
            actual = cache["sample_key"].astype(str)
            if not np.array_equal(expected, actual):
                raise ValueError(f"teacher cache ordering mismatch: {name}")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], str]:
        row = self.manifest.iloc[index]
        path = self.root / str(row["relative_path"])
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        targets: dict[str, torch.Tensor] = {}
        for teacher, cache in self.teacher_caches.items():
            targets[f"{teacher}_logit"] = torch.tensor(
                float(cache["logit"][index]), dtype=torch.float32
            )
            targets[f"{teacher}_embedding"] = torch.from_numpy(
                cache["embedding"][index].astype(np.float32, copy=False)
            )
        return tensor, targets, str(row["sample_key"])


class TimmDistillationStudent(nn.Module):
    """A timm student with separate logit/feature heads for each teacher."""

    def __init__(
        self,
        backbone_name: str,
        teacher_dims: dict[str, int],
        *,
        pretrained: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone_name
        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0
        )
        feature_dim = int(self.backbone.num_features)
        self.dropout = nn.Dropout(dropout)
        self.logit_heads = nn.ModuleDict(
            {name: nn.Linear(feature_dim, 1) for name in teacher_dims}
        )
        self.feature_heads = nn.ModuleDict(
            {name: nn.Linear(feature_dim, dim) for name, dim in teacher_dims.items()}
        )

    def forward(
        self, images: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        embedding = self.backbone(images)
        if embedding.ndim != 2:
            raise RuntimeError(
                f"student backbone must return [batch, features], got {embedding.shape}"
            )
        dropped = self.dropout(embedding)
        logits = {
            name: head(dropped).squeeze(1) for name, head in self.logit_heads.items()
        }
        features = {name: head(embedding) for name, head in self.feature_heads.items()}
        return logits, features, embedding

    def export_checkpoint(self) -> dict[str, object]:
        return {
            "format": "aigc-distilled-timm-v1",
            "backbone_name": self.backbone_name,
            "backbone": {
                name: value.detach().cpu()
                for name, value in self.backbone.state_dict().items()
            },
            "logit_heads": {
                teacher: {
                    name: value.detach().cpu()
                    for name, value in head.state_dict().items()
                }
                for teacher, head in self.logit_heads.items()
            },
        }


class CLIPMoLDDistillationStudent(nn.Module):
    """External-teacher distillation into a frozen CLIP MoLD detector head."""

    def __init__(
        self,
        backbone: str,
        feature_blocks: list[int],
        hidden_dim: int,
        gate_hidden_dim: int,
        teacher_dims: dict[str, int],
        dropout: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__()
        self.detector = CLIPMixtureOfLayersDetector(
            backbone,
            feature_blocks,
            0,
            hidden_dim,
            gate_hidden_dim,
            dropout,
            device=device,
            download_root=download_root,
        )
        self.logit_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, 1) for name in teacher_dims}
        )
        self.feature_heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, dim) for name, dim in teacher_dims.items()}
        )

    def forward(
        self, images: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], torch.Tensor]:
        _, embedding = self.detector(images)
        dropped = self.detector.dropout(embedding)
        logits = {
            name: head(dropped).squeeze(1) for name, head in self.logit_heads.items()
        }
        features = {
            name: head(embedding) for name, head in self.feature_heads.items()
        }
        return logits, features, embedding

    def backbone_parameters(self) -> list[nn.Parameter]:
        return self.detector.backbone_trainable_parameters()

    def head_parameters(self) -> list[nn.Parameter]:
        visual_ids = {id(parameter) for parameter in self.detector.visual.parameters()}
        return [
            parameter
            for parameter in self.parameters()
            if id(parameter) not in visual_ids
        ]

    def export_checkpoint(self, external_rows: int) -> dict[str, object]:
        head_state = self.detector.trainable_state_dict()
        with torch.no_grad():
            head_state["classifier.weight"] = torch.stack(
                [head.weight.detach().cpu() for head in self.logit_heads.values()]
            ).mean(dim=0)
            head_state["classifier.bias"] = torch.stack(
                [head.bias.detach().cpu() for head in self.logit_heads.values()]
            ).mean(dim=0)
        return {
            "format": "aigc-clip-mold-distilled-v1",
            "backbone": int(self.detector.visual.input_resolution),
            "feature_blocks": self.detector.feature_blocks,
            "detector_head": head_state,
            "external_rows": external_rows,
            "competition_rows_accessed": 0,
            "teachers": sorted(self.logit_heads),
        }


def load_teacher_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"sample_key", "logit", "embedding"}
        missing = required - set(payload.files)
        if missing:
            raise ValueError(f"teacher cache {path} misses {sorted(missing)}")
        return {name: payload[name] for name in required}
