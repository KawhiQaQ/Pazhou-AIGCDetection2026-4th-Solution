#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.special import expit
import torch
from torch.utils.data import DataLoader, Dataset
import yaml

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from aigc_detection.image_v1 import (  # noqa: E402
    CLIPAdapterMultiDepthPatchMILDetector,
    CLIPAdapterPatchMILDetector,
    CLIPForensicDetector,
    CLIPResidualAdapterDetector,
    LabeledImageDataset,
    build_train_transform,
    build_valid_transform,
    seed_everything,
)
from train_image_v1 import (  # noqa: E402
    autocast_context,
    load_external_trainable_checkpoint,
    make_optimizer,
    make_scheduler,
    train_epoch,
)


class TestImageDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, image_ids: list[str], image_dir: Path, transform) -> None:
        self.image_ids = image_ids
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        image_id = self.image_ids[index]
        with Image.open(self.image_dir / image_id) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, image_id


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_config(root: Path, path: Path) -> dict[str, object]:
    config = yaml.safe_load(resolve(root, str(path)).read_text(encoding="utf-8"))
    train_config = config["train"]
    assert isinstance(train_config, dict)
    # Reuse the fixed CV scheduler shape and stop at the selected full-data epoch.
    train_config["epochs"] = int(train_config["schedule_epochs"])
    return config


def build_model(
    root: Path, config: dict[str, object], device: torch.device
) -> torch.nn.Module:
    model_config = config["model"]
    assert isinstance(model_config, dict)
    architecture = str(model_config["architecture"])
    common = {
        "device": device,
        "download_root": resolve(root, str(model_config["clip_cache"])),
    }
    if architecture == "clip_forensic_finetune":
        model = CLIPForensicDetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["unfreeze_last_blocks"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            **common,
        ).to(device)
    elif architecture == "clip_residual_adapter":
        model = CLIPResidualAdapterDetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["adapter_last_blocks"]),
            int(model_config["adapter_dim"]),
            float(model_config["adapter_dropout"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            **common,
        ).to(device)
    elif architecture == "clip_adapter_patch_mil":
        model = CLIPAdapterPatchMILDetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["adapter_last_blocks"]),
            int(model_config["adapter_dim"]),
            float(model_config["adapter_dropout"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            int(model_config["patch_hidden_dim"]),
            float(model_config["patch_top_fraction"]),
            **common,
        ).to(device)
    elif architecture == "clip_adapter_multidepth_patch_mil":
        model = CLIPAdapterMultiDepthPatchMILDetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["adapter_last_blocks"]),
            int(model_config["adapter_dim"]),
            float(model_config["adapter_dropout"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            int(model_config["patch_hidden_dim"]),
            float(model_config["patch_top_fraction"]),
            **common,
        ).to(device)
    else:
        raise ValueError(f"unsupported validated submission architecture: {architecture}")

    if model_config.get("init_trainable_checkpoint"):
        load_external_trainable_checkpoint(
            model,
            resolve(root, str(model_config["init_trainable_checkpoint"])),
        )
    return model


def full_train(
    root: Path,
    config: dict[str, object],
    model: torch.nn.Module,
    device: torch.device,
    output_dir: Path,
) -> list[dict[str, object]]:
    data_config = config["data"]
    train_config = config["train"]
    assert isinstance(data_config, dict) and isinstance(train_config, dict)
    labels = pd.read_csv(resolve(root, str(data_config["labels_csv"])))
    if labels.id.duplicated().any() or set(labels.label.unique()) != {0, 1}:
        raise RuntimeError("full labels must contain unique IDs and both binary classes")
    image_dir = resolve(root, str(data_config["image_dir"]))
    dataset = LabeledImageDataset(
        labels[["id", "label"]].to_dict("records"), image_dir, build_train_transform(config)
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_config["train_batch_size"]),
        shuffle=True,
        drop_last=False,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        num_workers=int(data_config["workers"]),
        pin_memory=True,
        persistent_workers=int(data_config["workers"]) > 0,
    )
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, len(loader), config)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=128.0, growth_interval=2000
    )
    history: list[dict[str, object]] = []
    started = time.time()
    for epoch in range(1, int(train_config["full_train_epochs"]) + 1):
        metrics = train_epoch(model, loader, optimizer, scheduler, scaler, device, config)
        record = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - started,
            "train": metrics,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    checkpoint = output_dir / f"trainable_e{int(train_config['full_train_epochs']):03d}.pth"
    torch.save(model.trainable_state_dict(), checkpoint)
    return history


def load_trainable(model: torch.nn.Module, checkpoint: Path) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected checkpoint keys: {result.unexpected_keys}")
    missing_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in result.missing_keys
    ]
    if missing_trainable:
        raise RuntimeError(f"missing trainable checkpoint keys: {missing_trainable}")


@torch.no_grad()
def predict_test(
    root: Path,
    config: dict[str, object],
    model: torch.nn.Module,
    device: torch.device,
) -> pd.DataFrame:
    data_config = config["data"]
    assert isinstance(data_config, dict)
    test = pd.read_csv(resolve(root, str(data_config["test_csv"])))
    template = pd.read_csv(resolve(root, str(data_config["submission_example_csv"])))
    if list(template.columns) != ["id", "score"]:
        raise RuntimeError(f"unexpected submission template columns: {template.columns.tolist()}")
    if test.id.duplicated().any() or template.id.duplicated().any():
        raise RuntimeError("test or template IDs are duplicated")
    if test.id.astype(str).tolist() != template.id.astype(str).tolist():
        raise RuntimeError("test IDs and submission template IDs/order differ")

    dataset = TestImageDataset(
        test.id.astype(str).tolist(),
        resolve(root, str(data_config["test_image_dir"])),
        build_valid_transform(config),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_config["valid_batch_size"]),
        shuffle=False,
        drop_last=False,
        num_workers=int(data_config["workers"]),
        pin_memory=True,
        persistent_workers=int(data_config["workers"]) > 0,
    )
    model.eval()
    ids: list[str] = []
    logits_all: list[float] = []
    for batch_index, (images, image_ids) in enumerate(loader, start=1):
        batch_size, crops = images.shape[:2]
        images = images.flatten(0, 1).to(device, non_blocking=True)
        with autocast_context(device):
            logits, _ = model(images)
        logits = logits.float().view(batch_size, crops).mean(dim=1)
        ids.extend(image_ids)
        logits_all.extend(logits.cpu().numpy().astype(np.float64).tolist())
        if batch_index % 250 == 0 or batch_index == len(loader):
            print(f"predict {len(ids)}/{len(dataset)}", flush=True)

    if ids != template.id.astype(str).tolist():
        raise RuntimeError("prediction ID order differs from submission template")
    # Temperature scaling is strictly monotonic and avoids float sigmoid saturation/ties.
    scores = expit(np.asarray(logits_all, dtype=np.float64) / 4.0)
    submission = pd.DataFrame({"id": ids, "score": scores})
    if len(submission) != 20_000 or not np.isfinite(submission.score).all():
        raise RuntimeError("submission row count or score finiteness check failed")
    if not submission.score.between(0.0, 1.0, inclusive="both").all():
        raise RuntimeError("submission scores are outside [0, 1]")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/image_v1_clip.yaml"))
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = load_config(root, args.config)
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = resolve(root, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(root, config, device)
    train_config = config["train"]
    assert isinstance(train_config, dict)
    full_train_epochs = int(train_config["full_train_epochs"])
    checkpoint = output_dir / f"trainable_e{full_train_epochs:03d}.pth"
    submission_path = resolve(root, str(config["submission_path"]))
    manifest_path = submission_path.with_suffix(".manifest.json")

    if args.predict_only:
        load_trainable(model, checkpoint)
        history = (
            json.loads(manifest_path.read_text(encoding="utf-8")).get("history", [])
            if manifest_path.exists()
            else []
        )
    else:
        history = full_train(root, config, model, device, output_dir)

    submission = predict_test(root, config, model, device)
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    digest = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    manifest = {
        "experiment": config["experiment"],
        "cv_reference": config.get("cv_reference"),
        "full_train_rows": 1000,
        "full_train_epochs": full_train_epochs,
        "test_rows": int(len(submission)),
        "columns": submission.columns.tolist(),
        "score_min": float(submission.score.min()),
        "score_max": float(submission.score.max()),
        "score_unique": int(submission.score.nunique()),
        "submission": str(submission_path),
        "sha256": digest,
        "history": history,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
