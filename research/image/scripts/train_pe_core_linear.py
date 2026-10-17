#!/usr/bin/env python3
"""Frozen PE-Core feature extraction and deterministic linear probing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import yaml

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aigc_detection.metrics import binary_metrics  # noqa: E402
from safetensors.torch import load_file as load_safetensors  # noqa: E402


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ImageRows(Dataset):
    def __init__(self, rows: list[dict[str, object]], transform: transforms.Compose) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, float, str]:
        row = self.rows[index]
        with Image.open(str(row["path"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensor = self.transform(image)
        return tensor, float(row["label"]), str(row["id"])


def build_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(input_size, interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def load_encoder(config: dict[str, object], root: Path, device: torch.device) -> tuple[nn.Module, Path, str]:
    model_config = config["model"]
    assert isinstance(model_config, dict)
    checkpoint = resolve(root, str(model_config["checkpoint"]))
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing PE-Core checkpoint: {checkpoint}")
    actual_hash = sha256_file(checkpoint)
    expected_hash = str(model_config["checkpoint_sha256"])
    if actual_hash != expected_hash:
        raise RuntimeError(f"PE-Core checkpoint hash mismatch: {actual_hash} != {expected_hash}")
    model = timm.create_model(str(model_config["backbone"]), pretrained=False)
    state = load_safetensors(str(checkpoint), device="cpu")
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"PE-Core state mismatch: missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )
    del state
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval().to(device)
    return model, checkpoint, actual_hash


@torch.inference_mode()
def extract_features(
    model: nn.Module,
    rows: list[dict[str, object]],
    transform: transforms.Compose,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        ImageRows(rows, transform),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    feature_parts: list[np.ndarray] = []
    labels: list[float] = []
    ids: list[str] = []
    seen = 0
    started = time.time()
    progress_interval = 25 if len(rows) <= 2_000 else 500
    for images, batch_labels, batch_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            output = model(images)
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise RuntimeError(f"unexpected PE-Core output: {type(output)}")
        output = F.normalize(output.float(), dim=1)
        if not torch.isfinite(output).all():
            raise RuntimeError("PE-Core produced non-finite embeddings")
        feature_parts.append(output.cpu().numpy().astype(np.float16))
        labels.extend(batch_labels.numpy().tolist())
        ids.extend(str(value) for value in batch_ids)
        seen += len(batch_ids)
        if seen % progress_interval == 0 or seen == len(rows):
            print(
                json.dumps(
                    {
                        "features": f"{seen}/{len(rows)}",
                        "elapsed_seconds": round(time.time() - started, 2),
                    }
                ),
                flush=True,
            )
    return (
        np.concatenate(feature_parts, axis=0),
        np.asarray(labels, dtype=np.float32),
        np.asarray(ids),
    )


def load_or_extract_cache(
    cache_path: Path,
    rows: list[dict[str, object]],
    model: nn.Module | None,
    transform: transforms.Compose,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    checkpoint_sha256: str,
    input_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    expected_ids = np.asarray([str(row["id"]) for row in rows])
    if cache_path.is_file():
        cached = np.load(cache_path)
        if str(cached["checkpoint_sha256"].item()) != checkpoint_sha256:
            raise RuntimeError(f"stale feature checkpoint hash: {cache_path}")
        if int(cached["input_size"].item()) != input_size:
            raise RuntimeError(f"stale feature input size: {cache_path}")
        if not np.array_equal(cached["ids"].astype(str), expected_ids):
            raise RuntimeError(f"stale feature row identity: {cache_path}")
        return cached["features"], cached["labels"], cached["ids"]
    if model is None:
        raise RuntimeError(f"missing encoder for uncached feature extraction: {cache_path}")
    features, labels, ids = extract_features(
        model,
        rows,
        transform,
        batch_size=batch_size,
        workers=workers,
        device=device,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
        ids=ids,
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        input_size=np.asarray(input_size),
    )
    return features, labels, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = yaml.safe_load(resolve(root, str(args.config)).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["train"]
    assert isinstance(model_config, dict)
    assert isinstance(data_config, dict)
    assert isinstance(train_config, dict)

    labels = pd.read_csv(resolve(root, str(data_config["labels_csv"])))
    folds = pd.read_csv(resolve(root, str(data_config["folds_csv"])))
    competition = labels.merge(folds[["id", "fold", "group_id"]], on="id", validate="one_to_one")
    competition_rows = [
        {
            "id": str(row.id),
            "label": int(row.label),
            "path": resolve(root, str(data_config["image_dir"])) / str(row.id),
        }
        for row in competition.itertuples(index=False)
    ]
    external = pd.read_csv(resolve(root, str(data_config["external_manifest_csv"])))
    external_root = resolve(root, str(data_config["external_dir"])).resolve()
    if "competition" in external_root.parts:
        raise RuntimeError("external feature source resolves inside competition data")
    external_rows = [
        {
            "id": str(row.sample_key),
            "label": int(row.label),
            "path": external_root / str(row.relative_path),
        }
        for row in external.itertuples(index=False)
    ]
    if any(not Path(str(row["path"])).is_file() for row in competition_rows + external_rows):
        raise RuntimeError("PE-Core input manifest contains missing images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = int(data_config["input_size"])
    transform = build_transform(input_size)
    cache_dir = resolve(root, str(data_config["feature_cache_dir"]))
    competition_cache = cache_dir / "competition_center_features.npz"
    external_cache = cache_dir / "openfake_center_features.npz"
    checkpoint = resolve(root, str(model_config["checkpoint"]))
    expected_hash = str(model_config["checkpoint_sha256"])
    caches_complete = competition_cache.is_file() and external_cache.is_file()
    if args.smoke_only or not caches_complete:
        model, checkpoint, checkpoint_hash = load_encoder(config, root, device)
    else:
        if not checkpoint.is_file():
            raise FileNotFoundError(f"missing PE-Core checkpoint: {checkpoint}")
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != expected_hash:
            raise RuntimeError(
                f"PE-Core checkpoint hash mismatch: {checkpoint_hash} != {expected_hash}"
            )
        model = None
    if args.smoke_only:
        features, smoke_labels, _ = extract_features(
            model,
            competition_rows[:2],
            transform,
            batch_size=1,
            workers=0,
            device=device,
        )
        head = nn.Linear(features.shape[1], 1).to(device)
        optimizer = torch.optim.AdamW(head.parameters(), lr=float(train_config["lr"]))
        batch = torch.from_numpy(features.astype(np.float32)).to(device)
        targets = torch.from_numpy(smoke_labels).to(device)
        before = head.weight.detach().clone()
        loss = F.binary_cross_entropy_with_logits(head(batch).squeeze(1), targets)
        loss.backward()
        optimizer.step()
        print(
            json.dumps(
                {
                    "smoke": "passed",
                    "loss": float(loss.detach()),
                    "parameter_changed": bool(not torch.equal(before, head.weight.detach())),
                    "competition_training_rows_accessed": 0,
                }
            ),
            flush=True,
        )
        return

    competition_features, competition_labels, competition_ids = load_or_extract_cache(
        competition_cache,
        competition_rows,
        model,
        transform,
        batch_size=int(data_config["batch_size"]),
        workers=int(data_config["workers"]),
        device=device,
        checkpoint_sha256=checkpoint_hash,
        input_size=input_size,
    )
    external_features, external_labels, _ = load_or_extract_cache(
        external_cache,
        external_rows,
        model,
        transform,
        batch_size=int(data_config["batch_size"]),
        workers=int(data_config["workers"]),
        device=device,
        checkpoint_sha256=checkpoint_hash,
        input_size=input_size,
    )
    del model
    torch.cuda.empty_cache()

    fold = int(config["fold"])
    train_mask = competition.fold.to_numpy() != fold
    valid_mask = ~train_mask
    if set(competition.loc[train_mask, "group_id"]) & set(competition.loc[valid_mask, "group_id"]):
        raise RuntimeError("duplicate groups cross PE-Core train/validation")
    train_features = np.concatenate(
        [external_features.astype(np.float32), competition_features[train_mask].astype(np.float32)]
    )
    train_labels = np.concatenate([external_labels, competition_labels[train_mask]])
    valid_features = competition_features[valid_mask].astype(np.float32)
    valid_labels = competition_labels[valid_mask]
    valid_ids = competition_ids[valid_mask]

    generator = torch.Generator().manual_seed(int(config["seed"]))
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels)),
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )
    head = nn.Linear(train_features.shape[1], 1).to(device)
    nn.init.zeros_(head.weight)
    nn.init.zeros_(head.bias)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(train_config["epochs"]) + 1):
        head.train()
        loss_sum = 0.0
        rows_seen = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.binary_cross_entropy_with_logits(
                head(batch_features).squeeze(1), batch_labels
            )
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_labels)
            rows_seen += len(batch_labels)
        if epoch == 1 or epoch % 10 == 0 or epoch == int(train_config["epochs"]):
            entry = {"epoch": epoch, "train_bce": loss_sum / rows_seen}
            history.append(entry)
            print(json.dumps(entry), flush=True)

    head.eval()
    with torch.no_grad():
        logits = head(torch.from_numpy(valid_features).to(device)).squeeze(1)
        scores = torch.sigmoid(logits).cpu().numpy()
    metrics = binary_metrics(valid_labels.astype(np.int8), scores)
    output_dir = resolve(root, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame({"id": valid_ids, "label": valid_labels, "score": scores})
    predictions.to_csv(output_dir / "valid_predictions.csv", index=False)
    torch.save(
        {
            "format": "aigc-pe-core-linear-v1",
            "backbone": str(model_config["backbone"]),
            "checkpoint_sha256": checkpoint_hash,
            "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        },
        output_dir / "linear_head.pth",
    )
    result = {
        "experiment": str(config["experiment"]),
        "fold": fold,
        "metrics": metrics,
        "history": history,
        "audit": {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": checkpoint_hash,
            "safe_load": "safetensors strict state dict",
            "encoder_frozen": True,
            "embedding_l2_normalized": True,
            "external_rows": int(len(external_rows)),
            "competition_training_rows_accessed": int(train_mask.sum()),
            "competition_validation_rows": int(valid_mask.sum()),
            "validation_rows_used_for_model_selection": 0,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
