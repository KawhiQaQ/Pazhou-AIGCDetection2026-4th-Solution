#!/usr/bin/env python3
"""Train full-data V60 CLIP residual over frozen full-data PE detector and infer test."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from aigc_detection.image_v1 import CLIP_MEAN, CLIP_STD
from train_cross_backbone_residual import (
    ClipResidual,
    build_clip_encoder,
    load_or_extract_clip_cache,
    load_pe_cache,
    pe_base_logits,
    resolve,
    sha256_file,
)
from train_pe_core_linear import seed_everything
from train_pe_core_residual_mlp import ResidualMLP


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load(resolve(root, str(args.config)).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    clip_config = config["clip"]
    pe_config = config["pe_base"]
    data_config = config["data"]
    train_config = config["train"]
    labels = pd.read_csv(resolve(root, data_config["labels_csv"]), dtype={"id": str})
    test = pd.read_csv(resolve(root, data_config["test_csv"]), dtype={"id": str})
    template = pd.read_csv(resolve(root, data_config["submission_example_csv"]), dtype={"id": str})
    if test.id.astype(str).tolist() != template.id.astype(str).tolist() or len(test) != 20_000:
        raise RuntimeError("test IDs/order differ from submission template")
    competition_dir = resolve(root, data_config["competition_dir"])
    test_dir = resolve(root, data_config["test_image_dir"])
    external_root = resolve(root, data_config["external_root"]).resolve()
    if "competition" in external_root.parts:
        raise RuntimeError("external root resolves inside competition data")
    rows: dict[str, list[dict[str, object]]] = {
        "competition": [
            {"id": str(row.id), "label": int(row.label), "path": competition_dir / str(row.id)}
            for row in labels.itertuples(index=False)
        ],
        "test": [
            {"id": str(row.id), "label": 0, "path": test_dir / str(row.id)}
            for row in test.itertuples(index=False)
        ],
    }
    for source, manifest_value in data_config["external_manifests"].items():
        manifest = pd.read_csv(resolve(root, str(manifest_value)))
        rows[source] = [
            {
                "id": str(row.sample_key),
                "label": int(row.label),
                "path": external_root / str(row.relative_path),
            }
            for row in manifest.itertuples(index=False)
        ]
    if any(not Path(str(row["path"])).is_file() for items in rows.values() for row in items):
        raise RuntimeError("full V60 manifest contains missing images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize(int(clip_config["resize_size"]), InterpolationMode.BICUBIC),
            transforms.CenterCrop(int(clip_config["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    checkpoint_path = resolve(root, clip_config["checkpoint"])
    checkpoint_hash = sha256_file(checkpoint_path)
    cache_dir = resolve(root, data_config["clip_feature_cache_dir"])
    cache_paths = {name: cache_dir / f"{name}_center_features.npz" for name in rows}
    if all(path.is_file() for path in cache_paths.values()):
        clip_model = None
    else:
        clip_model, loaded_hash = build_clip_encoder(config, root, device)
        if loaded_hash != checkpoint_hash:
            raise RuntimeError("frozen CLIP checkpoint changed while loading")
    clip_sets = {
        name: load_or_extract_clip_cache(
            cache_paths[name],
            items,
            clip_model,
            transform,
            batch_size=int(clip_config["batch_size"]),
            workers=int(clip_config["workers"]),
            device=device,
            checkpoint_sha256=checkpoint_hash,
        )
        for name, items in rows.items()
    }
    del clip_model
    torch.cuda.empty_cache()
    pe_sets = {
        name: load_pe_cache(
            resolve(root, pe_config["feature_caches"][name]),
            clip_sets[name]["ids"],
            clip_sets[name]["labels"],
        )
        for name in rows
    }
    linear_path = resolve(root, pe_config["linear_head_checkpoint"])
    linear_payload = torch.load(linear_path, map_location="cpu", weights_only=True)
    if linear_payload.get("format") != "aigc-pe-core-linear-v1":
        raise RuntimeError("unexpected full PE linear format")
    pe_dim = int(pe_sets["competition"].shape[1])
    pe_head = nn.Linear(pe_dim, 1).to(device)
    pe_head.load_state_dict(linear_payload["state_dict"], strict=True)
    residual_path = resolve(root, pe_config["residual_checkpoint"])
    residual_payload = torch.load(residual_path, map_location="cpu", weights_only=True)
    if residual_payload.get("format") != "aigc-pe-core-residual-mlp-v1":
        raise RuntimeError("unexpected full PE residual format")
    pe_residual = ResidualMLP(
        pe_dim, int(pe_config["hidden_dim"]), float(pe_config["dropout"])
    ).to(device)
    pe_residual.load_state_dict(residual_payload["state_dict"], strict=True)
    for module in (pe_head, pe_residual):
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        module.eval()
    base_logits = {
        name: pe_base_logits(features, pe_head, pe_residual, device)
        for name, features in pe_sets.items()
    }
    train_names = ["competition", *data_config["external_manifests"].keys()]
    train_clip = np.concatenate([clip_sets[name]["features"] for name in train_names]).astype(
        np.float32
    )
    train_base = np.concatenate([base_logits[name] for name in train_names]).astype(np.float32)
    train_labels = np.concatenate([clip_sets[name]["labels"] for name in train_names]).astype(
        np.float32
    )
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_clip), torch.from_numpy(train_base), torch.from_numpy(train_labels)
        ),
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        drop_last=False,
    )
    residual = ClipResidual(
        train_clip.shape[1], int(train_config["hidden_dim"]), float(train_config["dropout"])
    ).to(device)
    optimizer = torch.optim.AdamW(
        residual.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(train_config["epochs"]) + 1):
        residual.train()
        loss_sum = 0.0
        seen = 0
        for clip_features, base_batch, targets in loader:
            clip_features = clip_features.to(device)
            base_batch = base_batch.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = base_batch + residual(clip_features)
            loss = F.binary_cross_entropy_with_logits(logits, targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            seen += len(targets)
        record = {"epoch": epoch, "train_bce": loss_sum / seen}
        history.append(record)
        print(json.dumps(record), flush=True)
    residual.eval()
    score_parts: list[np.ndarray] = []
    test_clip = clip_sets["test"]["features"]
    for start in range(0, len(test_clip), 2048):
        features = torch.from_numpy(test_clip[start : start + 2048].astype(np.float32)).to(device)
        base_batch = torch.from_numpy(base_logits["test"][start : start + 2048]).to(device)
        with torch.no_grad():
            score_parts.append(torch.sigmoid(base_batch + residual(features)).cpu().numpy())
    scores = np.concatenate(score_parts).astype(np.float64)
    if clip_sets["test"]["ids"].astype(str).tolist() != template.id.astype(str).tolist():
        raise RuntimeError("cached CLIP test IDs/order differ from template")
    output_dir = resolve(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "aigc-cross-backbone-residual-v1",
            "clip_checkpoint_sha256": checkpoint_hash,
            "state_dict": {key: value.detach().cpu() for key, value in residual.state_dict().items()},
        },
        output_dir / "clip_residual.pth",
    )
    submission_path = resolve(root, config["submission_path"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": template.id.astype(str), "score": scores}).to_csv(
        submission_path, index=False
    )
    manifest = {
        "experiment": str(config["experiment"]),
        "cv_reference": config["cv_reference"],
        "clip_checkpoint_sha256": checkpoint_hash,
        "clip_encoder_frozen": True,
        "pe_base_frozen": True,
        "external_rows": {name: len(rows[name]) for name in data_config["external_manifests"]},
        "competition_training_rows": len(rows["competition"]),
        "test_rows": len(scores),
        "history": history,
        "submission": str(submission_path),
        "sha256": sha256(submission_path),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_unique": int(pd.Series(scores).nunique()),
    }
    submission_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
