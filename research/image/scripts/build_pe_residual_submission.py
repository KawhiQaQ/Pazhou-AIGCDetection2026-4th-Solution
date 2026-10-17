#!/usr/bin/env python3
"""Train a fixed PE residual head on all labeled rows and infer cached test features."""

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
import yaml

from train_pe_core_linear import PROJECT_ROOT, resolve, seed_everything
from train_pe_core_residual_mlp import ResidualMLP


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_cache(path: Path, encoder_sha256: str) -> dict[str, np.ndarray]:
    cached = np.load(path)
    if str(cached["checkpoint_sha256"].item()) != encoder_sha256:
        raise RuntimeError(f"feature encoder mismatch: {path}")
    return {
        "features": cached["features"],
        "labels": cached["labels"].astype(np.float32),
        "ids": cached["ids"].astype(str),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load(resolve(root, str(args.config)).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["train"]
    encoder_sha256 = str(model_config["encoder_sha256"])

    competition = load_cache(resolve(root, data_config["competition_features"]), encoder_sha256)
    external = load_cache(resolve(root, data_config["external_features"]), encoder_sha256)
    test = load_cache(resolve(root, data_config["test_features"]), encoder_sha256)
    template = pd.read_csv(resolve(root, data_config["submission_example_csv"]), dtype={"id": str})
    if list(template.columns) != ["id", "score"] or len(template) != 20_000:
        raise RuntimeError("unexpected submission template")
    if test["ids"].tolist() != template.id.astype(str).tolist():
        raise RuntimeError("cached test IDs/order differ from template")

    base_path = resolve(root, model_config["base_head_checkpoint"])
    base_payload = torch.load(base_path, map_location="cpu", weights_only=True)
    if base_payload.get("format") != "aigc-pe-core-linear-v1":
        raise RuntimeError("unexpected PE base head format")
    feature_dim = int(competition["features"].shape[1])
    base = nn.Linear(feature_dim, 1)
    base.load_state_dict(base_payload["state_dict"], strict=True)
    for parameter in base.parameters():
        parameter.requires_grad_(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base.eval().to(device)
    residual = ResidualMLP(
        feature_dim, int(model_config["hidden_dim"]), float(model_config["dropout"])
    ).to(device)
    train_features = np.concatenate(
        [external["features"], competition["features"]]
    ).astype(np.float32)
    train_labels = np.concatenate([external["labels"], competition["labels"]]).astype(np.float32)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels)),
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        drop_last=False,
    )
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
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                base_logits = base(features).squeeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = base_logits + residual(features)
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
    for start in range(0, len(test["features"]), 2048):
        features = torch.from_numpy(
            test["features"][start : start + 2048].astype(np.float32)
        ).to(device)
        with torch.no_grad():
            logits = base(features).squeeze(1) + residual(features)
            score_parts.append(torch.sigmoid(logits).cpu().numpy())
    scores = np.concatenate(score_parts).astype(np.float64)
    if len(scores) != 20_000 or not np.isfinite(scores).all():
        raise RuntimeError("invalid residual test scores")
    output_dir = resolve(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "residual_mlp.pth"
    torch.save(
        {
            "format": "aigc-pe-core-residual-mlp-v1",
            "checkpoint_sha256": encoder_sha256,
            "base_head_checkpoint": str(base_path),
            "state_dict": {key: value.detach().cpu() for key, value in residual.state_dict().items()},
        },
        checkpoint_path,
    )
    submission_path = resolve(root, config["submission_path"])
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": template.id.astype(str), "score": scores}).to_csv(
        submission_path, index=False
    )
    manifest = {
        "experiment": str(config["experiment"]),
        "cv_reference": config["cv_reference"],
        "encoder_sha256": encoder_sha256,
        "base_head_checkpoint": str(base_path),
        "base_frozen": True,
        "external_rows": int(len(external["labels"])),
        "competition_training_rows": int(len(competition["labels"])),
        "test_rows": int(len(scores)),
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
