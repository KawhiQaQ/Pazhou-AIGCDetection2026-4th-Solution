#!/usr/bin/env python3
"""Train a fixed frozen-PE multi-depth head on all labels and infer cached test features."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from train_pe_core_linear import PROJECT_ROOT, resolve, seed_everything
from train_pe_core_multidepth_classifier import load_statistics
from train_pe_core_multidepth_residual import MultiDepthPatchResidual


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
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["train"]
    layers = [int(value) for value in model_config["layer_indices"]]
    encoder_sha256 = str(model_config["checkpoint_sha256"])
    competition = load_statistics(
        resolve(root, data_config["competition_multidepth_features"]), encoder_sha256, layers
    )
    external = load_statistics(
        resolve(root, data_config["external_multidepth_features"]), encoder_sha256, layers
    )
    test = load_statistics(
        resolve(root, data_config["test_multidepth_features"]), encoder_sha256, layers
    )
    template = pd.read_csv(resolve(root, data_config["submission_example_csv"]), dtype={"id": str})
    if test["ids"].tolist() != template.id.astype(str).tolist() or len(template) != 20_000:
        raise RuntimeError("cached test IDs/order differ from template")
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiDepthPatchResidual(
        train_features.shape[1],
        train_features.shape[2],
        int(model_config["projection_dim"]),
        float(model_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    ranking_weight = float(train_config.get("ranking_weight", 0.0))
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        total_sum = bce_sum = rank_sum = 0.0
        seen = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features)
            bce = F.binary_cross_entropy_with_logits(logits, targets)
            positive = logits[targets > 0.5]
            negative = logits[targets <= 0.5]
            if len(positive) and len(negative):
                ranking = F.softplus(-(positive[:, None] - negative[None, :])).mean()
            else:
                ranking = logits.sum() * 0.0
            loss = bce + ranking_weight * ranking
            loss.backward()
            optimizer.step()
            count = len(targets)
            total_sum += float(loss.detach()) * count
            bce_sum += float(bce.detach()) * count
            rank_sum += float(ranking.detach()) * count
            seen += count
        record = {
            "epoch": epoch,
            "train_total": total_sum / seen,
            "train_bce": bce_sum / seen,
            "train_pairwise_ranking": rank_sum / seen,
        }
        history.append(record)
        print(json.dumps(record), flush=True)
    model.eval()
    score_parts: list[np.ndarray] = []
    for start in range(0, len(test["features"]), 256):
        features = torch.from_numpy(
            test["features"][start : start + 256].astype(np.float32)
        ).to(device)
        with torch.no_grad():
            logits, _ = model(features)
            score_parts.append(torch.sigmoid(logits).cpu().numpy())
    scores = np.concatenate(score_parts).astype(np.float64)
    if len(scores) != 20_000 or not np.isfinite(scores).all():
        raise RuntimeError("invalid multi-depth test scores")
    output_dir = resolve(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "multidepth_classifier.pth"
    torch.save(
        {
            "format": "aigc-pe-core-multidepth-patch-classifier-v1",
            "encoder_checkpoint_sha256": encoder_sha256,
            "layer_indices": layers,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
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
        "encoder_checkpoint_sha256": encoder_sha256,
        "encoder_frozen": True,
        "layer_indices": layers,
        "external_rows": int(len(external["labels"])),
        "competition_training_rows": int(len(competition["labels"])),
        "test_rows": int(len(scores)),
        "pairwise_ranking_weight": ranking_weight,
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
