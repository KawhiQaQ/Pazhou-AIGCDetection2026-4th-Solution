#!/usr/bin/env python3
"""Train full-data V86 dual-PE fusion residual and infer cached test statistics."""

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

from train_dual_pe_multidepth_fusion import DualPEFusion, load_branch, load_cache
from train_pe_core_linear import PROJECT_ROOT, resolve, seed_everything


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
    core_hash = str(model_config["core_encoder_sha256"])
    spatial_hash = str(model_config["spatial_encoder_sha256"])
    core_competition = load_cache(
        resolve(root, data_config["core_competition_features"]), core_hash, layers
    )
    core_external = load_cache(
        resolve(root, data_config["core_external_features"]), core_hash, layers
    )
    core_test = load_cache(resolve(root, data_config["core_test_features"]), core_hash, layers)
    spatial_competition = load_cache(
        resolve(root, data_config["spatial_competition_features"]), spatial_hash, layers
    )
    spatial_external = load_cache(
        resolve(root, data_config["spatial_external_features"]), spatial_hash, layers
    )
    spatial_test = load_cache(
        resolve(root, data_config["spatial_test_features"]), spatial_hash, layers
    )
    for name, first, second in (
        ("competition", core_competition, spatial_competition),
        ("external", core_external, spatial_external),
        ("test", core_test, spatial_test),
    ):
        if not np.array_equal(first["ids"], second["ids"]):
            raise RuntimeError(f"{name} Core/Spatial identities differ")
        if not np.array_equal(first["labels"], second["labels"]):
            raise RuntimeError(f"{name} Core/Spatial labels differ")
    template = pd.read_csv(resolve(root, data_config["submission_example_csv"]), dtype={"id": str})
    if core_test["ids"].tolist() != template.id.astype(str).tolist() or len(template) != 20_000:
        raise RuntimeError("cached test IDs/order differ from template")

    statistic_dim = int(core_competition["features"].shape[2])
    projection_dim = int(model_config["projection_dim"])
    core_parent_path = resolve(root, model_config["core_parent_checkpoint"])
    spatial_parent_path = resolve(root, model_config["spatial_parent_checkpoint"])
    core_branch = load_branch(
        core_parent_path,
        core_hash,
        layers,
        statistic_dim,
        projection_dim,
        float(model_config["parent_dropout"]),
    )
    spatial_branch = load_branch(
        spatial_parent_path,
        spatial_hash,
        layers,
        statistic_dim,
        projection_dim,
        float(model_config["parent_dropout"]),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DualPEFusion(
        core_branch,
        spatial_branch,
        projection_dim,
        int(model_config["hidden_dim"]),
        float(model_config["dropout"]),
    ).to(device)
    train_core = np.concatenate(
        [core_external["features"], core_competition["features"]]
    ).astype(np.float32)
    train_spatial = np.concatenate(
        [spatial_external["features"], spatial_competition["features"]]
    ).astype(np.float32)
    train_labels = np.concatenate(
        [core_external["labels"], core_competition["labels"]]
    ).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_core),
            torch.from_numpy(train_spatial),
            torch.from_numpy(train_labels),
        ),
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        drop_last=False,
    )
    optimizer = torch.optim.AdamW(
        model.residual.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    ranking_weight = float(train_config["ranking_weight"])
    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        total_sum = bce_sum = rank_sum = 0.0
        seen = 0
        for core_features, spatial_features, targets in loader:
            core_features = core_features.to(device)
            spatial_features = spatial_features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(core_features, spatial_features)
            bce = F.binary_cross_entropy_with_logits(logits, targets)
            positive = logits[targets > 0.5]
            negative = logits[targets <= 0.5]
            ranking = (
                F.softplus(-(positive[:, None] - negative[None, :])).mean()
                if len(positive) and len(negative)
                else logits.sum() * 0.0
            )
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
    for start in range(0, len(core_test["features"]), 256):
        core_features = torch.from_numpy(
            core_test["features"][start : start + 256].astype(np.float32)
        ).to(device)
        spatial_features = torch.from_numpy(
            spatial_test["features"][start : start + 256].astype(np.float32)
        ).to(device)
        with torch.no_grad():
            logits, _, _ = model(core_features, spatial_features)
            score_parts.append(torch.sigmoid(logits).cpu().numpy())
    scores = np.concatenate(score_parts).astype(np.float64)
    if len(scores) != 20_000 or not np.isfinite(scores).all():
        raise RuntimeError("invalid dual-PE test scores")
    output_dir = resolve(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "dual_pe_fusion.pth"
    torch.save(
        {
            "format": "aigc-dual-pe-multidepth-fusion-v1",
            "core_encoder_sha256": core_hash,
            "spatial_encoder_sha256": spatial_hash,
            "layer_indices": layers,
            "core_parent_checkpoint": str(core_parent_path),
            "spatial_parent_checkpoint": str(spatial_parent_path),
            "residual_state_dict": {
                key: value.detach().cpu() for key, value in model.residual.state_dict().items()
            },
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
        "core_parent_checkpoint": str(core_parent_path),
        "spatial_parent_checkpoint": str(spatial_parent_path),
        "parents_frozen": True,
        "external_rows": int(len(core_external["labels"])),
        "competition_training_rows": int(len(core_competition["labels"])),
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
