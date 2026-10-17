#!/usr/bin/env python3
"""Add a zero-preserving nonlinear residual to a validated PE-Core linear head."""

from __future__ import annotations

import argparse
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

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from aigc_detection.metrics import binary_metrics  # noqa: E402


class ResidualMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


def load_cache(path: Path, expected_checkpoint: str) -> dict[str, np.ndarray]:
    cached = np.load(path)
    if str(cached["checkpoint_sha256"].item()) != expected_checkpoint:
        raise RuntimeError(f"feature checkpoint mismatch: {path}")
    return {
        "features": cached["features"].astype(np.float32),
        "labels": cached["labels"].astype(np.float32),
        "ids": cached["ids"].astype(str),
    }


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
    expected_checkpoint = str(model_config["checkpoint_sha256"])

    competition = load_cache(
        resolve(root, str(data_config["competition_features"])), expected_checkpoint
    )
    external = load_cache(
        resolve(root, str(data_config["external_features"])), expected_checkpoint
    )
    folds = pd.read_csv(resolve(root, str(data_config["folds_csv"])))
    fold_by_id = folds.assign(id=folds["id"].astype(str)).set_index("id")["fold"]
    group_by_id = folds.assign(id=folds["id"].astype(str)).set_index("id")["group_id"]
    fold_values = np.asarray([int(fold_by_id.loc[item]) for item in competition["ids"]])
    group_values = np.asarray([str(group_by_id.loc[item]) for item in competition["ids"]])
    fold = int(config["fold"])
    train_mask = fold_values != fold
    valid_mask = ~train_mask
    if set(group_values[train_mask]) & set(group_values[valid_mask]):
        raise RuntimeError("duplicate groups cross V54 training/validation")

    train_features = np.concatenate(
        [external["features"], competition["features"][train_mask]]
    )
    train_labels = np.concatenate(
        [external["labels"], competition["labels"][train_mask]]
    )
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels)),
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        drop_last=False,
    )

    base_checkpoint_path = resolve(root, str(model_config["base_head_checkpoint"]))
    base_checkpoint = torch.load(base_checkpoint_path, map_location="cpu", weights_only=True)
    if base_checkpoint.get("format") != "aigc-pe-core-linear-v1":
        raise RuntimeError("unexpected PE-Core base head format")
    feature_dim = int(train_features.shape[1])
    base_head = nn.Linear(feature_dim, 1)
    base_head.load_state_dict(base_checkpoint["state_dict"], strict=True)
    for parameter in base_head.parameters():
        parameter.requires_grad = False
    base_head.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_head.to(device)
    residual = ResidualMLP(
        feature_dim,
        int(model_config["hidden_dim"]),
        float(model_config["dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        residual.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )

    if args.smoke_only:
        batch_features, batch_labels = next(iter(loader))
        batch_features = batch_features.to(device)
        with torch.no_grad():
            base_logits = base_head(batch_features).squeeze(1)
        initial_logits = base_logits + residual(batch_features)
        if not torch.equal(base_logits, initial_logits):
            raise RuntimeError("V54 residual is not exactly zero at initialization")
        before = residual.network[-1].weight.detach().clone()
        loss = F.binary_cross_entropy_with_logits(initial_logits, batch_labels.to(device))
        loss.backward()
        optimizer.step()
        print(
            json.dumps(
                {
                    "smoke": "passed",
                    "loss": float(loss.detach()),
                    "exact_base_initialization": True,
                    "parameter_changed": not torch.equal(
                        before, residual.network[-1].weight.detach()
                    ),
                    "base_trainable_parameters": 0,
                    "validation_rows_accessed": 0,
                }
            ),
            flush=True,
        )
        return

    history: list[dict[str, float | int]] = []
    residual.train()
    for epoch in range(1, int(train_config["epochs"]) + 1):
        loss_sum = 0.0
        rows_seen = 0
        for batch_features, batch_labels in loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            with torch.no_grad():
                base_logits = base_head(batch_features).squeeze(1)
            optimizer.zero_grad(set_to_none=True)
            logits = base_logits + residual(batch_features)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_labels)
            rows_seen += len(batch_labels)
        record = {"epoch": epoch, "train_bce": loss_sum / rows_seen}
        history.append(record)
        print(json.dumps(record), flush=True)

    valid_features = torch.from_numpy(competition["features"][valid_mask]).to(device)
    valid_labels = competition["labels"][valid_mask]
    residual.eval()
    with torch.no_grad():
        logits = base_head(valid_features).squeeze(1) + residual(valid_features)
        scores = torch.sigmoid(logits).cpu().numpy()

    output_dir = resolve(root, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": competition["ids"][valid_mask],
            "label": valid_labels.astype(np.int8),
            "score": scores,
        }
    ).to_csv(output_dir / "valid_predictions.csv", index=False)
    torch.save(
        {
            "format": "aigc-pe-core-residual-mlp-v1",
            "checkpoint_sha256": expected_checkpoint,
            "base_head_checkpoint": str(base_checkpoint_path),
            "state_dict": {
                key: value.detach().cpu() for key, value in residual.state_dict().items()
            },
        },
        output_dir / "residual_mlp.pth",
    )
    result = {
        "experiment": str(config["experiment"]),
        "fold": fold,
        "metrics": binary_metrics(valid_labels.astype(np.int8), scores),
        "history": history,
        "audit": {
            "encoder_frozen": True,
            "base_head_frozen": True,
            "zero_preserving_initialization": True,
            "trainable_parameters": sum(p.numel() for p in residual.parameters()),
            "competition_training_rows": int(train_mask.sum()),
            "competition_validation_rows": int(valid_mask.sum()),
            "external_rows": int(len(external["labels"])),
            "validation_rows_used_for_model_selection": 0,
            "competition_test_rows_accessed": 0,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
