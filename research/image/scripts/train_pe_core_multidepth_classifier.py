#!/usr/bin/env python3
"""Train an independent frozen-PE multi-depth patch-statistics classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from train_pe_core_linear import PROJECT_ROOT, resolve, seed_everything
from train_pe_core_multidepth_residual import MultiDepthPatchResidual

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from aigc_detection.metrics import binary_metrics  # noqa: E402


def load_statistics(path: Path, expected_checkpoint: str, layer_indices: list[int]) -> dict[str, np.ndarray]:
    cached = np.load(path)
    if str(cached["checkpoint_sha256"].item()) != expected_checkpoint:
        raise RuntimeError(f"feature checkpoint mismatch: {path}")
    if not np.array_equal(cached["layer_indices"], np.asarray(layer_indices)):
        raise RuntimeError(f"feature layer mismatch: {path}")
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
    expected_checkpoint = str(model_config["checkpoint_sha256"])
    layer_indices = [int(value) for value in model_config["layer_indices"]]
    competition = load_statistics(
        resolve(root, str(data_config["competition_multidepth_features"])),
        expected_checkpoint,
        layer_indices,
    )
    external = load_statistics(
        resolve(root, str(data_config["external_multidepth_features"])),
        expected_checkpoint,
        layer_indices,
    )
    auxiliary_sources = 0
    if data_config.get("competition_auxiliary_features") and data_config.get(
        "external_auxiliary_features"
    ):
        competition_auxiliary = load_statistics(
            resolve(root, str(data_config["competition_auxiliary_features"])),
            expected_checkpoint,
            layer_indices,
        )
        external_auxiliary = load_statistics(
            resolve(root, str(data_config["external_auxiliary_features"])),
            expected_checkpoint,
            layer_indices,
        )
        for name, primary, auxiliary in (
            ("competition", competition, competition_auxiliary),
            ("external", external, external_auxiliary),
        ):
            if not np.array_equal(primary["ids"], auxiliary["ids"]):
                raise RuntimeError(f"{name} primary/auxiliary feature identity mismatch")
            if not np.array_equal(primary["labels"], auxiliary["labels"]):
                raise RuntimeError(f"{name} primary/auxiliary feature label mismatch")
            primary["features"] = np.concatenate(
                [primary["features"], auxiliary["features"]], axis=2
            )
        auxiliary_sources = 1

    folds = pd.read_csv(resolve(root, str(data_config["folds_csv"])))
    fold_by_id = folds.assign(id=folds["id"].astype(str)).set_index("id")["fold"]
    group_by_id = folds.assign(id=folds["id"].astype(str)).set_index("id")["group_id"]
    fold_values = np.asarray([int(fold_by_id.loc[item]) for item in competition["ids"]])
    group_values = np.asarray([str(group_by_id.loc[item]) for item in competition["ids"]])
    fold = int(config["fold"])
    train_mask = fold_values != fold
    valid_mask = ~train_mask
    if set(group_values[train_mask]) & set(group_values[valid_mask]):
        raise RuntimeError("duplicate groups cross multi-depth classifier training/validation")

    train_features = np.concatenate(
        [external["features"], competition["features"][train_mask]]
    ).astype(np.float32)
    train_labels = np.concatenate(
        [external["labels"], competition["labels"][train_mask]]
    ).astype(np.float32)
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
    parent_path = None
    if model_config.get("parent_checkpoint"):
        parent_path = resolve(root, str(model_config["parent_checkpoint"]))
        parent = torch.load(parent_path, map_location="cpu", weights_only=True)
        if parent.get("format") != "aigc-pe-core-multidepth-patch-classifier-v1":
            raise RuntimeError("unexpected multi-depth parent checkpoint format")
        if str(parent.get("encoder_checkpoint_sha256")) != expected_checkpoint:
            raise RuntimeError("multi-depth parent encoder mismatch")
        if list(parent.get("layer_indices", [])) != layer_indices:
            raise RuntimeError("multi-depth parent layer mismatch")
        model.load_state_dict(parent["state_dict"], strict=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    ranking_weight = float(train_config.get("ranking_weight", 0.0))

    history: list[dict[str, float | int]] = []
    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        loss_sum = 0.0
        binary_sum = 0.0
        ranking_sum = 0.0
        rows_seen = 0
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _ = model(features)
            binary_loss = F.binary_cross_entropy_with_logits(logits, targets)
            positive = logits[targets > 0.5]
            negative = logits[targets <= 0.5]
            if len(positive) and len(negative):
                ranking_loss = F.softplus(-(positive[:, None] - negative[None, :])).mean()
            else:
                ranking_loss = logits.sum() * 0.0
            loss = binary_loss + ranking_weight * ranking_loss
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            binary_sum += float(binary_loss.detach()) * len(targets)
            ranking_sum += float(ranking_loss.detach()) * len(targets)
            rows_seen += len(targets)
        record = {
            "epoch": epoch,
            "train_total": loss_sum / rows_seen,
            "train_bce": binary_sum / rows_seen,
            "train_pairwise_ranking": ranking_sum / rows_seen,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    model.eval()
    valid_features = torch.from_numpy(
        competition["features"][valid_mask].astype(np.float32)
    ).to(device)
    valid_labels = competition["labels"][valid_mask]
    with torch.no_grad():
        logits, gate_weights = model(valid_features)
        scores = torch.sigmoid(logits).cpu().numpy()
        gates = gate_weights.cpu().numpy()

    output_dir = resolve(root, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {"id": competition["ids"][valid_mask], "label": valid_labels.astype(np.int8), "score": scores}
    )
    for index, layer in enumerate(layer_indices):
        predictions[f"gate_layer_{layer}"] = gates[:, index]
    predictions.to_csv(output_dir / "valid_predictions.csv", index=False)
    torch.save(
        {
            "format": "aigc-pe-core-multidepth-patch-classifier-v1",
            "encoder_checkpoint_sha256": expected_checkpoint,
            "layer_indices": layer_indices,
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        },
        output_dir / "multidepth_classifier.pth",
    )
    result = {
        "experiment": str(config["experiment"]),
        "fold": fold,
        "metrics": binary_metrics(valid_labels.astype(np.int8), scores),
        "history": history,
        "diagnostics": {"validation_gate_mean": gates.mean(axis=0).tolist()},
        "audit": {
            "method": "independent MoLD-style dynamic fusion of PE-Core patch mean/std",
            "parent_checkpoint": None if parent_path is None else str(parent_path),
            "encoder_checkpoint_sha256": expected_checkpoint,
            "encoder_frozen": True,
            "layer_indices": layer_indices,
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "pairwise_ranking_weight": ranking_weight,
            "auxiliary_statistic_sources": auxiliary_sources,
            "external_rows": int(len(external["labels"])),
            "competition_training_rows": int(train_mask.sum()),
            "competition_validation_rows": int(valid_mask.sum()),
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
