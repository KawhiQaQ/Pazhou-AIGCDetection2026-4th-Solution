#!/usr/bin/env python3
"""Frozen PE-Core multi-depth patch-statistics residual over a validated detector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

from train_pe_core_linear import (
    PROJECT_ROOT,
    ImageRows,
    build_transform,
    load_encoder,
    resolve,
    seed_everything,
)
from train_pe_core_residual_mlp import ResidualMLP, load_cache

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from aigc_detection.metrics import binary_metrics  # noqa: E402


class MultiDepthPatchResidual(nn.Module):
    """MoLD-style dynamic pooling over evenly spaced PE-Core layer statistics."""

    def __init__(self, depths: int, statistic_dim: int, projection_dim: int, dropout: float) -> None:
        super().__init__()
        self.projections = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(statistic_dim),
                    nn.Linear(statistic_dim, projection_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(depths)
            ]
        )
        self.gate = nn.Linear(projection_dim, 1)
        self.output = nn.Sequential(nn.LayerNorm(projection_dim), nn.Linear(projection_dim, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)

    def forward(self, statistics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = torch.stack(
            [projection(statistics[:, index]) for index, projection in enumerate(self.projections)],
            dim=1,
        )
        gate_logits = self.gate(projected).squeeze(2)
        gate_weights = torch.softmax(gate_logits, dim=1)
        pooled = (gate_weights.unsqueeze(2) * projected).sum(dim=1)
        return self.output(pooled).squeeze(1), gate_weights


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.inference_mode()
def extract_statistics(
    encoder: nn.Module,
    rows: list[dict[str, object]],
    layer_indices: list[int],
    input_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(
        ImageRows(rows, build_transform(input_size)),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    parts: list[np.ndarray] = []
    labels: list[float] = []
    ids: list[str] = []
    seen = 0
    started = time.time()
    interval = 25 if len(rows) <= 2_000 else 500
    for images, batch_labels, batch_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            intermediate = encoder.forward_intermediates(
                images,
                indices=layer_indices,
                norm=True,
                output_fmt="NLC",
                intermediates_only=True,
            )
        if len(intermediate) != len(layer_indices):
            raise RuntimeError(f"unexpected PE-Core intermediate count: {len(intermediate)}")
        depth_statistics: list[torch.Tensor] = []
        for tokens in intermediate:
            tokens = tokens.float()
            if tokens.ndim != 3:
                raise RuntimeError(f"unexpected PE-Core token shape: {tuple(tokens.shape)}")
            mean = tokens.mean(dim=1)
            standard_deviation = tokens.var(dim=1, correction=0).clamp_min(1e-12).sqrt()
            depth_statistics.append(torch.cat([mean, standard_deviation], dim=1))
        statistics = torch.stack(depth_statistics, dim=1)
        if not torch.isfinite(statistics).all():
            raise RuntimeError("PE-Core produced non-finite multi-depth statistics")
        parts.append(statistics.cpu().numpy().astype(np.float16))
        labels.extend(batch_labels.numpy().tolist())
        ids.extend(str(value) for value in batch_ids)
        seen += len(batch_ids)
        if seen % interval == 0 or seen == len(rows):
            print(
                json.dumps(
                    {"multidepth_features": f"{seen}/{len(rows)}", "elapsed_seconds": round(time.time() - started, 2)}
                ),
                flush=True,
            )
    return np.concatenate(parts), np.asarray(labels, dtype=np.float32), np.asarray(ids)


def load_or_extract(
    cache_path: Path,
    rows: list[dict[str, object]],
    encoder: nn.Module,
    layer_indices: list[int],
    input_size: int,
    batch_size: int,
    workers: int,
    device: torch.device,
    checkpoint_hash: str,
) -> dict[str, np.ndarray]:
    expected_ids = np.asarray([str(row["id"]) for row in rows])
    if cache_path.is_file():
        cached = np.load(cache_path)
        if str(cached["checkpoint_sha256"].item()) != checkpoint_hash:
            raise RuntimeError(f"stale encoder hash: {cache_path}")
        if not np.array_equal(cached["layer_indices"], np.asarray(layer_indices)):
            raise RuntimeError(f"stale layer indices: {cache_path}")
        if not np.array_equal(cached["ids"].astype(str), expected_ids):
            raise RuntimeError(f"stale row identity: {cache_path}")
        return {key: cached[key] for key in ("features", "labels", "ids")}
    features, labels, ids = extract_statistics(
        encoder, rows, layer_indices, input_size, batch_size, workers, device
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
        ids=ids,
        checkpoint_sha256=np.asarray(checkpoint_hash),
        layer_indices=np.asarray(layer_indices),
        input_size=np.asarray(input_size),
    )
    return {"features": features, "labels": labels, "ids": ids}


def load_base_detector(
    feature_dim: int,
    base_path: Path,
    parent_path: Path,
    hidden_dim: int,
    dropout: float,
    device: torch.device,
) -> tuple[nn.Linear, ResidualMLP]:
    base_payload = torch.load(base_path, map_location="cpu", weights_only=True)
    if base_payload.get("format") != "aigc-pe-core-linear-v1":
        raise RuntimeError("unexpected PE-Core base head format")
    base_head = nn.Linear(feature_dim, 1)
    base_head.load_state_dict(base_payload["state_dict"], strict=True)
    parent_payload = torch.load(parent_path, map_location="cpu", weights_only=True)
    if parent_payload.get("format") != "aigc-pe-core-residual-mlp-v1":
        raise RuntimeError("unexpected parent residual format")
    parent = ResidualMLP(feature_dim, hidden_dim, dropout)
    parent.load_state_dict(parent_payload["state_dict"], strict=True)
    for module in (base_head, parent):
        for parameter in module.parameters():
            parameter.requires_grad = False
        module.eval().to(device)
    return base_head, parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--extract-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = yaml.safe_load(resolve(root, str(args.config)).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["train"]
    layer_indices = [int(value) for value in model_config["layer_indices"]]

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
        raise RuntimeError("external source resolves inside competition data")
    external_rows = [
        {
            "id": str(row.sample_key),
            "label": int(row.label),
            "path": external_root / str(row.relative_path),
        }
        for row in external.itertuples(index=False)
    ]
    if any(not Path(str(row["path"])).is_file() for row in competition_rows + external_rows):
        raise RuntimeError("multi-depth manifest contains missing images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, encoder_path, checkpoint_hash = load_encoder(config, root, device)
    input_size = int(data_config["input_size"])
    if args.smoke_only:
        smoke_features, smoke_labels, _ = extract_statistics(
            encoder, competition_rows[:2], layer_indices, input_size, 1, 0, device
        )
        residual = MultiDepthPatchResidual(
            len(layer_indices), smoke_features.shape[2], int(model_config["projection_dim"]), float(model_config["dropout"])
        ).to(device)
        batch = torch.from_numpy(smoke_features.astype(np.float32)).to(device)
        delta, _ = residual(batch)
        if not torch.equal(delta, torch.zeros_like(delta)):
            raise RuntimeError("multi-depth residual is not exactly zero at initialization")
        before = residual.output[-1].weight.detach().clone()
        loss = F.binary_cross_entropy_with_logits(delta, torch.from_numpy(smoke_labels).to(device))
        loss.backward()
        torch.optim.AdamW(residual.parameters(), lr=float(train_config["lr"])).step()
        print(
            json.dumps(
                {
                    "smoke": "passed",
                    "feature_shape": list(smoke_features.shape),
                    "exact_zero_initialization": True,
                    "parameter_changed": not torch.equal(before, residual.output[-1].weight.detach()),
                    "competition_training_rows_accessed": 0,
                    "validation_labels_accessed": 0,
                }
            ),
            flush=True,
        )
        return


    cache_dir = resolve(root, str(data_config["multidepth_cache_dir"]))
    competition_cache_path = resolve(
        root,
        str(
            data_config.get(
                "competition_multidepth_cache",
                cache_dir / "competition_multidepth_statistics.npz",
            )
        ),
    )
    external_cache_path = resolve(
        root,
        str(
            data_config.get(
                "external_multidepth_cache",
                cache_dir / "aigi_multidepth_statistics.npz",
            )
        ),
    )
    competition_multidepth = load_or_extract(
        competition_cache_path,
        competition_rows,
        encoder,
        layer_indices,
        input_size,
        int(data_config["batch_size"]),
        int(data_config["workers"]),
        device,
        checkpoint_hash,
    )
    external_multidepth = load_or_extract(
        external_cache_path,
        external_rows,
        encoder,
        layer_indices,
        input_size,
        int(data_config["batch_size"]),
        int(data_config["workers"]),
        device,
        checkpoint_hash,
    )
    if args.extract_only:
        print(
            json.dumps(
                {
                    "extract_only": "complete",
                    "competition_cache": str(competition_cache_path),
                    "competition_rows": len(competition_multidepth["ids"]),
                    "external_cache": str(external_cache_path),
                    "external_rows": len(external_multidepth["ids"]),
                    "encoder_checkpoint_sha256": checkpoint_hash,
                    "layer_indices": layer_indices,
                    "labels_used_for_training": 0,
                    "competition_test_rows_accessed": 0,
                }
            ),
            flush=True,
        )
        return
    del encoder
    torch.cuda.empty_cache()

    final_competition = load_cache(
        resolve(root, str(data_config["competition_final_features"])), checkpoint_hash
    )
    final_external = load_cache(
        resolve(root, str(data_config["external_final_features"])), checkpoint_hash
    )
    if not np.array_equal(competition_multidepth["ids"].astype(str), final_competition["ids"]):
        raise RuntimeError("competition multi-depth/final feature identity mismatch")
    if not np.array_equal(external_multidepth["ids"].astype(str), final_external["ids"]):
        raise RuntimeError("external multi-depth/final feature identity mismatch")

    fold = int(config["fold"])
    train_mask = competition.fold.to_numpy() != fold
    valid_mask = ~train_mask
    if set(competition.loc[train_mask, "group_id"]) & set(competition.loc[valid_mask, "group_id"]):
        raise RuntimeError("duplicate groups cross multi-depth training/validation")
    train_statistics = np.concatenate(
        [external_multidepth["features"], competition_multidepth["features"][train_mask]]
    ).astype(np.float32)
    train_final = np.concatenate(
        [final_external["features"], final_competition["features"][train_mask]]
    ).astype(np.float32)
    train_labels = np.concatenate(
        [external_multidepth["labels"], competition_multidepth["labels"][train_mask]]
    ).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_statistics),
            torch.from_numpy(train_final),
            torch.from_numpy(train_labels),
        ),
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        generator=torch.Generator().manual_seed(int(config["seed"])),
        drop_last=False,
    )

    base_path = resolve(root, str(model_config["base_head_checkpoint"]))
    parent_path = resolve(root, str(model_config["parent_residual_checkpoint"]))
    base_head, parent = load_base_detector(
        train_final.shape[1],
        base_path,
        parent_path,
        int(model_config["parent_hidden_dim"]),
        float(model_config["parent_dropout"]),
        device,
    )
    residual = MultiDepthPatchResidual(
        train_statistics.shape[1],
        train_statistics.shape[2],
        int(model_config["projection_dim"]),
        float(model_config["dropout"]),
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
        rows_seen = 0
        for statistics, final_features, targets in loader:
            statistics = statistics.to(device)
            final_features = final_features.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                base_logits = base_head(final_features).squeeze(1) + parent(final_features)
            optimizer.zero_grad(set_to_none=True)
            delta, _ = residual(statistics)
            loss = F.binary_cross_entropy_with_logits(base_logits + delta, targets)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(targets)
            rows_seen += len(targets)
        record = {"epoch": epoch, "train_bce": loss_sum / rows_seen}
        history.append(record)
        print(json.dumps(record), flush=True)

    residual.eval()
    valid_statistics = torch.from_numpy(
        competition_multidepth["features"][valid_mask].astype(np.float32)
    ).to(device)
    valid_final = torch.from_numpy(final_competition["features"][valid_mask]).to(device)
    valid_labels = competition_multidepth["labels"][valid_mask]
    with torch.no_grad():
        base_logits = base_head(valid_final).squeeze(1) + parent(valid_final)
        delta, gate_weights = residual(valid_statistics)
        scores = torch.sigmoid(base_logits + delta).cpu().numpy()
        base_scores = torch.sigmoid(base_logits).cpu().numpy()
        gates = gate_weights.cpu().numpy()

    output_dir = resolve(root, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions = pd.DataFrame(
        {
            "id": competition_multidepth["ids"][valid_mask],
            "label": valid_labels.astype(np.int8),
            "score": scores,
            "parent_score": base_scores,
        }
    )
    for index, layer in enumerate(layer_indices):
        predictions[f"gate_layer_{layer}"] = gates[:, index]
    predictions.to_csv(output_dir / "valid_predictions.csv", index=False)
    torch.save(
        {
            "format": "aigc-pe-core-multidepth-patch-residual-v1",
            "encoder_checkpoint_sha256": checkpoint_hash,
            "layer_indices": layer_indices,
            "base_head_checkpoint": str(base_path),
            "parent_residual_checkpoint": str(parent_path),
            "state_dict": {key: value.detach().cpu() for key, value in residual.state_dict().items()},
        },
        output_dir / "multidepth_residual.pth",
    )
    parent_reconstruction_difference = float(
        np.max(
            np.abs(
                base_scores
                - pd.read_csv(resolve(root, str(model_config["parent_valid_predictions"])))
                .set_index("id")
                .loc[predictions["id"], "score"]
                .to_numpy()
            )
        )
    )
    if parent_reconstruction_difference > 2e-6:
        raise RuntimeError(f"parent prediction reconstruction mismatch: {parent_reconstruction_difference}")
    result = {
        "experiment": str(config["experiment"]),
        "fold": fold,
        "metrics": binary_metrics(valid_labels.astype(np.int8), scores),
        "history": history,
        "diagnostics": {
            "parent_reconstruction_max_abs_difference": parent_reconstruction_difference,
            "validation_gate_mean": gates.mean(axis=0).tolist(),
        },
        "audit": {
            "method": "MoLD-style dynamic fusion of evenly spaced PE-Core layer patch mean/std",
            "encoder_checkpoint": str(encoder_path),
            "encoder_checkpoint_sha256": checkpoint_hash,
            "encoder_frozen": True,
            "base_detector_frozen": True,
            "zero_preserving_initialization": True,
            "layer_indices": layer_indices,
            "trainable_parameters": sum(parameter.numel() for parameter in residual.parameters()),
            "external_rows": len(external_rows),
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
