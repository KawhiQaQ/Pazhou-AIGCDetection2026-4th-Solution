#!/usr/bin/env python3
"""Fuse frozen PE-Core and PE-Spatial multi-depth detectors at feature level."""

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
from train_pe_core_multidepth_residual import MultiDepthPatchResidual

sys.path.insert(0, str(PROJECT_ROOT / "src"))
from aigc_detection.metrics import binary_metrics  # noqa: E402


def load_cache(path: Path, checkpoint_sha256: str, layers: list[int]) -> dict[str, np.ndarray]:
    cached = np.load(path)
    if str(cached["checkpoint_sha256"].item()) != checkpoint_sha256:
        raise RuntimeError(f"feature checkpoint mismatch: {path}")
    if not np.array_equal(cached["layer_indices"], np.asarray(layers)):
        raise RuntimeError(f"feature layer mismatch: {path}")
    return {
        "features": cached["features"],
        "labels": cached["labels"].astype(np.float32),
        "ids": cached["ids"].astype(str),
    }


def load_branch(
    checkpoint: Path,
    encoder_sha256: str,
    layers: list[int],
    statistic_dim: int,
    projection_dim: int,
    dropout: float,
) -> MultiDepthPatchResidual:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format") != "aigc-pe-core-multidepth-patch-classifier-v1":
        raise RuntimeError(f"unexpected parent format: {checkpoint}")
    if str(payload.get("encoder_checkpoint_sha256")) != encoder_sha256:
        raise RuntimeError(f"parent encoder mismatch: {checkpoint}")
    if list(payload.get("layer_indices", [])) != layers:
        raise RuntimeError(f"parent layers mismatch: {checkpoint}")
    branch = MultiDepthPatchResidual(len(layers), statistic_dim, projection_dim, dropout)
    branch.load_state_dict(payload["state_dict"], strict=True)
    branch.eval()
    for parameter in branch.parameters():
        parameter.requires_grad_(False)
    return branch


def branch_forward(
    branch: MultiDepthPatchResidual, statistics: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projected = torch.stack(
        [projection(statistics[:, index]) for index, projection in enumerate(branch.projections)],
        dim=1,
    )
    gates = torch.softmax(branch.gate(projected).squeeze(2), dim=1)
    pooled = (gates.unsqueeze(2) * projected).sum(dim=1)
    logits = branch.output(pooled).squeeze(1)
    return logits, pooled, gates


class DualPEFusion(nn.Module):
    """Zero-initialized nonlinear correction over two frozen forensic branches."""

    def __init__(
        self,
        core: MultiDepthPatchResidual,
        spatial: MultiDepthPatchResidual,
        projection_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.core = core
        self.spatial = spatial
        fusion_dim = projection_dim * 4 + 3
        self.residual = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def train(self, mode: bool = True) -> "DualPEFusion":
        super().train(mode)
        self.core.eval()
        self.spatial.eval()
        return self

    def forward(
        self, core_statistics: torch.Tensor, spatial_statistics: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            core_logit, core_pooled, core_gates = branch_forward(self.core, core_statistics)
            spatial_logit, spatial_pooled, spatial_gates = branch_forward(
                self.spatial, spatial_statistics
            )
        interaction = torch.cat(
            [
                core_pooled,
                spatial_pooled,
                torch.abs(core_pooled - spatial_pooled),
                core_pooled * spatial_pooled,
                core_logit[:, None],
                spatial_logit[:, None],
                (core_logit - spatial_logit)[:, None],
            ],
            dim=1,
        )
        base_logit = 0.5 * (core_logit + spatial_logit)
        return base_logit + self.residual(interaction).squeeze(1), core_gates, spatial_gates


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
    layers = [int(value) for value in model_config["layer_indices"]]

    core_competition = load_cache(
        resolve(root, data_config["core_competition_features"]),
        str(model_config["core_encoder_sha256"]),
        layers,
    )
    core_external = load_cache(
        resolve(root, data_config["core_external_features"]),
        str(model_config["core_encoder_sha256"]),
        layers,
    )
    spatial_competition = load_cache(
        resolve(root, data_config["spatial_competition_features"]),
        str(model_config["spatial_encoder_sha256"]),
        layers,
    )
    spatial_external = load_cache(
        resolve(root, data_config["spatial_external_features"]),
        str(model_config["spatial_encoder_sha256"]),
        layers,
    )
    for name, first, second in (
        ("competition", core_competition, spatial_competition),
        ("external", core_external, spatial_external),
    ):
        if not np.array_equal(first["ids"], second["ids"]):
            raise RuntimeError(f"{name} Core/Spatial identities differ")
        if not np.array_equal(first["labels"], second["labels"]):
            raise RuntimeError(f"{name} Core/Spatial labels differ")

    folds = pd.read_csv(resolve(root, data_config["folds_csv"]))
    fold_by_id = folds.assign(id=folds["id"].astype(str)).set_index("id")["fold"]
    group_by_id = folds.assign(id=folds["id"].astype(str)).set_index("id")["group_id"]
    fold_values = np.asarray([int(fold_by_id.loc[item]) for item in core_competition["ids"]])
    group_values = np.asarray([str(group_by_id.loc[item]) for item in core_competition["ids"]])
    fold = int(config["fold"])
    train_mask = fold_values != fold
    valid_mask = ~train_mask
    if set(group_values[train_mask]) & set(group_values[valid_mask]):
        raise RuntimeError("duplicate groups cross dual-PE training/validation")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    statistic_dim = int(core_competition["features"].shape[2])
    projection_dim = int(model_config["projection_dim"])
    core_branch = load_branch(
        resolve(root, model_config["core_parent_checkpoint"]),
        str(model_config["core_encoder_sha256"]),
        layers,
        statistic_dim,
        projection_dim,
        float(model_config["parent_dropout"]),
    )
    spatial_branch = load_branch(
        resolve(root, model_config["spatial_parent_checkpoint"]),
        str(model_config["spatial_encoder_sha256"]),
        layers,
        statistic_dim,
        projection_dim,
        float(model_config["parent_dropout"]),
    )
    model = DualPEFusion(
        core_branch,
        spatial_branch,
        projection_dim,
        int(model_config["hidden_dim"]),
        float(model_config["dropout"]),
    ).to(device)

    if args.smoke_only:
        core_batch = torch.from_numpy(core_competition["features"][:4].astype(np.float32)).to(device)
        spatial_batch = torch.from_numpy(
            spatial_competition["features"][:4].astype(np.float32)
        ).to(device)
        labels = torch.from_numpy(core_competition["labels"][:4]).to(device)
        with torch.no_grad():
            core_logits, _, _ = branch_forward(model.core, core_batch)
            spatial_logits, _, _ = branch_forward(model.spatial, spatial_batch)
            initial, _, _ = model(core_batch, spatial_batch)
        expected = 0.5 * (core_logits + spatial_logits)
        if not torch.equal(initial, expected):
            raise RuntimeError("fusion residual is not exactly zero at initialization")
        before = model.residual[-1].weight.detach().clone()
        logits, _, _ = model(core_batch, spatial_batch)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        torch.optim.AdamW(model.residual.parameters(), lr=float(train_config["lr"])).step()
        if torch.equal(before, model.residual[-1].weight.detach()):
            raise RuntimeError("fusion head did not update")
        print(json.dumps({"smoke": "passed", "loss": float(loss.detach())}))
        return

    include_external = bool(data_config.get("include_external", True))
    if include_external:
        train_core = np.concatenate(
            [core_external["features"], core_competition["features"][train_mask]]
        ).astype(np.float32)
        train_spatial = np.concatenate(
            [spatial_external["features"], spatial_competition["features"][train_mask]]
        ).astype(np.float32)
        train_labels = np.concatenate(
            [core_external["labels"], core_competition["labels"][train_mask]]
        ).astype(np.float32)
    else:
        train_core = core_competition["features"][train_mask].astype(np.float32)
        train_spatial = spatial_competition["features"][train_mask].astype(np.float32)
        train_labels = core_competition["labels"][train_mask].astype(np.float32)
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
        rows_seen = 0
        for core_features, spatial_features, targets in loader:
            core_features = core_features.to(device)
            spatial_features = spatial_features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, _, _ = model(core_features, spatial_features)
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
            rows_seen += count
        record = {
            "epoch": epoch,
            "train_total": total_sum / rows_seen,
            "train_bce": bce_sum / rows_seen,
            "train_pairwise_ranking": rank_sum / rows_seen,
        }
        history.append(record)
        print(json.dumps(record), flush=True)

    model.eval()
    valid_core = torch.from_numpy(
        core_competition["features"][valid_mask].astype(np.float32)
    ).to(device)
    valid_spatial = torch.from_numpy(
        spatial_competition["features"][valid_mask].astype(np.float32)
    ).to(device)
    with torch.no_grad():
        logits, core_gates, spatial_gates = model(valid_core, valid_spatial)
        scores = torch.sigmoid(logits).cpu().numpy()
    labels = core_competition["labels"][valid_mask]
    output_dir = resolve(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"id": core_competition["ids"][valid_mask], "label": labels.astype(np.int8), "score": scores}
    ).to_csv(output_dir / "valid_predictions.csv", index=False)
    torch.save(
        {
            "format": "aigc-dual-pe-multidepth-fusion-v1",
            "core_encoder_sha256": str(model_config["core_encoder_sha256"]),
            "spatial_encoder_sha256": str(model_config["spatial_encoder_sha256"]),
            "layer_indices": layers,
            "core_parent_checkpoint": str(model_config["core_parent_checkpoint"]),
            "spatial_parent_checkpoint": str(model_config["spatial_parent_checkpoint"]),
            "residual_state_dict": {
                key: value.detach().cpu() for key, value in model.residual.state_dict().items()
            },
        },
        output_dir / "dual_pe_fusion.pth",
    )
    result = {
        "experiment": str(config["experiment"]),
        "fold": fold,
        "metrics": binary_metrics(labels.astype(np.int8), scores),
        "history": history,
        "diagnostics": {
            "core_gate_mean": core_gates.mean(0).cpu().tolist(),
            "spatial_gate_mean": spatial_gates.mean(0).cpu().tolist(),
        },
        "audit": {
            "method": "frozen PE-Core/PE-Spatial parent branches plus feature-interaction residual",
            "parents_frozen": True,
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "external_rows": int(len(core_external["labels"])) if include_external else 0,
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
