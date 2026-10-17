#!/usr/bin/env python3
"""Train a zero-preserving CLIP-feature residual over a frozen PE-Core detector."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from aigc_detection.image_v1 import (  # noqa: E402
    CLIP_MEAN,
    CLIP_STD,
    CLIPResidualAdapterDetector,
    seed_everything,
)
from aigc_detection.metrics import binary_metrics  # noqa: E402
from train_pe_core_residual_mlp import ResidualMLP  # noqa: E402


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ImageRows(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(self, rows: list[dict[str, object]], transform: transforms.Compose) -> None:
        self.rows = rows
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.rows[index]
        with Image.open(Path(str(row["path"]))) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, torch.tensor(float(row["label"]), dtype=torch.float32), str(row["id"])


class ClipResidual(nn.Module):
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


def build_clip_encoder(config: dict[str, object], root: Path, device: torch.device) -> tuple[nn.Module, str]:
    clip_config = config["clip"]
    assert isinstance(clip_config, dict)
    model = CLIPResidualAdapterDetector(
        str(clip_config["backbone"]),
        [int(index) for index in clip_config["feature_blocks"]],
        int(clip_config["adapter_last_blocks"]),
        int(clip_config["adapter_dim"]),
        float(clip_config["adapter_dropout"]),
        int(clip_config["embedding_dim"]),
        float(clip_config["dropout"]),
        device=device,
        download_root=resolve(root, str(clip_config["clip_cache"])),
    ).to(device)
    checkpoint_path = resolve(root, str(clip_config["checkpoint"]))
    checkpoint_hash = sha256_file(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if payload.get("format") != "aigc-clip-adapter-external-supervised-v1":
        raise RuntimeError("unexpected external CLIP checkpoint format")
    if int(payload.get("competition_rows_accessed", -1)) != 0:
        raise RuntimeError("frozen CLIP checkpoint accessed competition rows")
    state = payload.get("trainable_state")
    if not isinstance(state, dict):
        raise RuntimeError("external CLIP checkpoint has no trainable state")
    loaded = model.load_state_dict(state, strict=False)
    if loaded.unexpected_keys:
        raise RuntimeError(f"unexpected frozen CLIP keys: {loaded.unexpected_keys}")
    missing_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in loaded.missing_keys
    ]
    if missing_trainable:
        raise RuntimeError(f"missing frozen CLIP trainable state: {missing_trainable}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model, checkpoint_hash


@torch.no_grad()
def extract_clip_features(
    model: nn.Module,
    rows: list[dict[str, object]],
    transform: transforms.Compose,
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
    features: list[np.ndarray] = []
    labels: list[float] = []
    ids: list[str] = []
    started = time.time()
    seen = 0
    for images, batch_labels, batch_ids in loader:
        images = images.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            _, embedding = model(images)
        features.append(embedding.float().cpu().numpy().astype(np.float16))
        labels.extend(batch_labels.numpy().tolist())
        ids.extend(batch_ids)
        seen += len(batch_ids)
        if seen % 500 < len(batch_ids) or seen == len(rows):
            print(
                json.dumps(
                    {"clip_features": f"{seen}/{len(rows)}", "elapsed_seconds": round(time.time() - started, 2)}
                ),
                flush=True,
            )
    return np.concatenate(features), np.asarray(labels, dtype=np.float32), np.asarray(ids)


def load_or_extract_clip_cache(
    path: Path,
    rows: list[dict[str, object]],
    model: nn.Module | None,
    transform: transforms.Compose,
    *,
    batch_size: int,
    workers: int,
    device: torch.device,
    checkpoint_sha256: str,
) -> dict[str, np.ndarray]:
    expected_ids = np.asarray([str(row["id"]) for row in rows])
    if path.is_file():
        cached = np.load(path)
        if str(cached["checkpoint_sha256"].item()) != checkpoint_sha256:
            raise RuntimeError(f"stale CLIP feature checkpoint: {path}")
        if not np.array_equal(cached["ids"].astype(str), expected_ids):
            raise RuntimeError(f"stale CLIP feature row identity: {path}")
        return {key: cached[key] for key in ("features", "labels", "ids")}
    if model is None:
        raise RuntimeError(f"missing CLIP encoder for uncached features: {path}")
    features, labels, ids = extract_clip_features(
        model, rows, transform, batch_size, workers, device
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        labels=labels,
        ids=ids,
        checkpoint_sha256=np.asarray(checkpoint_sha256),
    )
    return {"features": features, "labels": labels, "ids": ids}


def load_pe_cache(path: Path, expected_ids: np.ndarray, expected_labels: np.ndarray) -> np.ndarray:
    cached = np.load(path)
    ids = cached["ids"].astype(str)
    labels = cached["labels"].astype(np.float32)
    if not np.array_equal(ids, expected_ids.astype(str)):
        raise RuntimeError(f"PE-Core/CLIP cache ID mismatch: {path}")
    if not np.array_equal(labels, expected_labels.astype(np.float32)):
        raise RuntimeError(f"PE-Core/CLIP cache label mismatch: {path}")
    return cached["features"].astype(np.float32)


def pe_base_logits(
    features: np.ndarray,
    head: nn.Module,
    residual: nn.Module,
    device: torch.device,
    batch_size: int = 4096,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    head.eval()
    residual.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            logits = head(batch).squeeze(1) + residual(batch)
            parts.append(logits.cpu().numpy().astype(np.float32))
    return np.concatenate(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = yaml.safe_load(resolve(root, str(args.config)).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    data_config = config["data"]
    clip_config = config["clip"]
    pe_config = config["pe_base"]
    train_config = config["train"]
    assert isinstance(data_config, dict)
    assert isinstance(clip_config, dict)
    assert isinstance(pe_config, dict)
    assert isinstance(train_config, dict)

    labels = pd.read_csv(resolve(root, str(data_config["labels_csv"])))
    folds = pd.read_csv(resolve(root, str(data_config["folds_csv"])))
    competition = labels.merge(folds[["id", "fold", "group_id"]], on="id", validate="one_to_one")
    competition_dir = resolve(root, str(data_config["competition_dir"]))
    competition_rows = [
        {"id": str(row.id), "label": int(row.label), "path": competition_dir / str(row.id)}
        for row in competition.itertuples(index=False)
    ]

    external_root = resolve(root, str(data_config["external_root"])).resolve()
    if "competition" in external_root.parts:
        raise RuntimeError("V60 external root resolves inside competition data")
    external_rows: dict[str, list[dict[str, object]]] = {}
    for source, manifest_value in data_config["external_manifests"].items():
        manifest = pd.read_csv(resolve(root, str(manifest_value)))
        external_rows[source] = [
            {
                "id": str(row.sample_key),
                "label": int(row.label),
                "path": external_root / str(row.relative_path),
            }
            for row in manifest.itertuples(index=False)
        ]
    all_rows = competition_rows + [row for rows in external_rows.values() for row in rows]
    if any(not Path(str(row["path"])).is_file() for row in all_rows):
        raise RuntimeError("V60 input manifest contains missing images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [
            transforms.Resize(int(clip_config["resize_size"]), interpolation=InterpolationMode.BICUBIC),
            transforms.CenterCrop(int(clip_config["input_size"])),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )
    checkpoint_path = resolve(root, str(clip_config["checkpoint"]))
    checkpoint_hash = sha256_file(checkpoint_path)
    cache_dir = resolve(root, str(data_config["clip_feature_cache_dir"]))
    cache_paths = {
        "competition": cache_dir / "competition_center_features.npz",
        **{source: cache_dir / f"{source}_center_features.npz" for source in external_rows},
    }
    caches_complete = all(path.is_file() for path in cache_paths.values())
    if args.smoke_only or not caches_complete:
        clip_model, loaded_hash = build_clip_encoder(config, root, device)
        if loaded_hash != checkpoint_hash:
            raise RuntimeError("V60 frozen CLIP checkpoint changed while loading")
    else:
        clip_model = None
    if args.smoke_only:
        smoke_rows = competition_rows[:2]
        assert clip_model is not None
        features, smoke_labels, _ = extract_clip_features(
            clip_model, smoke_rows, transform, 2, 0, device
        )
        residual = ClipResidual(features.shape[1], int(train_config["hidden_dim"]), float(train_config["dropout"])).to(device)
        optimizer = torch.optim.AdamW(residual.parameters(), lr=float(train_config["lr"]))
        batch = torch.from_numpy(features.astype(np.float32)).to(device)
        base_logits = torch.zeros(len(batch), device=device)
        before = residual.network[-1].weight.detach().clone()
        initial = base_logits + residual(batch)
        if not torch.equal(initial, base_logits):
            raise RuntimeError("V60 residual is not exactly zero at initialization")
        loss = F.binary_cross_entropy_with_logits(initial, torch.from_numpy(smoke_labels).to(device))
        loss.backward()
        optimizer.step()
        print(json.dumps({"smoke": "passed", "loss": float(loss.detach()), "parameter_changed": not torch.equal(before, residual.network[-1].weight.detach()), "exact_zero_initialization": True, "validation_rows_accessed": 0}), flush=True)
        return
    clip_sets = {
        "competition": load_or_extract_clip_cache(
            cache_paths["competition"],
            competition_rows,
            clip_model,
            transform,
            batch_size=int(clip_config["batch_size"]),
            workers=0 if args.smoke_only else int(clip_config["workers"]),
            device=device,
            checkpoint_sha256=checkpoint_hash,
        )
    }
    for source, rows in external_rows.items():
        clip_sets[source] = load_or_extract_clip_cache(
            cache_paths[source],
            rows,
            clip_model,
            transform,
            batch_size=int(clip_config["batch_size"]),
            workers=int(clip_config["workers"]),
            device=device,
            checkpoint_sha256=checkpoint_hash,
        )
    del clip_model
    torch.cuda.empty_cache()

    pe_sets = {
        name: load_pe_cache(
            resolve(root, str(pe_config["feature_caches"][name])),
            clip_set["ids"],
            clip_set["labels"],
        )
        for name, clip_set in clip_sets.items()
    }
    pe_head_payload = torch.load(
        resolve(root, str(pe_config["linear_head_checkpoint"])), map_location="cpu", weights_only=True
    )
    if pe_head_payload.get("format") != "aigc-pe-core-linear-v1":
        raise RuntimeError("unexpected V60 PE linear head format")
    pe_dim = int(pe_sets["competition"].shape[1])
    pe_head = nn.Linear(pe_dim, 1).to(device)
    pe_head.load_state_dict(pe_head_payload["state_dict"], strict=True)
    pe_residual_payload = torch.load(
        resolve(root, str(pe_config["residual_checkpoint"])), map_location="cpu", weights_only=True
    )
    if pe_residual_payload.get("format") != "aigc-pe-core-residual-mlp-v1":
        raise RuntimeError("unexpected V60 PE residual format")
    pe_residual = ResidualMLP(
        pe_dim, int(pe_config["hidden_dim"]), float(pe_config["dropout"])
    ).to(device)
    pe_residual.load_state_dict(pe_residual_payload["state_dict"], strict=True)
    for module in (pe_head, pe_residual):
        for parameter in module.parameters():
            parameter.requires_grad = False
        module.eval()
    base_logits = {
        name: pe_base_logits(features, pe_head, pe_residual, device)
        for name, features in pe_sets.items()
    }

    fold = int(config["fold"])
    train_mask = competition.fold.to_numpy() != fold
    valid_mask = ~train_mask
    reference = pd.read_csv(resolve(root, str(pe_config["reference_predictions"]))).set_index("id")
    valid_ids = clip_sets["competition"]["ids"][valid_mask].astype(str)
    if set(valid_ids) != set(reference.index.astype(str)):
        raise RuntimeError("V60 PE baseline reference IDs differ")
    reference_scores = reference.loc[valid_ids, "score"].to_numpy(dtype=np.float32)
    reconstructed_scores = torch.sigmoid(
        torch.from_numpy(base_logits["competition"][valid_mask])
    ).numpy()
    base_max_abs_diff = float(np.max(np.abs(reference_scores - reconstructed_scores)))
    if base_max_abs_diff > 2e-6:
        raise RuntimeError(f"V60 did not exactly reconstruct V54 baseline: {base_max_abs_diff}")
    if set(competition.loc[train_mask, "group_id"]) & set(competition.loc[valid_mask, "group_id"]):
        raise RuntimeError("duplicate groups cross V60 training/validation")
    train_clip = np.concatenate(
        [clip_sets["competition"]["features"][train_mask]]
        + [clip_sets[source]["features"] for source in external_rows]
    ).astype(np.float32)
    train_base = np.concatenate(
        [base_logits["competition"][train_mask]]
        + [base_logits[source] for source in external_rows]
    ).astype(np.float32)
    train_labels = np.concatenate(
        [clip_sets["competition"]["labels"][train_mask]]
        + [clip_sets[source]["labels"] for source in external_rows]
    ).astype(np.float32)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_clip),
            torch.from_numpy(train_base),
            torch.from_numpy(train_labels),
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
        rows_seen = 0
        for clip_features, batch_base, batch_labels in loader:
            clip_features = clip_features.to(device)
            batch_base = batch_base.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = batch_base + residual(clip_features)
            loss = F.binary_cross_entropy_with_logits(logits, batch_labels)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(batch_labels)
            rows_seen += len(batch_labels)
        record = {"epoch": epoch, "train_bce": loss_sum / rows_seen}
        history.append(record)
        print(json.dumps(record), flush=True)

    residual.eval()
    valid_clip = torch.from_numpy(
        clip_sets["competition"]["features"][valid_mask].astype(np.float32)
    ).to(device)
    with torch.no_grad():
        scores = torch.sigmoid(
            torch.from_numpy(base_logits["competition"][valid_mask]).to(device)
            + residual(valid_clip)
        ).cpu().numpy()
    valid_labels = clip_sets["competition"]["labels"][valid_mask].astype(np.int8)
    output_dir = resolve(root, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "id": clip_sets["competition"]["ids"][valid_mask],
            "label": valid_labels,
            "score": scores,
        }
    ).to_csv(output_dir / "valid_predictions.csv", index=False)
    torch.save(
        {
            "format": "aigc-cross-backbone-residual-v1",
            "clip_checkpoint_sha256": checkpoint_hash,
            "state_dict": {key: value.detach().cpu() for key, value in residual.state_dict().items()},
        },
        output_dir / "clip_residual.pth",
    )
    result = {
        "experiment": str(config["experiment"]),
        "fold": fold,
        "metrics": binary_metrics(valid_labels, scores),
        "history": history,
        "audit": {
            "clip_checkpoint_sha256": checkpoint_hash,
            "clip_checkpoint_competition_rows_accessed": 0,
            "clip_encoder_frozen": True,
            "pe_encoder_frozen": True,
            "pe_linear_head_frozen": True,
            "pe_residual_head_frozen": True,
            "zero_preserving_initialization": True,
            "v54_baseline_max_abs_diff": base_max_abs_diff,
            "external_rows": {source: len(rows) for source, rows in external_rows.items()},
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
