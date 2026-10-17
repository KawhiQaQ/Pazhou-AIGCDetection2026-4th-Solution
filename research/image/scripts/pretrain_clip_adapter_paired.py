#!/usr/bin/env python3
"""External real/fake paired training for the CLIP residual adapter."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import pandas as pd
from PIL import Image, ImageOps
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
import yaml

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aigc_detection.image_v1 import (  # noqa: E402
    CLIPResidualAdapterDetector,
    build_train_transform,
    seed_everything,
)
from train_image_v1 import autocast_context, make_optimizer, make_scheduler  # noqa: E402


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


class PairedImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str, str]]):
    def __init__(
        self,
        pairs: list[dict[str, str]],
        image_root: Path,
        transform,
    ) -> None:
        self.pairs = pairs
        self.image_root = image_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def _read(self, relative_path: str) -> torch.Tensor:
        with Image.open(self.image_root / relative_path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            return self.transform(image)

    def __getitem__(self, index: int):
        pair = self.pairs[index]
        return (
            self._read(pair["real_path"]),
            self._read(pair["fake_path"]),
            pair["source_real_id"],
            pair["generator"],
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = yaml.safe_load(resolve(root, str(args.config)).read_text(encoding="utf-8"))
    seed = int(config["seed"])
    seed_everything(seed)
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    assert isinstance(data_config, dict)
    assert isinstance(model_config, dict)
    assert isinstance(train_config, dict)

    image_root = resolve(root, str(data_config["transfer_dir"])).resolve()
    if "competition" in image_root.parts:
        raise RuntimeError("paired external training must not access a competition path")
    manifest = pd.read_csv(image_root / str(data_config["manifest_csv"]))
    required = {"relative_path", "label", "model_name", "architecture", "source_real_id"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"paired manifest misses columns: {sorted(required-set(manifest.columns))}")
    relative_paths = manifest.relative_path.astype(str)
    if any(Path(path).is_absolute() or ".." in Path(path).parts for path in relative_paths):
        raise RuntimeError("paired manifest contains an unsafe relative path")

    paired_reals = manifest[
        (manifest.label == 0) & manifest.source_real_id.notna()
    ].copy()
    if paired_reals.source_real_id.astype(str).duplicated().any():
        raise RuntimeError("paired real source IDs are not unique")
    real_paths = dict(
        zip(
            paired_reals.source_real_id.astype(str),
            paired_reals.relative_path.astype(str),
            strict=True,
        )
    )
    generators = [str(value) for value in data_config["generators"]]
    paired_fakes = manifest[
        (manifest.label == 1)
        & manifest.source_real_id.notna()
        & manifest.architecture.astype(str).isin(generators)
        & manifest.model_name.astype(str).str.startswith("AIGIv2_")
    ].copy()
    pairs: list[dict[str, str]] = []
    missing_real_ids: list[str] = []
    for row in paired_fakes.itertuples(index=False):
        source_real_id = str(row.source_real_id)
        real_path = real_paths.get(source_real_id)
        if real_path is None:
            missing_real_ids.append(source_real_id)
            continue
        pairs.append(
            {
                "real_path": real_path,
                "fake_path": str(row.relative_path),
                "source_real_id": source_real_id,
                "generator": str(row.architecture),
            }
        )
    if missing_real_ids:
        raise RuntimeError(f"fake rows without paired real: {missing_real_ids[:5]}")
    pair_frame = pd.DataFrame(pairs)
    expected_sources = len(real_paths)
    coverage = pair_frame.groupby("generator").source_real_id.nunique().to_dict()
    if set(coverage) != set(generators) or any(
        int(coverage[generator]) != expected_sources for generator in generators
    ):
        raise RuntimeError(
            f"paired generator grid is incomplete: expected={expected_sources}, coverage={coverage}"
        )

    dataset = PairedImageDataset(pairs, image_root, build_train_transform(config))
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(data_config["pair_batch_size"]),
        shuffle=True,
        generator=loader_generator,
        drop_last=True,
        num_workers=int(data_config["workers"]),
        pin_memory=True,
        persistent_workers=int(data_config["workers"]) > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPResidualAdapterDetector(
        str(model_config["backbone"]),
        [int(index) for index in model_config["feature_blocks"]],
        int(model_config["adapter_last_blocks"]),
        int(model_config["adapter_dim"]),
        float(model_config["adapter_dropout"]),
        int(model_config["hidden_dim"]),
        float(model_config["dropout"]),
        device=device,
        download_root=resolve(root, str(model_config["clip_cache"])),
    ).to(device)
    init_path = resolve(root, str(model_config["init_trainable_checkpoint"]))
    payload = torch.load(init_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != "aigc-clip-adapter-external-supervised-v1":
        raise TypeError(f"invalid external initialization checkpoint: {init_path}")
    if int(payload.get("competition_rows_accessed", -1)) != 0:
        raise RuntimeError("external initialization checkpoint accessed competition rows")
    state = payload.get("trainable_state")
    if not isinstance(state, dict):
        raise TypeError("external initialization checkpoint has no trainable_state")
    result = model.load_state_dict(state, strict=False)
    missing_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and name in result.missing_keys
    ]
    if result.unexpected_keys or missing_trainable:
        raise RuntimeError(
            f"paired initialization mismatch: missing={missing_trainable}, "
            f"unexpected={list(result.unexpected_keys)}"
        )

    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, len(loader), config)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=128.0, growth_interval=2000
    )
    pairwise_weight = float(train_config["paired_margin_weight"])
    temperature = float(train_config["paired_margin_temperature"])
    output_dir = resolve(root, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config,
        "source": "AIGI-v2 complete six-generator paired grid",
        "pair_rows": len(pairs),
        "paired_real_sources": expected_sources,
        "generator_coverage": {key: int(value) for key, value in coverage.items()},
        "competition_rows_accessed": 0,
        "initialization": str(init_path),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "objective": "balanced BCE(real=0,fake=1) plus matched fake-over-real softplus margin",
        "device": str(device),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    history: list[dict[str, object]] = []
    started = time.time()
    for epoch in range(1, int(train_config["epochs"]) + 1):
        model.train()
        totals = {"loss": 0.0, "bce": 0.0, "paired_margin": 0.0}
        steps = 0
        skipped_steps = 0
        last_grad_norm = float("nan")
        tracked_parameter = next(
            parameter
            for _, parameter in reversed(list(model.named_parameters()))
            if parameter.requires_grad
        )
        before = tracked_parameter.detach().clone() if args.smoke_only else None
        for real_images, fake_images, _, _ in loader:
            real_images = real_images.to(device, non_blocking=True)
            fake_images = fake_images.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device):
                images = torch.cat([real_images, fake_images], dim=0)
                logits, _ = model(images)
                real_logits, fake_logits = logits.chunk(2)
                labels = torch.cat(
                    [torch.zeros_like(real_logits), torch.ones_like(fake_logits)]
                )
                bce = F.binary_cross_entropy_with_logits(logits.float(), labels.float())
                paired_margin = F.softplus(
                    -(fake_logits.float() - real_logits.float()) / temperature
                ).mean()
                loss = bce + pairwise_weight * paired_margin
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(train_config["grad_clip_norm"]),
            )
            last_grad_norm = float(grad_norm.detach())
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            if scaler.get_scale() >= scale_before:
                scheduler.step()
            else:
                skipped_steps += 1
            totals["loss"] += float(loss.detach())
            totals["bce"] += float(bce.detach())
            totals["paired_margin"] += float(paired_margin.detach())
            steps += 1
            if args.smoke_only:
                break
        metrics = {
            key: value / max(steps, 1) for key, value in totals.items()
        }
        metrics.update(
            {
                "steps": float(steps),
                "skipped_steps": float(skipped_steps),
                "last_grad_norm": last_grad_norm,
                "lr_head": float(optimizer.param_groups[-1]["lr"]),
            }
        )
        if not math.isfinite(metrics["loss"]):
            raise RuntimeError(f"non-finite paired pretraining loss: {metrics}")
        changed = bool(before is not None and not torch.equal(before, tracked_parameter.detach()))
        if args.smoke_only and not changed:
            raise RuntimeError("paired pretraining smoke did not update a trainable parameter")
        record = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - started,
            "train": metrics,
            "competition_rows_accessed": 0,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.smoke_only:
            print(json.dumps({"smoke": "passed", "parameter_changed": changed, "competition_rows_accessed": 0}))
            return

    checkpoint = {
        "format": "aigc-clip-adapter-external-supervised-v1",
        "source": metadata["source"],
        "external_rows": len(pairs) * 2,
        "external_model_count": len(generators) + 1,
        "competition_rows_accessed": 0,
        "sampling": {
            "policy": "complete matched real/fake pairs; each generator has full source coverage"
        },
        "architecture": "clip_residual_adapter",
        "initialization": str(init_path),
        "trainable_state": model.trainable_state_dict(),
    }
    torch.save(checkpoint, output_dir / "external_pretrain.pth")
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
