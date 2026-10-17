#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aigc_detection.image_v1 import (  # noqa: E402
    CLIPAdapterPatchMILDetector,
    CLIPAdapterMultiDepthPatchMILDetector,
    CLIPAdapterFrequencyResidualDetector,
    CLIPAdapterDegradationResponseDetector,
    CLIPResidualAdapterDetector,
    CLIPResidualAdapterLoRADetector,
    LabeledImageDataset,
    build_train_transform,
    seed_everything,
)
from train_image_v1 import make_optimizer, make_scheduler, train_epoch  # noqa: E402


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def balanced_generator_weights(
    manifest: pd.DataFrame,
    *,
    balance_real_models: bool = False,
    balance_fake_models: bool = True,
) -> tuple[np.ndarray, dict[str, object]]:
    real = manifest[manifest.label == 0]
    fake = manifest[manifest.label == 1]
    fake_models = sorted(fake.model_name.astype(str).unique())
    if real.empty or fake.empty or not fake_models:
        raise RuntimeError("external manifest must contain real and multiple fake sources")
    weights = np.zeros(len(manifest), dtype=np.float64)
    real_models = sorted(real.model_name.astype(str).unique())
    if balance_real_models:
        real_mass = 0.5 / len(real_models)
        for model_name in real_models:
            mask = (manifest.label == 0) & (
                manifest.model_name.astype(str) == model_name
            )
            weights[mask.to_numpy()] = real_mass / int(mask.sum())
    else:
        weights[manifest.label.to_numpy() == 0] = 0.5 / len(real)
    if balance_fake_models:
        fake_mass = 0.5 / len(fake_models)
        for model_name in fake_models:
            mask = (manifest.label == 1) & (manifest.model_name.astype(str) == model_name)
            weights[mask.to_numpy()] = fake_mass / int(mask.sum())
    else:
        weights[manifest.label.to_numpy() == 1] = 0.5 / len(fake)
    if not np.isclose(weights.sum(), 1.0):
        raise RuntimeError(f"external sampling weights do not sum to one: {weights.sum()}")
    audit = {
        "policy": (
            "0.5 real uniformly divided across real model_name groups; "
            "0.5 fake uniformly divided across fake model_name groups"
            if balance_real_models and balance_fake_models
            else (
                "0.5 real uniformly divided across real model_name groups; "
                "0.5 fake uniformly across rows"
                if balance_real_models
                else (
                    "0.5 real uniformly across rows; "
                    "0.5 fake uniformly divided across fake model_name groups"
                    if balance_fake_models
                    else "0.5 real uniformly across rows; 0.5 fake uniformly across rows"
                )
            )
        ),
        "real_rows": int(len(real)),
        "real_models": real_models,
        "fake_rows": int(len(fake)),
        "fake_models": fake_models,
        "weight_sum": float(weights.sum()),
    }
    return weights, audit


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
    model_config = config["model"]
    train_config = config["train"]
    assert isinstance(data_config, dict)
    assert isinstance(model_config, dict)
    assert isinstance(train_config, dict)

    transfer_dir = resolve(root, str(data_config["transfer_dir"])).resolve()
    if "competition" in transfer_dir.parts:
        raise RuntimeError("external pretraining must not access a competition path")
    manifest = pd.read_csv(transfer_dir / str(data_config["manifest_csv"]))
    required = {"relative_path", "label", "model_name"}
    if not required.issubset(manifest.columns):
        raise RuntimeError(f"external manifest misses columns: {sorted(required-set(manifest.columns))}")
    if set(manifest.label.unique()) != {0, 1}:
        raise RuntimeError("external manifest labels are not binary")
    relative_paths = manifest.relative_path.astype(str)
    if any(Path(path).is_absolute() or ".." in Path(path).parts for path in relative_paths):
        raise RuntimeError("external manifest contains an unsafe relative path")

    weights, sampling_audit = balanced_generator_weights(
        manifest,
        balance_real_models=bool(data_config.get("balance_real_models", False)),
        balance_fake_models=bool(data_config.get("balance_fake_models", True)),
    )
    dataset = LabeledImageDataset(
        [
            {"id": path, "label": int(label)}
            for path, label in zip(relative_paths, manifest.label, strict=True)
        ],
        transfer_dir,
        build_train_transform(config),
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights),
        num_samples=len(dataset),
        replacement=True,
        generator=generator,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(data_config["train_batch_size"]),
        sampler=sampler,
        drop_last=True,
        num_workers=int(data_config["workers"]),
        pin_memory=True,
        persistent_workers=int(data_config["workers"]) > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    architecture = str(model_config.get("architecture", "clip_residual_adapter"))
    common_model_args = (
        str(model_config["backbone"]),
        [int(index) for index in model_config["feature_blocks"]],
        int(model_config["adapter_last_blocks"]),
        int(model_config["adapter_dim"]),
        float(model_config["adapter_dropout"]),
        int(model_config["hidden_dim"]),
        float(model_config["dropout"]),
    )
    common_model_kwargs = {
        "device": device,
        "download_root": resolve(root, str(model_config["clip_cache"])),
    }
    if architecture == "clip_residual_adapter":
        model = CLIPResidualAdapterDetector(
            *common_model_args, **common_model_kwargs
        ).to(device)
    elif architecture == "clip_adapter_patch_mil":
        model = CLIPAdapterPatchMILDetector(
            *common_model_args,
            int(model_config["patch_hidden_dim"]),
            float(model_config["patch_top_fraction"]),
            **common_model_kwargs,
        ).to(device)
    elif architecture == "clip_adapter_multidepth_patch_mil":
        model = CLIPAdapterMultiDepthPatchMILDetector(
            *common_model_args,
            int(model_config["patch_hidden_dim"]),
            float(model_config["patch_top_fraction"]),
            **common_model_kwargs,
        ).to(device)
    elif architecture == "clip_adapter_frequency_residual":
        model = CLIPAdapterFrequencyResidualDetector(
            *common_model_args,
            int(model_config["frequency_hidden_dim"]),
            **common_model_kwargs,
        ).to(device)
    elif architecture == "clip_adapter_degradation_response":
        model = CLIPAdapterDegradationResponseDetector(
            *common_model_args,
            int(model_config["response_hidden_dim"]),
            float(model_config["degradation_scale"]),
            **common_model_kwargs,
        ).to(device)
    elif architecture == "clip_residual_adapter_lora":
        model = CLIPResidualAdapterLoRADetector(
            *common_model_args,
            int(model_config["lora_last_blocks"]),
            int(model_config["lora_rank"]),
            float(model_config["lora_alpha"]),
            **common_model_kwargs,
        ).to(device)
    else:
        raise ValueError(f"unsupported external-pretraining architecture: {architecture}")

    initialization_audit = None
    if model_config.get("init_trainable_checkpoint"):
        init_path = resolve(root, str(model_config["init_trainable_checkpoint"]))
        payload = torch.load(init_path, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or payload.get("format") != (
            "aigc-clip-adapter-external-supervised-v1"
        ):
            raise TypeError(f"invalid external initialization checkpoint: {init_path}")
        if int(payload.get("competition_rows_accessed", -1)) != 0:
            raise RuntimeError("external initialization checkpoint accessed competition rows")
        state = payload.get("trainable_state")
        if not isinstance(state, dict):
            raise TypeError("external initialization checkpoint has no trainable_state")
        result = model.load_state_dict(state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError(
                f"unexpected external initialization keys: {result.unexpected_keys}"
            )
        new_module_prefixes = tuple(
            str(prefix)
            for prefix in train_config.get(
                "new_module_prefixes",
                ["patch_", "frequency_", "fusion_classifier."],
            )
        )
        missing_trainable = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and name in result.missing_keys
            and not name.startswith(new_module_prefixes)
        ]
        if missing_trainable:
            raise RuntimeError(
                f"missing inherited external trainable keys: {missing_trainable}"
            )
        initialization_audit = {
            "checkpoint": str(init_path),
            "format": payload["format"],
            "competition_rows_accessed": 0,
            "new_trainable_module_prefixes": list(new_module_prefixes),
        }
    initially_trainable_names = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    if bool(train_config.get("new_modules_only", False)):
        if initialization_audit is None:
            raise RuntimeError("new_modules_only requires an initialization checkpoint")
        for name, parameter in model.named_parameters():
            if name in initially_trainable_names:
                parameter.requires_grad_(
                    name.startswith(new_module_prefixes)
                )
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, len(loader), config)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=128.0, growth_interval=2000
    )
    output_dir = resolve(root, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    external_source = str(
        data_config.get(
            "source",
            "OwensLab/CommunityForensics-Small external transfer subset",
        )
    )
    metadata = {
        "config": config,
        "source": external_source,
        "external_rows": int(len(manifest)),
        "external_model_count": int(manifest.model_name.nunique()),
        "label_counts": {
            str(key): int(value)
            for key, value in manifest.label.value_counts().sort_index().items()
        },
        "sampling": sampling_audit,
        "competition_rows_accessed": 0,
        "initialization": initialization_audit,
        "optimization_policy": (
            "configured new modules only"
            if bool(train_config.get("new_modules_only", False))
            else "all trainable modules"
        ),
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "device": str(device),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    history: list[dict[str, object]] = []
    started = time.time()
    for epoch in range(1, int(train_config["epochs"]) + 1):
        tracked_parameter = next(
            parameter
            for _, parameter in reversed(list(model.named_parameters()))
            if parameter.requires_grad
        )
        before = tracked_parameter.detach().clone() if args.smoke_only else None
        metrics = train_epoch(
            model,
            loader,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
            max_steps=1 if args.smoke_only else None,
        )
        if not math.isfinite(metrics["loss"]):
            raise RuntimeError(f"non-finite external pretraining loss: {metrics}")
        parameter_changed = bool(
            before is not None and not torch.equal(before, tracked_parameter.detach())
        )
        if args.smoke_only and not parameter_changed:
            raise RuntimeError("external pretraining smoke did not update a trainable parameter")
        record = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - started,
            "train": metrics,
            "competition_rows_accessed": 0,
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if args.smoke_only:
            print(
                json.dumps(
                    {
                        "smoke": "passed",
                        "parameter_changed": parameter_changed,
                        "competition_rows_accessed": 0,
                    }
                )
            )
            return

    # Restore the inherited modules' trainability before serializing so the
    # compact checkpoint remains complete for downstream competition finetuning.
    for name, parameter in model.named_parameters():
        if name in initially_trainable_names:
            parameter.requires_grad_(True)
    checkpoint = {
        "format": "aigc-clip-adapter-external-supervised-v1",
        "source": metadata["source"],
        "external_rows": int(len(manifest)),
        "external_model_count": int(manifest.model_name.nunique()),
        "competition_rows_accessed": 0,
        "sampling": sampling_audit,
        "architecture": architecture,
        "initialization": initialization_audit,
        "trainable_state": model.trainable_state_dict(),
    }
    torch.save(checkpoint, output_dir / "external_pretrain.pth")
    (output_dir / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
