#!/usr/bin/env python3
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import yaml

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aigc_detection.image_v1 import (  # noqa: E402
    CLIPAdapterPatchMILDetector,
    CLIPAdapterMultiDepthPatchMILDetector,
    CLIPAdapterFrequencyResidualDetector,
    CLIPAdapterDegradationResponseDetector,
    CLIPAttentionLoRADetector,
    CLIPEffortOrthogonalDetector,
    CLIPForensicDetector,
    CLIPLocalForensicDetector,
    CLIPMixtureOfLayersDetector,
    CLIPOfficialMoLDDetector,
    CLIPResidualAdapterDetector,
    CLIPResidualAdapterLoRADetector,
    CommunityForensicsDetector,
    DDADinoV2Detector,
    DRCTConvNextDetector,
    DistilledTimmForensicDetector,
    ExternalTimmForensicDetector,
    GAPLDetector,
    NPRDetector,
    OmniAIDOpenAIClipDetector,
    OmniAIDResidualAdapterDetector,
    LabeledImageDataset,
    RINEDetector,
    SAFEDetector,
    TimmEffortOrthogonalDetector,
    TimmForensicDetector,
    build_train_transform,
    build_valid_transform,
    seed_everything,
    supervised_contrastive_loss,
)
from aigc_detection.metrics import binary_metrics  # noqa: E402


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_split(root: Path, config: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    data_config = config["data"]
    assert isinstance(data_config, dict)
    labels = pd.read_csv(resolve(root, str(data_config["labels_csv"])))
    folds = pd.read_csv(resolve(root, str(data_config["folds_csv"])))
    frame = labels.merge(folds[["id", "fold", "group_id"]], on="id", validate="one_to_one")
    fold = int(config["fold"])
    train = frame[frame.fold != fold].reset_index(drop=True)
    valid = frame[frame.fold == fold].reset_index(drop=True)

    train_groups = set(train.group_id)
    valid_groups = set(valid.group_id)
    overlap = train_groups & valid_groups
    if overlap:
        raise RuntimeError(f"duplicate groups cross train/validation: {sorted(overlap)[:5]}")
    if set(train.id) & set(valid.id):
        raise RuntimeError("image IDs cross train/validation")
    if set(train.label.unique()) != {0, 1} or set(valid.label.unique()) != {0, 1}:
        raise RuntimeError("both classes must be present in train and validation")
    audit = {
        "fold": fold,
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "train_label_counts": {str(k): int(v) for k, v in train.label.value_counts().sort_index().items()},
        "valid_label_counts": {str(k): int(v) for k, v in valid.label.value_counts().sort_index().items()},
        "duplicate_group_overlap": 0,
        "test_rows_accessed": 0,
    }
    return train, valid, audit


def make_loaders(
    root: Path, config: dict[str, object], train: pd.DataFrame, valid: pd.DataFrame
) -> tuple[DataLoader, DataLoader]:
    data_config = config["data"]
    assert isinstance(data_config, dict)
    image_dir = resolve(root, str(data_config["image_dir"]))
    train_dataset = LabeledImageDataset(
        train[["id", "label"]].to_dict("records"), image_dir, build_train_transform(config)
    )
    valid_dataset = LabeledImageDataset(
        valid[["id", "label"]].to_dict("records"), image_dir, build_valid_transform(config)
    )
    generator = torch.Generator().manual_seed(int(config["seed"]))
    common = {
        "num_workers": int(data_config["workers"]),
        "pin_memory": True,
        "persistent_workers": int(data_config["workers"]) > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(data_config["train_batch_size"]),
        shuffle=True,
        drop_last=True,
        generator=generator,
        **common,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(data_config["valid_batch_size"]),
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, valid_loader


def autocast_context(device: torch.device):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


@torch.no_grad()
def evaluate(
    model: torch.nn.Module, loader: DataLoader, device: torch.device
) -> tuple[dict[str, float | int], pd.DataFrame]:
    model.eval()
    all_ids: list[str] = []
    all_labels: list[float] = []
    all_scores: list[float] = []
    for images, labels, image_ids in loader:
        batch_size, crops = images.shape[:2]
        images = images.flatten(0, 1).to(device, non_blocking=True)
        with autocast_context(device):
            logits, _ = model(images)
        logits = logits.float().view(batch_size, crops).mean(dim=1)
        scores = torch.sigmoid(logits)
        all_ids.extend(image_ids)
        all_labels.extend(labels.numpy().tolist())
        all_scores.extend(scores.cpu().numpy().tolist())
    metrics = binary_metrics(np.asarray(all_labels, dtype=np.int8), all_scores)
    predictions = pd.DataFrame({"id": all_ids, "label": all_labels, "score": all_scores})
    return metrics, predictions


def make_optimizer(model: torch.nn.Module, config: dict[str, object]) -> torch.optim.Optimizer:
    train_config = config["train"]
    assert isinstance(train_config, dict)
    if isinstance(
        model,
        (
            CLIPForensicDetector,
            CLIPEffortOrthogonalDetector,
            TimmEffortOrthogonalDetector,
            CLIPAdapterPatchMILDetector,
            CLIPAttentionLoRADetector,
            CLIPLocalForensicDetector,
            CLIPMixtureOfLayersDetector,
            CLIPOfficialMoLDDetector,
            CLIPResidualAdapterDetector,
            GAPLDetector,
            NPRDetector,
            OmniAIDOpenAIClipDetector,
            OmniAIDResidualAdapterDetector,
            SAFEDetector,
            CommunityForensicsDetector,
            DRCTConvNextDetector,
            DistilledTimmForensicDetector,
            ExternalTimmForensicDetector,
            DDADinoV2Detector,
            TimmForensicDetector,
        ),
    ):
        groups = [
            {
                "params": model.backbone_trainable_parameters(),
                "lr": float(train_config["backbone_lr"]),
            },
            {"params": model.head_parameters(), "lr": float(train_config["head_lr"])},
        ]
    elif isinstance(model, RINEDetector):
        groups = [
            {"params": [model.alpha], "lr": float(train_config["alpha_lr"])},
            {
                "params": list(model.proj1.parameters()) + list(model.proj2.parameters()),
                "lr": float(train_config["projection_lr"]),
            },
            {"params": model.head.parameters(), "lr": float(train_config["head_lr"])},
        ]
    else:
        raise TypeError(f"unsupported model type: {type(model)}")
    return torch.optim.AdamW(groups, weight_decay=float(train_config["weight_decay"]))


def make_scheduler(
    optimizer: torch.optim.Optimizer, steps_per_epoch: int, config: dict[str, object]
) -> torch.optim.lr_scheduler.LambdaLR:
    train_config = config["train"]
    assert isinstance(train_config, dict)
    total_steps = int(train_config["epochs"]) * steps_per_epoch
    warmup_steps = int(train_config["warmup_epochs"]) * steps_per_epoch

    def scale(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    config: dict[str, object],
    ema_state: dict[str, torch.Tensor] | None = None,
    *,
    max_steps: int | None = None,
) -> dict[str, float]:
    train_config = config["train"]
    assert isinstance(train_config, dict)
    model.train()
    running_loss = 0.0
    running_bce = 0.0
    running_pairwise = 0.0
    running_contrastive = 0.0
    running_auxiliary = 0.0
    count = 0
    skipped_steps = 0
    last_grad_norm = float("nan")
    for step, (images, labels, _) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device):
            logits, embeddings = model(images)
            bce = F.binary_cross_entropy_with_logits(logits.float(), labels)
            contrastive_weight = float(train_config["contrastive_weight"])
            loss = bce
            auxiliary_weight = float(
                train_config.get(
                    "auxiliary_weight",
                    train_config.get("patch_auxiliary_weight", 0.0),
                )
            )
            auxiliary_logits = getattr(model, "auxiliary_logits", None)
            if auxiliary_weight > 0:
                if not isinstance(auxiliary_logits, torch.Tensor):
                    raise RuntimeError(
                        "auxiliary_weight requires model.auxiliary_logits"
                    )
                auxiliary = F.binary_cross_entropy_with_logits(
                    auxiliary_logits.float(), labels
                )
                loss = loss + auxiliary_weight * auxiliary
            else:
                auxiliary = logits.new_zeros(())
            if contrastive_weight > 0:
                contrastive = supervised_contrastive_loss(
                    embeddings.float(), labels
                )
                loss = loss + contrastive_weight * contrastive
            else:
                contrastive = logits.new_zeros(())
            pairwise_weight = float(train_config.get("pairwise_auc_weight", 0.0))
            positive_logits = logits[labels > 0.5].float()
            negative_logits = logits[labels <= 0.5].float()
            if pairwise_weight > 0 and len(positive_logits) and len(negative_logits):
                temperature = float(train_config.get("pairwise_temperature", 1.0))
                margins = (
                    positive_logits[:, None] - negative_logits[None, :]
                ) / temperature
                pairwise = F.softplus(-margins).mean()
                loss = loss + pairwise_weight * pairwise
            else:
                pairwise = logits.new_zeros(())
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
            if ema_state is not None:
                decay = float(train_config["ema_decay"])
                with torch.no_grad():
                    for name, parameter in model.named_parameters():
                        if name in ema_state:
                            ema_state[name].mul_(decay).add_(
                                parameter.detach(), alpha=1.0 - decay
                            )
        else:
            skipped_steps += 1
        running_loss += float(loss.detach())
        running_bce += float(bce.detach())
        running_pairwise += float(pairwise.detach())
        running_contrastive += float(contrastive.detach())
        running_auxiliary += float(auxiliary.detach())
        count += 1
        if max_steps is not None and count >= max_steps:
            break
    return {
        "loss": running_loss / max(count, 1),
        "bce": running_bce / max(count, 1),
        "pairwise": running_pairwise / max(count, 1),
        "contrastive": running_contrastive / max(count, 1),
        "auxiliary": running_auxiliary / max(count, 1),
        "steps": float(count),
        "skipped_steps": float(skipped_steps),
        "last_grad_norm": last_grad_norm,
        "lr_head": float(optimizer.param_groups[-1]["lr"]),
    }


@contextmanager
def use_ema_parameters(
    model: torch.nn.Module, ema_state: dict[str, torch.Tensor] | None
):
    if ema_state is None:
        yield
        return
    original: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name in ema_state:
                original[name] = parameter.detach().clone()
                parameter.copy_(ema_state[name])
    try:
        yield
    finally:
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name in original:
                    parameter.copy_(original[name])


def load_external_trainable_checkpoint(
    model: torch.nn.Module,
    checkpoint: Path,
    *,
    allowed_missing_trainable: tuple[str, ...] = (),
) -> dict[str, object]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"invalid external trainable checkpoint: {checkpoint}")
    if payload.get("format") != "aigc-clip-adapter-external-supervised-v1":
        raise ValueError(f"unexpected external checkpoint format: {payload.get('format')}")
    if int(payload.get("competition_rows_accessed", -1)) != 0:
        raise RuntimeError("external pretraining checkpoint accessed competition rows")
    state = payload.get("trainable_state")
    if not isinstance(state, dict):
        raise TypeError("external checkpoint has no trainable_state")
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys:
        raise RuntimeError(f"unexpected external checkpoint keys: {result.unexpected_keys}")
    missing_trainable = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name in result.missing_keys
        and name not in allowed_missing_trainable
    ]
    if missing_trainable:
        raise RuntimeError(f"missing external trainable keys: {missing_trainable}")
    return {
        "format": payload["format"],
        "checkpoint": str(checkpoint),
        "external_rows": int(payload["external_rows"]),
        "external_model_count": int(payload["external_model_count"]),
        "competition_rows_accessed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, default=Path("configs/image_v1.yaml"))
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config_path = resolve(root, str(args.config))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_frame, valid_frame, split_audit = load_split(root, config)
    train_loader, valid_loader = make_loaders(root, config, train_frame, valid_frame)

    model_config = config["model"]
    assert isinstance(model_config, dict)
    architecture = str(model_config.get("architecture", "rine"))
    if architecture == "clip_forensic_finetune":
        model = CLIPForensicDetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["unfreeze_last_blocks"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = {
            "source": "OpenAI CLIP pretrained visual encoder",
            "unfrozen_last_blocks": int(model_config["unfreeze_last_blocks"]),
        }
    elif architecture == "clip_effort_orthogonal":
        model = CLIPEffortOrthogonalDetector(
            str(model_config["backbone"]),
            int(model_config["effort_last_blocks"]),
            float(model_config["dropout"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = {
            "source": "OpenAI CLIP pretrained visual encoder",
            "method": "Effort rank-one smallest-singular attention adaptation",
            "effort_last_blocks": int(model_config["effort_last_blocks"]),
            "external_checkpoint": None,
        }
    elif architecture == "timm_effort_orthogonal":
        checkpoint = resolve(root, str(model_config["checkpoint"]))
        model = TimmEffortOrthogonalDetector(
            str(model_config["backbone"]),
            checkpoint,
            int(model_config["effort_last_blocks"]),
            float(model_config["dropout"]),
            resolve(root, str(model_config["head_init_checkpoint"]))
            if model_config.get("head_init_checkpoint")
            else None,
        ).to(device)
        checkpoint_audit = {
            "source": str(model_config.get("repository", "timm pretrained visual encoder")),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": str(model_config.get("checkpoint_sha256", "")),
            "method": "Effort rank-one smallest-singular attention adaptation",
            "effort_last_blocks": int(model_config["effort_last_blocks"]),
            "head_init_checkpoint": model_config.get("head_init_checkpoint"),
            "external_checkpoint": None,
        }
    elif architecture == "clip_attention_lora":
        model = CLIPAttentionLoRADetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["lora_last_blocks"]),
            int(model_config["lora_rank"]),
            float(model_config["lora_alpha"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = {
            "source": "OpenAI CLIP pretrained visual encoder",
            "method": "late-attention QKV and output-projection LoRA",
            "lora_last_blocks": int(model_config["lora_last_blocks"]),
            "lora_rank": int(model_config["lora_rank"]),
            "lora_alpha": float(model_config["lora_alpha"]),
        }
    elif architecture == "clip_residual_adapter":
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
        checkpoint_audit = {
            "source": "OpenAI CLIP pretrained visual encoder",
            "method": "zero-initialized late-block bottleneck residual adapters",
            "adapter_last_blocks": int(model_config["adapter_last_blocks"]),
            "adapter_dim": int(model_config["adapter_dim"]),
            "fully_frozen_visual_parameters": True,
        }
    elif architecture == "clip_adapter_patch_mil":
        model = CLIPAdapterPatchMILDetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["adapter_last_blocks"]),
            int(model_config["adapter_dim"]),
            float(model_config["adapter_dropout"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            int(model_config["patch_hidden_dim"]),
            float(model_config["patch_top_fraction"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = {
            "source": "OpenAI CLIP pretrained visual encoder",
            "method": "late residual adapters plus patch-token multiple-instance pooling",
            "adapter_last_blocks": int(model_config["adapter_last_blocks"]),
            "patch_evidence": "mean/top-k/bottom-k/std of learned final-layer patch scores",
        }
    elif architecture == "clip_adapter_multidepth_patch_mil":
        model = CLIPAdapterMultiDepthPatchMILDetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["adapter_last_blocks"]),
            int(model_config["adapter_dim"]),
            float(model_config["adapter_dropout"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            int(model_config["patch_hidden_dim"]),
            float(model_config["patch_top_fraction"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = {
            "source": "OpenAI CLIP pretrained visual encoder",
            "method": "late residual adapters plus multi-depth patch-token MIL",
            "adapter_last_blocks": int(model_config["adapter_last_blocks"]),
            "patch_evidence": (
                "mean/top-k/bottom-k/std learned independently at every selected depth"
            ),
        }
    elif architecture == "clip_adapter_frequency_residual":
        model = CLIPAdapterFrequencyResidualDetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["adapter_last_blocks"]),
            int(model_config["adapter_dim"]),
            float(model_config["adapter_dropout"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            int(model_config["frequency_hidden_dim"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = {
            "source": "OpenAI CLIP pretrained visual encoder",
            "method": "late residual adapters plus RGB high-pass residual CNN",
            "adapter_last_blocks": int(model_config["adapter_last_blocks"]),
            "frequency_evidence": (
                "multi-scale RGB residuals and fixed Sobel/Laplacian responses"
            ),
        }
    elif architecture == "clip_adapter_degradation_response":
        model = CLIPAdapterDegradationResponseDetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["adapter_last_blocks"]),
            int(model_config["adapter_dim"]),
            float(model_config["adapter_dropout"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            int(model_config["response_hidden_dim"]),
            float(model_config["degradation_scale"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = {
            "source": "OpenAI CLIP pretrained visual encoder",
            "method": "zero-preserving global detector plus resampling-response evidence",
            "adapter_last_blocks": int(model_config["adapter_last_blocks"]),
            "degradation_scale": float(model_config["degradation_scale"]),
            "response_evidence": "original/degraded multi-depth embedding delta",
        }
    elif architecture == "clip_residual_adapter_lora":
        model = CLIPResidualAdapterLoRADetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["adapter_last_blocks"]),
            int(model_config["adapter_dim"]),
            float(model_config["adapter_dropout"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            int(model_config["lora_last_blocks"]),
            int(model_config["lora_rank"]),
            float(model_config["lora_alpha"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = {
            "source": "OpenAI CLIP pretrained visual encoder",
            "method": "late residual adapters plus zero-initialized attention LoRA",
            "adapter_last_blocks": int(model_config["adapter_last_blocks"]),
            "lora_last_blocks": int(model_config["lora_last_blocks"]),
            "lora_rank": int(model_config["lora_rank"]),
        }
    elif architecture == "clip_mixture_of_layers_finetune":
        model = CLIPMixtureOfLayersDetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["unfreeze_last_blocks"]),
            int(model_config["hidden_dim"]),
            int(model_config["gate_hidden_dim"]),
            float(model_config["dropout"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = {
            "source": "OpenAI CLIP pretrained visual encoder",
            "method": "MoLD paper-formula data-dependent layer gating",
            "selected_blocks": [int(index) for index in model_config["feature_blocks"]],
            "unfrozen_last_blocks": int(model_config["unfreeze_last_blocks"]),
        }
        if model_config.get("init_checkpoint"):
            distilled_audit = model.load_distilled_head_checkpoint(
                resolve(root, str(model_config["init_checkpoint"]))
            )
            checkpoint_audit["distilled_head"] = distilled_audit
    elif architecture == "clip_official_mold":
        model = CLIPOfficialMoLDDetector(
            str(model_config["backbone"]),
            int(model_config["hidden_dim"]),
            float(model_config["init_gain"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = {
            "source": "OpenAI CLIP pretrained visual encoder",
            "method": "official nahyeonkaty/mold released all-layer score head",
            "fully_frozen_visual_parameters": True,
            "selected_blocks": "all",
        }
    elif architecture == "omniaid_openai_clip_svd":
        model = OmniAIDOpenAIClipDetector(
            str(model_config["backbone"]),
            resolve(root, str(model_config["delta_checkpoint"])),
            float(model_config["expert_dropout"]),
            trainable_experts=bool(model_config["trainable_experts"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = dict(model.checkpoint_audit)
        checkpoint_audit["method"] = (
            "official OmniAID v2 Mirage SVD experts over bit-identical cached OpenAI CLIP"
        )
    elif architecture == "omniaid_residual_adapter":
        model = OmniAIDResidualAdapterDetector(
            str(model_config["backbone"]),
            resolve(root, str(model_config["delta_checkpoint"])),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["adapter_last_blocks"]),
            int(model_config["adapter_dim"]),
            float(model_config["adapter_dropout"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = dict(model.checkpoint_audit)
        checkpoint_audit.update(
            {
                "method": (
                    "V21 external-pretrained residual adapters plus zero-initialized "
                    "per-layer OmniAID forensic prior scales"
                ),
                "domain_scale_initialization": 0.0,
                "official_router_and_experts_frozen": True,
                "selected_blocks": [
                    int(index) for index in model_config["feature_blocks"]
                ],
            }
        )
    elif architecture == "clip_local_forensic_finetune":
        model = CLIPLocalForensicDetector(
            str(model_config["backbone"]),
            [int(index) for index in model_config["feature_blocks"]],
            int(model_config["unfreeze_last_blocks"]),
            int(model_config["hidden_dim"]),
            float(model_config["dropout"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_audit = {
            "source": "OpenAI CLIP pretrained visual encoder",
            "unfrozen_last_blocks": int(model_config["unfreeze_last_blocks"]),
            "local_evidence": "final-layer patch-token mean and standard deviation",
        }
    elif architecture == "gapl":
        checkpoint_path = resolve(root, str(model_config["checkpoint"]))
        model = GAPLDetector(
            resolve(root, str(model_config["clip_config"])),
            checkpoint_path,
            unfreeze_lora=bool(model_config["unfreeze_lora"]),
        ).to(device)
        checkpoint_audit = {
            "source": "UltraCapture/GAPL CVPR 2026 official checkpoint",
            "checkpoint": str(checkpoint_path),
            "safe_load": "torch.load(weights_only=True), strict state dict",
            "generator_prototypes": 64,
            "unfreeze_lora": bool(model_config["unfreeze_lora"]),
        }
    elif architecture == "npr":
        checkpoint_path = resolve(root, str(model_config["checkpoint"]))
        model = NPRDetector(
            checkpoint_path,
            unfreeze_layer2=bool(model_config["unfreeze_layer2"]),
        ).to(device)
        checkpoint_audit = {
            "source": "chuangchuangtan/NPR-DeepfakeDetection official CVPR 2024 checkpoint",
            "checkpoint": str(checkpoint_path),
            "safe_load": "torch.load(weights_only=True), strict state dict",
            "unfreeze_layer2": bool(model_config["unfreeze_layer2"]),
        }
    elif architecture == "safe":
        checkpoint_path = resolve(root, str(model_config["checkpoint"]))
        model = SAFEDetector(
            checkpoint_path,
            unfreeze_layer2=bool(model_config["unfreeze_layer2"]),
        ).to(device)
        checkpoint_audit = {
            "source": "Ouxiang-Li/SAFE official KDD 2025 checkpoint",
            "checkpoint": str(checkpoint_path),
            "sha256": str(model_config["checkpoint_sha256"]),
            "safe_load": "torch.load(weights_only=True), strict official model keys",
            "unfreeze_layer2": bool(model_config["unfreeze_layer2"]),
        }
    elif architecture == "community_forensics":
        checkpoint_path = resolve(root, str(model_config["checkpoint"]))
        model = CommunityForensicsDetector(
            str(model_config["backbone"]),
            checkpoint_path,
            int(model_config["unfreeze_last_blocks"]),
        ).to(device)
        checkpoint_audit = {
            "source": "OwensLab/commfor-model-384",
            "checkpoint": str(checkpoint_path),
            "unfrozen_last_blocks": int(model_config["unfreeze_last_blocks"]),
        }
    elif architecture == "dda_dinov2_lora":
        model = DDADinoV2Detector(
            resolve(root, str(model_config["repository"])),
            resolve(root, str(model_config["checkpoint"])),
            str(model_config["checkpoint_sha256"]),
        ).to(device)
        checkpoint_audit = dict(model.checkpoint_audit)
    elif architecture == "timm_forensic_finetune":
        checkpoint_path = resolve(root, str(model_config["checkpoint"]))
        model = TimmForensicDetector(
            str(model_config["backbone"]),
            checkpoint_path,
            int(model_config["unfreeze_last_blocks"]),
            float(model_config["dropout"]),
        ).to(device)
        checkpoint_audit = {
            "source": f"timm pretrained {model_config['backbone']}",
            "checkpoint": str(checkpoint_path),
            "unfrozen_last_blocks": int(model_config["unfreeze_last_blocks"]),
        }
    elif architecture == "distilled_timm_finetune":
        checkpoint_path = resolve(root, str(model_config["distilled_checkpoint"]))
        model = DistilledTimmForensicDetector(
            str(model_config["backbone"]),
            checkpoint_path,
            int(model_config["unfreeze_last_units"]),
            float(model_config["dropout"]),
        ).to(device)
        checkpoint_audit = {
            "source": "external multi-teacher distillation; no competition rows used",
            "checkpoint": str(checkpoint_path),
            "unfrozen_last_units": int(model_config["unfreeze_last_units"]),
            "competition_rows_accessed_during_distillation": 0,
        }
    elif architecture == "external_timm_finetune":
        checkpoint_path = resolve(root, str(model_config["external_checkpoint"]))
        model = ExternalTimmForensicDetector(
            str(model_config["backbone"]),
            checkpoint_path,
            int(model_config["unfreeze_last_units"]),
            float(model_config["dropout"]),
        ).to(device)
        checkpoint_audit = {
            "source": "external hard-label real/fake supervision; no competition rows used",
            "checkpoint": str(checkpoint_path),
            "unfrozen_last_units": int(model_config["unfreeze_last_units"]),
            "competition_rows_accessed_during_external_training": 0,
        }
    elif architecture == "drct_convnext":
        checkpoint_path = resolve(root, str(model_config["checkpoint"]))
        model = DRCTConvNextDetector(
            str(model_config["backbone"]),
            checkpoint_path,
            str(model_config["checkpoint_sha256"]),
            int(model_config["embedding_dim"]),
            int(model_config["unfreeze_last_units"]),
            float(model_config["dropout"]),
        ).to(device)
        checkpoint_audit = {
            "source": "beibuwandeluori/DRCT official DRCT-2M SDv1.4 ConvNeXt-Base checkpoint",
            "checkpoint": str(checkpoint_path),
            "sha256": str(model_config["checkpoint_sha256"]),
            "safe_load": "torch.load(weights_only=True), strict encoder and classifier states",
            "unfrozen_last_units": int(model_config["unfreeze_last_units"]),
            "binary_head_initialization": "official fake logit minus official real logit",
        }
    elif architecture == "rine":
        model = RINEDetector(
            str(model_config["backbone"]),
            int(model_config["projection_layers"]),
            int(model_config["projection_dim"]),
            device=device,
            download_root=resolve(root, str(model_config["clip_cache"])),
        ).to(device)
        checkpoint_path = resolve(root, str(model_config["init_checkpoint"]))
        checkpoint_audit = model.load_trainable_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"unknown model architecture: {architecture}")
    if model_config.get("init_trainable_checkpoint"):
        allowed_missing = (
            ("domain_scales",)
            if isinstance(model, OmniAIDResidualAdapterDetector)
            else ()
        )
        checkpoint_audit["external_supervised_initialization"] = (
            load_external_trainable_checkpoint(
                model,
                resolve(root, str(model_config["init_trainable_checkpoint"])),
                allowed_missing_trainable=allowed_missing,
            )
        )
    optimizer = make_optimizer(model, config)
    scheduler = make_scheduler(optimizer, len(train_loader), config)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda", init_scale=128.0, growth_interval=2000
    )
    train_config = config["train"]
    assert isinstance(train_config, dict)
    ema_state: dict[str, torch.Tensor] | None = None
    if float(train_config.get("ema_decay", 0.0)) > 0:
        ema_state = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }

    output_dir = resolve(root, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.jsonl"
    history_path.unlink(missing_ok=True)
    run_metadata = {
        "config": config,
        "split_audit": split_audit,
        "checkpoint_audit": checkpoint_audit,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "trainable_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
        ),
        "frozen_parameters": int(
            sum(parameter.numel() for parameter in model.parameters() if not parameter.requires_grad)
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    if args.smoke_only:
        tracked_parameter = next(
            parameter
            for _, parameter in reversed(list(model.named_parameters()))
            if parameter.requires_grad
        )
        before = tracked_parameter.detach().clone()
        result = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
            ema_state,
            max_steps=1,
        )
        changed = not torch.equal(before, tracked_parameter.detach())
        if not changed or not math.isfinite(result["loss"]):
            raise RuntimeError(f"smoke check failed: changed={changed}, result={result}")
        print(json.dumps({"smoke": "passed", "parameter_changed": changed, **result}, indent=2))
        return

    history: list[dict[str, object]] = []
    initial_metrics, initial_predictions = evaluate(model, valid_loader, device)
    initial_predictions.to_csv(output_dir / "valid_predictions_epoch000.csv", index=False)
    initial_predictions.to_csv(output_dir / "valid_predictions_best.csv", index=False)
    print(f"initial {json.dumps(initial_metrics)}", flush=True)
    best_auc = float(initial_metrics["auroc"])
    best_epoch = 0
    torch.save(model.trainable_state_dict(), output_dir / "best_trainable.pth")

    started = time.time()
    for epoch in range(1, int(train_config["epochs"]) + 1):
        train_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            device,
            config,
            ema_state,
        )
        with use_ema_parameters(model, ema_state):
            valid_metrics, predictions = evaluate(model, valid_loader, device)
        record: dict[str, object] = {
            "epoch": epoch,
            "elapsed_seconds": time.time() - started,
            "train": train_metrics,
            "valid": valid_metrics,
        }
        history.append(record)
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps(record, ensure_ascii=False), flush=True)
        if float(valid_metrics["auroc"]) > best_auc:
            best_auc = float(valid_metrics["auroc"])
            best_epoch = epoch
            predictions.to_csv(output_dir / "valid_predictions_best.csv", index=False)
            with use_ema_parameters(model, ema_state):
                torch.save(model.trainable_state_dict(), output_dir / "best_trainable.pth")

    # Persist the predeclared final epoch as well as the historical best-epoch
    # artifact.  New strict-CV experiments can consume ``last`` without using
    # outer-fold labels to choose an epoch, while older configs remain fully
    # backward compatible.
    predictions.to_csv(output_dir / "valid_predictions_last.csv", index=False)
    with use_ema_parameters(model, ema_state):
        torch.save(model.trainable_state_dict(), output_dir / "last_trainable.pth")

    final_report = {
        "experiment": config["experiment"],
        "fold": config["fold"],
        "initial": initial_metrics,
        "best_epoch": best_epoch,
        "best_auroc": best_auc,
        "last_epoch": int(train_config["epochs"]),
        "last_auroc": float(valid_metrics["auroc"]),
        "history": history,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(final_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(final_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
