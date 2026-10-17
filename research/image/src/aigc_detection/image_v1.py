from __future__ import annotations

from collections.abc import Callable
import hashlib
from io import BytesIO
from pathlib import Path
import random
import sys

import clip
import timm
import numpy as np
from PIL import Image, ImageFilter, ImageOps
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import parametrize
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.models import resnet50 as torchvision_resnet50
from torchvision.transforms import InterpolationMode
from safetensors.torch import load_file as load_safetensors


CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SIGLIP_MEAN = (0.5, 0.5, 0.5)
SIGLIP_STD = (0.5, 0.5, 0.5)


class _ActivationHook:
    def __init__(self, module: nn.Module) -> None:
        self.output: torch.Tensor | None = None
        self.handle = module.register_forward_hook(self._capture)

    def _capture(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        del module, inputs
        self.output = output

    def close(self) -> None:
        self.handle.remove()


class _BottleneckAdapter(nn.Module):
    """Small zero-initialized residual adapter for a frozen ViT block."""

    def __init__(self, width: int, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, width)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.activation(self.down(self.norm(tokens)))))


class _ResidualAdapterHook:
    def __init__(self, module: nn.Module, adapter: _BottleneckAdapter) -> None:
        self.adapter = adapter
        self.handle = module.register_forward_hook(self._apply)

    def _apply(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> torch.Tensor:
        del module, inputs
        return output + self.adapter(output)

    def close(self) -> None:
        self.handle.remove()


class _LoRAWeightUpdate(nn.Module):
    """Low-rank additive parametrization for a frozen matrix."""

    def __init__(self, out_features: int, in_features: int, rank: int, alpha: float) -> None:
        super().__init__()
        if not 0 < rank <= min(out_features, in_features):
            raise ValueError("LoRA rank is outside the weight dimensions")
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        nn.init.kaiming_uniform_(self.lora_a, a=5**0.5)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        update = torch.matmul(self.lora_b, self.lora_a) * self.scale
        return original + update.to(dtype=original.dtype)


def _smallest_singular_component(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the smallest singular triplet without retaining a full SVD graph."""
    with torch.no_grad():
        original_device = weight.device
        matrix = weight.detach().float()
        if matrix.device.type == "cpu" and torch.cuda.is_available():
            matrix = matrix.cuda()
        u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
        return (
            u[:, -1].to(original_device).clone(),
            singular[-1:].to(original_device).clone(),
            vh[-1].to(original_device).clone(),
        )


class _EffortRankOneUpdate(nn.Module):
    """Effort residual parametrization initialized at the pretrained weight.

    Keeping the original matrix fixed and subtracting the initial rank-one
    component is algebraically identical to freezing Effort's top-r SVD
    reconstruction and training the remaining singular component.
    """

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        u, singular, v = _smallest_singular_component(weight)
        initial = torch.outer(u, v) * singular
        self.u = nn.Parameter(u)
        self.singular = nn.Parameter(singular)
        self.v = nn.Parameter(v)
        self.register_buffer("initial", initial, persistent=True)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        current = torch.outer(self.u, self.v) * self.singular
        return original + (current - self.initial).to(dtype=original.dtype)


class _EffortQKVUpdate(nn.Module):
    """Independent Effort rank-one residuals for packed CLIP Q/K/V weights."""

    def __init__(self, weight: torch.Tensor) -> None:
        super().__init__()
        if weight.shape[0] != 3 * weight.shape[1]:
            raise ValueError(f"expected packed square QKV weight, got {tuple(weight.shape)}")
        width = weight.shape[1]
        initial_parts: list[torch.Tensor] = []
        for name, part in zip(("q", "k", "v"), weight.split(width, dim=0), strict=True):
            u, singular, vector = _smallest_singular_component(part)
            self.register_parameter(f"{name}_u", nn.Parameter(u))
            self.register_parameter(f"{name}_singular", nn.Parameter(singular))
            self.register_parameter(f"{name}_v", nn.Parameter(vector))
            initial_parts.append(torch.outer(u, vector) * singular)
        self.register_buffer("initial", torch.cat(initial_parts), persistent=True)

    def forward(self, original: torch.Tensor) -> torch.Tensor:
        parts = []
        for name in ("q", "k", "v"):
            u = getattr(self, f"{name}_u")
            singular = getattr(self, f"{name}_singular")
            vector = getattr(self, f"{name}_v")
            parts.append(torch.outer(u, vector) * singular)
        current = torch.cat(parts)
        return original + (current - self.initial).to(dtype=original.dtype)


class CLIPEffortOrthogonalDetector(nn.Module):
    """CLIP detector adapted only in low-energy attention subspaces."""

    def __init__(
        self,
        backbone: str,
        effort_last_blocks: int,
        dropout: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__()
        clip_model, _ = clip.load(
            backbone,
            device=device,
            jit=False,
            download_root=str(download_root),
        )
        self.visual = clip_model.visual.float()
        del clip_model
        for parameter in self.visual.parameters():
            parameter.requires_grad = False

        blocks = self.visual.transformer.resblocks
        if not 0 < effort_last_blocks <= len(blocks):
            raise ValueError("effort_last_blocks is outside CLIP depth")
        self.effort_last_blocks = int(effort_last_blocks)
        for block in blocks[-self.effort_last_blocks :]:
            attention = block.attn
            parametrize.register_parametrization(
                attention,
                "in_proj_weight",
                _EffortQKVUpdate(attention.in_proj_weight),
                unsafe=True,
            )
            attention.parametrizations.in_proj_weight.original.requires_grad = False
            parametrize.register_parametrization(
                attention.out_proj,
                "weight",
                _EffortRankOneUpdate(attention.out_proj.weight),
                unsafe=True,
            )
            attention.out_proj.parametrizations.weight.original.requires_grad = False

        width = int(self.visual.ln_post.normalized_shape[0])
        self._pooled: torch.Tensor | None = None

        def capture_pooled(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ) -> None:
            del module, inputs
            self._pooled = output

        self._pool_hook = self.visual.ln_post.register_forward_hook(capture_pooled)
        self.feature_norm = nn.LayerNorm(width)
        self.feature_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(width, 1)

    def train(self, mode: bool = True) -> CLIPEffortOrthogonalDetector:
        super().train(mode)
        self.visual.eval()
        self.feature_norm.train(mode)
        self.feature_dropout.train(mode)
        self.classifier.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._pooled = None
        self.visual(images)
        if self._pooled is None or self._pooled.ndim != 2:
            raise RuntimeError("CLIP Effort failed to capture the pre-projection CLS feature")
        embedding = self.feature_norm(self._pooled.float())
        logits = self.classifier(self.feature_dropout(embedding)).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.visual.parameters() if parameter.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.feature_norm.parameters()) + list(self.classifier.parameters())

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable
        }


class TimmEffortOrthogonalDetector(nn.Module):
    """Effort-style low-energy attention adaptation for a timm ViT."""

    def __init__(
        self,
        backbone: str,
        checkpoint: Path,
        effort_last_blocks: int,
        dropout: float,
        head_init_checkpoint: Path | None = None,
    ) -> None:
        super().__init__()
        # PE-Core's pretrained 1280-D projection head is part of the released
        # representation and must remain present for strict checkpoint parity.
        self.backbone = timm.create_model(backbone, pretrained=False)
        state = load_safetensors(str(checkpoint), device="cpu")
        result = self.backbone.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"timm checkpoint mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        del state
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

        blocks = self.backbone.blocks
        if not 0 < effort_last_blocks <= len(blocks):
            raise ValueError("effort_last_blocks is outside the visual Transformer depth")
        self.effort_last_blocks = int(effort_last_blocks)
        for block in blocks[-self.effort_last_blocks :]:
            attention = block.attn
            parametrize.register_parametrization(
                attention.qkv,
                "weight",
                _EffortQKVUpdate(attention.qkv.weight),
                unsafe=True,
            )
            attention.qkv.parametrizations.weight.original.requires_grad = False
            parametrize.register_parametrization(
                attention.proj,
                "weight",
                _EffortRankOneUpdate(attention.proj.weight),
                unsafe=True,
            )
            attention.proj.parametrizations.weight.original.requires_grad = False

        width = int(self.backbone.num_classes)
        if width <= 0:
            raise ValueError("timm Effort backbone must expose a pretrained embedding head")
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(width, 1)
        if head_init_checkpoint is not None:
            payload = torch.load(head_init_checkpoint, map_location="cpu", weights_only=True)
            if not isinstance(payload, dict) or payload.get("format") != "aigc-pe-core-linear-v1":
                raise TypeError(f"invalid PE-Core linear head: {head_init_checkpoint}")
            if payload.get("backbone") != backbone:
                raise ValueError("PE-Core linear head backbone mismatch")
            state_dict = payload.get("state_dict")
            if not isinstance(state_dict, dict):
                raise TypeError("PE-Core linear head has no state_dict")
            self.head.load_state_dict(state_dict, strict=True)

    def train(self, mode: bool = True) -> TimmEffortOrthogonalDetector:
        super().train(mode)
        self.backbone.eval()
        self.dropout.train(mode)
        self.head.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.backbone(images)
        with torch.autocast(device_type=images.device.type, enabled=False):
            embedding = F.normalize(embedding.float(), dim=1)
            logits = self.head(self.dropout(embedding)).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class RINEDetector(nn.Module):
    """RINE detector with a frozen OpenAI CLIP visual encoder.

    The trainable architecture follows the official Apache-2.0 RINE implementation:
    https://github.com/mever-team/rine
    """

    def __init__(
        self,
        backbone: str,
        projection_layers: int,
        projection_dim: int,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__()
        clip_model, _ = clip.load(
            backbone,
            device=device,
            jit=False,
            download_root=str(download_root),
        )
        self.visual = clip_model.visual
        del clip_model
        self.visual.eval()
        for parameter in self.visual.parameters():
            parameter.requires_grad = False

        self.hooks = [
            _ActivationHook(module)
            for name, module in self.visual.named_modules()
            if "ln_2" in name
        ]
        if not self.hooks:
            raise RuntimeError("no CLIP Transformer ln_2 blocks were found")
        backbone_dim = int(self.visual.ln_post.normalized_shape[0])

        self.alpha = nn.Parameter(torch.randn(1, len(self.hooks), projection_dim))
        projection_1: list[nn.Module] = [nn.Dropout()]
        for index in range(projection_layers):
            projection_1.extend(
                [
                    nn.Linear(backbone_dim if index == 0 else projection_dim, projection_dim),
                    nn.ReLU(),
                    nn.Dropout(),
                ]
            )
        self.proj1 = nn.Sequential(*projection_1)

        projection_2: list[nn.Module] = [nn.Dropout()]
        for _ in range(projection_layers):
            projection_2.extend(
                [nn.Linear(projection_dim, projection_dim), nn.ReLU(), nn.Dropout()]
            )
        self.proj2 = nn.Sequential(*projection_2)
        self.head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(projection_dim, 1),
        )

    def train(self, mode: bool = True) -> RINEDetector:
        super().train(mode)
        self.visual.eval()
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.visual.eval()
        with torch.no_grad():
            self.visual(images)
            if any(hook.output is None for hook in self.hooks):
                raise RuntimeError("a CLIP intermediate activation was not captured")
            # Each activation is [tokens, batch, width]. The CLS token is index 0.
            intermediate = torch.stack(
                [hook.output for hook in self.hooks if hook.output is not None], dim=2
            )[0]

        projected = self.proj1(intermediate.float())
        importance = torch.softmax(self.alpha, dim=1)
        embedding = self.proj2((importance * projected).sum(dim=1))
        logits = self.head(embedding).squeeze(1)
        return logits, embedding

    def load_trainable_checkpoint(self, path: Path) -> dict[str, list[str]]:
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"expected a state dict in {path}")
        result = self.load_state_dict(state, strict=False)
        unexpected = list(result.unexpected_keys)
        missing_trainable = [
            name
            for name in result.missing_keys
            if not name.startswith("visual.")
        ]
        if unexpected or missing_trainable:
            raise RuntimeError(
                f"checkpoint mismatch: unexpected={unexpected}, missing_trainable={missing_trainable}"
            )
        return {"missing_frozen": list(result.missing_keys), "unexpected": unexpected}

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("visual.")
        }


class CLIPForensicDetector(nn.Module):
    """Fine-tune CLIP's last visual blocks while classifying multi-depth features."""

    def __init__(
        self,
        backbone: str,
        feature_blocks: list[int],
        unfreeze_last_blocks: int,
        hidden_dim: int,
        dropout: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__()
        clip_model, _ = clip.load(
            backbone,
            device=device,
            jit=False,
            download_root=str(download_root),
        )
        # Keep master weights in fp32; autocast provides fp16 compute without fp16 optimizer states.
        self.visual = clip_model.visual.float()
        del clip_model
        blocks = self.visual.transformer.resblocks
        block_count = len(blocks)
        if not feature_blocks or min(feature_blocks) < 0 or max(feature_blocks) >= block_count:
            raise ValueError(f"feature_blocks must be within [0, {block_count - 1}]")
        if not 0 < unfreeze_last_blocks <= block_count:
            raise ValueError("unfreeze_last_blocks is outside the visual Transformer depth")

        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        for block in blocks[-unfreeze_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        self.feature_blocks = list(feature_blocks)
        self.feature_hooks = [_ActivationHook(blocks[index]) for index in self.feature_blocks]
        width = int(self.visual.ln_post.normalized_shape[0])
        feature_dim = width * len(self.feature_blocks)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.feature_activation = nn.GELU()
        self.feature_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.visual(images)
        if any(hook.output is None for hook in self.feature_hooks):
            raise RuntimeError("a selected CLIP block activation was not captured")
        # Keep the small classification head in fp32. This avoids artificial score ties from
        # fp16 logit quantization while the expensive CLIP encoder still runs under autocast.
        with torch.autocast(device_type=images.device.type, enabled=False):
            cls_tokens = [
                hook.output[0].float()
                for hook in self.feature_hooks
                if hook.output is not None
            ]
            features = self.feature_norm(torch.cat(cls_tokens, dim=1))
            embedding = self.feature_activation(self.feature_projection(features))
            logits = self.classifier(self.feature_dropout(embedding)).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.visual.parameters() if parameter.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        modules = [
            self.feature_norm,
            self.feature_projection,
            self.classifier,
        ]
        return [parameter for module in modules for parameter in module.parameters()]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class CLIPResidualAdapterDetector(nn.Module):
    """Frozen CLIP with trainable residual adapters in its late transformer blocks.

    Full late-block fine-tuning exposes tens of millions of parameters to only
    800 training images per fold.  Zero-initialized bottleneck adapters preserve
    the pretrained representation at initialization and learn a compact forensic
    residual while the multi-depth classification head remains identical to V4.
    """

    def __init__(
        self,
        backbone: str,
        feature_blocks: list[int],
        adapter_last_blocks: int,
        adapter_dim: int,
        adapter_dropout: float,
        hidden_dim: int,
        dropout: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__()
        clip_model, _ = clip.load(
            backbone,
            device=device,
            jit=False,
            download_root=str(download_root),
        )
        self.visual = clip_model.visual.float()
        del clip_model
        blocks = self.visual.transformer.resblocks
        block_count = len(blocks)
        if not feature_blocks or min(feature_blocks) < 0 or max(feature_blocks) >= block_count:
            raise ValueError(f"feature_blocks must be within [0, {block_count - 1}]")
        if not 0 < adapter_last_blocks <= block_count:
            raise ValueError("adapter_last_blocks is outside the visual Transformer depth")
        if adapter_dim <= 0:
            raise ValueError("adapter_dim must be positive")

        for parameter in self.visual.parameters():
            parameter.requires_grad = False

        width = int(self.visual.ln_post.normalized_shape[0])
        adapter_blocks = list(blocks[-adapter_last_blocks:])
        self.adapters = nn.ModuleList(
            [
                _BottleneckAdapter(width, adapter_dim, adapter_dropout)
                for _ in adapter_blocks
            ]
        )
        # Register the mutating adapter hooks before the read-only feature hooks,
        # so selected late-depth features include the accumulated adaptation.
        self._adapter_hooks = [
            _ResidualAdapterHook(block, adapter)
            for block, adapter in zip(adapter_blocks, self.adapters, strict=True)
        ]
        self.feature_blocks = list(feature_blocks)
        self.feature_hooks = [_ActivationHook(blocks[index]) for index in self.feature_blocks]

        feature_dim = width * len(self.feature_blocks)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.feature_activation = nn.GELU()
        self.feature_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def train(self, mode: bool = True) -> CLIPResidualAdapterDetector:
        super().train(mode)
        # The encoder itself is frozen; only adapters and the head change state.
        self.visual.eval()
        self.adapters.train(mode)
        self.feature_norm.train(mode)
        self.feature_projection.train(mode)
        self.feature_dropout.train(mode)
        self.classifier.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.visual(images)
        if any(hook.output is None for hook in self.feature_hooks):
            raise RuntimeError("a selected CLIP block activation was not captured")
        with torch.autocast(device_type=images.device.type, enabled=False):
            cls_tokens = [
                hook.output[0].float()
                for hook in self.feature_hooks
                if hook.output is not None
            ]
            features = self.feature_norm(torch.cat(cls_tokens, dim=1))
            embedding = self.feature_activation(self.feature_projection(features))
            logits = self.classifier(self.feature_dropout(embedding)).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.adapters.parameters())

    def head_parameters(self) -> list[nn.Parameter]:
        modules = [self.feature_norm, self.feature_projection, self.classifier]
        return [parameter for module in modules for parameter in module.parameters()]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class CLIPResidualAdapterLoRADetector(CLIPResidualAdapterDetector):
    """Strong residual-adapter detector with zero-preserving attention LoRA.

    The inherited adapters retain the established forensic representation while
    low-rank QKV/output updates provide an independently parameterized way to
    absorb new generator domains.  LoRA B matrices start at zero, so loading an
    adapter checkpoint preserves its logits exactly before LoRA training.
    """

    def __init__(
        self,
        backbone: str,
        feature_blocks: list[int],
        adapter_last_blocks: int,
        adapter_dim: int,
        adapter_dropout: float,
        hidden_dim: int,
        dropout: float,
        lora_last_blocks: int,
        lora_rank: int,
        lora_alpha: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__(
            backbone,
            feature_blocks,
            adapter_last_blocks,
            adapter_dim,
            adapter_dropout,
            hidden_dim,
            dropout,
            device=device,
            download_root=download_root,
        )
        blocks = self.visual.transformer.resblocks
        if not 0 < lora_last_blocks <= len(blocks):
            raise ValueError("lora_last_blocks is outside the visual Transformer depth")
        for block in blocks[-lora_last_blocks:]:
            attention = block.attn
            in_projection = attention.in_proj_weight
            parametrize.register_parametrization(
                attention,
                "in_proj_weight",
                _LoRAWeightUpdate(
                    int(in_projection.shape[0]),
                    int(in_projection.shape[1]),
                    lora_rank,
                    lora_alpha,
                ).to(device=in_projection.device, dtype=in_projection.dtype),
            )
            output_projection = attention.out_proj.weight
            parametrize.register_parametrization(
                attention.out_proj,
                "weight",
                _LoRAWeightUpdate(
                    int(output_projection.shape[0]),
                    int(output_projection.shape[1]),
                    lora_rank,
                    lora_alpha,
                ).to(
                    device=output_projection.device,
                    dtype=output_projection.dtype,
                ),
            )

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return list(self.adapters.parameters()) + [
            parameter
            for parameter in self.visual.parameters()
            if parameter.requires_grad
        ]


class _OmniAIDSVDProjection(nn.Module):
    """Compressed OmniAID SVD projection over an existing OpenAI CLIP weight."""

    def __init__(
        self,
        state: dict[str, torch.Tensor],
        prefix: str,
        *,
        num_experts: int,
        dropout: float,
        trainable_experts: bool,
    ) -> None:
        super().__init__()
        self.dropout = dropout
        correction_left = state[f"{prefix}.correction_left"].detach().float()
        correction_right = state[f"{prefix}.correction_right"].detach().float()
        if correction_left.ndim == 1:
            correction_left = correction_left.unsqueeze(1)
        if correction_right.ndim == 1:
            correction_right = correction_right.unsqueeze(0)
        self.register_buffer(
            "correction_left", correction_left
        )
        self.register_buffer(
            "correction_right", correction_right
        )
        self.expert_u = nn.Parameter(
            torch.stack(
                [state[f"{prefix}.expert_u.{index}"] for index in range(num_experts)]
            ).float(),
            requires_grad=trainable_experts,
        )
        self.expert_s = nn.Parameter(
            torch.stack(
                [state[f"{prefix}.expert_s.{index}"] for index in range(num_experts)]
            ).float(),
            requires_grad=trainable_experts,
        )
        self.expert_v = nn.Parameter(
            torch.stack(
                [state[f"{prefix}.expert_v.{index}"] for index in range(num_experts)]
            ).float(),
            requires_grad=trainable_experts,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        base_weight: torch.Tensor,
        base_bias: torch.Tensor | None,
        expert_indices: torch.Tensor,
        expert_gates: torch.Tensor,
        domain_scale: torch.Tensor | float = 1.0,
    ) -> torch.Tensor:
        # Official OmniAID removes the last singular component from the shared
        # CLIP matrix. The compact checkpoint stores that component as two
        # vectors, avoiding a duplicated 1024x1024 matrix for every projection.
        output = F.linear(inputs, base_weight, base_bias)
        correction_score = F.linear(inputs, self.correction_right)
        domain_update = -torch.matmul(
            correction_score, self.correction_left.transpose(0, 1)
        )

        expert_inputs = F.dropout(inputs, p=self.dropout, training=self.training)
        expert_output = torch.zeros_like(output)
        for selection in range(expert_indices.shape[1]):
            chosen = expert_indices[:, selection]
            gates = expert_gates[:, selection, None, None]
            selected_u = self.expert_u[chosen]
            selected_s = self.expert_s[chosen]
            selected_v = self.expert_v[chosen]
            projected = torch.bmm(expert_inputs, selected_v.transpose(1, 2))
            projected = projected * selected_s.unsqueeze(1)
            projected = torch.bmm(projected, selected_u.transpose(1, 2))
            expert_output = expert_output + projected * gates
        scale = torch.as_tensor(domain_scale, dtype=output.dtype, device=output.device)
        return output + scale * (domain_update + expert_output)


class OmniAIDOpenAIClipDetector(nn.Module):
    """OmniAID v2 Mirage experts re-used over the cached OpenAI CLIP-L/336.

    The official checkpoint redundantly contains a clean routing backbone and
    an SVD-MoE backbone. Its frozen base tensors are bit-identical to OpenAI
    CLIP. A verified compact delta therefore keeps only the low-rank experts,
    shared-space corrections, router, and classifier.
    """

    def __init__(
        self,
        backbone: str,
        delta_checkpoint: Path,
        expert_dropout: float,
        *,
        trainable_experts: bool,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__()
        payload = torch.load(delta_checkpoint, map_location="cpu", weights_only=True)
        expected_format = "omniaid-v2-openai-clip-svd-delta-v1"
        if not isinstance(payload, dict) or payload.get("format") != expected_format:
            raise TypeError(f"invalid compact OmniAID checkpoint: {delta_checkpoint}")
        if payload.get("base_backbone") != backbone:
            raise ValueError(
                f"OmniAID base mismatch: {payload.get('base_backbone')} != {backbone}"
            )
        if int(payload.get("num_layers", -1)) != 24:
            raise ValueError("OmniAID compact checkpoint must contain 24 layers")
        if float(payload.get("base_max_abs_error", 1.0)) > 2e-5:
            raise ValueError("OmniAID compact checkpoint failed base-weight audit")
        if float(payload.get("rank_one_max_abs_error", 1.0)) > 2e-5:
            raise ValueError("OmniAID compact checkpoint failed compression audit")
        state = payload.get("state")
        if not isinstance(state, dict):
            raise TypeError("OmniAID compact checkpoint has no tensor state")

        clip_model, _ = clip.load(
            backbone,
            device=device,
            jit=False,
            download_root=str(download_root),
        )
        self.visual = clip_model.visual.float()
        del clip_model
        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        self.visual.eval()
        self._routing_hook = _ActivationHook(self.visual.ln_post)

        self.num_experts = int(payload["num_experts"])
        self.artifact_expert_index = int(payload["artifact_expert_index"])
        self.semantic_top_k = int(payload["semantic_top_k"])
        width = int(self.visual.ln_post.normalized_shape[0])
        self.router = nn.Sequential(
            nn.Linear(width, state["gating_network.network.0.weight"].shape[0]),
            nn.ReLU(),
            nn.Linear(
                state["gating_network.network.2.weight"].shape[1],
                state["gating_network.network.2.weight"].shape[0],
            ),
        )
        self.router[0].load_state_dict(
            {
                "weight": state["gating_network.network.0.weight"],
                "bias": state["gating_network.network.0.bias"],
            }
        )
        self.router[2].load_state_dict(
            {
                "weight": state["gating_network.network.2.weight"],
                "bias": state["gating_network.network.2.bias"],
            }
        )
        self.head = nn.Linear(width, 2)
        self.head.load_state_dict(
            {"weight": state["head.weight"], "bias": state["head.bias"]}
        )

        projections: dict[str, nn.Module] = {}
        for layer_index in range(24):
            for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
                key = f"layer{layer_index}_{projection}"
                prefix = f"layers.{layer_index}.{projection}"
                projections[key] = _OmniAIDSVDProjection(
                    state,
                    prefix,
                    num_experts=self.num_experts,
                    dropout=expert_dropout,
                    trainable_experts=trainable_experts,
                )
        self.projections = nn.ModuleDict(projections)
        self.checkpoint_audit = {
            "format": expected_format,
            "source": str(payload["source"]),
            "source_sha256": str(payload["source_sha256"]),
            "source_training_data": str(payload["source_training_data"]),
            "compressed_checkpoint": str(delta_checkpoint),
            "base_max_abs_error": float(payload["base_max_abs_error"]),
            "rank_one_max_abs_error": float(payload["rank_one_max_abs_error"]),
            "trainable_experts": trainable_experts,
        }

    def train(self, mode: bool = True) -> OmniAIDOpenAIClipDetector:
        super().train(mode)
        self.visual.eval()
        self.router.train(mode)
        self.head.train(mode)
        self.projections.train(mode)
        return self

    def _routing(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.visual(images)
            if self._routing_hook.output is None:
                raise RuntimeError("OmniAID routing feature was not captured")
            routing_features = self._routing_hook.output.detach()
        router_logits = self.router(routing_features)
        top_logits, semantic_indices = router_logits.topk(self.semantic_top_k, dim=1)
        semantic_gates = top_logits.softmax(dim=1)
        artifact_indices = torch.full(
            (len(images), 1),
            self.artifact_expert_index,
            dtype=torch.long,
            device=images.device,
        )
        artifact_gates = torch.ones(
            (len(images), 1), dtype=semantic_gates.dtype, device=images.device
        )
        return (
            torch.cat([semantic_indices, artifact_indices], dim=1),
            torch.cat([semantic_gates, artifact_gates], dim=1),
        )

    def _projection(
        self,
        layer_index: int,
        name: str,
        inputs: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        expert_indices: torch.Tensor,
        expert_gates: torch.Tensor,
    ) -> torch.Tensor:
        module = self.projections[f"layer{layer_index}_{name}"]
        assert isinstance(module, _OmniAIDSVDProjection)
        return module(
            inputs,
            weight,
            bias,
            expert_indices,
            expert_gates,
            self._domain_scale(layer_index, inputs),
        )

    def _domain_scale(
        self, layer_index: int, reference: torch.Tensor
    ) -> torch.Tensor | float:
        del layer_index, reference
        return 1.0

    def _post_moe_block(self, layer_index: int, tokens: torch.Tensor) -> torch.Tensor:
        del layer_index
        return tokens

    def _encode_moe(
        self,
        images: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_gates: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.visual
        tokens = visual.conv1(images)
        tokens = tokens.reshape(tokens.shape[0], tokens.shape[1], -1).permute(0, 2, 1)
        class_token = visual.class_embedding.to(tokens.dtype).expand(
            tokens.shape[0], 1, -1
        )
        tokens = torch.cat([class_token, tokens], dim=1)
        tokens = visual.ln_pre(tokens + visual.positional_embedding.to(tokens.dtype))

        for layer_index, block in enumerate(visual.transformer.resblocks):
            residual = tokens
            normalized = block.ln_1(tokens)
            q_weight, k_weight, v_weight = block.attn.in_proj_weight.chunk(3)
            q_bias, k_bias, v_bias = block.attn.in_proj_bias.chunk(3)
            query = self._projection(
                layer_index,
                "q_proj",
                normalized,
                q_weight,
                q_bias,
                expert_indices,
                expert_gates,
            )
            key = self._projection(
                layer_index,
                "k_proj",
                normalized,
                k_weight,
                k_bias,
                expert_indices,
                expert_gates,
            )
            value = self._projection(
                layer_index,
                "v_proj",
                normalized,
                v_weight,
                v_bias,
                expert_indices,
                expert_gates,
            )
            batch, sequence, width = query.shape
            heads = block.attn.num_heads
            head_width = width // heads
            query = query.view(batch, sequence, heads, head_width).transpose(1, 2)
            key = key.view(batch, sequence, heads, head_width).transpose(1, 2)
            value = value.view(batch, sequence, heads, head_width).transpose(1, 2)
            attention = torch.matmul(
                query * (head_width**-0.5), key.transpose(-1, -2)
            ).softmax(dim=-1)
            attended = torch.matmul(attention, value)
            attended = attended.transpose(1, 2).reshape(batch, sequence, width)
            attended = self._projection(
                layer_index,
                "out_proj",
                attended,
                block.attn.out_proj.weight,
                block.attn.out_proj.bias,
                expert_indices,
                expert_gates,
            )
            tokens = residual + attended
            tokens = tokens + block.mlp(block.ln_2(tokens))
            tokens = self._post_moe_block(layer_index, tokens)
        return visual.ln_post(tokens[:, 0, :])

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expert_indices, expert_gates = self._routing(images)
        embedding = self._encode_moe(images, expert_indices, expert_gates)
        class_logits = self.head(embedding)
        # softmax([real, fake])[:, 1] == sigmoid(fake_logit - real_logit)
        logits = class_logits[:, 1] - class_logits[:, 0]
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in self.projections.parameters()
            if parameter.requires_grad
        ]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.router.parameters()) + list(self.head.parameters())

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class OmniAIDResidualAdapterDetector(OmniAIDOpenAIClipDetector):
    """V21 residual adapters with gated, frozen OmniAID domain directions.

    Domain scales start at zero, so initialization exactly reproduces the
    externally pretrained V21 representation. Competition supervision may then
    admit or reject each layer's public forensic direction independently.
    """

    def __init__(
        self,
        backbone: str,
        delta_checkpoint: Path,
        feature_blocks: list[int],
        adapter_last_blocks: int,
        adapter_dim: int,
        adapter_dropout: float,
        hidden_dim: int,
        dropout: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__(
            backbone,
            delta_checkpoint,
            expert_dropout=0.0,
            trainable_experts=False,
            device=device,
            download_root=download_root,
        )
        blocks = self.visual.transformer.resblocks
        block_count = len(blocks)
        if not feature_blocks or min(feature_blocks) < 0 or max(feature_blocks) >= block_count:
            raise ValueError(f"feature_blocks must be within [0, {block_count - 1}]")
        if not 0 < adapter_last_blocks <= block_count:
            raise ValueError("adapter_last_blocks is outside the visual Transformer depth")
        width = int(self.visual.ln_post.normalized_shape[0])
        self.adapter_start_block = block_count - adapter_last_blocks
        self.adapters = nn.ModuleList(
            [
                _BottleneckAdapter(width, adapter_dim, adapter_dropout)
                for _ in range(adapter_last_blocks)
            ]
        )
        self.feature_blocks = list(feature_blocks)
        self._feature_block_set = set(feature_blocks)
        self._captured_features: list[torch.Tensor] = []
        self.domain_scales = nn.Parameter(torch.zeros(block_count))

        feature_dim = width * len(feature_blocks)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.feature_activation = nn.GELU()
        self.feature_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

        # The released router, expert directions, and original detector head are
        # priors rather than free parameters in the 800-row fold training.
        for module in (self.router, self.head, self.projections):
            for parameter in module.parameters():
                parameter.requires_grad = False

    def _domain_scale(
        self, layer_index: int, reference: torch.Tensor
    ) -> torch.Tensor:
        return self.domain_scales[layer_index].to(dtype=reference.dtype)

    def _post_moe_block(self, layer_index: int, tokens: torch.Tensor) -> torch.Tensor:
        if layer_index >= self.adapter_start_block:
            tokens = tokens + self.adapters[layer_index - self.adapter_start_block](tokens)
        if layer_index in self._feature_block_set:
            self._captured_features.append(tokens[:, 0, :])
        return tokens

    def train(self, mode: bool = True) -> OmniAIDResidualAdapterDetector:
        super().train(mode)
        self.visual.eval()
        self.router.eval()
        self.head.eval()
        self.projections.eval()
        self.adapters.train(mode)
        self.feature_norm.train(mode)
        self.feature_projection.train(mode)
        self.feature_dropout.train(mode)
        self.classifier.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        expert_indices, expert_gates = self._routing(images)
        self._captured_features = []
        self._encode_moe(images, expert_indices, expert_gates)
        if len(self._captured_features) != len(self.feature_blocks):
            raise RuntimeError("not all requested OmniAID-CLIP depths were captured")
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = self.feature_norm(
                torch.cat([feature.float() for feature in self._captured_features], dim=1)
            )
            embedding = self.feature_activation(self.feature_projection(features))
            logits = self.classifier(self.feature_dropout(embedding)).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [self.domain_scales, *self.adapters.parameters()]

    def head_parameters(self) -> list[nn.Parameter]:
        modules = [self.feature_norm, self.feature_projection, self.classifier]
        return [parameter for module in modules for parameter in module.parameters()]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class CLIPAdapterPatchMILDetector(CLIPResidualAdapterDetector):
    """Residual-adapter CLIP with an image-level multiple-instance patch head."""

    def __init__(
        self,
        backbone: str,
        feature_blocks: list[int],
        adapter_last_blocks: int,
        adapter_dim: int,
        adapter_dropout: float,
        hidden_dim: int,
        dropout: float,
        patch_hidden_dim: int,
        patch_top_fraction: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__(
            backbone,
            feature_blocks,
            adapter_last_blocks,
            adapter_dim,
            adapter_dropout,
            hidden_dim,
            dropout,
            device=device,
            download_root=download_root,
        )
        if not 0.0 < patch_top_fraction <= 0.5:
            raise ValueError("patch_top_fraction must be within (0, 0.5]")
        width = int(self.visual.ln_post.normalized_shape[0])
        self.patch_top_fraction = patch_top_fraction
        self.patch_norm = nn.LayerNorm(width)
        self.patch_projection = nn.Linear(width, patch_hidden_dim)
        self.patch_activation = nn.GELU()
        self.patch_dropout = nn.Dropout(dropout)
        self.patch_classifier = nn.Linear(patch_hidden_dim, 1)
        # Start exactly as the global V17 branch; local evidence earns weight
        # only through supervised training rather than perturbing initialization.
        self.fusion_classifier = nn.Linear(5, 1)
        nn.init.zeros_(self.fusion_classifier.weight)
        nn.init.zeros_(self.fusion_classifier.bias)
        with torch.no_grad():
            self.fusion_classifier.weight[0, 0] = 1.0

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.visual(images)
        if any(hook.output is None for hook in self.feature_hooks):
            raise RuntimeError("a selected CLIP block activation was not captured")
        with torch.autocast(device_type=images.device.type, enabled=False):
            outputs = [
                hook.output.float()
                for hook in self.feature_hooks
                if hook.output is not None
            ]
            cls_tokens = [output[0] for output in outputs]
            features = self.feature_norm(torch.cat(cls_tokens, dim=1))
            embedding = self.feature_activation(self.feature_projection(features))
            global_logit = self.classifier(self.feature_dropout(embedding)).squeeze(1)

            patch_tokens = outputs[-1][1:].permute(1, 0, 2)
            patch_embeddings = self.patch_activation(
                self.patch_projection(self.patch_norm(patch_tokens))
            )
            patch_scores = self.patch_classifier(
                self.patch_dropout(patch_embeddings)
            ).squeeze(2)
            top_count = max(1, round(patch_scores.shape[1] * self.patch_top_fraction))
            patch_mean = patch_scores.mean(dim=1)
            patch_top = patch_scores.topk(top_count, dim=1).values.mean(dim=1)
            patch_bottom = patch_scores.topk(top_count, dim=1, largest=False).values.mean(dim=1)
            patch_std = patch_scores.std(dim=1, unbiased=False)
            # Exposed for an optional auxiliary image-level objective. Training the
            # patch detector independently prevents a strong global initialization
            # from starving the zero-weight local fusion branch of gradients.
            self.auxiliary_logits = patch_mean
            evidence = torch.stack(
                [global_logit, patch_mean, patch_top, patch_bottom, patch_std], dim=1
            )
            logits = self.fusion_classifier(evidence).squeeze(1)
        return logits, embedding

    def head_parameters(self) -> list[nn.Parameter]:
        modules = [
            self.feature_norm,
            self.feature_projection,
            self.classifier,
            self.patch_norm,
            self.patch_projection,
            self.patch_classifier,
            self.fusion_classifier,
        ]
        return [parameter for module in modules for parameter in module.parameters()]


class CLIPAdapterMultiDepthPatchMILDetector(CLIPResidualAdapterDetector):
    """Residual-adapter CLIP with independent local evidence at several depths."""

    def __init__(
        self,
        backbone: str,
        feature_blocks: list[int],
        adapter_last_blocks: int,
        adapter_dim: int,
        adapter_dropout: float,
        hidden_dim: int,
        dropout: float,
        patch_hidden_dim: int,
        patch_top_fraction: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__(
            backbone,
            feature_blocks,
            adapter_last_blocks,
            adapter_dim,
            adapter_dropout,
            hidden_dim,
            dropout,
            device=device,
            download_root=download_root,
        )
        if not 0.0 < patch_top_fraction <= 0.5:
            raise ValueError("patch_top_fraction must be within (0, 0.5]")
        width = int(self.visual.ln_post.normalized_shape[0])
        depth_count = len(feature_blocks)
        self.patch_top_fraction = patch_top_fraction
        self.patch_norms = nn.ModuleList(
            [nn.LayerNorm(width) for _ in range(depth_count)]
        )
        self.patch_projections = nn.ModuleList(
            [nn.Linear(width, patch_hidden_dim) for _ in range(depth_count)]
        )
        self.patch_classifiers = nn.ModuleList(
            [nn.Linear(patch_hidden_dim, 1) for _ in range(depth_count)]
        )
        self.patch_activation = nn.GELU()
        self.patch_dropout = nn.Dropout(dropout)
        # Four local statistics per selected depth plus the inherited global logit.
        self.fusion_classifier = nn.Linear(1 + 4 * depth_count, 1)
        nn.init.zeros_(self.fusion_classifier.weight)
        nn.init.zeros_(self.fusion_classifier.bias)
        with torch.no_grad():
            self.fusion_classifier.weight[0, 0] = 1.0

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.visual(images)
        if any(hook.output is None for hook in self.feature_hooks):
            raise RuntimeError("a selected CLIP block activation was not captured")
        with torch.autocast(device_type=images.device.type, enabled=False):
            outputs = [
                hook.output.float()
                for hook in self.feature_hooks
                if hook.output is not None
            ]
            cls_tokens = [output[0] for output in outputs]
            features = self.feature_norm(torch.cat(cls_tokens, dim=1))
            embedding = self.feature_activation(self.feature_projection(features))
            global_logit = self.classifier(self.feature_dropout(embedding)).squeeze(1)

            local_evidence: list[torch.Tensor] = []
            patch_means: list[torch.Tensor] = []
            for output, norm, projection, classifier in zip(
                outputs,
                self.patch_norms,
                self.patch_projections,
                self.patch_classifiers,
                strict=True,
            ):
                patch_tokens = output[1:].permute(1, 0, 2)
                patch_embeddings = self.patch_activation(
                    projection(norm(patch_tokens))
                )
                patch_scores = classifier(
                    self.patch_dropout(patch_embeddings)
                ).squeeze(2)
                top_count = max(
                    1, round(patch_scores.shape[1] * self.patch_top_fraction)
                )
                patch_mean = patch_scores.mean(dim=1)
                patch_top = patch_scores.topk(top_count, dim=1).values.mean(dim=1)
                patch_bottom = patch_scores.topk(
                    top_count, dim=1, largest=False
                ).values.mean(dim=1)
                patch_std = patch_scores.std(dim=1, unbiased=False)
                patch_means.append(patch_mean)
                local_evidence.extend(
                    [patch_mean, patch_top, patch_bottom, patch_std]
                )
            self.auxiliary_logits = torch.stack(patch_means, dim=1).mean(dim=1)
            evidence = torch.stack([global_logit, *local_evidence], dim=1)
            logits = self.fusion_classifier(evidence).squeeze(1)
        return logits, embedding

    def head_parameters(self) -> list[nn.Parameter]:
        modules = [
            self.feature_norm,
            self.feature_projection,
            self.classifier,
            self.patch_norms,
            self.patch_projections,
            self.patch_classifiers,
            self.fusion_classifier,
        ]
        return [parameter for module in modules for parameter in module.parameters()]


class CLIPAdapterFrequencyResidualDetector(CLIPResidualAdapterDetector):
    """Global CLIP evidence plus a compact high-pass residual CNN branch."""

    def __init__(
        self,
        backbone: str,
        feature_blocks: list[int],
        adapter_last_blocks: int,
        adapter_dim: int,
        adapter_dropout: float,
        hidden_dim: int,
        dropout: float,
        frequency_hidden_dim: int,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__(
            backbone,
            feature_blocks,
            adapter_last_blocks,
            adapter_dim,
            adapter_dropout,
            hidden_dim,
            dropout,
            device=device,
            download_root=download_root,
        )
        self.register_buffer(
            "_frequency_pixel_mean",
            torch.tensor(CLIP_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_frequency_pixel_std",
            torch.tensor(CLIP_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        kernels = torch.tensor(
            [
                [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
                [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
                [[0.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 0.0]],
            ]
        ).unsqueeze(1)
        self.register_buffer("_frequency_kernels", kernels, persistent=False)
        channels = [9, 32, 64, 128, frequency_hidden_dim]
        layers: list[nn.Module] = []
        for input_channels, output_channels in zip(channels[:-1], channels[1:], strict=True):
            layers.extend(
                [
                    nn.Conv2d(
                        input_channels,
                        output_channels,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                    nn.GroupNorm(min(8, output_channels), output_channels),
                    nn.GELU(),
                ]
            )
        layers.append(nn.AdaptiveAvgPool2d(1))
        self.frequency_encoder = nn.Sequential(*layers)
        self.frequency_classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(frequency_hidden_dim, frequency_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(frequency_hidden_dim, 1),
        )
        self.fusion_classifier = nn.Linear(2, 1)
        nn.init.zeros_(self.fusion_classifier.weight)
        nn.init.zeros_(self.fusion_classifier.bias)
        with torch.no_grad():
            self.fusion_classifier.weight[0, 0] = 1.0

    def _frequency_input(self, images: torch.Tensor) -> torch.Tensor:
        pixels = (
            images.float() * self._frequency_pixel_std + self._frequency_pixel_mean
        ).clamp(0.0, 1.0)
        residual_3 = pixels - F.avg_pool2d(pixels, 3, stride=1, padding=1)
        residual_5 = pixels - F.avg_pool2d(pixels, 5, stride=1, padding=2)
        luminance = (
            pixels
            * pixels.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        ).sum(dim=1, keepdim=True)
        edges = F.conv2d(luminance, self._frequency_kernels, padding=1)
        residuals = torch.cat([residual_3, residual_5, edges], dim=1)
        return torch.tanh(8.0 * residuals)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_logit, embedding = super().forward(images)
        with torch.autocast(device_type=images.device.type, enabled=False):
            frequency_features = self.frequency_encoder(self._frequency_input(images))
            frequency_logit = self.frequency_classifier(frequency_features).squeeze(1)
            self.auxiliary_logits = frequency_logit
            logits = self.fusion_classifier(
                torch.stack([global_logit.float(), frequency_logit], dim=1)
            ).squeeze(1)
        return logits, embedding

    def head_parameters(self) -> list[nn.Parameter]:
        modules = [
            self.feature_norm,
            self.feature_projection,
            self.classifier,
            self.frequency_encoder,
            self.frequency_classifier,
            self.fusion_classifier,
        ]
        return [parameter for module in modules for parameter in module.parameters()]


class CLIPAdapterDegradationResponseDetector(CLIPResidualAdapterDetector):
    """Global detector plus a learned response to fixed resampling loss.

    Zero-initialized fusion exactly preserves the original-view detector until
    the response branch has learned useful evidence.
    """

    def __init__(
        self,
        backbone: str,
        feature_blocks: list[int],
        adapter_last_blocks: int,
        adapter_dim: int,
        adapter_dropout: float,
        hidden_dim: int,
        dropout: float,
        response_hidden_dim: int,
        degradation_scale: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__(
            backbone,
            feature_blocks,
            adapter_last_blocks,
            adapter_dim,
            adapter_dropout,
            hidden_dim,
            dropout,
            device=device,
            download_root=download_root,
        )
        if not 0.25 <= degradation_scale < 1.0:
            raise ValueError("degradation_scale must be in [0.25, 1.0)")
        self.degradation_scale = degradation_scale
        self.register_buffer(
            "_degradation_pixel_mean",
            torch.tensor(CLIP_MEAN).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_degradation_pixel_std",
            torch.tensor(CLIP_STD).view(1, 3, 1, 1),
            persistent=False,
        )
        response_dim = hidden_dim * 3
        self.response_norm = nn.LayerNorm(response_dim)
        self.response_head = nn.Sequential(
            nn.Linear(response_dim, response_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(response_hidden_dim, 1),
        )
        self.fusion_classifier = nn.Linear(3, 1)
        nn.init.zeros_(self.fusion_classifier.weight)
        nn.init.zeros_(self.fusion_classifier.bias)
        with torch.no_grad():
            self.fusion_classifier.weight[0, 0] = 1.0

    def _degrade(self, images: torch.Tensor) -> torch.Tensor:
        pixels = (
            images.float() * self._degradation_pixel_std + self._degradation_pixel_mean
        ).clamp(0.0, 1.0)
        height, width = pixels.shape[-2:]
        reduced = F.interpolate(
            pixels,
            size=(
                max(1, round(height * self.degradation_scale)),
                max(1, round(width * self.degradation_scale)),
            ),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        restored = F.interpolate(
            reduced,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        quantized = torch.round(restored * 255.0) / 255.0
        restored = restored + (quantized - restored).detach()
        return (restored - self._degradation_pixel_mean) / self._degradation_pixel_std

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        global_logit, embedding = super().forward(images)
        degraded_logit, degraded_embedding = super().forward(self._degrade(images))
        with torch.autocast(device_type=images.device.type, enabled=False):
            original = embedding.float()
            degraded = degraded_embedding.float()
            delta = original - degraded
            response_features = self.response_norm(
                torch.cat([delta, delta.abs(), original * degraded], dim=1)
            )
            response_logit = self.response_head(response_features).squeeze(1)
            self.auxiliary_logits = response_logit
            logits = self.fusion_classifier(
                torch.stack(
                    [global_logit.float(), degraded_logit.float(), response_logit],
                    dim=1,
                )
            ).squeeze(1)
        return logits, embedding

    def head_parameters(self) -> list[nn.Parameter]:
        return (
            super().head_parameters()
            + list(self.response_norm.parameters())
            + list(self.response_head.parameters())
            + list(self.fusion_classifier.parameters())
        )


class CLIPAttentionLoRADetector(nn.Module):
    """CLIP multi-depth detector with low-rank late-attention adaptation."""

    def __init__(
        self,
        backbone: str,
        feature_blocks: list[int],
        lora_last_blocks: int,
        lora_rank: int,
        lora_alpha: float,
        hidden_dim: int,
        dropout: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__()
        clip_model, _ = clip.load(
            backbone,
            device=device,
            jit=False,
            download_root=str(download_root),
        )
        self.visual = clip_model.visual.float()
        del clip_model
        blocks = self.visual.transformer.resblocks
        block_count = len(blocks)
        if not feature_blocks or min(feature_blocks) < 0 or max(feature_blocks) >= block_count:
            raise ValueError(f"feature_blocks must be within [0, {block_count - 1}]")
        if not 0 < lora_last_blocks <= block_count:
            raise ValueError("lora_last_blocks is outside the visual Transformer depth")

        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        for block in blocks[-lora_last_blocks:]:
            attention = block.attn
            in_projection = attention.in_proj_weight
            in_projection_lora = _LoRAWeightUpdate(
                int(in_projection.shape[0]),
                int(in_projection.shape[1]),
                lora_rank,
                lora_alpha,
            ).to(device=in_projection.device, dtype=in_projection.dtype)
            parametrize.register_parametrization(
                attention,
                "in_proj_weight",
                in_projection_lora,
            )
            output_projection = attention.out_proj.weight
            output_projection_lora = _LoRAWeightUpdate(
                int(output_projection.shape[0]),
                int(output_projection.shape[1]),
                lora_rank,
                lora_alpha,
            ).to(device=output_projection.device, dtype=output_projection.dtype)
            parametrize.register_parametrization(
                attention.out_proj,
                "weight",
                output_projection_lora,
            )

        self.feature_blocks = list(feature_blocks)
        self.feature_hooks = [_ActivationHook(blocks[index]) for index in self.feature_blocks]
        width = int(self.visual.ln_post.normalized_shape[0])
        feature_dim = width * len(self.feature_blocks)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.feature_activation = nn.GELU()
        self.feature_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def train(self, mode: bool = True) -> CLIPAttentionLoRADetector:
        super().train(mode)
        self.visual.eval()
        self.feature_norm.train(mode)
        self.feature_projection.train(mode)
        self.feature_dropout.train(mode)
        self.classifier.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.visual(images)
        if any(hook.output is None for hook in self.feature_hooks):
            raise RuntimeError("a selected CLIP block activation was not captured")
        with torch.autocast(device_type=images.device.type, enabled=False):
            cls_tokens = [
                hook.output[0].float()
                for hook in self.feature_hooks
                if hook.output is not None
            ]
            features = self.feature_norm(torch.cat(cls_tokens, dim=1))
            embedding = self.feature_activation(self.feature_projection(features))
            logits = self.classifier(self.feature_dropout(embedding)).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.visual.parameters() if parameter.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        modules = [self.feature_norm, self.feature_projection, self.classifier]
        return [parameter for module in modules for parameter in module.parameters()]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class CLIPMixtureOfLayersDetector(nn.Module):
    """MoLD-style adaptive fusion over CLIP transformer depths.

    The detector follows the data-dependent gating formulation in Park et al.
    (arXiv:2512.04969): CLS features from every selected depth are projected to
    a common space and a gate conditioned on the top-layer CLS feature assigns
    per-image layer weights.  The projection is shared across layers, matching
    the authors' released implementation and keeping the head appropriately
    small for the competition's 1,000 labelled development images.
    """

    def __init__(
        self,
        backbone: str,
        feature_blocks: list[int],
        unfreeze_last_blocks: int,
        hidden_dim: int,
        gate_hidden_dim: int,
        dropout: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__()
        clip_model, _ = clip.load(
            backbone,
            device=device,
            jit=False,
            download_root=str(download_root),
        )
        self.visual = clip_model.visual.float()
        del clip_model
        blocks = self.visual.transformer.resblocks
        block_count = len(blocks)
        if (
            not feature_blocks
            or len(set(feature_blocks)) != len(feature_blocks)
            or min(feature_blocks) < 0
            or max(feature_blocks) >= block_count
        ):
            raise ValueError(f"feature_blocks must be unique and within [0, {block_count - 1}]")
        if feature_blocks != sorted(feature_blocks):
            raise ValueError("feature_blocks must be ordered from shallow to deep")
        if not 0 <= unfreeze_last_blocks <= block_count:
            raise ValueError("unfreeze_last_blocks is outside the visual Transformer depth")

        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        if unfreeze_last_blocks:
            for block in blocks[-unfreeze_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad = True

        self.feature_blocks = list(feature_blocks)
        self.feature_hooks = [_ActivationHook(blocks[index]) for index in self.feature_blocks]
        width = int(self.visual.ln_post.normalized_shape[0])
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(width) for _ in self.feature_blocks]
        )
        self.shared_projection = nn.Linear(width, hidden_dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(width),
            nn.Linear(width, gate_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden_dim, len(self.feature_blocks)),
        )
        self.fused_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)
        self.last_gate_weights: torch.Tensor | None = None

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.visual(images)
        if any(hook.output is None for hook in self.feature_hooks):
            raise RuntimeError("a selected CLIP block activation was not captured")
        with torch.autocast(device_type=images.device.type, enabled=False):
            cls_tokens = [
                hook.output[0].float()
                for hook in self.feature_hooks
                if hook.output is not None
            ]
            projected = torch.stack(
                [
                    F.gelu(self.shared_projection(norm(token)))
                    for norm, token in zip(self.layer_norms, cls_tokens, strict=True)
                ],
                dim=1,
            )
            gate_weights = torch.softmax(self.gate(cls_tokens[-1]), dim=1)
            embedding = self.fused_norm(
                (gate_weights.unsqueeze(2) * projected).sum(dim=1)
            )
            logits = self.classifier(self.dropout(embedding)).squeeze(1)
            self.last_gate_weights = gate_weights.detach()
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.visual.parameters() if parameter.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        modules: list[nn.Module] = [
            self.layer_norms,
            self.shared_projection,
            self.gate,
            self.fused_norm,
            self.classifier,
        ]
        return [parameter for module in modules for parameter in module.parameters()]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }

    def load_distilled_head_checkpoint(self, path: Path) -> dict[str, object]:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != "aigc-clip-mold-distilled-v1":
            raise TypeError(f"invalid CLIP MoLD distillation checkpoint: {path}")
        if checkpoint.get("backbone") != self.visual.input_resolution:
            # OpenAI CLIP variants can share width while using incompatible
            # positional embeddings, so the distillation artifact records resolution.
            raise ValueError(
                f"distilled CLIP input resolution mismatch: {checkpoint.get('backbone')} "
                f"!= {self.visual.input_resolution}"
            )
        if checkpoint.get("feature_blocks") != self.feature_blocks:
            raise ValueError("distilled CLIP feature block selection mismatch")
        state = checkpoint.get("detector_head")
        if not isinstance(state, dict):
            raise TypeError("distillation checkpoint has no detector_head state")
        result = self.load_state_dict(state, strict=False)
        unexpected = list(result.unexpected_keys)
        missing_head = [name for name in result.missing_keys if not name.startswith("visual.")]
        if unexpected or missing_head:
            raise RuntimeError(
                f"distilled head mismatch: unexpected={unexpected}, missing_head={missing_head}"
            )
        return {
            "format": checkpoint["format"],
            "external_rows": int(checkpoint["external_rows"]),
            "competition_rows_accessed": int(checkpoint["competition_rows_accessed"]),
            "teachers": list(checkpoint["teachers"]),
        }


class CLIPOfficialMoLDDetector(nn.Module):
    """Released MoLD head applied to every frozen CLIP transformer layer.

    This intentionally follows the authors' repository rather than the
    data-dependent gating equation in the paper: a shared per-layer MLP emits
    one score for each CLS token, then a learned linear layer combines all
    layer scores into the binary logit.
    """

    def __init__(
        self,
        backbone: str,
        hidden_dim: int,
        init_gain: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__()
        clip_model, _ = clip.load(
            backbone,
            device=device,
            jit=False,
            download_root=str(download_root),
        )
        self.visual = clip_model.visual.float()
        del clip_model
        for parameter in self.visual.parameters():
            parameter.requires_grad = False

        blocks = self.visual.transformer.resblocks
        self.feature_hooks = [_ActivationHook(block) for block in blocks]
        width = int(self.visual.ln_post.normalized_shape[0])
        self.fc1 = nn.Linear(width, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.fc3 = nn.Linear(len(blocks), 1)
        nn.init.normal_(self.fc1.weight, 0.0, init_gain)
        nn.init.normal_(self.fc2.weight, 0.0, init_gain)
        nn.init.normal_(self.fc3.weight, 0.0, init_gain)

    def train(self, mode: bool = True) -> CLIPOfficialMoLDDetector:
        super().train(mode)
        self.visual.eval()
        self.fc1.train(mode)
        self.fc2.train(mode)
        self.fc3.train(mode)
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.visual(images)
        if any(hook.output is None for hook in self.feature_hooks):
            raise RuntimeError("a CLIP layer activation was not captured")
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = torch.stack(
                [
                    hook.output[0].float()
                    for hook in self.feature_hooks
                    if hook.output is not None
                ],
                dim=1,
            )
            layer_scores = self.fc2(F.gelu(self.fc1(features))).squeeze(2)
            logits = self.fc3(layer_scores).squeeze(1)
        return logits, layer_scores

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return []

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.fc1.parameters()) + list(self.fc2.parameters()) + list(self.fc3.parameters())

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if not name.startswith("visual.")
        }


class CLIPLocalForensicDetector(nn.Module):
    """CLIP detector combining multi-depth semantics with local token statistics.

    The same visual encoder handles global and native-resolution crops.  Patch-token
    moments expose spatially local artifacts without adding a second backbone.
    """

    def __init__(
        self,
        backbone: str,
        feature_blocks: list[int],
        unfreeze_last_blocks: int,
        hidden_dim: int,
        dropout: float,
        *,
        device: torch.device,
        download_root: Path,
    ) -> None:
        super().__init__()
        clip_model, _ = clip.load(
            backbone,
            device=device,
            jit=False,
            download_root=str(download_root),
        )
        self.visual = clip_model.visual.float()
        del clip_model
        blocks = self.visual.transformer.resblocks
        block_count = len(blocks)
        if not feature_blocks or min(feature_blocks) < 0 or max(feature_blocks) >= block_count:
            raise ValueError(f"feature_blocks must be within [0, {block_count - 1}]")
        if not 0 < unfreeze_last_blocks <= block_count:
            raise ValueError("unfreeze_last_blocks is outside the visual Transformer depth")

        for parameter in self.visual.parameters():
            parameter.requires_grad = False
        for block in blocks[-unfreeze_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

        self.feature_blocks = list(feature_blocks)
        self.feature_hooks = [_ActivationHook(blocks[index]) for index in self.feature_blocks]
        width = int(self.visual.ln_post.normalized_shape[0])
        # One CLS vector per selected depth plus mean/std of final-layer patch tokens.
        feature_dim = width * (len(self.feature_blocks) + 2)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_projection = nn.Linear(feature_dim, hidden_dim)
        self.feature_activation = nn.GELU()
        self.feature_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.visual(images)
        if any(hook.output is None for hook in self.feature_hooks):
            raise RuntimeError("a selected CLIP block activation was not captured")
        with torch.autocast(device_type=images.device.type, enabled=False):
            outputs = [
                hook.output.float()
                for hook in self.feature_hooks
                if hook.output is not None
            ]
            cls_tokens = [output[0] for output in outputs]
            patch_tokens = outputs[-1][1:].permute(1, 0, 2)
            patch_mean = patch_tokens.mean(dim=1)
            patch_std = patch_tokens.std(dim=1, unbiased=False)
            features = self.feature_norm(
                torch.cat([*cls_tokens, patch_mean, patch_std], dim=1)
            )
            embedding = self.feature_activation(self.feature_projection(features))
            logits = self.classifier(self.feature_dropout(embedding)).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.visual.parameters() if parameter.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        modules = [self.feature_norm, self.feature_projection, self.classifier]
        return [parameter for module in modules for parameter in module.parameters()]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class SAFEDetector(nn.Module):
    """Official SAFE high-frequency detector (KDD 2025).

    SAFE applies a one-level bior1.3 wavelet transform, retains its diagonal
    high-frequency band, and classifies it with the first two stages of a
    ResNet-50.  The compact architecture and checkpoint layout follow the
    authors' Apache-2.0 implementation.
    """

    def __init__(self, checkpoint: Path, *, unfreeze_layer2: bool) -> None:
        super().__init__()
        from pytorch_wavelets import DWTForward

        base = torchvision_resnet50(weights=None)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 2)
        self.dwt = DWTForward(J=1, mode="symmetric", wave="bior1.3")
        del base

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise TypeError(f"invalid SAFE checkpoint: {checkpoint}")
        result = self.load_state_dict(payload["model"], strict=False)
        missing = sorted(result.missing_keys)
        expected_missing = sorted(f"dwt.{name}" for name in self.dwt.state_dict())
        if missing != expected_missing or result.unexpected_keys:
            raise RuntimeError(
                f"SAFE checkpoint mismatch: missing={missing}, "
                f"unexpected={list(result.unexpected_keys)}"
            )

        for parameter in self.parameters():
            parameter.requires_grad = False
        if unfreeze_layer2:
            for parameter in self.layer2.parameters():
                parameter.requires_grad = True
        for parameter in self.fc1.parameters():
            parameter.requires_grad = True
        self.unfreeze_layer2 = unfreeze_layer2

    def train(self, mode: bool = True) -> SAFEDetector:
        super().train(mode)
        if mode:
            self.bn1.eval()
            self.layer1.eval()
            if not self.unfreeze_layer2:
                self.layer2.eval()
        return self

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        _, high_bands = self.dwt(images)
        evidence = high_bands[0][:, :, 2]
        evidence = F.interpolate(
            evidence,
            size=images.shape[-2:],
            mode="bilinear",
            align_corners=False,
            antialias=True,
        )
        evidence = self.conv1(evidence)
        evidence = self.bn1(evidence)
        evidence = self.relu(evidence)
        evidence = self.maxpool(evidence)
        evidence = self.layer1(evidence)
        evidence = self.layer2(evidence)
        embedding = self.avgpool(evidence).flatten(1)
        class_logits = self.fc1(embedding)
        logits = class_logits[:, 1] - class_logits[:, 0]
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.layer2.parameters() if parameter.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.fc1.parameters())

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class DDADinoV2Detector(nn.Module):
    """Official DDA DINOv2-L/14 detector with its released full-model checkpoint.

    The official constructor downloads an additional pretrained DINOv2-L weight even
    though the released checkpoint already contains the complete backbone.  Constructing
    the same architecture without that redundant download keeps this experiment within
    the server's disk budget; the checkpoint is then loaded strictly.
    """

    def __init__(
        self,
        repository: Path,
        checkpoint: Path,
        expected_sha256: str,
    ) -> None:
        super().__init__()
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                f"DDA checkpoint SHA-256 mismatch: {actual_sha256} != {expected_sha256}"
            )

        inference_dir = repository / "Inference"
        if not (inference_dir / "models" / "dinov2_models_lora.py").is_file():
            raise FileNotFoundError(f"invalid official DDA repository: {repository}")
        sys.path.insert(0, str(inference_dir))
        try:
            from models.dinov2_models_lora import DINOv2ModelWithLoRA

            original_hub_load = torch.hub.load

            def load_architecture_only(
                repo_or_dir: str, model: str, *args: object, **kwargs: object
            ) -> nn.Module:
                del args
                kwargs["pretrained"] = False
                kwargs["trust_repo"] = True
                cached_repository = (
                    Path(torch.hub.get_dir()) / "facebookresearch_dinov2_main"
                )
                if cached_repository.is_dir():
                    kwargs.pop("trust_repo")
                    return original_hub_load(
                        str(cached_repository), model, source="local", **kwargs
                    )
                return original_hub_load(repo_or_dir, model, **kwargs)

            torch.hub.load = load_architecture_only
            try:
                detector = DINOv2ModelWithLoRA(
                    name="dinov2_vitl14",
                    lora_rank=8,
                    lora_alpha=1,
                    lora_targets=None,
                )
            finally:
                torch.hub.load = original_hub_load
        finally:
            sys.path.remove(str(inference_dir))

        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(payload, dict) or not isinstance(payload.get("model"), dict):
            raise TypeError(f"invalid official DDA checkpoint: {checkpoint}")
        result = detector.load_state_dict(payload["model"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"DDA checkpoint mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        self.detector = detector

        # Preserve the domain-trained LoRA initialization and adapt only its compact
        # low-rank updates plus the binary head on the competition training fold.
        for parameter in self.detector.parameters():
            parameter.requires_grad = False
        for name, parameter in self.detector.named_parameters():
            if ".lora." in name or name.startswith("base_model.fc."):
                parameter.requires_grad = True

        self.checkpoint_audit = {
            "source": "roy-ch/Dual-Data-Alignment official NeurIPS 2025 checkpoint",
            "checkpoint": str(checkpoint),
            "sha256": actual_sha256,
            "backbone": "DINOv2-L/14",
            "method": "full-depth rank-8 DDA LoRA plus binary head fine-tuning",
            "safe_load": "torch.load(weights_only=True), strict state dict",
        }

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding, logits = self.detector(images, return_feature=True)
        return logits.squeeze(1), embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.detector.named_parameters()
            if ".lora." in name and parameter.requires_grad
        ]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.detector.base_model.fc.parameters())

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class GAPLDetector(nn.Module):
    """Generator-Aware Prototype Learning detector (CVPR 2026)."""

    def __init__(
        self,
        config_path: Path,
        checkpoint_path: Path,
        *,
        unfreeze_lora: bool,
    ) -> None:
        super().__init__()
        from peft import LoraConfig, get_peft_model
        from transformers import CLIPConfig, CLIPModel

        clip_config = CLIPConfig.from_json_file(str(config_path))
        clip_model = CLIPModel(clip_config)
        peft_config = LoraConfig(
            task_type="FEATURE_EXTRACTION",
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj"],
        )
        peft_model = get_peft_model(clip_model, peft_config)
        # This mirrors the official implementation: retain the LoRA-injected inner
        # vision transformer and discard the unused text tower.
        self.feature_extractor = peft_model.vision_model
        del peft_model, clip_model
        self.foren_proj = nn.Linear(1024, 128, bias=False)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=128, num_heads=4, batch_first=True
        )
        self.fc = nn.Linear(128, 1, bias=False)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise TypeError(f"invalid GAPL checkpoint structure: {checkpoint_path}")
        result = self.load_state_dict(checkpoint["model"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"GAPL checkpoint mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        prototype = checkpoint.get("prototype")
        if not isinstance(prototype, torch.Tensor) or tuple(prototype.shape) != (64, 128):
            raise ValueError("GAPL checkpoint prototype must have shape [64, 128]")
        self.register_buffer("prototype", prototype.detach().float(), persistent=True)

        for parameter in self.parameters():
            parameter.requires_grad = False
        if unfreeze_lora:
            for name, parameter in self.feature_extractor.named_parameters():
                if "lora_" in name:
                    parameter.requires_grad = True
        for module in (self.foren_proj, self.cross_attention, self.fc):
            for parameter in module.parameters():
                parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.feature_extractor(images)
        pooled = output.pooler_output
        with torch.autocast(device_type=images.device.type, enabled=False):
            embedding = F.normalize(self.foren_proj(pooled.float()), dim=1)
            prototypes = self.prototype.unsqueeze(0).expand(len(images), -1, -1)
            attended, _ = self.cross_attention(
                query=embedding.unsqueeze(1), key=prototypes, value=prototypes
            )
            attended = attended.squeeze(1)
            logits = self.fc(attended).squeeze(1)
        return logits, attended

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for parameter in self.feature_extractor.parameters()
            if parameter.requires_grad
        ]

    def head_parameters(self) -> list[nn.Parameter]:
        modules = [self.foren_proj, self.cross_attention, self.fc]
        return [parameter for module in modules for parameter in module.parameters()]

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names or name == "prototype"
        }


class NPRDetector(nn.Module):
    """Neighboring Pixel Relationships detector (CVPR 2024)."""

    def __init__(self, checkpoint_path: Path, *, unfreeze_layer2: bool) -> None:
        super().__init__()
        from torchvision.models import resnet50

        backbone = resnet50(weights=None)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)
        del backbone

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
            raise TypeError(f"invalid NPR checkpoint structure: {checkpoint_path}")
        state = {
            name.removeprefix("module."): value
            for name, value in checkpoint["model"].items()
        }
        result = self.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"NPR checkpoint mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )

        for parameter in self.parameters():
            parameter.requires_grad = False
        if unfreeze_layer2:
            for parameter in self.layer2.parameters():
                parameter.requires_grad = True
        for parameter in self.fc1.parameters():
            parameter.requires_grad = True

    @staticmethod
    def neighboring_pixel_residual(images: torch.Tensor) -> torch.Tensor:
        down = F.interpolate(
            images, scale_factor=0.5, mode="nearest", recompute_scale_factor=True
        )
        reconstructed = F.interpolate(
            down, scale_factor=2.0, mode="nearest", recompute_scale_factor=True
        )
        return images - reconstructed

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        images = images[:, :, : images.shape[2] // 2 * 2, : images.shape[3] // 2 * 2]
        x = self.conv1(self.neighboring_pixel_residual(images) * (2.0 / 3.0))
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        embedding = self.avgpool(x).flatten(1)
        logits = self.fc1(embedding).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        modules = [self.conv1, self.bn1, self.layer1, self.layer2]
        return [
            parameter
            for module in modules
            for parameter in module.parameters()
            if parameter.requires_grad
        ]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.fc1.parameters())

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        state = self.state_dict()
        # Persist BN buffers from the adapted layer together with trainable weights.
        return {
            name: value.detach().cpu()
            for name, value in state.items()
            if name in trainable_names or name.startswith("layer2.")
        }


class CommunityForensicsDetector(nn.Module):
    """CVPR 2025 Community Forensics ViT-S detector with partial fine-tuning."""

    def __init__(
        self,
        backbone: str,
        checkpoint: Path,
        unfreeze_last_blocks: int,
    ) -> None:
        super().__init__()
        self.vit = timm.create_model(backbone, pretrained=False)
        in_features = int(self.vit.head.in_features)
        self.vit.head = nn.Linear(in_features, 1)
        state = load_safetensors(str(checkpoint), device="cpu")
        result = self.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"Community Forensics checkpoint mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )

        blocks = self.vit.blocks
        if not 0 < unfreeze_last_blocks <= len(blocks):
            raise ValueError("unfreeze_last_blocks is outside the ViT depth")
        for parameter in self.vit.parameters():
            parameter.requires_grad = False
        for block in blocks[-unfreeze_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        for parameter in self.vit.head.parameters():
            parameter.requires_grad = True

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.vit.forward_features(images)
        embedding = self.vit.forward_head(tokens, pre_logits=True)
        logits = self.vit.head(embedding).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [
            parameter
            for name, parameter in self.vit.named_parameters()
            if parameter.requires_grad and not name.startswith("head.")
        ]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.vit.head.parameters())

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class TimmForensicDetector(nn.Module):
    """Partially fine-tuned timm visual encoder with a binary forensic head."""

    def __init__(
        self,
        backbone: str,
        checkpoint: Path,
        unfreeze_last_blocks: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        state = load_safetensors(str(checkpoint), device="cpu")
        result = self.backbone.load_state_dict(state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"timm checkpoint mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
        blocks = self.backbone.blocks
        if not 0 < unfreeze_last_blocks <= len(blocks):
            raise ValueError("unfreeze_last_blocks is outside the visual Transformer depth")
        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        for block in blocks[-unfreeze_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        # Keep the pretrained final normalization/pooling trainable with the adapted blocks.
        for module_name in ("norm", "fc_norm", "attn_pool"):
            module = getattr(self.backbone, module_name, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    parameter.requires_grad = True
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(int(self.backbone.num_features), 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone.forward_features(images)
        embedding = self.backbone.forward_head(tokens, pre_logits=True)
        with torch.autocast(device_type=images.device.type, enabled=False):
            embedding = embedding.float()
            logits = self.head(self.dropout(embedding)).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class DistilledTimmForensicDetector(nn.Module):
    """A heterogeneous timm student initialized by external multi-teacher distillation."""

    def __init__(
        self,
        backbone: str,
        distilled_checkpoint: Path,
        unfreeze_last_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        checkpoint = torch.load(distilled_checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != "aigc-distilled-timm-v1":
            raise TypeError(f"invalid distilled checkpoint: {distilled_checkpoint}")
        if checkpoint.get("backbone_name") != backbone:
            raise ValueError(
                f"distilled backbone mismatch: {checkpoint.get('backbone_name')} != {backbone}"
            )
        backbone_state = checkpoint.get("backbone")
        if not isinstance(backbone_state, dict):
            raise TypeError("distilled checkpoint has no backbone state")
        result = self.backbone.load_state_dict(backbone_state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"distilled backbone mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        units: nn.Module | None = None
        for name in ("stages", "layers", "blocks"):
            candidate = getattr(self.backbone, name, None)
            if isinstance(candidate, nn.Module) and hasattr(candidate, "__len__"):
                units = candidate
                break
        if units is None or not 0 < unfreeze_last_units <= len(units):  # type: ignore[arg-type]
            raise ValueError("unfreeze_last_units is outside the backbone depth")
        for unit in list(units)[-unfreeze_last_units:]:  # type: ignore[arg-type]
            for parameter in unit.parameters():
                parameter.requires_grad = True
        for module_name in ("norm", "fc_norm", "head"):
            module = getattr(self.backbone, module_name, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    parameter.requires_grad = True

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(int(self.backbone.num_features), 1)
        teacher_heads = checkpoint.get("logit_heads")
        if not isinstance(teacher_heads, dict) or not teacher_heads:
            raise TypeError("distilled checkpoint has no teacher logit heads")
        head_states = list(teacher_heads.values())
        with torch.no_grad():
            self.head.weight.copy_(
                torch.stack([state["weight"] for state in head_states]).mean(dim=0)
            )
            self.head.bias.copy_(
                torch.stack([state["bias"] for state in head_states]).mean(dim=0)
            )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone.forward_features(images)
        embedding = self.backbone.forward_head(tokens, pre_logits=True)
        with torch.autocast(device_type=images.device.type, enabled=False):
            embedding = embedding.float()
            logits = self.head(self.dropout(embedding)).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class ExternalTimmForensicDetector(nn.Module):
    """A timm detector initialized by external real/fake supervised training."""

    def __init__(
        self,
        backbone: str,
        external_checkpoint: Path,
        unfreeze_last_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        checkpoint = torch.load(external_checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(checkpoint, dict) or checkpoint.get("format") != (
            "aigc-timm-external-supervised-v1"
        ):
            raise TypeError(f"invalid external timm checkpoint: {external_checkpoint}")
        if checkpoint.get("backbone_name") != backbone:
            raise ValueError(
                f"external backbone mismatch: {checkpoint.get('backbone_name')} != {backbone}"
            )
        if int(checkpoint.get("competition_rows_accessed", -1)) != 0:
            raise RuntimeError("external timm checkpoint accessed competition rows")

        self.backbone = timm.create_model(backbone, pretrained=False, num_classes=0)
        backbone_state = checkpoint.get("backbone")
        if not isinstance(backbone_state, dict):
            raise TypeError("external timm checkpoint has no backbone state")
        result = self.backbone.load_state_dict(backbone_state, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(
                f"external backbone mismatch: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        units: nn.Module | None = None
        for name in ("stages", "layers", "blocks"):
            candidate = getattr(self.backbone, name, None)
            if isinstance(candidate, nn.Module) and hasattr(candidate, "__len__"):
                units = candidate
                break
        if units is None or not 0 < unfreeze_last_units <= len(units):  # type: ignore[arg-type]
            raise ValueError("unfreeze_last_units is outside the backbone depth")
        for unit in list(units)[-unfreeze_last_units:]:  # type: ignore[arg-type]
            for parameter in unit.parameters():
                parameter.requires_grad = True
        for module_name in ("norm", "fc_norm", "head"):
            module = getattr(self.backbone, module_name, None)
            if isinstance(module, nn.Module):
                for parameter in module.parameters():
                    parameter.requires_grad = True

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(int(self.backbone.num_features), 1)
        head_state = checkpoint.get("binary_head")
        if not isinstance(head_state, dict):
            raise TypeError("external timm checkpoint has no binary head")
        head_result = self.head.load_state_dict(head_state, strict=True)
        if head_result.missing_keys or head_result.unexpected_keys:
            raise RuntimeError(
                f"external head mismatch: missing={head_result.missing_keys}, "
                f"unexpected={head_result.unexpected_keys}"
            )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = self.backbone.forward_features(images)
        embedding = self.backbone.forward_head(tokens, pre_logits=True)
        with torch.autocast(device_type=images.device.type, enabled=False):
            embedding = embedding.float()
            logits = self.head(self.dropout(embedding)).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class DRCTConvNextDetector(nn.Module):
    """Official DRCT ConvNeXt encoder adapted to a binary competition head."""

    def __init__(
        self,
        backbone: str,
        checkpoint: Path,
        checkpoint_sha256: str,
        embedding_dim: int,
        unfreeze_last_units: int,
        dropout: float,
    ) -> None:
        super().__init__()
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != checkpoint_sha256:
            raise RuntimeError(
                f"DRCT checkpoint SHA-256 mismatch: "
                f"{actual_sha256} != {checkpoint_sha256}"
            )
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if not isinstance(state, dict):
            raise TypeError(f"invalid DRCT checkpoint: {checkpoint}")
        encoder_state = {
            name.removeprefix("model."): value
            for name, value in state.items()
            if name.startswith("model.")
        }
        classifier_state = {
            name.removeprefix("fc."): value
            for name, value in state.items()
            if name.startswith("fc.")
        }
        if not encoder_state or not classifier_state:
            raise TypeError("DRCT checkpoint has no encoder or classifier state")

        self.backbone = timm.create_model(
            backbone,
            pretrained=False,
            num_classes=embedding_dim,
        )
        encoder_result = self.backbone.load_state_dict(encoder_state, strict=True)
        if encoder_result.missing_keys or encoder_result.unexpected_keys:
            raise RuntimeError(
                f"DRCT encoder mismatch: missing={encoder_result.missing_keys}, "
                f"unexpected={encoder_result.unexpected_keys}"
            )
        official_classifier = nn.Linear(embedding_dim, 2)
        classifier_result = official_classifier.load_state_dict(
            classifier_state, strict=True
        )
        if classifier_result.missing_keys or classifier_result.unexpected_keys:
            raise RuntimeError(
                f"DRCT classifier mismatch: missing={classifier_result.missing_keys}, "
                f"unexpected={classifier_result.unexpected_keys}"
            )

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False
        units = getattr(self.backbone, "stages", None)
        if not isinstance(units, nn.Module) or not hasattr(units, "__len__"):
            raise TypeError("DRCT ConvNeXt backbone has no stages")
        if not 0 < unfreeze_last_units <= len(units):  # type: ignore[arg-type]
            raise ValueError("unfreeze_last_units is outside the DRCT backbone depth")
        for unit in list(units)[-unfreeze_last_units:]:  # type: ignore[arg-type]
            for parameter in unit.parameters():
                parameter.requires_grad = True
        encoder_head = getattr(self.backbone, "head", None)
        if not isinstance(encoder_head, nn.Module):
            raise TypeError("DRCT ConvNeXt backbone has no embedding head")
        for parameter in encoder_head.parameters():
            parameter.requires_grad = True

        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(embedding_dim, 1)
        # For a two-class softmax, P(fake) = sigmoid(logit_fake - logit_real).
        # This preserves the official zero-shot ranking exactly.
        with torch.no_grad():
            self.head.weight.copy_(
                official_classifier.weight[1:2] - official_classifier.weight[0:1]
            )
            self.head.bias.copy_(
                official_classifier.bias[1:2] - official_classifier.bias[0:1]
            )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.backbone(images)
        with torch.autocast(device_type=images.device.type, enabled=False):
            embedding = embedding.float()
            logits = self.head(self.dropout(embedding)).squeeze(1)
        return logits, embedding

    def backbone_trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.backbone.parameters() if parameter.requires_grad]

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.head.parameters())

    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        trainable_names = {
            name for name, parameter in self.named_parameters() if parameter.requires_grad
        }
        return {
            name: value.detach().cpu()
            for name, value in self.state_dict().items()
            if name in trainable_names
        }


class RandomJPEG:
    def __init__(self, probability: float, quality_min: int, quality_max: int) -> None:
        self.probability = probability
        self.quality_min = quality_min
        self.quality_max = quality_max

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        buffer = BytesIO()
        image.save(
            buffer,
            format="JPEG",
            quality=random.randint(self.quality_min, self.quality_max),
        )
        buffer.seek(0)
        result = Image.open(buffer).convert("RGB")
        result.load()
        return result


class RandomGaussianBlur:
    def __init__(self, probability: float, sigma_min: float, sigma_max: float) -> None:
        self.probability = probability
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.probability:
            return image
        return image.filter(ImageFilter.GaussianBlur(random.uniform(self.sigma_min, self.sigma_max)))


class RandomPatchMask:
    def __init__(
        self, probability: float, ratio_min: float, ratio_max: float, patch_size: int
    ) -> None:
        self.probability = probability
        self.ratio_min = ratio_min
        self.ratio_max = ratio_max
        self.patch_size = patch_size

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() >= self.probability:
            return tensor
        _, height, width = tensor.shape
        cells = [
            (top, left)
            for top in range(0, height - self.patch_size + 1, self.patch_size)
            for left in range(0, width - self.patch_size + 1, self.patch_size)
        ]
        ratio = random.uniform(self.ratio_min, self.ratio_max)
        count = min(round(len(cells) * ratio), len(cells))
        if count == 0:
            return tensor
        result = tensor.clone()
        for top, left in random.sample(cells, count):
            result[:, top : top + self.patch_size, left : left + self.patch_size] = 0
        return result


class RandomGlobalOrNativeCrop:
    """Choose a semantic global crop or a pixel-preserving native crop."""

    def __init__(self, input_size: int, resize_size: int, local_probability: float) -> None:
        self.input_size = input_size
        self.resize_size = resize_size
        self.local_probability = local_probability

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() >= self.local_probability:
            image = transforms.functional.resize(
                image, self.resize_size, interpolation=InterpolationMode.BICUBIC
            )
            return transforms.RandomCrop(self.input_size)(image)
        width, height = image.size
        if min(width, height) < self.input_size:
            image = transforms.functional.resize(
                image, self.input_size, interpolation=InterpolationMode.BICUBIC
            )
        return transforms.RandomCrop(self.input_size)(image)


class GlobalLocalValidationCrops:
    """Five resized global crops plus five native-resolution crops."""

    def __init__(
        self,
        input_size: int,
        resize_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.input_size = input_size
        self.resize_size = resize_size
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        global_image = transforms.functional.resize(
            image, self.resize_size, interpolation=InterpolationMode.BICUBIC
        )
        global_crops = transforms.FiveCrop(self.input_size)(global_image)
        native_image = image
        if min(native_image.size) < self.input_size:
            native_image = transforms.functional.resize(
                native_image, self.input_size, interpolation=InterpolationMode.BICUBIC
            )
        native_crops = transforms.FiveCrop(self.input_size)(native_image)
        return torch.stack(
            [self.normalize(crop) for crop in (*global_crops, *native_crops)]
        )


class NativeValidationCrops:
    """Five deterministic crops without downsampling away forensic traces."""

    def __init__(
        self,
        input_size: int,
        mean: tuple[float, float, float],
        std: tuple[float, float, float],
    ) -> None:
        self.input_size = input_size
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if min(image.size) < self.input_size:
            image = transforms.functional.resize(
                image, self.input_size, interpolation=InterpolationMode.BICUBIC
            )
        crops = transforms.FiveCrop(self.input_size)(image)
        return torch.stack([self.normalize(crop) for crop in crops])


def build_train_transform(config: dict[str, object]) -> Callable[[Image.Image], torch.Tensor]:
    augmentation = config["augmentation"]
    data = config["data"]
    assert isinstance(augmentation, dict) and isinstance(data, dict)
    model = config["model"]
    assert isinstance(model, dict)
    normalization = str(model.get("normalization", "clip"))
    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif normalization == "siglip":
        mean, std = SIGLIP_MEAN, SIGLIP_STD
    elif normalization == "none":
        mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    else:
        mean, std = CLIP_MEAN, CLIP_STD
    view_strategy = str(data.get("view_strategy", "global"))
    if view_strategy == "pad_crop":
        spatial_transform = transforms.RandomCrop(
            int(data["input_size"]), pad_if_needed=True
        )
    elif view_strategy in {"global_local", "native"}:
        spatial_transform: Callable[[Image.Image], Image.Image] = RandomGlobalOrNativeCrop(
            int(data["input_size"]),
            int(data["resize_size"]),
            1.0 if view_strategy == "native" else float(data.get("local_probability", 0.5)),
        )
    else:
        spatial_transform = transforms.Compose(
            [
                transforms.Resize(int(data["resize_size"]), InterpolationMode.BICUBIC),
                transforms.RandomCrop(int(data["input_size"])),
            ]
        )
    pipeline: list[Callable] = [
            RandomGaussianBlur(
                float(augmentation["blur_probability"]),
                float(augmentation["blur_sigma_min"]),
                float(augmentation["blur_sigma_max"]),
            ),
            RandomJPEG(
                float(augmentation["jpeg_probability"]),
                int(augmentation["jpeg_quality_min"]),
                int(augmentation["jpeg_quality_max"]),
            ),
            spatial_transform,
            transforms.RandomHorizontalFlip(),
    ]
    rotation_degrees = float(augmentation.get("rotation_degrees", 0.0))
    if rotation_degrees > 0:
        pipeline.append(transforms.RandomRotation(rotation_degrees))
    color_jitter = float(augmentation.get("color_jitter", 0.0))
    if color_jitter > 0:
        pipeline.append(
            transforms.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
            )
        )
    pipeline.append(transforms.ToTensor())
    mask_probability = float(augmentation.get("mask_probability", 0.0))
    if mask_probability > 0:
        pipeline.append(
            RandomPatchMask(
                mask_probability,
                float(augmentation.get("mask_ratio_min", 0.0)),
                float(augmentation.get("mask_ratio_max", 0.75)),
                int(augmentation.get("mask_patch_size", 16)),
            )
        )
    pipeline.append(transforms.Normalize(mean, std))
    return transforms.Compose(pipeline)


def build_valid_transform(config: dict[str, object]) -> Callable[[Image.Image], torch.Tensor]:
    data = config["data"]
    assert isinstance(data, dict)
    input_size = int(data["input_size"])
    crops = int(data["validation_crops"])
    crop_transform: nn.Module
    if crops == 10:
        crop_transform = transforms.TenCrop(input_size)
    elif crops == 5:
        crop_transform = transforms.FiveCrop(input_size)
    elif crops == 1:
        crop_transform = transforms.CenterCrop(input_size)
    else:
        raise ValueError("validation_crops must be one of 1, 5, or 10")
    model = config["model"]
    assert isinstance(model, dict)
    normalization = str(model.get("normalization", "clip"))
    if normalization == "imagenet":
        mean, std = IMAGENET_MEAN, IMAGENET_STD
    elif normalization == "siglip":
        mean, std = SIGLIP_MEAN, SIGLIP_STD
    elif normalization == "none":
        mean, std = (0.0, 0.0, 0.0), (1.0, 1.0, 1.0)
    else:
        mean, std = CLIP_MEAN, CLIP_STD
    if str(data.get("validation_view_strategy", "global")) == "center_crop_native1":
        normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(mean, std)]
        )
        return transforms.Compose(
            [
                transforms.CenterCrop(input_size),
                normalize,
                transforms.Lambda(lambda tensor: tensor.unsqueeze(0)),
            ]
        )
    if str(data.get("validation_view_strategy", "global")) == "global_local10":
        return GlobalLocalValidationCrops(
            input_size,
            int(data["resize_size"]),
            mean,
            std,
        )
    if str(data.get("validation_view_strategy", "global")) == "native5":
        return NativeValidationCrops(input_size, mean, std)
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if crops == 1:
        return transforms.Compose(
            [
                transforms.Resize(int(data["resize_size"]), InterpolationMode.BICUBIC),
                crop_transform,
                normalize,
                transforms.Lambda(lambda tensor: tensor.unsqueeze(0)),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(data["resize_size"]), InterpolationMode.BICUBIC),
            crop_transform,
            transforms.Lambda(lambda images: torch.stack([normalize(image) for image in images])),
        ]
    )


class LabeledImageDataset(Dataset[tuple[torch.Tensor, torch.Tensor, str]]):
    def __init__(
        self,
        records: list[dict[str, object]],
        image_dir: Path,
        transform: Callable[[Image.Image], torch.Tensor],
    ) -> None:
        self.records = records
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        record = self.records[index]
        image_id = str(record["id"])
        with Image.open(self.image_dir / image_id) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, torch.tensor(float(record["label"]), dtype=torch.float32), image_id


def supervised_contrastive_loss(
    embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:
    features = F.normalize(embeddings, dim=1)
    logits = features @ features.T / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    allowed = ~identity
    log_probability = logits - torch.logsumexp(logits.masked_fill(~allowed, -torch.inf), dim=1)[:, None]
    log_probability = log_probability.masked_fill(~positive, 0.0)
    positive_count = positive.sum(dim=1).clamp_min(1)
    return -(log_probability.sum(dim=1) / positive_count).mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
