from __future__ import annotations

import math
from pathlib import Path

import clip
import timm
import torch
from torch import nn
from torch.nn import functional as F
from transformers import SiglipVisionConfig, SiglipVisionModel
from peft import LoraConfig, get_peft_model


class ActivationHook:
    def __init__(self, module: nn.Module) -> None:
        self.output: torch.Tensor | None = None
        self.handle = module.register_forward_hook(self._capture)

    def _capture(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        del module, inputs
        self.output = output


class BottleneckAdapter(nn.Module):
    def __init__(self, width: int, bottleneck_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.down = nn.Linear(width, bottleneck_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.up = nn.Linear(bottleneck_dim, width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.up(self.dropout(self.activation(self.down(self.norm(tokens)))))


class ResidualAdapterHook:
    def __init__(self, module: nn.Module, adapter: BottleneckAdapter) -> None:
        self.adapter = adapter
        self.handle = module.register_forward_hook(self._apply)

    def _apply(self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        del module, inputs
        return output + self.adapter(output)


class CLIPResidualAdapterDetector(nn.Module):
    def __init__(self, cache: Path, *, multi_patch: bool = False) -> None:
        super().__init__()
        clip_model, _ = clip.load("ViT-L/14@336px", device="cpu", jit=False, download_root=str(cache))
        self.visual = clip_model.visual.float()
        del clip_model
        blocks = self.visual.transformer.resblocks
        self.adapters = nn.ModuleList([BottleneckAdapter(1024, 64, 0.1) for _ in blocks[-8:]])
        self.adapter_hooks = [ResidualAdapterHook(block, adapter) for block, adapter in zip(blocks[-8:], self.adapters, strict=True)]
        self.feature_hooks = [ActivationHook(blocks[index]) for index in [5, 11, 17, 23]]
        self.feature_norm = nn.LayerNorm(4096)
        self.feature_projection = nn.Linear(4096, 512)
        self.feature_activation = nn.GELU()
        self.feature_dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(512, 1)
        self.multi_patch = multi_patch
        if multi_patch:
            self.patch_top_fraction = 0.1
            self.patch_norms = nn.ModuleList([nn.LayerNorm(1024) for _ in range(4)])
            self.patch_projections = nn.ModuleList([nn.Linear(1024, 128) for _ in range(4)])
            self.patch_classifiers = nn.ModuleList([nn.Linear(128, 1) for _ in range(4)])
            self.patch_activation = nn.GELU()
            self.patch_dropout = nn.Dropout(0.2)
            self.fusion_classifier = nn.Linear(17, 1)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.visual(images)
        if any(hook.output is None for hook in self.feature_hooks):
            raise RuntimeError("CLIP activation hook did not fire")
        with torch.autocast(device_type=images.device.type, enabled=False):
            outputs = [hook.output.float() for hook in self.feature_hooks if hook.output is not None]
            cls_tokens = [output[0] for output in outputs]
            features = self.feature_norm(torch.cat(cls_tokens, dim=1))
            embedding = self.feature_activation(self.feature_projection(features))
            global_logit = self.classifier(self.feature_dropout(embedding)).squeeze(1)
            if not self.multi_patch:
                return global_logit, embedding
            local: list[torch.Tensor] = []
            for output, norm, projection, classifier in zip(
                outputs, self.patch_norms, self.patch_projections, self.patch_classifiers, strict=True
            ):
                patch_tokens = output[1:].permute(1, 0, 2)
                patch_embeddings = self.patch_activation(projection(norm(patch_tokens)))
                patch_scores = classifier(self.patch_dropout(patch_embeddings)).squeeze(2)
                top_count = max(1, round(patch_scores.shape[1] * self.patch_top_fraction))
                local.extend([
                    patch_scores.mean(dim=1),
                    patch_scores.topk(top_count, dim=1).values.mean(dim=1),
                    patch_scores.topk(top_count, dim=1, largest=False).values.mean(dim=1),
                    patch_scores.std(dim=1, unbiased=False),
                ])
            return self.fusion_classifier(torch.stack([global_logit, *local], dim=1)).squeeze(1), embedding


class ResidualMLP(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(feature_dim), nn.Linear(feature_dim, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(1)


class ClipResidual(ResidualMLP):
    pass


class MultiDepthPatchResidual(nn.Module):
    def __init__(self, depths: int = 5, statistic_dim: int = 3072, projection_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(statistic_dim), nn.Linear(statistic_dim, projection_dim), nn.GELU(), nn.Dropout(dropout))
            for _ in range(depths)
        ])
        self.gate = nn.Linear(projection_dim, 1)
        self.output = nn.Sequential(nn.LayerNorm(projection_dim), nn.Linear(projection_dim, 1))

    def forward(self, statistics: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        projected = torch.stack([projection(statistics[:, i]) for i, projection in enumerate(self.projections)], dim=1)
        weights = torch.softmax(self.gate(projected).squeeze(2), dim=1)
        pooled = (weights.unsqueeze(2) * projected).sum(dim=1)
        return self.output(pooled).squeeze(1), weights

    def pooled(self, statistics: torch.Tensor) -> torch.Tensor:
        projected = torch.stack([projection(statistics[:, i]) for i, projection in enumerate(self.projections)], dim=1)
        weights = torch.softmax(self.gate(projected).squeeze(2), dim=1)
        return (weights.unsqueeze(2) * projected).sum(dim=1)


class DualPEFusion(nn.Module):
    def __init__(self, core: MultiDepthPatchResidual, spatial: MultiDepthPatchResidual) -> None:
        super().__init__()
        self.core, self.spatial = core, spatial
        self.residual = nn.Sequential(
            nn.LayerNorm(515), nn.Linear(515, 128), nn.GELU(), nn.Dropout(0.2),
            nn.LayerNorm(128), nn.Linear(128, 1),
        )

    def forward(self, core_statistics: torch.Tensor, spatial_statistics: torch.Tensor) -> torch.Tensor:
        core_pooled = self.core.pooled(core_statistics)
        spatial_pooled = self.spatial.pooled(spatial_statistics)
        core_logit = self.core.output(core_pooled).squeeze(1)
        spatial_logit = self.spatial.output(spatial_pooled).squeeze(1)
        interaction = torch.cat([
            core_pooled, spatial_pooled, torch.abs(core_pooled - spatial_pooled), core_pooled * spatial_pooled,
            core_logit[:, None], spatial_logit[:, None], (core_logit - spatial_logit)[:, None],
        ], dim=1)
        return 0.5 * (core_logit + spatial_logit) + self.residual(interaction).squeeze(1)


class RINEHook(ActivationHook):
    pass


class RINE(nn.Module):
    def __init__(self, clip_cache: Path) -> None:
        super().__init__()
        self.clip, _ = clip.load("ViT-L/14", device="cpu", download_root=str(clip_cache))
        self.hooks = [RINEHook(module) for name, module in self.clip.visual.named_modules() if "ln_2" in name]
        self.alpha = nn.Parameter(torch.randn(1, len(self.hooks), 1024))
        def projection() -> nn.Sequential:
            layers: list[nn.Module] = [nn.Dropout()]
            for _ in range(4):
                layers.extend([nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout()])
            return nn.Sequential(*layers)
        self.proj1, self.proj2 = projection(), projection()
        self.head = nn.Sequential(
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(), nn.Linear(1024, 1024),
            nn.ReLU(), nn.Dropout(), nn.Linear(1024, 1),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.clip.encode_image(images)
            intermediate = torch.stack([hook.output for hook in self.hooks if hook.output is not None], dim=2)[0]
        projected = self.proj1(intermediate.float())
        pooled = (torch.softmax(self.alpha, dim=1) * projected).sum(dim=1)
        representation = self.proj2(pooled)
        return self.head(representation).squeeze(1), representation


def conv3x3(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)


def conv1x1(cin: int, cout: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(cin, cout, 1, stride=stride, bias=False)


class NPRBottleneck(nn.Module):
    expansion = 4
    def __init__(self, cin: int, channels: int, stride: int = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1, self.bn1 = conv1x1(cin, channels), nn.BatchNorm2d(channels)
        self.conv2, self.bn2 = conv3x3(channels, channels, stride), nn.BatchNorm2d(channels)
        self.conv3, self.bn3 = conv1x1(channels, channels * 4), nn.BatchNorm2d(channels * 4)
        self.relu, self.downsample = nn.ReLU(inplace=True), downsample
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None: identity = self.downsample(x)
        return self.relu(out + identity)


class NPR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1, self.bn1 = nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64)
        self.relu, self.maxpool = nn.ReLU(inplace=True), nn.MaxPool2d(3, 2, 1)
        self.layer1 = self._make_layer(64, 3)
        self.layer2 = self._make_layer(128, 4, 2)
        self.avgpool, self.fc1 = nn.AdaptiveAvgPool2d((1, 1)), nn.Linear(512, 1)
    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        out = channels * 4
        down = None if stride == 1 and self.inplanes == out else nn.Sequential(conv1x1(self.inplanes, out, stride), nn.BatchNorm2d(out))
        layers = [NPRBottleneck(self.inplanes, channels, stride, down)]
        self.inplanes = out
        layers.extend(NPRBottleneck(self.inplanes, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)
    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if images.shape[2] % 2: images = images[:, :, :-1, :]
        if images.shape[3] % 2: images = images[:, :, :, :-1]
        low = F.interpolate(images, scale_factor=0.5, mode="nearest", recompute_scale_factor=True)
        residual = (images - F.interpolate(low, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)) * (2.0 / 3.0)
        out = self.maxpool(self.relu(self.bn1(self.conv1(residual))))
        out = self.layer2(self.layer1(out))
        features = self.avgpool(out).flatten(1)
        return self.fc1(features).squeeze(1), features


class SiglipVisionWrapper(nn.Module):
    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionModel(config)
    def forward(self, pixel_values: torch.Tensor):
        return self.vision_model(pixel_values=pixel_values)


class SiglipForensic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        config = SiglipVisionConfig(hidden_size=1152, image_size=384, intermediate_size=4304,
                                    num_attention_heads=16, num_hidden_layers=27, patch_size=14)
        wrapped = SiglipVisionWrapper(config)
        self.siglip = get_peft_model(wrapped, LoraConfig(r=32, lora_alpha=64,
            target_modules=["q_proj", "v_proj"], lora_dropout=0.1, bias="none"))
    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        return self.siglip(pixel_values=pixels).pooler_output


def l2(values: torch.Tensor) -> torch.Tensor:
    return F.normalize(values.float(), dim=1)
