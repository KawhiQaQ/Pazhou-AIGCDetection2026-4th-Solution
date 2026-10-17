from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from PIL import Image, ImageOps
from safetensors.torch import load_file as load_safetensors
import timm
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import ViTForImageClassification

from .components import (
    CLIPResidualAdapterDetector, ClipResidual, DualPEFusion, MultiDepthPatchResidual,
    NPR, RINE, ResidualMLP, SiglipForensic,
)


from dualtrace.paths import checkpoints

ASSETS = Path(os.environ.get("V178_ASSETS", str(checkpoints() / "image")))
LAYERS = [9, 19, 29, 39, 49]
V90_WEIGHTS = {
    "V29": 0.08859375, "V36": 0.08859375, "V47": 0.08859375,
    "V54": 0.08859375, "V56": 0.08859375, "V67": 0.08859375,
    "V71": 0.08859375, "V76": 0.044296875, "V77": 0.044296875,
    "V86": 0.10125, "V60": 0.09, "V58": 0.10,
}
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _state(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def _load_raw_state(module: nn.Module, path: Path) -> None:
    state = _state(path)
    result = module.load_state_dict(state, strict=False)
    trainable_missing = [name for name, _ in module.named_parameters() if name in result.missing_keys and not name.startswith("visual.")]
    if result.unexpected_keys or trainable_missing:
        raise RuntimeError(f"checkpoint mismatch {path.name}: unexpected={result.unexpected_keys[:3]} missing={trainable_missing[:3]}")


def _load_linear(path: Path, dim: int = 1280) -> nn.Linear:
    payload = _state(path)
    model = nn.Linear(dim, 1)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def _load_residual(path: Path, dim: int = 1280) -> ResidualMLP:
    payload = _state(path)
    model = ResidualMLP(dim)
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def _load_multidepth(path: Path) -> MultiDepthPatchResidual:
    payload = _state(path)
    model = MultiDepthPatchResidual()
    model.load_state_dict(payload["state_dict"], strict=True)
    return model.eval()


def _unit(array: np.ndarray) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    return array / np.maximum(np.linalg.norm(array, axis=-1, keepdims=True), 1e-12)


def _rbf(left: np.ndarray, right: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    cosine = np.clip(left @ right.T, -1.0, 1.0)
    return np.exp(-gamma * np.maximum(2.0 - 2.0 * cosine, 0.0)).astype(np.float32)


def _view_kernel(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return sum(_rbf(left[:, i], right[:, i]) for i in range(left.shape[1])) / left.shape[1]


def _ecdf(value: float, sorted_values: np.ndarray) -> float:
    left = int(np.searchsorted(sorted_values, value, side="left"))
    right = int(np.searchsorted(sorted_values, value, side="right"))
    return float((left + right + 1) / (2.0 * len(sorted_values)))


def _five_global_native(image: Image.Image) -> torch.Tensor:
    normalize = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    global_image = transforms.functional.resize(image, 512, interpolation=InterpolationMode.BICUBIC)
    global_crops = transforms.FiveCrop(448)(global_image)
    native = image
    if min(native.size) < 448:
        native = transforms.functional.resize(native, 448, interpolation=InterpolationMode.BICUBIC)
    native_crops = transforms.FiveCrop(448)(native)
    return torch.stack([normalize(crop) for crop in (*global_crops, *native_crops)])


def _statistics(intermediates: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> torch.Tensor:
    parts = []
    for tokens in intermediates:
        tokens = tokens.float()
        parts.append(torch.cat([tokens.mean(1), tokens.var(1, correction=0).clamp_min(1e-12).sqrt()], dim=1))
    return torch.stack(parts, dim=1)


class V178Runtime:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("V178 requires an NVIDIA CUDA GPU")
        self.device = torch.device("cuda")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        started = time.time()
        self._load_models()
        self.load_seconds = time.time() - started

    def _load_pe(self, architecture: str, filename: str) -> nn.Module:
        model = timm.create_model(architecture, pretrained=False).half()
        weights = load_safetensors(str(ASSETS / "backbones" / filename), device="cpu")
        weights = {key: value.half() if value.is_floating_point() else value for key, value in weights.items()}
        result = model.load_state_dict(weights, strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError(f"PE checkpoint mismatch: {filename}")
        del weights
        return model.eval().to(self.device)

    def _load_models(self) -> None:
        b, h = ASSETS / "backbones", ASSETS / "heads"
        self.core_encoder = self._load_pe("vit_pe_core_gigantic_patch14_448.fb", "pe_core_g14.safetensors")
        self.spatial_encoder = self._load_pe("vit_pe_spatial_gigantic_patch14_448.fb", "pe_spatial_g14.safetensors")

        self.clip_v29 = CLIPResidualAdapterDetector(b / "clip", multi_patch=True)
        _load_raw_state(self.clip_v29, h / "V29.pth")
        self.clip_v36 = CLIPResidualAdapterDetector(b / "clip")
        _load_raw_state(self.clip_v36, h / "V36.pth")
        self.clip_external = CLIPResidualAdapterDetector(b / "clip")
        external = _state(h / "V36_external.pth")
        ext_state = external.get("trainable_state", external)
        result = self.clip_external.load_state_dict(ext_state, strict=False)
        if result.unexpected_keys:
            raise RuntimeError("V36 external checkpoint mismatch")
        for model in (self.clip_v29, self.clip_v36, self.clip_external):
            model.eval().to(self.device)

        self.pe_v47 = _load_linear(h / "V47_linear.pth").to(self.device)
        self.pe_v51 = _load_linear(h / "V51_linear.pth").to(self.device)
        self.pe_v54 = _load_residual(h / "V54_residual.pth").to(self.device)
        self.pe_v56 = _load_residual(h / "V56_residual.pth").to(self.device)
        self.pe_v58 = _load_residual(h / "V58_residual.pth").to(self.device)
        self.clip_v60 = ClipResidual(512).to(self.device).eval()
        self.clip_v60.load_state_dict(_state(h / "V60_residual.pth")["state_dict"], strict=True)

        self.md_core_v67 = _load_multidepth(h / "V67_multidepth.pth").to(self.device)
        self.md_core_v71 = _load_multidepth(h / "V71_multidepth.pth").to(self.device)
        self.md_spatial_v76 = _load_multidepth(h / "V76_multidepth.pth").to(self.device)
        self.md_spatial_v77 = _load_multidepth(h / "V77_multidepth.pth").to(self.device)
        core_parent, spatial_parent = _load_multidepth(h / "V71_multidepth.pth"), _load_multidepth(h / "V77_multidepth.pth")
        self.dual_v86 = DualPEFusion(core_parent, spatial_parent)
        self.dual_v86.residual.load_state_dict(_state(h / "V86_fusion.pth")["residual_state_dict"], strict=True)
        self.dual_v86.eval().to(self.device)

        self.face = ViTForImageClassification.from_pretrained(b / "deepfake_vit", local_files_only=True, use_safetensors=True).eval().to(self.device)
        self.rine = RINE(b / "clip")
        rine_result = self.rine.load_state_dict(_state(b / "rine.pth"), strict=False)
        if rine_result.unexpected_keys or any(not key.startswith("clip.") for key in rine_result.missing_keys):
            raise RuntimeError("RINE checkpoint mismatch")
        self.rine.eval().to(self.device)
        self.npr = NPR().eval().to(self.device)
        self.npr.load_state_dict(_state(b / "npr.pth"), strict=True)
        self.siglip = SiglipForensic()
        siglip_result = self.siglip.load_state_dict(_state(b / "bombek_siglip.pth"), strict=True)
        if siglip_result.missing_keys or siglip_result.unexpected_keys:
            raise RuntimeError("Bombek SigLIP checkpoint mismatch")
        self.siglip.eval().to(self.device)

        self.svc = {name: joblib.load(h / f"{name}.joblib") for name in ["V109", "V122", "V155", "V167", "V171", "RINE", "NPR"]}
        self.meta = joblib.load(h / "V178_meta.joblib")
        ref = np.load(ASSETS / "calibration" / "train_references.npz")
        self.ref = {key: ref[key] for key in ref.files}
        cal = np.load(ASSETS / "calibration" / "score_ecdfs.npz")
        self.ecdfs = {key: cal[key].astype(np.float64) for key in cal.files}
        self.face_detector = cv2.CascadeClassifier(str(ASSETS / "haarcascade_frontalface_default.xml"))
        if self.face_detector.empty():
            raise RuntimeError("cannot load Haar face detector")
        gc.collect()
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def _pe_features(self, image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        center_tf = transforms.Compose([
            transforms.Resize(448, InterpolationMode.BICUBIC), transforms.CenterCrop(448),
            transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
        gray_tf = transforms.Compose([
            transforms.Resize(512, InterpolationMode.BICUBIC), transforms.CenterCrop(448),
            transforms.Grayscale(3), transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3),
        ])
        center = center_tf(image).unsqueeze(0).to(self.device)
        gray = gray_tf(image).unsqueeze(0).to(self.device)
        views = _five_global_native(image).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            pe_color = F.normalize(self.core_encoder(center).float(), dim=1)
            pe_gray = F.normalize(self.core_encoder(gray).float(), dim=1)
            view_parts = [F.normalize(self.core_encoder(views[i:i + 1]).float(), dim=1) for i in range(10)]
            core_intermediate = self.core_encoder.forward_intermediates(center, indices=LAYERS, norm=True, output_fmt="NLC", intermediates_only=True)
            spatial_intermediate = self.spatial_encoder.forward_intermediates(center, indices=LAYERS, norm=True, output_fmt="NLC", intermediates_only=True)
        # Match the float16 feature caches used to train all downstream heads.
        quantize = lambda x: x.detach().cpu().numpy().astype(np.float16).astype(np.float32)
        return quantize(pe_color), quantize(pe_gray), quantize(torch.cat(view_parts).unsqueeze(0)), quantize(_statistics(core_intermediate)), quantize(_statistics(spatial_intermediate))

    @torch.inference_mode()
    def _clip_features(self, image: Image.Image) -> tuple[float, float, np.ndarray]:
        norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])
        resized = transforms.functional.resize(image, 384, interpolation=InterpolationMode.BICUBIC)
        crops = torch.stack([norm(crop) for crop in transforms.TenCrop(336)(resized)]).to(self.device)
        center = norm(transforms.CenterCrop(336)(resized)).unsqueeze(0).to(self.device)
        with torch.autocast("cuda", dtype=torch.float16):
            # The original V29/V36 submission builder used monotonic T=4
            # temperature scaling before emitting the member CSVs.
            v29 = torch.sigmoid(self.clip_v29(crops)[0].float().mean() / 4.0).item()
            v36 = torch.sigmoid(self.clip_v36(crops)[0].float().mean() / 4.0).item()
            clip_embedding = self.clip_external(center)[1]
        clip_feature = clip_embedding.float().cpu().numpy().astype(np.float16).astype(np.float32)
        return v29, v36, clip_feature

    def _face_crop(self, image: Image.Image) -> tuple[Image.Image, bool]:
        rgb = np.asarray(image)
        detections = self.face_detector.detectMultiScale(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), 1.1, 4, minSize=(24, 24))
        if len(detections) == 0:
            return image, False
        x, y, width, height = max(detections, key=lambda box: box[2] * box[3])
        padding = max(width, height) // 2
        return image.crop((max(0, x-padding), max(0, y-padding), min(image.width, x+width+padding), min(image.height, y+height+padding))), True

    @torch.inference_mode()
    def _forensics(self, image: Image.Image) -> tuple[np.ndarray, bool, np.ndarray, float, np.ndarray, float, np.ndarray]:
        face_image, detected = self._face_crop(image)
        face_tf = transforms.Compose([transforms.Resize((224, 224), InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        face_pixels = face_tf(face_image).unsqueeze(0).to(self.device)
        face_feature = self.face.vit(pixel_values=face_pixels).last_hidden_state[:, 0].float().cpu().numpy().astype(np.float16).astype(np.float32)

        rine_tf = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])
        rine_logit, rine_feature = self.rine(rine_tf(image).unsqueeze(0).to(self.device))
        rine_feature_np = rine_feature.float().cpu().numpy().astype(np.float32)

        npr_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
        npr_logit, npr_feature = self.npr(npr_tf(image).unsqueeze(0).to(self.device))
        npr_feature_np = npr_feature.float().cpu().numpy().astype(np.float32)

        siglip_tf = transforms.Compose([transforms.Resize((384, 384), InterpolationMode.BICUBIC), transforms.ToTensor(), transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            siglip = self.siglip(siglip_tf(image).unsqueeze(0).to(self.device))
        siglip_np = siglip.float().cpu().numpy().astype(np.float16).astype(np.float32)
        return face_feature, detected, rine_feature_np, float(rine_logit.item()), npr_feature_np, float(npr_logit.item()), siglip_np

    def _visual_scores(self, pe: np.ndarray, gray: np.ndarray, clip_feature: np.ndarray, face: np.ndarray,
                       views: np.ndarray, siglip: np.ndarray, square: bool, low: bool, high: bool) -> dict[str, float]:
        test = {"pe": _unit(pe), "gray": _unit(gray), "clip": _unit(clip_feature), "face": _unit(face),
                "views": _unit(views), "siglip": _unit(siglip)}
        pe_k = _rbf(test["pe"], self.ref["pe_color"])
        gray_k = _rbf(test["gray"], self.ref["pe_gray"])
        clip_k = _rbf(test["clip"], self.ref["clip"])
        face_k = _rbf(test["face"], self.ref["face"])
        siglip_k = _rbf(test["siglip"], self.ref["siglip"])
        global_k = _view_kernel(test["views"][:, :5], self.ref["views"][:, :5])
        native_k = _view_kernel(test["views"][:, 5:], self.ref["views"][:, 5:])
        common_color = 0.45 * pe_k + 0.45 * clip_k + 0.10 * face_k
        common_tta = 0.45 * (0.5 * pe_k + 0.5 * gray_k) + 0.45 * clip_k + 0.10 * face_k
        train_square, train_low, train_high = self.ref["square"], self.ref["low"], self.ref["high"]
        v155_conditioned = square * train_square * global_k + (not square) * (~train_square) * native_k
        v167_conditioned = square * train_square * global_k + (low * train_low + high * train_high) * native_k
        same = square * train_square + low * train_low + high * train_high
        kernels = {
            "V155": 0.90 * (0.80 * common_color + 0.20 * v155_conditioned) + 0.10 * siglip_k,
            "V167": 0.90 * (0.80 * common_tta + 0.20 * v167_conditioned) + 0.10 * siglip_k,
            "V171": 0.90 * (0.80 * same * common_tta + 0.20 * v167_conditioned) + 0.10 * siglip_k,
        }
        return {name: float(self.svc[name].decision_function(kernel.astype(np.float32))[0]) for name, kernel in kernels.items()}

    def predict(self, source: Image.Image) -> tuple[float, dict[str, float]]:
        started = time.time()
        image = ImageOps.exif_transpose(source).convert("RGB")
        width, height = image.size
        if width < 2 or height < 2 or width * height > 100_000_000:
            raise ValueError("unsupported image dimensions")
        square = abs(np.log(width / height)) < 0.12
        low = (not square) and min(width, height) < 448
        high = (not square) and not low

        pe, gray, views, core_stats, spatial_stats = self._pe_features(image)
        v29, v36, clip_feature = self._clip_features(image)
        face, face_detected, rine_feature, rine_logit, npr_feature, npr_logit, siglip = self._forensics(image)

        device_core = torch.from_numpy(core_stats).to(self.device)
        device_spatial = torch.from_numpy(spatial_stats).to(self.device)
        device_pe = torch.from_numpy(pe).to(self.device)
        device_clip = torch.from_numpy(clip_feature).to(self.device)
        with torch.inference_mode():
            v47 = torch.sigmoid(self.pe_v47(device_pe).squeeze(1)).item()
            v54_logit = self.pe_v51(device_pe).squeeze(1) + self.pe_v54(device_pe)
            v54 = torch.sigmoid(v54_logit).item()
            v56 = torch.sigmoid(self.pe_v47(device_pe).squeeze(1) + self.pe_v56(device_pe)).item()
            v58 = torch.sigmoid(self.pe_v51(device_pe).squeeze(1) + self.pe_v58(device_pe)).item()
            v67 = torch.sigmoid(self.md_core_v67(device_core)[0]).item()
            v71 = torch.sigmoid(self.md_core_v71(device_core)[0]).item()
            v76 = torch.sigmoid(self.md_spatial_v76(device_spatial)[0]).item()
            v77 = torch.sigmoid(self.md_spatial_v77(device_spatial)[0]).item()
            v86 = torch.sigmoid(self.dual_v86(device_core, device_spatial)).item()
            v60 = torch.sigmoid(v54_logit + self.clip_v60(device_clip)).item()
            core_pooled = self.md_core_v71.pooled(device_core).cpu().numpy()
            spatial_pooled = self.md_spatial_v77.pooled(device_spatial).cpu().numpy()
        members = {"V29": v29, "V36": v36, "V47": v47, "V54": v54, "V56": v56, "V58": v58,
                   "V60": v60, "V67": v67, "V71": v71, "V76": v76, "V77": v77, "V86": v86}
        member_ranks = {name: _ecdf(value, self.ecdfs[f"V90_{name}"]) for name, value in members.items()}
        v90 = sum(V90_WEIGHTS[name] * member_ranks[name] for name in members)

        v109_feature = np.concatenate([_unit(pe) * np.sqrt(0.30), _unit(clip_feature) * np.sqrt(0.30),
                                       _unit(core_pooled) * np.sqrt(0.20), _unit(spatial_pooled) * np.sqrt(0.20)], axis=1)
        v109 = float(self.svc["V109"].decision_function(v109_feature.astype(np.float32))[0])
        flat_views = _unit(views).reshape(10, -1)
        instance = self.svc["V122"].decision_function(flat_views).astype(np.float64)
        mil_mean = float(instance.mean())
        mil_route = float(instance[:5].mean() if square else instance[5:].mean())
        visual = self._visual_scores(pe, gray, clip_feature, face, views, siglip, square, low, high)
        v168 = (v90 + _ecdf(mil_mean, self.ecdfs["mil_mean"]) + _ecdf(mil_route, self.ecdfs["mil_route"])
                + _ecdf(visual["V167"], self.ecdfs["V167"])) / 4.0
        uncertainty = float(np.clip(1.0 - abs(v168 - 0.5) / 0.25, 0.0, 1.0))
        routed_raw, routed_name = (visual["V171"], "V171") if low else ((v109, "V109") if high else (visual["V155"], "V155"))
        routed = _ecdf(routed_raw, self.ecdfs[routed_name])
        score = (1.0 - 0.25 * uncertainty) * v168 + 0.25 * uncertainty * routed

        square_meta = float(self.meta.predict_proba(np.asarray([[rine_logit, npr_logit]], dtype=np.float64))[0, 1])
        square_meta_rank = _ecdf(square_meta, self.ecdfs["square_meta"])
        rine_score = float(self.svc["RINE"].decision_function(_unit(rine_feature))[0])
        npr_score = float(self.svc["NPR"].decision_function(_unit(npr_feature))[0])
        rine_rank, npr_rank = _ecdf(rine_score, self.ecdfs["RINE"]), _ecdf(npr_score, self.ecdfs["NPR"])
        if square:
            score = (1.0 - 0.10 * uncertainty) * score + 0.10 * uncertainty * square_meta_rank
        elif face_detected and low and npr_rank >= 0.80:
            score = (1.0 - 0.10 * uncertainty) * score + 0.10 * uncertainty * npr_rank
        elif face_detected and high and rine_rank >= 0.90:
            score = (1.0 - 0.10 * uncertainty) * score + 0.10 * uncertainty * rine_rank
        score = float(np.clip(score, 0.0, 1.0))
        return score, {"elapsed_seconds": time.time() - started, "V90": v90, "V168": v168,
                       "uncertainty": uncertainty, "face_detected": float(face_detected),
                       "members": members, "member_ranks": member_ranks, "mil_mean": mil_mean,
                       "mil_route": mil_route, "V109": v109, **visual,
                       "RINE": rine_score, "NPR": npr_score, "square_meta": square_meta}


def audit_assets() -> dict:
    manifest = json.loads((ASSETS / "manifest.json").read_text())
    return {"version": manifest["version"], "asset_count": len(manifest["files"]), "submission_sha256": manifest["submission_sha256"]}
