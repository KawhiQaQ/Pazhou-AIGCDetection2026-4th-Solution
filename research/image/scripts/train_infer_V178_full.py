#!/usr/bin/env python3
"""Fit every target-data head in V178 on all 1,000 labels and infer 20k test images."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import rankdata
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_V109_multiscale_rbf import (  # noqa: E402
    BLOCK_WEIGHTS,
    CORE_SHA256,
    LAYERS,
    SPATIAL_SHA256,
    pooled_features,
)
from evaluate_V178_content_protocol_forensic_cascade import build_meta_model  # noqa: E402


SQUARE_LOG_ASPECT_THRESHOLD = 0.12
NATIVE_UPSAMPLE_THRESHOLD = 448
UNCERTAINTY_HALF_WIDTH = 0.25
PROTOCOL_MAX_WEIGHT = 0.25
FORENSIC_MAX_WEIGHT = 0.10
NPR_HIGH_CONFIDENCE_RANK = 0.80
RINE_HIGH_CONFIDENCE_RANK = 0.90


def event(stage: str, **values: object) -> None:
    print(json.dumps({"stage": stage, **values}), flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def load_cache(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    cached = np.load(path)
    required = {"features", "ids", "labels"}
    if not required.issubset(cached.files):
        raise RuntimeError(f"invalid feature cache: {path}")
    return {key: cached[key] for key in cached.files}


def reorder(cache: dict[str, np.ndarray], ids: np.ndarray, name: str) -> dict[str, np.ndarray]:
    cache_ids = cache["ids"].astype(str)
    index = pd.Index(cache_ids).get_indexer(ids)
    if (index < 0).any() or len(np.unique(cache_ids)) != len(cache_ids):
        raise RuntimeError(f"{name} identity mismatch")
    result = dict(cache)
    for key in ("features", "labels", "logits", "parent_scores", "face_detected"):
        if key in result and np.asarray(result[key]).ndim > 0 and len(result[key]) == len(cache_ids):
            result[key] = result[key][index]
    result["ids"] = cache_ids[index]
    return result


def unit(values: np.ndarray) -> np.ndarray:
    return normalize(values.astype(np.float32)).astype(np.float32)


def unit_views(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=2, keepdims=True), 1e-12)


def rbf_cross(left: np.ndarray, right: np.ndarray, gamma: float = 0.5) -> np.ndarray:
    cosine = np.clip(left @ right.T, -1.0, 1.0)
    return np.exp(-gamma * np.maximum(2.0 - 2.0 * cosine, 0.0)).astype(np.float32)


def view_cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    result = np.zeros((len(left), len(right)), dtype=np.float32)
    for view in range(left.shape[1]):
        result += rbf_cross(left[:, view], right[:, view])
    return result / float(left.shape[1])


def regime_masks(width: np.ndarray, height: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    aspect = width.astype(np.float64) / height.astype(np.float64)
    square = np.abs(np.log(aspect)) < SQUARE_LOG_ASPECT_THRESHOLD
    low = (~square) & (np.minimum(width, height) < NATIVE_UPSAMPLE_THRESHOLD)
    high = (~square) & ~low
    return square, low, high


def geometry(root: Path, ids: np.ndarray, output: Path) -> pd.DataFrame:
    if output.is_file():
        frame = pd.read_csv(output, dtype={"id": str}).set_index("id").loc[ids].reset_index()
        return frame
    image_dir = root / "data/competition/image/data/image_test"
    rows: list[dict[str, object]] = []
    started = time.time()
    for index, image_id in enumerate(ids, start=1):
        with Image.open(image_dir / image_id) as source:
            image = ImageOps.exif_transpose(source)
            width, height = image.size
        rows.append({"id": image_id, "width": width, "height": height})
        if index % 2000 == 0:
            event("test_geometry", rows=index, elapsed=round(time.time() - started, 1))
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame


def multiscale_features(
    root: Path,
    pe: np.ndarray,
    clip: np.ndarray,
    core_statistics: np.ndarray,
    spatial_statistics: np.ndarray,
    *,
    device: torch.device,
) -> np.ndarray:
    core = pooled_features(
        core_statistics,
        root / "outputs/V71_pe_core_multidepth_pairwise/full_e10/multidepth_classifier.pth",
        CORE_SHA256,
        device=device,
    )
    spatial = pooled_features(
        spatial_statistics,
        root / "outputs/V77_pe_spatial_multidepth_pairwise/full_e10/multidepth_classifier.pth",
        SPATIAL_SHA256,
        device=device,
    )
    result = np.concatenate(
        [
            unit(pe) * np.sqrt(BLOCK_WEIGHTS["pe_core_final"]),
            unit(clip) * np.sqrt(BLOCK_WEIGHTS["clip_v36_final"]),
            unit(core) * np.sqrt(BLOCK_WEIGHTS["pe_core_multidepth"]),
            unit(spatial) * np.sqrt(BLOCK_WEIGHTS["pe_spatial_multidepth"]),
        ],
        axis=1,
    ).astype(np.float32)
    if not np.isfinite(result).all():
        raise RuntimeError("non-finite V109 features")
    return result


def batched_decision(model: SVC, values: np.ndarray, batch_size: int) -> np.ndarray:
    parts = [
        model.decision_function(values[start : start + batch_size])
        for start in range(0, len(values), batch_size)
    ]
    return np.concatenate(parts).astype(np.float64)


@torch.inference_mode()
def rbf_svc_decision_torch(
    model: SVC,
    values: np.ndarray,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> np.ndarray:
    """Evaluate a fitted binary RBF SVC with batched GEMMs on the GPU."""
    if model.kernel != "rbf" or model.dual_coef_.shape[0] != 1:
        raise RuntimeError("GPU SVC evaluator requires a fitted binary RBF model")
    support = torch.from_numpy(model.support_vectors_.astype(np.float32)).to(device)
    coefficients = torch.from_numpy(model.dual_coef_[0].astype(np.float32)).to(device)
    support_norm = (support * support).sum(dim=1)[None]
    gamma = float(model._gamma)

    def evaluate(batch: np.ndarray) -> np.ndarray:
        inputs = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to(device)
        squared = torch.clamp(
            (inputs * inputs).sum(dim=1, keepdim=True) + support_norm - 2.0 * inputs @ support.T,
            min=0.0,
        )
        scores = torch.exp(-gamma * squared) @ coefficients + float(model.intercept_[0])
        return scores.cpu().numpy().astype(np.float64)

    # Guard the manually accelerated path against coefficient/sign mistakes.
    check_n = min(32, len(values))
    accelerated_check = evaluate(values[:check_n])
    reference_check = model.decision_function(values[:check_n]).astype(np.float64)
    maximum_error = float(np.max(np.abs(accelerated_check - reference_check)))
    if maximum_error > 5e-4:
        event(
            "rbf_gpu_fallback",
            rows=check_n,
            maximum_absolute_error=maximum_error,
            reason="strict_parity_threshold_exceeded",
        )
        return batched_decision(model, values, 4096)
    event("rbf_gpu_parity", rows=check_n, maximum_absolute_error=maximum_error)
    return np.concatenate(
        [evaluate(values[start : start + batch_size]) for start in range(0, len(values), batch_size)]
    )


def train_visual_kernels(
    labels: np.ndarray,
    pe_color: np.ndarray,
    pe_gray: np.ndarray,
    clip: np.ndarray,
    face: np.ndarray,
    views: np.ndarray,
    siglip: np.ndarray,
    square: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> tuple[dict[str, SVC], dict[str, np.ndarray]]:
    pe_color_n, pe_gray_n = unit(pe_color), unit(pe_gray)
    clip_n, face_n, siglip_n = unit(clip), unit(face), unit(siglip)
    views_n = unit_views(views)
    pe_color_k = rbf_cross(pe_color_n, pe_color_n)
    pe_gray_k = rbf_cross(pe_gray_n, pe_gray_n)
    clip_k = rbf_cross(clip_n, clip_n)
    face_k = rbf_cross(face_n, face_n)
    siglip_k = rbf_cross(siglip_n, siglip_n)
    global_k = view_cross(views_n[:, :5], views_n[:, :5])
    native_k = view_cross(views_n[:, 5:], views_n[:, 5:])
    common_color = 0.45 * pe_color_k + 0.45 * clip_k + 0.10 * face_k
    common_tta = 0.45 * (0.5 * pe_color_k + 0.5 * pe_gray_k) + 0.45 * clip_k + 0.10 * face_k
    nonsquare = ~square
    v155_conditioned = np.outer(square, square) * global_k + np.outer(nonsquare, nonsquare) * native_k
    same_regime = np.outer(square, square) + np.outer(low, low) + np.outer(high, high)
    v167_conditioned = np.outer(square, square) * global_k + (np.outer(low, low) + np.outer(high, high)) * native_k
    kernels = {
        "V155": (0.90 * (0.80 * common_color + 0.20 * v155_conditioned) + 0.10 * siglip_k).astype(np.float32),
        "V167": (0.90 * (0.80 * common_tta + 0.20 * v167_conditioned) + 0.10 * siglip_k).astype(np.float32),
        "V171": (0.90 * (0.80 * same_regime * common_tta + 0.20 * v167_conditioned) + 0.10 * siglip_k).astype(np.float32),
    }
    models: dict[str, SVC] = {}
    for name, kernel in kernels.items():
        if not np.isfinite(kernel).all() or not np.allclose(kernel, kernel.T, atol=2e-6, rtol=0):
            raise RuntimeError(f"invalid {name} train kernel")
        model = SVC(C=10.0, kernel="precomputed", cache_size=4096)
        model.fit(kernel, labels)
        models[name] = model
        event("fit_visual_kernel", model=name, support_vectors=model.n_support_.tolist())
    normalized = {
        "pe_color": pe_color_n,
        "pe_gray": pe_gray_n,
        "clip": clip_n,
        "face": face_n,
        "views": views_n,
        "siglip": siglip_n,
    }
    return models, normalized


def infer_visual_kernels(
    models: dict[str, SVC],
    train: dict[str, np.ndarray],
    test: dict[str, np.ndarray],
    train_regimes: tuple[np.ndarray, np.ndarray, np.ndarray],
    test_regimes: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    train_square, train_low, train_high = train_regimes
    test_square, test_low, test_high = test_regimes
    outputs = {name: [] for name in models}
    for start in range(0, len(test["pe_color"]), batch_size):
        stop = min(start + batch_size, len(test["pe_color"]))
        sl = slice(start, stop)
        pe_color_k = rbf_cross(test["pe_color"][sl], train["pe_color"])
        pe_gray_k = rbf_cross(test["pe_gray"][sl], train["pe_gray"])
        clip_k = rbf_cross(test["clip"][sl], train["clip"])
        face_k = rbf_cross(test["face"][sl], train["face"])
        siglip_k = rbf_cross(test["siglip"][sl], train["siglip"])
        global_k = view_cross(test["views"][sl, :5], train["views"][:, :5])
        native_k = view_cross(test["views"][sl, 5:], train["views"][:, 5:])
        common_color = 0.45 * pe_color_k + 0.45 * clip_k + 0.10 * face_k
        common_tta = 0.45 * (0.5 * pe_color_k + 0.5 * pe_gray_k) + 0.45 * clip_k + 0.10 * face_k
        sq = test_square[sl]
        lo = test_low[sl]
        hi = test_high[sl]
        v155_conditioned = np.outer(sq, train_square) * global_k + np.outer(~sq, ~train_square) * native_k
        same_regime = np.outer(sq, train_square) + np.outer(lo, train_low) + np.outer(hi, train_high)
        v167_conditioned = np.outer(sq, train_square) * global_k + (np.outer(lo, train_low) + np.outer(hi, train_high)) * native_k
        kernels = {
            "V155": 0.90 * (0.80 * common_color + 0.20 * v155_conditioned) + 0.10 * siglip_k,
            "V167": 0.90 * (0.80 * common_tta + 0.20 * v167_conditioned) + 0.10 * siglip_k,
            "V171": 0.90 * (0.80 * same_regime * common_tta + 0.20 * v167_conditioned) + 0.10 * siglip_k,
        }
        for name, kernel in kernels.items():
            outputs[name].append(models[name].decision_function(kernel.astype(np.float32)))
        event("infer_visual_kernel", rows=stop)
    return {name: np.concatenate(parts).astype(np.float64) for name, parts in outputs.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/V178_full"))
    parser.add_argument("--kernel-batch-size", type=int, default=512)
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()

    train_manifest = pd.read_csv(root / "data/folds/image_folds_v1.csv", dtype={"id": str})
    train_ids = train_manifest.id.to_numpy(str)
    labels = train_manifest.label.to_numpy(np.int8)
    test_ids = pd.read_csv(root / "data/competition/image/image_test.csv", dtype={"id": str}).id.to_numpy(str)
    if len(train_ids) != 1000 or len(test_ids) != 20000 or len(np.unique(test_ids)) != len(test_ids):
        raise RuntimeError("unexpected train/test identities")

    paths = {
        "pe_train": root / "outputs/V51_pe_core_aigi_breadth_linear/features/competition_center_features.npz",
        "pe_test": root / "outputs/V51_pe_core_aigi_breadth_linear/features/test_center_features.npz",
        "gray_train": root / "outputs/V161_pe_core_grayscale_rbf/features/competition.npz",
        "gray_test": root / "outputs/V161_pe_core_grayscale_rbf/features/test.npz",
        "clip_train": root / "outputs/V60_cross_backbone_residual/features/competition_center_features.npz",
        "clip_test": root / "outputs/V60_cross_backbone_residual/features/test_center_features.npz",
        "face_train": root / "outputs/V107_face_crop_rbf/features/competition.npz",
        "face_test": root / "outputs/V107_face_crop_rbf/features/test.npz",
        "views_train": root / "outputs/V118_pe_core_multiview_rbf/features/competition.npz",
        "views_test": root / "outputs/V118_pe_core_multiview_rbf/features/test.npz",
        "bombek_train": root / "outputs/V99_bombek_dual_residual/features/competition_features.npz",
        "bombek_test": root / "outputs/V99_bombek_dual_residual/features/test_features.npz",
        "rine_train": root / "outputs/V111_rine_ldm_rbf/features/competition_center_features.npz",
        "rine_test": root / "outputs/V111_rine_ldm_rbf/features/test_center_features.npz",
        "npr_train": root / "outputs/V113_npr_rbf/features/competition_full_features.npz",
        "npr_test": root / "outputs/V113_npr_rbf/features/test_full_features.npz",
    }
    caches: dict[str, dict[str, np.ndarray]] = {}
    for name, path in paths.items():
        expected_ids = train_ids if name.endswith("train") else test_ids
        caches[name] = reorder(load_cache(path), expected_ids, name)
    for name in ("pe_train", "gray_train", "clip_train", "face_train", "views_train", "bombek_train", "rine_train", "npr_train"):
        if not np.array_equal(caches[name]["labels"].astype(np.int8), labels):
            raise RuntimeError(f"{name} label mismatch")
    event("feature_audit", train_rows=len(train_ids), test_rows=len(test_ids), caches=len(caches))

    test_geometry = geometry(root, test_ids, output / "test_geometry.csv")
    train_regimes = regime_masks(train_manifest.width.to_numpy(), train_manifest.height.to_numpy())
    test_regimes = regime_masks(test_geometry.width.to_numpy(), test_geometry.height.to_numpy())
    event(
        "regimes",
        train=[int(mask.sum()) for mask in train_regimes],
        test=[int(mask.sum()) for mask in test_regimes],
    )

    # Minimal launch check requested by the competition workflow.
    if args.smoke_only:
        smoke_labels = np.asarray([0, 0, 1, 1], dtype=np.int8)
        smoke = unit(caches["pe_train"]["features"][:4])
        model = SVC(C=10.0, gamma=0.5).fit(smoke, smoke_labels)
        score = model.decision_function(smoke)
        if score.shape != (4,) or not np.isfinite(score).all():
            raise RuntimeError("V178 full smoke failed")
        event("smoke", status="passed")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    core_train = load_cache(root / "outputs/V66_pe_core_multidepth_residual/features/competition_multidepth_statistics.npz")
    core_test = load_cache(root / "outputs/V66_pe_core_multidepth_residual/features/test_multidepth_statistics.npz")
    spatial_train = load_cache(root / "outputs/V76_pe_spatial_multidepth_classifier/features/competition_multidepth_statistics.npz")
    spatial_test = load_cache(root / "outputs/V76_pe_spatial_multidepth_classifier/features/test_multidepth_statistics.npz")
    core_train = reorder(core_train, train_ids, "core_train")
    core_test = reorder(core_test, test_ids, "core_test")
    spatial_train = reorder(spatial_train, train_ids, "spatial_train")
    spatial_test = reorder(spatial_test, test_ids, "spatial_test")
    v109_train = multiscale_features(root, caches["pe_train"]["features"], caches["clip_train"]["features"], core_train["features"], spatial_train["features"], device=device)
    v109_test = multiscale_features(root, caches["pe_test"]["features"], caches["clip_test"]["features"], core_test["features"], spatial_test["features"], device=device)
    v109_model = SVC(C=10, gamma=0.5, kernel="rbf", cache_size=4096).fit(v109_train, labels)
    v109_score = rbf_svc_decision_torch(v109_model, v109_test, device=device)
    joblib.dump(v109_model, models_dir / "V109_multiscale_rbf_full.joblib")
    event("fit_infer", model="V109", support_vectors=v109_model.n_support_.tolist())
    del core_train, core_test, spatial_train, spatial_test, v109_train, v109_test

    train_views = unit_views(caches["views_train"]["features"])
    test_views = unit_views(caches["views_test"]["features"])
    instance_model = SVC(C=10.0, gamma=0.5, kernel="rbf", cache_size=4096).fit(
        train_views.reshape(-1, train_views.shape[-1]), np.repeat(labels, 10)
    )
    flat_test_views = test_views.reshape(-1, test_views.shape[-1])
    instance_scores = rbf_svc_decision_torch(
        instance_model, flat_test_views, device=device
    ).reshape(len(test_ids), 10)
    mil_mean = instance_scores.mean(axis=1)
    mil_route = np.where(test_regimes[0], instance_scores[:, :5].mean(axis=1), instance_scores[:, 5:].mean(axis=1))
    joblib.dump(instance_model, models_dir / "V122_instance_rbf_full.joblib")
    event("fit_infer", model="V122", support_vectors=instance_model.n_support_.tolist())

    train_siglip = reorder(caches["bombek_train"], train_ids, "bombek_train")["features"][:, :1152]
    test_siglip = reorder(caches["bombek_test"], test_ids, "bombek_test")["features"][:, :1152]
    visual_models, normalized_train = train_visual_kernels(
        labels,
        caches["pe_train"]["features"], caches["gray_train"]["features"],
        caches["clip_train"]["features"], caches["face_train"]["features"],
        caches["views_train"]["features"], train_siglip, *train_regimes,
    )
    normalized_test = {
        "pe_color": unit(caches["pe_test"]["features"]),
        "pe_gray": unit(caches["gray_test"]["features"]),
        "clip": unit(caches["clip_test"]["features"]),
        "face": unit(caches["face_test"]["features"]),
        "views": test_views,
        "siglip": unit(test_siglip),
    }
    visual_scores = infer_visual_kernels(
        visual_models, normalized_train, normalized_test, train_regimes, test_regimes,
        batch_size=args.kernel_batch_size,
    )
    for name, model in visual_models.items():
        joblib.dump(model, models_dir / f"{name}_full.joblib")

    def fit_forensic(train_name: str, test_name: str, model_name: str) -> tuple[np.ndarray, np.ndarray]:
        train_features = unit(caches[train_name]["features"])
        test_features = unit(caches[test_name]["features"])
        model = SVC(C=10, gamma=0.5, kernel="rbf", cache_size=4096).fit(train_features, labels)
        score = rbf_svc_decision_torch(model, test_features, device=device)
        joblib.dump(model, models_dir / f"{model_name}_full.joblib")
        event("fit_infer", model=model_name, support_vectors=model.n_support_.tolist())
        return score, caches[test_name]["logits"].astype(np.float64)

    rine_score, rine_logit = fit_forensic("rine_train", "rine_test", "V111_RINE")
    npr_score, npr_logit = fit_forensic("npr_train", "npr_test", "V113_NPR")
    meta_train = np.column_stack([
        caches["rine_train"]["logits"].astype(np.float64),
        caches["npr_train"]["logits"].astype(np.float64),
    ])
    meta_model = build_meta_model().fit(meta_train, labels)
    square_meta_rank = percentile_rank(meta_model.predict_proba(np.column_stack([rine_logit, npr_logit]))[:, 1])
    joblib.dump(meta_model, models_dir / "V178_rine_npr_square_meta_full.joblib")

    v90_frame = pd.read_csv(root / "outputs/submissions/V90_image_weighted_rank_ensemble.csv", dtype={"id": str}).set_index("id")
    if set(v90_frame.index) != set(test_ids):
        raise RuntimeError("V90 submission identity mismatch")
    v90_rank = percentile_rank(v90_frame.loc[test_ids, "score"].to_numpy(np.float64))
    v168 = (
        v90_rank + percentile_rank(mil_mean) + percentile_rank(mil_route) + percentile_rank(visual_scores["V167"])
    ) / 4.0
    uncertainty = np.clip(1.0 - np.abs(v168 - 0.5) / UNCERTAINTY_HALF_WIDTH, 0.0, 1.0)
    routed = np.where(test_regimes[1], percentile_rank(visual_scores["V171"]), np.where(test_regimes[2], percentile_rank(v109_score), percentile_rank(visual_scores["V155"])))
    v174 = (1.0 - PROTOCOL_MAX_WEIGHT * uncertainty) * v168 + PROTOCOL_MAX_WEIGHT * uncertainty * routed
    score = v174.copy()
    square = test_regimes[0]
    score[square] = (1.0 - FORENSIC_MAX_WEIGHT * uncertainty[square]) * score[square] + FORENSIC_MAX_WEIGHT * uncertainty[square] * square_meta_rank[square]
    face_detected = caches["face_test"]["face_detected"].astype(bool)
    npr_rank = percentile_rank(npr_score)
    rine_rank = percentile_rank(rine_score)
    npr_mask = face_detected & test_regimes[1] & (npr_rank >= NPR_HIGH_CONFIDENCE_RANK)
    rine_mask = face_detected & test_regimes[2] & (rine_rank >= RINE_HIGH_CONFIDENCE_RANK)
    for mask, expert_rank in ((npr_mask, npr_rank), (rine_mask, rine_rank)):
        score[mask] = (1.0 - FORENSIC_MAX_WEIGHT * uncertainty[mask]) * score[mask] + FORENSIC_MAX_WEIGHT * uncertainty[mask] * expert_rank[mask]
    if score.shape != (20000,) or not np.isfinite(score).all() or score.min() < 0 or score.max() > 1:
        raise RuntimeError("invalid V178 final score")

    components = pd.DataFrame({
        "id": test_ids, "V90_rank": v90_rank, "mil_mean": mil_mean, "mil_route": mil_route,
        "V109": v109_score, "V155": visual_scores["V155"], "V167": visual_scores["V167"],
        "V171": visual_scores["V171"], "RINE": rine_score, "NPR": npr_score,
        "V168": v168, "V174": v174, "uncertainty_gate": uncertainty,
        "square_meta_rank": square_meta_rank, "npr_rescue": npr_mask.astype(np.int8),
        "rine_rescue": rine_mask.astype(np.int8), "score": score,
    })
    components.to_csv(output / "test_component_predictions.csv", index=False)
    submission = components[["id", "score"]]
    submission_path = output / "V178_image_submission.csv"
    submission.to_csv(submission_path, index=False)

    manifest = {
        "version": "V178_content_protocol_forensic_cascade_full",
        "cv": {
            "fold_auc": [0.9998, 0.9984158415841585, 0.9995, 0.9986, 0.9987878787878788],
            "mean_auc": 0.9990207440744074,
        },
        "training": {"labeled_rows": 1000, "all_labeled_rows_used": True, "test_labels_used": False},
        "inference": {
            "rows": 20000,
            "score_min": float(score.min()), "score_max": float(score.max()),
            "score_mean": float(score.mean()), "unique_scores": int(len(np.unique(score))),
            "regime_rows": [int(mask.sum()) for mask in test_regimes],
            "npr_rescue_rows": int(npr_mask.sum()), "rine_rescue_rows": int(rine_mask.sum()),
        },
        "submission": {"path": str(submission_path), "sha256": sha256_file(submission_path)},
        "script_sha256": sha256_file(Path(__file__)),
        "elapsed_seconds": round(time.time() - started, 1),
        "audit": {
            "test_ids_used_only_for_alignment": True,
            "geometry_used_only_for_visual_preprocessing_routes": True,
            "file_size_or_filename_score_feature": False,
            "full_target_heads_fit_on_all_1000_labels": True,
            "rank_averaging_applied_over_full_20000_test_rows": True,
        },
    }
    (output / "submission_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    event("complete", submission=str(submission_path), sha256=manifest["submission"]["sha256"], elapsed=manifest["elapsed_seconds"])


if __name__ == "__main__":
    main()
