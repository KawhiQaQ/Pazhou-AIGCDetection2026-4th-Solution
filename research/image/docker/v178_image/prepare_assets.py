#!/usr/bin/env python3
"""Materialize the self-contained, ID-free inference assets for V178."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import joblib
import numpy as np
import pandas as pd
import torch


VERSION = "V178_content_protocol_forensic_cascade_full"
SUBMISSION_SHA256 = "dc9981768cef1dbd8dcf05e5a3eb8abcb83852dbf7b9406dd9b3d2b8ae32f9ee"


def copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def unit(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-12)


def reorder(path: Path, ids: np.ndarray) -> dict[str, np.ndarray]:
    cached = np.load(path)
    cached_ids = cached["ids"].astype(str)
    index = pd.Index(cached_ids).get_indexer(ids)
    if (index < 0).any() or len(np.unique(cached_ids)) != len(cached_ids):
        raise RuntimeError(f"cache identity mismatch: {path}")
    return {key: cached[key][index] if np.asarray(cached[key]).ndim and len(cached[key]) == len(cached_ids) else cached[key] for key in cached.files}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root, out = args.project_root.resolve(), args.output.resolve()
    backbones, heads, calibration = out / "backbones", out / "heads", out / "calibration"
    for directory in (backbones, heads, calibration):
        directory.mkdir(parents=True, exist_ok=True)

    large = {
        root / ".cache/pe_core_g14/model.safetensors": backbones / "pe_core_g14.safetensors",
        root / ".cache/pe_spatial_g14/model.safetensors": backbones / "pe_spatial_g14.safetensors",
        root / ".cache/clip/ViT-L-14-336px.pt": backbones / "clip/ViT-L-14-336px.pt",
        root / ".cache/clip/ViT-L-14.pt": backbones / "clip/ViT-L-14.pt",
        root / ".cache/rine/model_ldm_trainable.pth": backbones / "rine.pth",
        root / ".cache/npr/model_epoch_last_3090.pth": backbones / "npr.pth",
    }
    for source, target in large.items():
        copy(source, target)
    for source in (root / ".cache/deepfake_vit").iterdir():
        if source.is_file():
            copy(source, backbones / "deepfake_vit" / source.name)
    copy(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml", out / "haarcascade_frontalface_default.xml")

    # V178 consumes only the 1,152-dimensional SigLIP branch, never Bombek's DINO branch.
    # Exporting that branch alone reduces the container without changing any score.
    with torch.serialization.safe_globals([np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]):
        bombek = torch.load(root / ".cache/bombek_dual/pytorch_model.pt", map_location="cpu", weights_only=True)
    siglip = {key: value for key, value in bombek["model_state_dict"].items() if key.startswith("siglip.")}
    if not siglip:
        raise RuntimeError("Bombek checkpoint has no SigLIP branch")
    torch.save(siglip, backbones / "bombek_siglip.pth")
    del bombek, siglip

    files = {
        "V29.pth": "outputs/V29_clip_adapter_multidepth_patch_external/full_e7/trainable_e007.pth",
        "V36.pth": "outputs/V36_clip_adapter_paired_grid/full_e7/trainable_e007.pth",
        "V36_external.pth": "outputs/V36_clip_adapter_paired_grid/pretrain/external_pretrain.pth",
        "V47_linear.pth": "outputs/V47_pe_core_g14_linear/full_e100/linear_head.pth",
        "V51_linear.pth": "outputs/V51_pe_core_aigi_breadth_linear/full_e22/linear_head.pth",
        "V54_residual.pth": "outputs/V54_pe_core_residual_mlp/full_e3/residual_mlp.pth",
        "V56_residual.pth": "outputs/V56_pe_core_openfake_residual_mlp/full_e3/residual_mlp.pth",
        "V58_residual.pth": "outputs/V58_pe_core_aigi_residual_mlp_seed2/full_e3/residual_mlp.pth",
        "V60_residual.pth": "outputs/V60_cross_backbone_residual/full_e3/clip_residual.pth",
        "V67_multidepth.pth": "outputs/V67_pe_core_multidepth_classifier/full_e10/multidepth_classifier.pth",
        "V71_multidepth.pth": "outputs/V71_pe_core_multidepth_pairwise/full_e10/multidepth_classifier.pth",
        "V76_multidepth.pth": "outputs/V76_pe_spatial_multidepth_classifier/full_e10/multidepth_classifier.pth",
        "V77_multidepth.pth": "outputs/V77_pe_spatial_multidepth_pairwise/full_e10/multidepth_classifier.pth",
        "V86_fusion.pth": "outputs/V86_dual_pe_multidepth_fusion/full_e5/dual_pe_fusion.pth",
        "V109.joblib": "outputs/V178_full/models/V109_multiscale_rbf_full.joblib",
        "V122.joblib": "outputs/V178_full/models/V122_instance_rbf_full.joblib",
        "V155.joblib": "outputs/V178_full/models/V155_full.joblib",
        "V167.joblib": "outputs/V178_full/models/V167_full.joblib",
        "V171.joblib": "outputs/V178_full/models/V171_full.joblib",
        "RINE.joblib": "outputs/V178_full/models/V111_RINE_full.joblib",
        "NPR.joblib": "outputs/V178_full/models/V113_NPR_full.joblib",
        "V178_meta.joblib": "outputs/V178_full/models/V178_rine_npr_square_meta_full.joblib",
    }
    for target, source in files.items():
        copy(root / source, heads / target)

    fold = pd.read_csv(root / "data/folds/image_folds_v1.csv", dtype={"id": str})
    ids = fold.id.to_numpy(str)
    cache_paths = {
        "pe": "outputs/V51_pe_core_aigi_breadth_linear/features/competition_center_features.npz",
        "gray": "outputs/V161_pe_core_grayscale_rbf/features/competition.npz",
        "clip": "outputs/V60_cross_backbone_residual/features/competition_center_features.npz",
        "face": "outputs/V107_face_crop_rbf/features/competition.npz",
        "views": "outputs/V118_pe_core_multiview_rbf/features/competition.npz",
        "bombek": "outputs/V99_bombek_dual_residual/features/competition_features.npz",
    }
    caches = {name: reorder(root / path, ids) for name, path in cache_paths.items()}
    aspect = fold.width.to_numpy(np.float64) / fold.height.to_numpy(np.float64)
    square = np.abs(np.log(aspect)) < 0.12
    low = (~square) & (np.minimum(fold.width.to_numpy(), fold.height.to_numpy()) < 448)
    high = (~square) & (~low)
    np.savez_compressed(
        calibration / "train_references.npz",
        pe_color=unit(caches["pe"]["features"]), pe_gray=unit(caches["gray"]["features"]),
        clip=unit(caches["clip"]["features"]), face=unit(caches["face"]["features"]),
        views=unit(caches["views"]["features"]), siglip=unit(caches["bombek"]["features"][:, :1152]),
        square=square, low=low, high=high,
    )

    v90_members = {
        "V29": "V29_clip_adapter_multidepth_patch_external_full_e7.csv",
        "V36": "V36_clip_adapter_paired_grid_full_e7.csv", "V47": "V47_pe_core_g14_linear_full_e100.csv",
        "V54": "V54_pe_core_residual_mlp_full_e3.csv", "V56": "V56_pe_core_openfake_residual_mlp_full_e3.csv",
        "V58": "V58_pe_core_aigi_residual_mlp_seed2_full_e3.csv", "V60": "V60_cross_backbone_residual_full_e3.csv",
        "V67": "V67_pe_core_multidepth_classifier_full_e10.csv", "V71": "V71_pe_core_multidepth_pairwise_full_e10.csv",
        "V76": "V76_pe_spatial_multidepth_classifier_full_e10.csv", "V77": "V77_pe_spatial_multidepth_pairwise_full_e10.csv",
        "V86": "V86_dual_pe_multidepth_fusion_full_e5.csv",
    }
    ecdfs = {f"V90_{name}": np.sort(pd.read_csv(root / "outputs/submissions" / filename).score.to_numpy(np.float64)) for name, filename in v90_members.items()}
    components = pd.read_csv(root / "outputs/V178_full/test_component_predictions.csv")
    for name, column in {"mil_mean": "mil_mean", "mil_route": "mil_route", "V109": "V109", "V155": "V155",
                         "V167": "V167", "V171": "V171", "RINE": "RINE", "NPR": "NPR"}.items():
        ecdfs[name] = np.sort(components[column].to_numpy(np.float64))
    test_ids = components.id.astype(str).to_numpy()
    rine = reorder(root / "outputs/V111_rine_ldm_rbf/features/test_center_features.npz", test_ids)["logits"].astype(np.float64)
    npr = reorder(root / "outputs/V113_npr_rbf/features/test_full_features.npz", test_ids)["logits"].astype(np.float64)
    meta = joblib.load(root / files["V178_meta.joblib"])
    ecdfs["square_meta"] = np.sort(meta.predict_proba(np.column_stack([rine, npr]))[:, 1].astype(np.float64))
    np.savez_compressed(calibration / "score_ecdfs.npz", **ecdfs)

    manifest_files = []
    for path in sorted(item for item in out.rglob("*") if item.is_file() and item.name != "manifest.json"):
        manifest_files.append({"path": str(path.relative_to(out)), "bytes": path.stat().st_size, "sha256": sha256(path)})
    manifest = {
        "version": VERSION, "submission_sha256": SUBMISSION_SHA256,
        "test_labels_used": False, "sample_id_or_filename_lookup": False,
        "calibration": "unlabeled empirical score CDFs only; no IDs, labels, filenames, or per-sample answer map",
        "files": manifest_files,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"assets": str(out), "files": len(manifest_files), "bytes": sum(x["bytes"] for x in manifest_files)}, indent=2))


if __name__ == "__main__":
    main()
