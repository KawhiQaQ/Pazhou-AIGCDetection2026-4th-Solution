#!/usr/bin/env python3
"""Extract the three frozen forensic test representations used by V178."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from evaluate_V111_rine_ldm_rbf import (  # noqa: E402
    CompetitionImages as RineImages,
    RINE,
    RINE_SHA256,
    RINE_URL,
    sha256_file,
)
from evaluate_V113_npr_rbf import (  # noqa: E402
    CompetitionImages as NprImages,
    NPR,
    NPR_SHA256,
    NPR_URL,
)
from evaluate_bombek_dual import DualDetector, Images as BombekImages  # noqa: E402


def test_frame(root: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        root / "data/competition/image/image_test.csv", dtype={"id": str}
    )
    frame["label"] = -1
    if len(frame) != 20_000 or frame.id.duplicated().any():
        raise RuntimeError("unexpected image test manifest")
    return frame


def rows_for(frame: pd.DataFrame, image_dir: Path) -> list[dict[str, object]]:
    rows = [
        {"id": row.id, "label": -1, "path": image_dir / row.id}
        for row in frame.itertuples(index=False)
    ]
    if any(not Path(str(row["path"])).is_file() for row in rows):
        raise RuntimeError("test manifest contains missing images")
    return rows


def extract_rine(root: Path, frame: pd.DataFrame, *, workers: int) -> Path:
    output = root / "outputs/V111_rine_ldm_rbf/features/test_center_features.npz"
    if output.is_file():
        print(json.dumps({"model": "RINE", "cache": str(output), "status": "reused"}))
        return output
    checkpoint = root / ".cache/rine/model_ldm_trainable.pth"
    actual_hash = sha256_file(checkpoint)
    if actual_hash != RINE_SHA256:
        raise RuntimeError(f"RINE checkpoint hash mismatch: {actual_hash}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RINE(root / ".cache/clip", device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(
        not key.startswith("clip.") for key in result.missing_keys
    ):
        raise RuntimeError("RINE state mismatch")
    model.eval().to(device)
    loader = DataLoader(
        RineImages(frame[["id", "label"]], root / "data/competition/image/data/image_test"),
        batch_size=24,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    ids: list[str] = []
    started = time.time()
    with torch.inference_mode():
        for batch_index, (images, batch_ids, _) in enumerate(loader, start=1):
            batch_logits, batch_features = model(images.to(device, non_blocking=True))
            features.append(batch_features.cpu().numpy().astype(np.float32))
            logits.append(batch_logits.cpu().numpy().astype(np.float32))
            ids.extend(batch_ids)
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(json.dumps({"model": "RINE", "rows": len(ids), "elapsed": round(time.time() - started, 1)}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.concatenate(features),
        logits=np.concatenate(logits),
        ids=np.asarray(ids),
        labels=np.full(len(ids), -1, dtype=np.int8),
        checkpoint_sha256=np.asarray(actual_hash),
        checkpoint_url=np.asarray(RINE_URL),
        transform=np.asarray("official center-crop-224"),
    )
    return output


def extract_npr(root: Path, frame: pd.DataFrame, *, workers: int) -> Path:
    output = root / "outputs/V113_npr_rbf/features/test_full_features.npz"
    if output.is_file():
        print(json.dumps({"model": "NPR", "cache": str(output), "status": "reused"}))
        return output
    checkpoint = root / ".cache/npr/model_epoch_last_3090.pth"
    actual_hash = sha256_file(checkpoint)
    if actual_hash != NPR_SHA256:
        raise RuntimeError(f"NPR checkpoint hash mismatch: {actual_hash}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NPR()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval().to(device)
    loader = DataLoader(
        NprImages(frame[["id", "label"]], root / "data/competition/image/data/image_test"),
        batch_size=1,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    ids: list[str] = []
    started = time.time()
    with torch.inference_mode():
        for batch_index, (images, batch_ids, _) in enumerate(loader, start=1):
            batch_logits, batch_features = model(images.to(device, non_blocking=True))
            features.append(batch_features.cpu().numpy().astype(np.float32))
            logits.append(batch_logits.cpu().numpy().astype(np.float32))
            ids.extend(batch_ids)
            if batch_index % 500 == 0 or batch_index == len(loader):
                print(json.dumps({"model": "NPR", "rows": len(ids), "elapsed": round(time.time() - started, 1)}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.concatenate(features),
        logits=np.concatenate(logits),
        ids=np.asarray(ids),
        labels=np.full(len(ids), -1, dtype=np.int8),
        checkpoint_sha256=np.asarray(actual_hash),
        checkpoint_url=np.asarray(NPR_URL),
        transform=np.asarray("official AIGCDetectBenchmark full-image/no-resize/no-crop"),
    )
    return output


def extract_bombek(root: Path, frame: pd.DataFrame, *, workers: int) -> Path:
    output = root / "outputs/V99_bombek_dual_residual/features/test_features.npz"
    if output.is_file():
        print(json.dumps({"model": "Bombek", "cache": str(output), "status": "reused"}))
        return output
    checkpoint_path = root / ".cache/bombek_dual/pytorch_model.pt"
    with torch.serialization.safe_globals(
        [np._core.multiarray.scalar, np.dtype, np.dtypes.Float64DType]
    ):
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise RuntimeError("unexpected Bombek checkpoint structure")
    model = DualDetector()
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    del checkpoint
    for parameter in model.parameters():
        parameter.requires_grad = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    rows = rows_for(frame, root / "data/competition/image/data/image_test")
    loader = DataLoader(
        BombekImages(rows),
        batch_size=6,
        shuffle=False,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    ids: list[str] = []
    started = time.time()
    with torch.inference_mode():
        for batch_index, (siglip_pixels, dinov2_pixels, _, batch_ids) in enumerate(loader, start=1):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                batch_logits, batch_features = model(
                    siglip_pixels.to(device, non_blocking=True),
                    dinov2_pixels.to(device, non_blocking=True),
                )
            if not torch.isfinite(batch_logits).all() or batch_features.shape[1] != 2176:
                raise RuntimeError("invalid Bombek output")
            features.append(batch_features.to(torch.float16).cpu().numpy())
            logits.append(batch_logits.float().cpu().numpy())
            ids.extend(batch_ids)
            if batch_index % 100 == 0 or batch_index == len(loader):
                print(json.dumps({"model": "Bombek", "rows": len(ids), "elapsed": round(time.time() - started, 1)}), flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.concatenate(features),
        parent_scores=np.concatenate(logits).astype(np.float32),
        labels=np.full(len(ids), -1, dtype=np.float32),
        ids=np.asarray(ids),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--models", nargs="+", choices=["rine", "npr", "bombek"],
        default=["rine", "npr", "bombek"],
    )
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    root = args.project_root.resolve()
    frame = test_frame(root)
    outputs: list[str] = []
    for name in args.models:
        extractor = {"rine": extract_rine, "npr": extract_npr, "bombek": extract_bombek}[name]
        outputs.append(str(extractor(root, frame, workers=args.workers)))
        torch.cuda.empty_cache()
    print(json.dumps({"status": "complete", "outputs": outputs}, indent=2))


if __name__ == "__main__":
    main()
