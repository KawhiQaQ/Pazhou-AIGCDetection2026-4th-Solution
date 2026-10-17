#!/usr/bin/env python3
"""Evaluate the official NPR low-level detector and a fold-safe RBF transfer head."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NPR_URL = "https://raw.githubusercontent.com/chuangchuangtan/NPR-DeepfakeDetection/main/model_epoch_last_3090.pth"
NPR_SHA256 = "b67a91555ce786a6d0463ff0cb2b0b874d1c3f971b0e3febd2ae5618a80f7e8a"
BASELINE_FOLD_AUC = [1.0, 0.9971782178217822, 0.99945, 0.9942, 0.9966666666666667]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def conv3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_channels: int,
        channels: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1(in_channels, channels)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = conv3x3(channels, channels, stride)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = conv1x1(channels, channels * self.expansion)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        identity = inputs
        output = self.relu(self.bn1(self.conv1(inputs)))
        output = self.relu(self.bn2(self.conv2(output)))
        output = self.bn3(self.conv3(output))
        if self.downsample is not None:
            identity = self.downsample(inputs)
        return self.relu(output + identity)


class NPR(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, blocks=3)
        self.layer2 = self._make_layer(128, blocks=4, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(512, 1)

    def _make_layer(self, channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        out_channels = channels * Bottleneck.expansion
        downsample = None
        if stride != 1 or self.inplanes != out_channels:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, out_channels, stride),
                nn.BatchNorm2d(out_channels),
            )
        layers: list[nn.Module] = [
            Bottleneck(self.inplanes, channels, stride, downsample)
        ]
        self.inplanes = out_channels
        layers.extend(Bottleneck(self.inplanes, channels) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # This is the odd-size fix specified by the official NPR repository.
        if images.shape[2] % 2:
            images = images[:, :, :-1, :]
        if images.shape[3] % 2:
            images = images[:, :, :, :-1]
        resized = F.interpolate(images, scale_factor=0.5, mode="nearest", recompute_scale_factor=True)
        restored = F.interpolate(resized, scale_factor=2.0, mode="nearest", recompute_scale_factor=True)
        residual = (images - restored) * (2.0 / 3.0)
        output = self.maxpool(self.relu(self.bn1(self.conv1(residual))))
        output = self.layer1(output)
        output = self.layer2(output)
        features = self.avgpool(output).flatten(1)
        return self.fc1(features).squeeze(1), features


class CompetitionImages(Dataset):
    def __init__(self, rows: pd.DataFrame, image_dir: Path) -> None:
        self.rows = rows.to_dict(orient="records")
        self.image_dir = image_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str, int]:
        row = self.rows[index]
        with Image.open(self.image_dir / str(row["id"])) as source:
            image = ImageOps.exif_transpose(source).convert("RGB")
            tensor = self.transform(image)
        return tensor, str(row["id"]), int(row["label"])


def extract_features(
    root: Path, checkpoint: Path, checkpoint_sha256: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cache_path = root / "outputs/V113_npr_rbf/features/competition_full_features.npz"
    if cache_path.is_file():
        cache = np.load(cache_path)
        if str(cache["checkpoint_sha256"].item()) != checkpoint_sha256:
            raise RuntimeError("V113 feature cache checkpoint mismatch")
        return (
            cache["features"].astype(np.float32),
            cache["logits"].astype(np.float32),
            cache["ids"].astype(str),
            cache["labels"].astype(np.int8),
        )
    folds = pd.read_csv(root / "data/folds/image_folds_v1.csv")
    folds["id"] = folds.id.astype(str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NPR()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    loader = DataLoader(
        CompetitionImages(
            folds[["id", "label"]],
            root / "data/competition/image/data/image_sample_data",
        ),
        batch_size=1,
        shuffle=False,
        num_workers=6,
        pin_memory=True,
    )
    features: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    ids: list[str] = []
    labels: list[int] = []
    with torch.inference_mode():
        for images, batch_ids, batch_labels in loader:
            batch_logits, batch_features = model(images.to(device, non_blocking=True))
            features.append(batch_features.cpu().numpy().astype(np.float32))
            logits.append(batch_logits.cpu().numpy().astype(np.float32))
            ids.extend(batch_ids)
            labels.extend(batch_labels.numpy().tolist())
    feature_array = np.concatenate(features)
    logit_array = np.concatenate(logits)
    id_array = np.asarray(ids)
    label_array = np.asarray(labels, dtype=np.int8)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=feature_array,
        logits=logit_array,
        ids=id_array,
        labels=label_array,
        checkpoint_sha256=np.asarray(checkpoint_sha256),
        checkpoint_url=np.asarray(NPR_URL),
        transform=np.asarray("official AIGCDetectBenchmark full-image/no-resize/no-crop"),
    )
    return feature_array, logit_array, id_array, label_array


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    checkpoint = root / ".cache/npr/model_epoch_last_3090.pth"
    actual_sha256 = sha256_file(checkpoint)
    if actual_sha256 != NPR_SHA256:
        raise RuntimeError(f"NPR checkpoint hash mismatch: {actual_sha256}")
    features, zero_shot_logits, ids, labels = extract_features(
        root, checkpoint, actual_sha256
    )
    folds_frame = pd.read_csv(root / "data/folds/image_folds_v1.csv")
    folds_frame["id"] = folds_frame.id.astype(str)
    fold_by_id = folds_frame.set_index("id").fold
    group_by_id = folds_frame.set_index("id").group_id
    fold_values = np.asarray([int(fold_by_id.loc[item]) for item in ids])
    group_values = np.asarray([str(group_by_id.loc[item]) for item in ids])
    baseline_oof = pd.read_csv(
        root / "outputs/fivefold_milestones/oof_predictions.csv"
    ).assign(id=lambda frame: frame.id.astype(str)).set_index("id")
    normalized_features = normalize(features).astype(np.float32)
    output_root = root / "outputs/V113_npr_rbf"

    rows: list[dict[str, object]] = []
    for fold in sorted(args.folds):
        train_mask = fold_values != fold
        valid_mask = ~train_mask
        if set(group_values[train_mask]) & set(group_values[valid_mask]):
            raise RuntimeError(f"Fold{fold} group leakage")
        model = SVC(C=10, gamma=0.5, kernel="rbf")
        if args.smoke_only:
            smoke_indices = np.concatenate(
                [
                    np.flatnonzero(train_mask & (labels == 0))[:16],
                    np.flatnonzero(train_mask & (labels == 1))[:16],
                ]
            )
            model.fit(normalized_features[smoke_indices], labels[smoke_indices])
            scores = model.decision_function(normalized_features[valid_mask][:4])
            if len(scores) != 4 or not np.isfinite(scores).all():
                raise RuntimeError("V113 smoke inference failed")
            print(json.dumps({"smoke": "passed", "feature_dim": features.shape[1]}))
            return
        model.fit(normalized_features[train_mask], labels[train_mask])
        raw_score = model.decision_function(normalized_features[valid_mask])
        candidate_rank = percentile_rank(raw_score)
        valid_ids = ids[valid_mask]
        fold_oof = baseline_oof.loc[valid_ids]
        v90_rank = percentile_rank(fold_oof["V90_score"].to_numpy(dtype=np.float64))
        v107_frame = pd.read_csv(
            root / f"outputs/V107_face_crop_rbf/fold{fold}/valid_predictions.csv"
        ).assign(id=lambda frame: frame.id.astype(str)).set_index("id")
        v107_rank = percentile_rank(
            v107_frame.loc[valid_ids, "score"].to_numpy(dtype=np.float64)
        )
        baseline = 0.5 * v90_rank + 0.5 * v107_rank
        equal_three = (v90_rank + v107_rank + candidate_rank) / 3.0
        baseline_auc = float(roc_auc_score(labels[valid_mask], baseline))
        if not np.isclose(baseline_auc, BASELINE_FOLD_AUC[fold], atol=5e-13, rtol=0):
            raise RuntimeError(f"Fold{fold} true-entry baseline mismatch")
        output_dir = output_root / f"fold{fold}"
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_dir / "rbf_svm.joblib")
        pd.DataFrame(
            {
                "id": valid_ids,
                "label": labels[valid_mask],
                "score": raw_score,
                "percentile_rank": candidate_rank,
                "zero_shot_logit": zero_shot_logits[valid_mask],
                "V114_equal_three_score": equal_three,
            }
        ).to_csv(output_dir / "valid_predictions.csv", index=False)
        row = {
            "fold": fold,
            "n": int(valid_mask.sum()),
            "V108_submission_replay_auc": baseline_auc,
            "NPR_zero_shot_auc": float(
                roc_auc_score(labels[valid_mask], zero_shot_logits[valid_mask])
            ),
            "V113_single_auc": float(roc_auc_score(labels[valid_mask], raw_score)),
            "V114_equal_three_auc": float(
                roc_auc_score(labels[valid_mask], equal_three)
            ),
        }
        row["delta"] = row["V114_equal_three_auc"] - baseline_auc
        row["support_vectors"] = model.n_support_.tolist()
        rows.append(row)
        (output_dir / "metrics.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(row), flush=True)

    summary = {
        "version": "V113_npr_rbf_and_V114_equal_three",
        "folds": rows,
        "V108_submission_replay_mean": float(
            np.mean([row["V108_submission_replay_auc"] for row in rows])
        ),
        "NPR_zero_shot_mean": float(
            np.mean([row["NPR_zero_shot_auc"] for row in rows])
        ),
        "V113_single_mean": float(np.mean([row["V113_single_auc"] for row in rows])),
        "V114_equal_three_mean": float(
            np.mean([row["V114_equal_three_auc"] for row in rows])
        ),
        "checkpoint": {"url": NPR_URL, "sha256": actual_sha256},
        "audit": {
            "true_submission_entry_replayed": True,
            "weight_search": False,
            "validation_rows_used_for_training": 0,
            "group_leakage": False,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "fivefold_metrics.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
