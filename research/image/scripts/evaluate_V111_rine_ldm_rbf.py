#!/usr/bin/env python3
"""Evaluate an official LDM-pretrained RINE representation with fold-safe RBF heads."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import clip
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RINE_URL = "https://raw.githubusercontent.com/mever-team/rine/main/ckpt/model_ldm_trainable.pth"
RINE_SHA256 = "13986ffc86e1bd9d57fd640f891358c4f14c0f8aea8e590e52c598541f69fdc1"
BASELINE_FOLD_AUC = [1.0, 0.9971782178217822, 0.99945, 0.9942, 0.9966666666666667]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class Hook:
    def __init__(self, module: nn.Module) -> None:
        self.output: torch.Tensor | None = None
        self.handle = module.register_forward_hook(self._capture)

    def _capture(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs
        self.output = output


class RINE(nn.Module):
    def __init__(self, clip_cache: Path, device: torch.device) -> None:
        super().__init__()
        self.clip, _ = clip.load(
            "ViT-L/14", device=device, download_root=str(clip_cache)
        )
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self.hooks = [
            Hook(module)
            for name, module in self.clip.visual.named_modules()
            if "ln_2" in name
        ]
        projection_dim = 1024
        self.alpha = nn.Parameter(torch.randn(1, len(self.hooks), projection_dim))
        projection_layers: list[nn.Module] = [nn.Dropout()]
        for _ in range(4):
            projection_layers.extend(
                [nn.Linear(projection_dim, projection_dim), nn.ReLU(), nn.Dropout()]
            )
        self.proj1 = nn.Sequential(*projection_layers)
        projection_layers = [nn.Dropout()]
        for _ in range(4):
            projection_layers.extend(
                [nn.Linear(projection_dim, projection_dim), nn.ReLU(), nn.Dropout()]
            )
        self.proj2 = nn.Sequential(*projection_layers)
        self.head = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(projection_dim, projection_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(projection_dim, 1),
        )

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            self.clip.encode_image(images)
            if any(hook.output is None for hook in self.hooks):
                raise RuntimeError("RINE intermediate hooks did not fire")
            # CLIP residual blocks expose [tokens, batch, channels].
            intermediate = torch.stack(
                [hook.output for hook in self.hooks if hook.output is not None], dim=2
            )[0]
        projected = self.proj1(intermediate.float())
        pooled = (torch.softmax(self.alpha, dim=1) * projected).sum(dim=1)
        representation = self.proj2(pooled)
        return self.head(representation).squeeze(1), representation


class CompetitionImages(Dataset):
    def __init__(self, rows: pd.DataFrame, image_dir: Path) -> None:
        self.rows = rows.to_dict(orient="records")
        self.image_dir = image_dir
        self.transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.48145466, 0.4578275, 0.40821073),
                    std=(0.26862954, 0.26130258, 0.27577711),
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
    cache_path = root / "outputs/V111_rine_ldm_rbf/features/competition_center_features.npz"
    if cache_path.is_file():
        cache = np.load(cache_path)
        if str(cache["checkpoint_sha256"].item()) != checkpoint_sha256:
            raise RuntimeError("V111 feature cache checkpoint mismatch")
        return (
            cache["features"].astype(np.float32),
            cache["logits"].astype(np.float32),
            cache["ids"].astype(str),
            cache["labels"].astype(np.int8),
        )

    folds = pd.read_csv(root / "data/folds/image_folds_v1.csv")
    folds["id"] = folds.id.astype(str)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RINE(root / ".cache/clip", device)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    result = model.load_state_dict(state, strict=False)
    if result.unexpected_keys or any(
        not key.startswith("clip.") for key in result.missing_keys
    ):
        raise RuntimeError(
            f"RINE state mismatch: missing={result.missing_keys[:5]}, unexpected={result.unexpected_keys}"
        )
    model.eval().to(device)
    loader = DataLoader(
        CompetitionImages(
            folds[["id", "label"]],
            root / "data/competition/image/data/image_sample_data",
        ),
        batch_size=24,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )
    feature_parts: list[np.ndarray] = []
    logit_parts: list[np.ndarray] = []
    ids: list[str] = []
    labels: list[int] = []
    with torch.inference_mode():
        for images, batch_ids, batch_labels in loader:
            logits, features = model(images.to(device, non_blocking=True))
            feature_parts.append(features.cpu().numpy().astype(np.float32))
            logit_parts.append(logits.cpu().numpy().astype(np.float32))
            ids.extend(batch_ids)
            labels.extend(batch_labels.numpy().tolist())
    feature_array = np.concatenate(feature_parts)
    logit_array = np.concatenate(logit_parts)
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
        checkpoint_url=np.asarray(RINE_URL),
        transform=np.asarray("official center-crop-224"),
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
    checkpoint = root / ".cache/rine/model_ldm_trainable.pth"
    actual_sha256 = sha256_file(checkpoint)
    if RINE_SHA256 == "TO_BE_FILLED":
        raise RuntimeError(f"fill audited RINE SHA-256 before running: {actual_sha256}")
    if actual_sha256 != RINE_SHA256:
        raise RuntimeError(f"RINE checkpoint hash mismatch: {actual_sha256}")
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
    output_root = root / "outputs/V111_rine_ldm_rbf"

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
                raise RuntimeError("V111 smoke inference failed")
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
        equal_three = (v90_rank + v107_rank + candidate_rank) / 3.0
        baseline = 0.5 * v90_rank + 0.5 * v107_rank
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
                "V112_equal_three_score": equal_three,
            }
        ).to_csv(output_dir / "valid_predictions.csv", index=False)
        row = {
            "fold": fold,
            "n": int(valid_mask.sum()),
            "V108_submission_replay_auc": baseline_auc,
            "RINE_LDM_zero_shot_auc": float(
                roc_auc_score(labels[valid_mask], zero_shot_logits[valid_mask])
            ),
            "V111_single_auc": float(roc_auc_score(labels[valid_mask], raw_score)),
            "V112_equal_three_auc": float(
                roc_auc_score(labels[valid_mask], equal_three)
            ),
        }
        row["delta"] = row["V112_equal_three_auc"] - baseline_auc
        row["support_vectors"] = model.n_support_.tolist()
        rows.append(row)
        (output_dir / "metrics.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(row), flush=True)

    summary = {
        "version": "V111_rine_ldm_rbf_and_V112_equal_three",
        "folds": rows,
        "V108_submission_replay_mean": float(
            np.mean([row["V108_submission_replay_auc"] for row in rows])
        ),
        "RINE_LDM_zero_shot_mean": float(
            np.mean([row["RINE_LDM_zero_shot_auc"] for row in rows])
        ),
        "V111_single_mean": float(np.mean([row["V111_single_auc"] for row in rows])),
        "V112_equal_three_mean": float(
            np.mean([row["V112_equal_three_auc"] for row in rows])
        ),
        "checkpoint": {
            "url": RINE_URL,
            "sha256": actual_sha256,
            "training_domain": "official latent-diffusion checkpoint",
        },
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
