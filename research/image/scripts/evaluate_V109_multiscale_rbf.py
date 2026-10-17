#!/usr/bin/env python3
"""Evaluate V109: a four-block multi-scale dual-backbone RBF detector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import normalize
from sklearn.svm import SVC
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_dual_pe_multidepth_fusion import branch_forward, load_branch, load_cache  # noqa: E402


CORE_SHA256 = "a0a499d4a7c6d35bb5f11aa2b076eaacde06e66a08176eb3b3d362d80eb91bd8"
SPATIAL_SHA256 = "db69c2e41d38a72cf3f852707367934edbea3f47c3d8b07dc3e082b8caff8d8d"
LAYERS = [9, 19, 29, 39, 49]
BLOCK_WEIGHTS = {
    "pe_core_final": 0.30,
    "clip_v36_final": 0.30,
    "pe_core_multidepth": 0.20,
    "pe_spatial_multidepth": 0.20,
}
BASELINE_FOLD_AUC = [1.0, 0.9971782178217822, 0.99945, 0.9942, 0.9966666666666667]


def load_flat_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cached = np.load(path)
    return (
        cached["features"].astype(np.float32),
        cached["ids"].astype(str),
        cached["labels"].astype(np.int8),
    )


@torch.inference_mode()
def pooled_features(
    statistics: np.ndarray,
    checkpoint: Path,
    encoder_sha256: str,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> np.ndarray:
    branch = load_branch(
        checkpoint,
        encoder_sha256,
        LAYERS,
        int(statistics.shape[2]),
        projection_dim=128,
        dropout=0.2,
    ).to(device)
    outputs: list[np.ndarray] = []
    for start in range(0, len(statistics), batch_size):
        batch = torch.from_numpy(
            statistics[start : start + batch_size].astype(np.float32)
        ).to(device)
        _, pooled, _ = branch_forward(branch, batch)
        outputs.append(pooled.cpu().numpy())
    return np.concatenate(outputs).astype(np.float32)


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()
    folds = sorted(args.folds)
    if any(fold not in range(5) for fold in folds):
        raise ValueError(f"invalid folds: {folds}")
    if not np.isclose(sum(BLOCK_WEIGHTS.values()), 1.0):
        raise RuntimeError("V109 block weights do not sum to one")

    pe, ids, labels = load_flat_cache(
        root / "outputs/V51_pe_core_aigi_breadth_linear/features/competition_center_features.npz"
    )
    clip, clip_ids, clip_labels = load_flat_cache(
        root / "outputs/V60_cross_backbone_residual/features/competition_center_features.npz"
    )
    core = load_cache(
        root / "outputs/V66_pe_core_multidepth_residual/features/competition_multidepth_statistics.npz",
        CORE_SHA256,
        LAYERS,
    )
    spatial = load_cache(
        root / "outputs/V76_pe_spatial_multidepth_classifier/features/competition_multidepth_statistics.npz",
        SPATIAL_SHA256,
        LAYERS,
    )
    if not (
        np.array_equal(ids, clip_ids)
        and np.array_equal(ids, core["ids"])
        and np.array_equal(ids, spatial["ids"])
        and np.array_equal(labels, clip_labels)
        and np.array_equal(labels, core["labels"].astype(np.int8))
        and np.array_equal(labels, spatial["labels"].astype(np.int8))
    ):
        raise RuntimeError("V109 feature cache identity/label mismatch")

    fold_frame = pd.read_csv(root / "data/folds/image_folds_v1.csv")
    fold_frame["id"] = fold_frame.id.astype(str)
    fold_by_id = fold_frame.set_index("id").fold
    group_by_id = fold_frame.set_index("id").group_id
    fold_values = np.asarray([int(fold_by_id.loc[item]) for item in ids])
    group_values = np.asarray([str(group_by_id.loc[item]) for item in ids])
    baseline_oof = pd.read_csv(
        root / "outputs/fivefold_milestones/oof_predictions.csv"
    ).set_index("id")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_root = root / "outputs/V109_multiscale_rbf"
    rows: list[dict[str, object]] = []
    for fold in folds:
        train_mask = fold_values != fold
        valid_mask = ~train_mask
        if set(group_values[train_mask]) & set(group_values[valid_mask]):
            raise RuntimeError(f"Fold{fold} group leakage")
        core_pooled = pooled_features(
            core["features"],
            root / f"outputs/V71_pe_core_multidepth_pairwise/fold{fold}/multidepth_classifier.pth",
            CORE_SHA256,
            device=device,
        )
        spatial_pooled = pooled_features(
            spatial["features"],
            root / f"outputs/V77_pe_spatial_multidepth_pairwise/fold{fold}/multidepth_classifier.pth",
            SPATIAL_SHA256,
            device=device,
        )
        features = np.concatenate(
            [
                normalize(pe) * np.sqrt(BLOCK_WEIGHTS["pe_core_final"]),
                normalize(clip) * np.sqrt(BLOCK_WEIGHTS["clip_v36_final"]),
                normalize(core_pooled) * np.sqrt(BLOCK_WEIGHTS["pe_core_multidepth"]),
                normalize(spatial_pooled) * np.sqrt(BLOCK_WEIGHTS["pe_spatial_multidepth"]),
            ],
            axis=1,
        ).astype(np.float32)
        model = SVC(C=10, gamma=0.5, kernel="rbf")
        if args.smoke_only:
            model.fit(features[train_mask][:32], labels[train_mask][:32])
            if len(model.decision_function(features[valid_mask][:4])) != 4:
                raise RuntimeError("V109 smoke inference failed")
            print(json.dumps({"smoke": "passed", "feature_dim": features.shape[1]}))
            return
        model.fit(features[train_mask], labels[train_mask])
        raw_score = model.decision_function(features[valid_mask])
        candidate_rank = percentile_rank(raw_score)
        valid_ids = ids[valid_mask]
        baseline = baseline_oof.loc[
            valid_ids, "V108-submission-replay_score"
        ].to_numpy(dtype=np.float64)
        baseline_rank = percentile_rank(baseline)
        # True submission path: each top-level system is ranked exactly once,
        # then V90, V107, and V109 are averaged as three equal systems.
        fold_oof = baseline_oof.loc[valid_ids]
        v90_rank = percentile_rank(fold_oof["V90_score"].to_numpy(dtype=np.float64))
        # The stored submission-replay score does not expose raw V107, so load it.
        v107_frame = pd.read_csv(
            root / f"outputs/V107_face_crop_rbf/fold{fold}/valid_predictions.csv"
        ).assign(id=lambda frame: frame.id.astype(str)).set_index("id")
        v107_rank = percentile_rank(
            v107_frame.loc[valid_ids, "score"].to_numpy(dtype=np.float64)
        )
        equal_three = (v90_rank + v107_rank + candidate_rank) / 3.0
        candidate_auc = float(roc_auc_score(labels[valid_mask], raw_score))
        equal_three_auc = float(roc_auc_score(labels[valid_mask], equal_three))
        baseline_auc = float(roc_auc_score(labels[valid_mask], baseline))
        if not np.isclose(baseline_auc, BASELINE_FOLD_AUC[fold], atol=5e-13, rtol=0):
            raise RuntimeError(
                f"Fold{fold} true-entry baseline mismatch: {baseline_auc} != {BASELINE_FOLD_AUC[fold]}"
            )
        output_dir = output_root / f"fold{fold}"
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, output_dir / "rbf_svm.joblib")
        pd.DataFrame(
            {
                "id": valid_ids,
                "label": labels[valid_mask],
                "score": raw_score,
                "percentile_rank": candidate_rank,
                "V110_equal_three_score": equal_three,
            }
        ).to_csv(output_dir / "valid_predictions.csv", index=False)
        row = {
            "fold": fold,
            "n": int(valid_mask.sum()),
            "V108_submission_replay_auc": baseline_auc,
            "V109_single_auc": candidate_auc,
            "V110_equal_three_auc": equal_three_auc,
            "delta": equal_three_auc - baseline_auc,
            "support_vectors": model.n_support_.tolist(),
        }
        rows.append(row)
        (output_dir / "metrics.json").write_text(
            json.dumps(row, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(row), flush=True)

    summary = {
        "version": "V109_multiscale_rbf_and_V110_equal_three",
        "folds": rows,
        "V108_submission_replay_mean": float(
            np.mean([row["V108_submission_replay_auc"] for row in rows])
        ),
        "V109_single_mean": float(np.mean([row["V109_single_auc"] for row in rows])),
        "V110_equal_three_mean": float(
            np.mean([row["V110_equal_three_auc"] for row in rows])
        ),
        "all_folds_non_decreasing": all(row["delta"] >= 0 for row in rows),
        "block_weights": BLOCK_WEIGHTS,
        "classifier": {"kernel": "rbf", "C": 10, "gamma": 0.5},
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
