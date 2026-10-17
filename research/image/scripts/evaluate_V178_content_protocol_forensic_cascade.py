#!/usr/bin/env python3
"""Evaluate V178: content/protocol-routed high-confidence forensic cascade.

V178 replays the true V168 score, applies the fixed V174 protocol MoE, and
then lets legacy domain-forensic specialists intervene only in the visual
regimes where their evidence is meaningful.  A strictly outer-trained
RINE/NPR meta expert handles square images; high-confidence NPR and RINE
specialists handle low-resolution and native-resolution detected faces.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
V168_FOLD_AUC = [0.9998, 0.997970297029703, 0.9995, 0.9967, 0.9981818181818182]
V174_FOLD_AUC = [0.9998, 0.998019801980198, 0.9995, 0.9983, 0.9983838383838384]
SQUARE_LOG_ASPECT_THRESHOLD = 0.12
NATIVE_UPSAMPLE_THRESHOLD = 448
UNCERTAINTY_HALF_WIDTH = 0.25
PROTOCOL_MAX_WEIGHT = 0.25
FORENSIC_MAX_WEIGHT = 0.10
NPR_HIGH_CONFIDENCE_RANK = 0.80
RINE_HIGH_CONFIDENCE_RANK = 0.90


def percentile_rank(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / len(values)


def read_fold_predictions(root: Path, version: str, fold: int) -> pd.DataFrame:
    return pd.read_csv(
        root / f"outputs/{version}/fold{fold}/valid_predictions.csv",
        dtype={"id": str},
    ).set_index("id")


def collect_zero_shot_feature(root: Path, version: str, ids: np.ndarray) -> np.ndarray:
    parts = [read_fold_predictions(root, version, fold) for fold in range(5)]
    frame = pd.concat(parts)
    if frame.index.has_duplicates or set(frame.index) != set(ids):
        raise RuntimeError(f"V178 {version} OOF identity mismatch")
    return frame.loc[ids, "zero_shot_logit"].to_numpy(np.float64)


def build_meta_model():
    return make_pipeline(
        StandardScaler(),
        PolynomialFeatures(degree=2, include_bias=False),
        LogisticRegression(C=1.0, max_iter=1000),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--folds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()
    root = args.project_root.resolve()

    fold_frame = pd.read_csv(
        root / "data/folds/image_folds_v1.csv", dtype={"id": str}
    ).set_index("id")
    ids = fold_frame.index.to_numpy(str)
    labels = fold_frame.label.to_numpy(np.int8)
    fold_values = fold_frame.fold.to_numpy(np.int8)
    group_values = fold_frame.group_id.astype(str).to_numpy()
    square = (
        np.abs(np.log(fold_frame.aspect_ratio.to_numpy(np.float64)))
        < SQUARE_LOG_ASPECT_THRESHOLD
    )
    low_resolution = (
        np.minimum(fold_frame.width.to_numpy(), fold_frame.height.to_numpy())
        < NATIVE_UPSAMPLE_THRESHOLD
    )
    regimes = np.where(square, "square", np.where(low_resolution, "low", "high"))

    face_cache = np.load(root / "outputs/V107_face_crop_rbf/features/competition.npz")
    face_ids = face_cache["ids"].astype(str)
    face_index = pd.Index(face_ids).get_indexer(ids)
    if (face_index < 0).any():
        raise RuntimeError("V178 face cache identity mismatch")
    face_detected = face_cache["face_detected"][face_index] > 0

    zero_shot_features = np.column_stack(
        [
            collect_zero_shot_feature(root, "V111_rine_ldm_rbf", ids),
            collect_zero_shot_feature(root, "V113_npr_rbf", ids),
        ]
    )
    if zero_shot_features.shape != (len(ids), 2) or not np.isfinite(
        zero_shot_features
    ).all():
        raise RuntimeError("V178 invalid zero-shot meta features")
    oof = pd.read_csv(
        root / "outputs/fivefold_milestones/oof_predictions.csv", dtype={"id": str}
    ).set_index("id")
    output_root = root / "outputs/V178_content_protocol_forensic_cascade"
    rows: list[dict[str, object]] = []

    for fold in sorted(args.folds):
        train_indices = np.flatnonzero(fold_values != fold)
        valid_indices = np.flatnonzero(fold_values == fold)
        if set(group_values[train_indices]) & set(group_values[valid_indices]):
            raise RuntimeError(f"Fold{fold} group leakage")

        meta_model = build_meta_model()
        if args.smoke_only:
            subset = np.concatenate(
                [
                    train_indices[labels[train_indices] == 0][:32],
                    train_indices[labels[train_indices] == 1][:32],
                ]
            )
            meta_model.fit(zero_shot_features[subset], labels[subset])
            score = meta_model.predict_proba(zero_shot_features[valid_indices[:4]])[:, 1]
            if score.shape != (4,) or not np.isfinite(score).all():
                raise RuntimeError("V178 smoke inference failed")
            print(json.dumps({"smoke": "passed", "fold": fold, "rows": len(ids)}))
            return

        meta_model.fit(zero_shot_features[train_indices], labels[train_indices])
        square_meta_rank = percentile_rank(
            meta_model.predict_proba(zero_shot_features[valid_indices])[:, 1]
        )
        valid_ids = ids[valid_indices]
        y_valid = labels[valid_indices]
        valid_regime = regimes[valid_indices]
        valid_face = face_detected[valid_indices]

        previous = read_fold_predictions(
            root, "V167_grayscale_protocol_multikernel", fold
        ).loc[valid_ids]
        protocol = read_fold_predictions(
            root, "V171_protocol_expert_cascade", fold
        ).loc[valid_ids]
        multiscale = read_fold_predictions(root, "V109_multiscale_rbf", fold).loc[
            valid_ids
        ]
        siglip_aspect = read_fold_predictions(
            root, "V155_siglip_aspect_multikernel", fold
        ).loc[valid_ids]
        rine = read_fold_predictions(root, "V111_rine_ldm_rbf", fold).loc[valid_ids]
        npr = read_fold_predictions(root, "V113_npr_rbf", fold).loc[valid_ids]

        v90_rank = percentile_rank(
            oof.loc[valid_ids, "V90_score"].to_numpy(np.float64)
        )
        mil_mean_rank = percentile_rank(
            previous.mil_mean_score.to_numpy(np.float64)
        )
        mil_route_rank = percentile_rank(
            previous.mil_route_score.to_numpy(np.float64)
        )
        v167_rank = percentile_rank(previous.score.to_numpy(np.float64))
        v168_score = (v90_rank + mil_mean_rank + mil_route_rank + v167_rank) / 4.0
        v168_auc = float(roc_auc_score(y_valid, v168_score))
        if not np.isclose(v168_auc, V168_FOLD_AUC[fold], atol=5e-13, rtol=0):
            raise RuntimeError(f"Fold{fold} V168 replay mismatch")

        protocol_rank = percentile_rank(
            protocol.protocol_expert_score.to_numpy(np.float64)
        )
        multiscale_rank = percentile_rank(multiscale.score.to_numpy(np.float64))
        siglip_aspect_rank = percentile_rank(
            siglip_aspect.score.to_numpy(np.float64)
        )
        routed_expert = np.where(
            valid_regime == "low",
            protocol_rank,
            np.where(valid_regime == "high", multiscale_rank, siglip_aspect_rank),
        )
        uncertainty_gate = np.clip(
            1.0 - np.abs(v168_score - 0.5) / UNCERTAINTY_HALF_WIDTH, 0.0, 1.0
        )
        v174_score = (
            (1.0 - PROTOCOL_MAX_WEIGHT * uncertainty_gate) * v168_score
            + PROTOCOL_MAX_WEIGHT * uncertainty_gate * routed_expert
        )
        v174_auc = float(roc_auc_score(y_valid, v174_score))
        if not np.isclose(v174_auc, V174_FOLD_AUC[fold], atol=5e-13, rtol=0):
            raise RuntimeError(f"Fold{fold} V174 replay mismatch")

        score = v174_score.copy()
        square_mask = valid_regime == "square"
        score[square_mask] = (
            (1.0 - FORENSIC_MAX_WEIGHT * uncertainty_gate[square_mask])
            * score[square_mask]
            + FORENSIC_MAX_WEIGHT
            * uncertainty_gate[square_mask]
            * square_meta_rank[square_mask]
        )
        rine_rank = percentile_rank(rine.score.to_numpy(np.float64))
        npr_rank = percentile_rank(npr.score.to_numpy(np.float64))
        npr_mask = (
            valid_face
            & (valid_regime == "low")
            & (npr_rank >= NPR_HIGH_CONFIDENCE_RANK)
        )
        rine_mask = (
            valid_face
            & (valid_regime == "high")
            & (rine_rank >= RINE_HIGH_CONFIDENCE_RANK)
        )
        for mask, expert_rank in ((npr_mask, npr_rank), (rine_mask, rine_rank)):
            score[mask] = (
                (1.0 - FORENSIC_MAX_WEIGHT * uncertainty_gate[mask]) * score[mask]
                + FORENSIC_MAX_WEIGHT * uncertainty_gate[mask] * expert_rank[mask]
            )

        auc = float(roc_auc_score(y_valid, score))
        output_dir = output_root / f"fold{fold}"
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(meta_model, output_dir / "rine_npr_square_meta.joblib")
        pd.DataFrame(
            {
                "id": valid_ids,
                "label": y_valid,
                "regime": valid_regime,
                "face_detected": valid_face.astype(np.int8),
                "V168_score": v168_score,
                "V174_score": v174_score,
                "square_meta_rank": square_meta_rank,
                "npr_rank": npr_rank,
                "rine_rank": rine_rank,
                "npr_rescue": npr_mask.astype(np.int8),
                "rine_rescue": rine_mask.astype(np.int8),
                "score": score,
            }
        ).to_csv(output_dir / "valid_predictions.csv", index=False)
        row = {
            "fold": fold,
            "V168_auc": v168_auc,
            "V174_auc": v174_auc,
            "V178_auc": auc,
            "delta_vs_V174": auc - v174_auc,
            "npr_rescue_rows": int(npr_mask.sum()),
            "rine_rescue_rows": int(rine_mask.sum()),
        }
        rows.append(row)
        (output_dir / "metrics.json").write_text(json.dumps(row, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    summary = {
        "version": "V178_content_protocol_forensic_cascade",
        "folds": rows,
        "V168_mean": float(np.mean([row["V168_auc"] for row in rows])),
        "V174_mean": float(np.mean([row["V174_auc"] for row in rows])),
        "V178_mean": float(np.mean([row["V178_auc"] for row in rows])),
        "non_decreasing_folds_vs_V174": int(
            sum(row["delta_vs_V174"] >= -5e-13 for row in rows)
        ),
        "fixed_cascade": {
            "uncertainty_half_width": UNCERTAINTY_HALF_WIDTH,
            "protocol_max_weight": PROTOCOL_MAX_WEIGHT,
            "forensic_max_weight": FORENSIC_MAX_WEIGHT,
            "npr_high_confidence_rank": NPR_HIGH_CONFIDENCE_RANK,
            "rine_high_confidence_rank": RINE_HIGH_CONFIDENCE_RANK,
        },
        "audit": {
            "true_V168_and_V174_entries_replayed": True,
            "outer_validation_rows_used_for_training": 0,
            "meta_model_fit_on_outer_training_rows_only": True,
            "zero_shot_meta_features_label_independent": True,
            "group_leakage": False,
            "id_filename_or_file_size_feature": False,
            "fold_specific_routes_weights_or_thresholds": False,
            "external_download_for_V178": False,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "fivefold_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
