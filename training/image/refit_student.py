#!/usr/bin/env python3
"""Refit the V696 frozen-feature deployment student, NOT image training from scratch.

Defaults use locally preserved 20k component predictions and teacher scores.
IDs are used only to align training rows, never as model input or checkpoint data.
Only load the trusted project checkpoint; joblib is executable serialization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import ExtraTreesRegressor

ROOT = Path(__file__).resolve().parents[2]
COLUMNS = ["mil_mean", "mil_route", "V109", "V155", "V167", "V171", "RINE", "NPR", "width", "height"]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def read_frame(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"id": str})
    if not set(["id", *columns]).issubset(frame.columns):
        raise ValueError(f"Missing columns in {path}: expected {columns}")
    if len(frame) != 20_000 or frame.id.isna().any() or frame.id.duplicated().any():
        raise ValueError(f"Expected exactly 20,000 unique nonempty IDs: {path}")
    return frame.set_index("id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", type=Path, default=ROOT / ".local/legacy")
    parser.add_argument("--reference", type=Path, default=ROOT / ".local/checkpoints/image/distill/v696_distill.joblib")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("v696_student.json"))
    parser.add_argument("--output", type=Path, default=ROOT / "runs/image_refit")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--components", type=Path)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--teacher", type=Path)
    parser.add_argument("--require-reference-parity", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if sklearn.__version__ != "1.9.0":
        raise RuntimeError(f"Exact recipe requires scikit-learn 1.9.0; got {sklearn.__version__}")
    params = json.loads(args.config.read_text())
    reference = {}
    if args.reference.is_file():
        reference = joblib.load(args.reference)
        if reference.get("columns") != COLUMNS or reference.get("sample_identifiers_stored") is not False:
            raise ValueError("Unexpected reference checkpoint schema")
        if reference["model"].get_params() != params:
            raise ValueError("Config differs from recovered original ExtraTrees parameters")
    elif args.require_reference_parity:
        raise FileNotFoundError(f"Parity verification requires the original checkpoint: {args.reference}")
    base = args.legacy_root / "workspaces/semifinal_image/outputs/V178_full"
    inputs = {
        "components": base / "test_component_predictions.csv",
        "geometry": base / "test_geometry.csv",
        "teacher": args.legacy_root / "submissions/V696_semifinal_image_submission.csv",
    }
    for name in inputs:
        override = getattr(args, name)
        if override is not None:
            inputs[name] = override
    component = read_frame(inputs["components"], COLUMNS[:8])
    geometry = read_frame(inputs["geometry"], COLUMNS[8:])
    teacher = read_frame(inputs["teacher"], ["score"])
    if set(component.index) != set(teacher.index) or set(geometry.index) != set(teacher.index):
        raise ValueError("Input ID sets differ")
    features = pd.concat([component.loc[teacher.index, COLUMNS[:8]], geometry.loc[teacher.index, COLUMNS[8:]]], axis=1)
    x = features[COLUMNS].to_numpy(np.float32)
    y = teacher.score.to_numpy(np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError("Nonfinite features or invalid teacher scores")
    if np.any(x[:, -2:] <= 0):
        raise ValueError("Invalid image dimensions")
    audit = {"training_mode": "frozen_feature_teacher_distillation", "rows": len(x),
             "columns": COLUMNS, "sklearn_version": sklearn.__version__, "parameters": params,
             "inputs": {name: {"sha256": digest(path), "bytes": path.stat().st_size} for name, path in inputs.items()},
             "reference_sha256": digest(args.reference) if reference else None, "pseudo_labels_used": True,
             "image_pixels_used": False, "trainable_visual_backbone": False}
    if args.audit_only:
        print(json.dumps(audit, indent=2))
        return
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose another --output")
    start = time.monotonic()
    model = ExtraTreesRegressor(**params).fit(x, y)
    new_score = model.predict(x)
    old_score = reference["model"].predict(x) if reference else None
    difference = np.abs(new_score - old_score) if reference else None
    audit.update({"seconds": time.monotonic() - start,
                  "training_spearman": float(spearmanr(new_score, y).statistic),
                  "max_absolute_prediction_difference_to_original": float(difference.max()) if reference else None,
                  "mean_absolute_prediction_difference_to_original": float(difference.mean()) if reference else None,
                  "prediction_parity_at_1e-10": bool(np.allclose(new_score, old_score, atol=1e-10, rtol=0)) if reference else None,
                  "historical_oof_spearman_not_rerun": reference.get("validation_oof_spearman"),
                  "generalization_revalidated": False})
    output.mkdir(parents=True)
    payload = {"model": model, "columns": COLUMNS, "target": "V696",
               "pseudo_labels_used": True, "sample_identifiers_stored": False,
               "training_spearman": audit["training_spearman"],
               "historical_validation_oof_spearman": reference.get("validation_oof_spearman"),
               "validation_oof_spearman": None,
               "training_mode": "frozen_feature_teacher_distillation"}
    joblib.dump(payload, output / "v696_distill.joblib", compress=3)
    audit["checkpoint_sha256"] = digest(output / "v696_distill.joblib")
    (output / "report.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    if args.require_reference_parity and not audit["prediction_parity_at_1e-10"]:
        raise RuntimeError("Refit completed but differs from original; report retained; do not claim exact reproduction")


if __name__ == "__main__":
    main()
