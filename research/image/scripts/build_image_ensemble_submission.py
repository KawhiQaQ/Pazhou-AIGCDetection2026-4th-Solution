#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata


MEMBERS = {
    "V20": ("outputs/submissions/V20_clip_adapter_patch_mil_336_full_e5.csv", 0.125),
    "V29": (
        "outputs/submissions/V29_clip_adapter_multidepth_patch_external_full_e7.csv",
        0.475,
    ),
    "V31": (
        "outputs/submissions/V31_clip_adapter_paired_external_full_e11.csv",
        0.400,
    ),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project-root", type=Path, default=Path(__file__).resolve().parents[1]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/submissions/V33_image_rank_ensemble.csv"),
    )
    args = parser.parse_args()

    root = args.project_root.resolve()
    template_path = root / "data/competition/image/image_submission_example.csv"
    template = pd.read_csv(template_path)
    if list(template.columns) != ["id", "score"] or len(template) != 20_000:
        raise RuntimeError("unexpected submission template schema or row count")
    if template.id.duplicated().any():
        raise RuntimeError("submission template IDs are duplicated")

    blend = np.zeros(len(template), dtype=np.float64)
    member_records = {}
    for name, (relative_path, weight) in MEMBERS.items():
        path = root / relative_path
        frame = pd.read_csv(path)
        if list(frame.columns) != ["id", "score"] or len(frame) != len(template):
            raise RuntimeError(f"invalid member schema or row count: {name}")
        if frame.id.astype(str).tolist() != template.id.astype(str).tolist():
            raise RuntimeError(f"member ID order differs from template: {name}")
        scores = frame.score.to_numpy(dtype=np.float64)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"member contains non-finite scores: {name}")
        ranks = rankdata(scores, method="average") / len(scores)
        blend += weight * ranks
        member_records[name] = {
            "path": str(path),
            "weight": weight,
            "sha256": sha256(path),
            "raw_score_min": float(scores.min()),
            "raw_score_max": float(scores.max()),
            "raw_score_unique": int(pd.Series(scores).nunique()),
        }

    if not np.isclose(sum(weight for _, weight in MEMBERS.values()), 1.0):
        raise RuntimeError("ensemble weights do not sum to one")
    submission = pd.DataFrame({"id": template.id.astype(str), "score": blend})
    if not np.isfinite(submission.score).all() or not submission.score.between(
        0.0, 1.0, inclusive="both"
    ).all():
        raise RuntimeError("blended scores failed range/finiteness validation")

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output, index=False)
    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output, arcname=output.name)

    manifest = {
        "experiment": "V33_image_rank_ensemble",
        "metric": "ROC-AUC",
        "method": "percentile-rank averaging",
        "weights_source": "fixed three-fold OOF constrained grid",
        "cv_mean_auc": 0.9895054455445544,
        "cv_lofo_mean_held_auc": 0.9893074257425742,
        "members": member_records,
        "submission": {
            "path": str(output),
            "sha256": sha256(output),
            "rows": int(len(submission)),
            "columns": submission.columns.tolist(),
            "score_min": float(submission.score.min()),
            "score_max": float(submission.score.max()),
            "score_unique": int(submission.score.nunique()),
        },
        "package": {
            "path": str(zip_path),
            "sha256": sha256(zip_path),
            "members": [output.name],
        },
    }
    manifest_path = output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
