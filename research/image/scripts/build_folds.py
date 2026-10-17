#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from aigc_detection.folds import (  # noqa: E402
    assign_balanced_group_folds,
    image_near_duplicate_groups,
    text_near_duplicate_groups,
)


def audit_frame(frame: pd.DataFrame, balance_column: str) -> list[dict[str, float | int]]:
    result = []
    for fold, part in frame.groupby("fold", sort=True):
        result.append(
            {
                "fold": int(fold),
                "n": int(len(part)),
                "n_negative": int((part.label == 0).sum()),
                "n_positive": int((part.label == 1).sum()),
                "balance_median": float(part[balance_column].median()),
                "balance_mean": float(part[balance_column].mean()),
                "n_groups": int(part.group_id.nunique()),
            }
        )
    return result


def build_text(project_root: Path, output_dir: Path, threshold: float) -> dict[str, object]:
    source = project_root / "data/competition/text/text_sample_data.csv"
    frame = pd.read_csv(source)
    frame["char_length"] = frame.text.fillna("").str.len()
    groups, grouping_audit = text_near_duplicate_groups(
        frame.text.fillna("").tolist(), cosine_threshold=threshold
    )
    folds, fold_config = assign_balanced_group_folds(
        frame.label.to_numpy(), groups, np.log1p(frame.char_length.to_numpy())
    )
    frame["group_id"] = groups
    frame["fold"] = folds
    output = frame[["id", "label", "fold", "group_id", "char_length"]]
    output.to_csv(output_dir / "text_folds_v1.csv", index=False)
    return {
        "source": str(source),
        "grouping": grouping_audit,
        "fold_config": fold_config,
        "folds": audit_frame(output, "char_length"),
    }


def build_image(project_root: Path, output_dir: Path, phash_distance: int) -> dict[str, object]:
    source = project_root / "data/competition/image/image_sample_data.csv"
    image_dir = project_root / "data/competition/image/data/image_sample_data"
    frame = pd.read_csv(source)
    paths = [image_dir / image_id for image_id in frame.id]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} images; first={missing[0]}")
    groups, metadata, grouping_audit = image_near_duplicate_groups(paths, phash_distance=phash_distance)
    folds, fold_config = assign_balanced_group_folds(
        frame.label.to_numpy(), groups, np.log1p(metadata.pixels.to_numpy())
    )
    output = pd.concat([frame.reset_index(drop=True), metadata], axis=1)
    output["group_id"] = groups
    output["fold"] = folds
    columns = [
        "id",
        "label",
        "fold",
        "group_id",
        "width",
        "height",
        "pixels",
        "aspect_ratio",
        "file_bytes",
        "sha256",
        "phash64",
    ]
    output[columns].to_csv(output_dir / "image_folds_v1.csv", index=False)
    return {
        "source": str(source),
        "grouping": grouping_audit,
        "fold_config": fold_config,
        "folds": audit_frame(output, "pixels"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--task", choices=("text", "image", "all"), default="all")
    parser.add_argument("--text-threshold", type=float, default=0.93)
    parser.add_argument("--phash-distance", type=int, default=4)
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    output_dir = project_root / "data/folds"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit: dict[str, object] = {}
    if args.task in ("text", "all"):
        audit["text"] = build_text(project_root, output_dir, args.text_threshold)
    if args.task in ("image", "all"):
        audit["image"] = build_image(project_root, output_dir, args.phash_distance)
    audit_path = output_dir / "fold_audit_v1.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
