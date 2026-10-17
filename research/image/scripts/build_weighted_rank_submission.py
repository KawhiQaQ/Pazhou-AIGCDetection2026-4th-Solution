#!/usr/bin/env python3
"""Build and package a fixed weighted percentile-rank ensemble from full predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load(resolve(root, str(args.config)).read_text(encoding="utf-8"))
    template = pd.read_csv(resolve(root, config["submission_example_csv"]), dtype={"id": str})
    if list(template.columns) != ["id", "score"] or len(template) != 20_000:
        raise RuntimeError("unexpected submission template")
    total_weight = float(sum(float(member["weight"]) for member in config["members"]))
    if not np.isclose(total_weight, 1.0, atol=1e-12):
        raise RuntimeError(f"ensemble weights do not sum to one: {total_weight}")
    blend = np.zeros(len(template), dtype=np.float64)
    records: dict[str, dict[str, object]] = {}
    for member in config["members"]:
        name = str(member["name"])
        weight = float(member["weight"])
        path = resolve(root, str(member["path"]))
        frame = pd.read_csv(path, dtype={"id": str})
        if list(frame.columns) != ["id", "score"] or len(frame) != len(template):
            raise RuntimeError(f"invalid member: {name}")
        if frame.id.astype(str).tolist() != template.id.astype(str).tolist():
            raise RuntimeError(f"member ID/order mismatch: {name}")
        scores = frame.score.to_numpy(dtype=np.float64)
        if not np.isfinite(scores).all():
            raise RuntimeError(f"non-finite scores: {name}")
        ranks = rankdata(scores, method="average") / len(scores)
        blend += weight * ranks
        records[name] = {"path": str(path), "weight": weight, "sha256": sha256(path)}
    output = resolve(root, config["output_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": template.id.astype(str), "score": blend}).to_csv(output, index=False)
    written = pd.read_csv(output, dtype={"id": str})
    if written.id.astype(str).tolist() != template.id.astype(str).tolist():
        raise RuntimeError("written submission ID/order mismatch")
    if not np.isfinite(written.score).all() or not written.score.between(0, 1).all():
        raise RuntimeError("written submission contains invalid scores")
    zip_path = output.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(output, arcname=output.name)
    with zipfile.ZipFile(zip_path) as archive:
        if archive.namelist() != [output.name] or archive.read(output.name) != output.read_bytes():
            raise RuntimeError("ZIP round-trip verification failed")
    manifest = {
        "experiment": str(config["experiment"]),
        "metric": "ROC-AUC",
        "method": str(config["method"]),
        "weights_source": str(config["weights_source"]),
        "cv_folds": [float(value) for value in config["cv_folds"]],
        "cv_mean_auc": float(config["cv_mean_auc"]),
        "members": records,
        "submission": {
            "path": str(output),
            "sha256": sha256(output),
            "rows": int(len(written)),
            "score_unique": int(written.score.nunique()),
            "score_min": float(written.score.min()),
            "score_max": float(written.score.max()),
        },
        "package": {"path": str(zip_path), "sha256": sha256(zip_path)},
    }
    output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
