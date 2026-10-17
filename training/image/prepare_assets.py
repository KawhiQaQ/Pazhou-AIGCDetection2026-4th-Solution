#!/usr/bin/env python3
"""Pack newly trained weights with original exporter, without historical score claims."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f"Refusing to overwrite assets: {a.output}")
    subprocess.run([sys.executable, str(a.source_root / "docker/v178_image/prepare_assets.py"), "--project-root", str(a.project_root), "--output", str(a.output)], check=True)
    manifest = a.output / "manifest.json"
    info = json.loads(manifest.read_text())
    csv = a.project_root / "outputs/V178_full/V178_image_submission.csv"
    info["historical_reference_submission_sha256"] = info.pop("submission_sha256")
    info["submission_sha256"] = hashlib.sha256(csv.read_bytes()).hexdigest()
    info["version"] = "V178_retrained_detector_assets"
    info["training_scope"] = "original_full_data_detector_recipes_from_public_pretrained_encoders"
    info["leaderboard_score_revalidated"] = False
    manifest.write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    main()
