#!/usr/bin/env python3
"""Call original training-feature extractors without unrelated historical CV reports."""
import argparse
import importlib
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--project-root", type=Path, required=True)
    p.add_argument("--model", choices=["rine", "npr"], required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.source_root / "scripts"))
    name, relative, constant = {
        "rine": ("evaluate_V111_rine_ldm_rbf", ".cache/rine/model_ldm_trainable.pth", "RINE_SHA256"),
        "npr": ("evaluate_V113_npr_rbf", ".cache/npr/model_epoch_last_3090.pth", "NPR_SHA256"),
    }[a.model]
    mod = importlib.import_module(name)
    root = a.project_root.resolve()
    checkpoint = root / relative
    actual = mod.sha256_file(checkpoint)
    if actual != getattr(mod, constant):
        raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")
    mod.extract_features(root, checkpoint, actual)


if __name__ == "__main__":
    main()
