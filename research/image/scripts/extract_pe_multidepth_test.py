#!/usr/bin/env python3
"""Extract frozen PE multi-depth patch statistics for the 20k image test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from train_pe_core_linear import PROJECT_ROOT, load_encoder, resolve, seed_everything
from train_pe_core_multidepth_residual import extract_statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = yaml.safe_load(resolve(root, str(args.config)).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    data_config = config["data"]
    model_config = config["model"]
    test = pd.read_csv(resolve(root, data_config["test_csv"]), dtype={"id": str})
    template = pd.read_csv(resolve(root, data_config["submission_example_csv"]), dtype={"id": str})
    if test.id.astype(str).tolist() != template.id.astype(str).tolist() or len(test) != 20_000:
        raise RuntimeError("test IDs/order differ from submission template")
    image_dir = resolve(root, data_config["test_image_dir"])
    rows = [
        {"id": str(row.id), "label": 0, "path": image_dir / str(row.id)}
        for row in test.itertuples(index=False)
    ]
    if any(not Path(str(row["path"])).is_file() for row in rows):
        raise RuntimeError("test manifest contains missing images")
    cache_path = resolve(root, data_config["test_multidepth_features"])
    layers = [int(value) for value in model_config["layer_indices"]]
    expected_hash = str(model_config["checkpoint_sha256"])
    if cache_path.is_file():
        cached = np.load(cache_path)
        if str(cached["checkpoint_sha256"].item()) != expected_hash:
            raise RuntimeError("stale test feature encoder")
        if not np.array_equal(cached["layer_indices"], np.asarray(layers)):
            raise RuntimeError("stale test feature layers")
        if cached["ids"].astype(str).tolist() != template.id.astype(str).tolist():
            raise RuntimeError("stale test feature IDs")
        print(json.dumps({"cache": str(cache_path), "rows": len(cached["ids"]), "reused": True}))
        return
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _, actual_hash = load_encoder(config, root, device)
    if actual_hash != expected_hash:
        raise RuntimeError("encoder hash changed while loading")
    features, labels, ids = extract_statistics(
        encoder,
        rows,
        layers,
        int(data_config["input_size"]),
        int(data_config["batch_size"]),
        int(data_config["workers"]),
        device,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features,
        labels=labels,
        ids=ids,
        checkpoint_sha256=np.asarray(actual_hash),
        layer_indices=np.asarray(layers),
        input_size=np.asarray(int(data_config["input_size"])),
    )
    print(json.dumps({"cache": str(cache_path), "rows": len(ids), "reused": False}))


if __name__ == "__main__":
    main()
