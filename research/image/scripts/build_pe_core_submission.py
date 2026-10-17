#!/usr/bin/env python3
"""Train the fixed V47 linear probe on all labeled rows and infer the test set."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from train_pe_core_linear import (  # noqa: E402
    build_transform,
    load_encoder,
    load_or_extract_cache,
    resolve,
    seed_everything,
    sha256_file,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/V47_pe_core_g14_linear_full_e100.yaml")
    )
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    config = yaml.safe_load(resolve(root, str(args.config)).read_text(encoding="utf-8"))
    seed_everything(int(config["seed"]))
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["train"]
    assert isinstance(model_config, dict)
    assert isinstance(data_config, dict)
    assert isinstance(train_config, dict)

    labels = pd.read_csv(resolve(root, str(data_config["labels_csv"])))
    external = pd.read_csv(resolve(root, str(data_config["external_manifest_csv"])))
    test = pd.read_csv(resolve(root, str(data_config["test_csv"])))
    template = pd.read_csv(resolve(root, str(data_config["submission_example_csv"])))
    if list(template.columns) != ["id", "score"] or len(template) != 20_000:
        raise RuntimeError("unexpected submission template")
    if test.id.astype(str).tolist() != template.id.astype(str).tolist():
        raise RuntimeError("test IDs/order differ from submission template")
    if labels.id.duplicated().any() or set(labels.label.unique()) != {0, 1}:
        raise RuntimeError("invalid full training labels")

    image_dir = resolve(root, str(data_config["image_dir"]))
    external_root = resolve(root, str(data_config["external_dir"])).resolve()
    test_image_dir = resolve(root, str(data_config["test_image_dir"]))
    if "competition" in external_root.parts:
        raise RuntimeError("external feature source resolves inside competition data")
    competition_rows = [
        {"id": str(row.id), "label": int(row.label), "path": image_dir / str(row.id)}
        for row in labels.itertuples(index=False)
    ]
    external_rows = [
        {
            "id": str(row.sample_key),
            "label": int(row.label),
            "path": external_root / str(row.relative_path),
        }
        for row in external.itertuples(index=False)
    ]
    test_rows = [
        {"id": str(row.id), "label": 0, "path": test_image_dir / str(row.id)}
        for row in test.itertuples(index=False)
    ]
    if any(not Path(str(row["path"])).is_file() for row in competition_rows + external_rows + test_rows):
        raise RuntimeError("full V47 manifest contains missing images")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = int(data_config["input_size"])
    cache_dir = resolve(root, str(data_config["feature_cache_dir"]))
    competition_cache = cache_dir / "competition_center_features.npz"
    external_cache = cache_dir / "openfake_center_features.npz"
    test_cache = cache_dir / "test_center_features.npz"
    checkpoint = resolve(root, str(model_config["checkpoint"]))
    expected_hash = str(model_config["checkpoint_sha256"])
    if test_cache.is_file() and competition_cache.is_file() and external_cache.is_file():
        checkpoint_hash = sha256_file(checkpoint)
        if checkpoint_hash != expected_hash:
            raise RuntimeError("PE-Core checkpoint hash mismatch")
        encoder = None
    else:
        encoder, checkpoint, checkpoint_hash = load_encoder(config, root, device)
    transform = build_transform(input_size)
    common = {
        "batch_size": int(data_config["batch_size"]),
        "workers": int(data_config["workers"]),
        "device": device,
        "checkpoint_sha256": checkpoint_hash,
        "input_size": input_size,
    }
    competition_features, competition_labels, _ = load_or_extract_cache(
        competition_cache, competition_rows, encoder, transform, **common
    )
    external_features, external_labels, _ = load_or_extract_cache(
        external_cache, external_rows, encoder, transform, **common
    )
    test_features, _, test_ids = load_or_extract_cache(
        test_cache, test_rows, encoder, transform, **common
    )
    del encoder
    torch.cuda.empty_cache()

    output_dir = resolve(root, str(config["output_dir"]))
    output_dir.mkdir(parents=True, exist_ok=True)
    head_path = output_dir / "linear_head.pth"
    head = nn.Linear(competition_features.shape[1], 1).to(device)
    history: list[dict[str, float | int]] = []
    if args.predict_only:
        state = torch.load(head_path, map_location="cpu", weights_only=True)
        result = head.load_state_dict(state["state_dict"], strict=True)
        if result.missing_keys or result.unexpected_keys:
            raise RuntimeError("invalid V47 full linear head")
    else:
        nn.init.zeros_(head.weight)
        nn.init.zeros_(head.bias)
        train_features = np.concatenate(
            [external_features.astype(np.float32), competition_features.astype(np.float32)]
        )
        train_labels = np.concatenate([external_labels, competition_labels])
        loader = DataLoader(
            TensorDataset(torch.from_numpy(train_features), torch.from_numpy(train_labels)),
            batch_size=int(train_config["batch_size"]),
            shuffle=True,
            generator=torch.Generator().manual_seed(int(config["seed"])),
            drop_last=False,
        )
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=float(train_config["lr"]),
            weight_decay=float(train_config["weight_decay"]),
        )
        for epoch in range(1, int(train_config["epochs"]) + 1):
            loss_sum = 0.0
            rows_seen = 0
            for batch_features, batch_labels in loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.binary_cross_entropy_with_logits(
                    head(batch_features).squeeze(1), batch_labels
                )
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch_labels)
                rows_seen += len(batch_labels)
            if epoch == 1 or epoch % 10 == 0 or epoch == int(train_config["epochs"]):
                record = {"epoch": epoch, "train_bce": loss_sum / rows_seen}
                history.append(record)
                print(json.dumps(record), flush=True)
        torch.save(
            {
                "format": "aigc-pe-core-linear-v1",
                "backbone": str(model_config["backbone"]),
                "checkpoint_sha256": checkpoint_hash,
                "state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
            },
            head_path,
        )

    head.eval()
    score_parts: list[np.ndarray] = []
    for start in range(0, len(test_features), 1024):
        batch = torch.from_numpy(
            test_features[start : start + 1024].astype(np.float32)
        ).to(device)
        with torch.no_grad():
            score_parts.append(torch.sigmoid(head(batch).squeeze(1)).cpu().numpy())
    scores = np.concatenate(score_parts).astype(np.float64)
    if test_ids.astype(str).tolist() != template.id.astype(str).tolist():
        raise RuntimeError("cached V47 test IDs/order differ from template")
    submission = pd.DataFrame({"id": template.id.astype(str), "score": scores})
    if len(submission) != 20_000 or not np.isfinite(scores).all():
        raise RuntimeError("invalid V47 full submission")
    submission_path = resolve(root, str(config["submission_path"]))
    submission_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(submission_path, index=False)
    manifest = {
        "experiment": str(config["experiment"]),
        "cv_reference": config["cv_reference"],
        "encoder": str(model_config["backbone"]),
        "encoder_checkpoint_sha256": checkpoint_hash,
        "encoder_frozen": True,
        "embedding_l2_normalized": True,
        "external_rows": int(len(external_rows)),
        "competition_training_rows": int(len(competition_rows)),
        "test_rows": int(len(submission)),
        "validation_rows_used_for_model_selection": 0,
        "submission": str(submission_path),
        "sha256": sha256(submission_path),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "score_unique": int(pd.Series(scores).nunique()),
        "history": history,
    }
    submission_path.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
