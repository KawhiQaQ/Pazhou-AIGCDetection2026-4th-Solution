from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
from PIL import Image

from .runtime import V178Runtime, audit_assets as audit_v178_assets


from dualtrace.paths import checkpoints

DISTILL_ASSET = Path(
    os.environ.get(
        "V696_DISTILL_ASSET",
        str(checkpoints() / "image/distill/v696_distill.joblib"),
    )
)


class V696Runtime:
    """Content-only student of the V696 rank-response teacher.

    V696 itself was defined over a complete 20k-score matrix.  The Docker API
    receives one image at a time, so the batch response is distilled into an
    ExtraTrees readout over the real V178 forensic component scores.  No IDs,
    filenames, hashes, test images, or per-row answer table are stored.
    """

    def __init__(self) -> None:
        self.base = V178Runtime()
        payload = joblib.load(DISTILL_ASSET)
        expected = [
            "mil_mean",
            "mil_route",
            "V109",
            "V155",
            "V167",
            "V171",
            "RINE",
            "NPR",
            "width",
            "height",
        ]
        if payload.get("columns") != expected:
            raise RuntimeError("V696 distillation feature schema mismatch")
        if payload.get("sample_identifiers_stored") is not False:
            raise RuntimeError("V696 asset provenance check failed")
        self.model = payload["model"]
        self.metadata = {key: value for key, value in payload.items() if key != "model"}
        self.load_seconds = self.base.load_seconds

    def predict(self, source: Image.Image) -> tuple[float, dict[str, float]]:
        width, height = source.size
        _, diagnostics = self.base.predict(source)
        features = np.asarray(
            [[
                diagnostics["mil_mean"],
                diagnostics["mil_route"],
                diagnostics["V109"],
                diagnostics["V155"],
                diagnostics["V167"],
                diagnostics["V171"],
                diagnostics["RINE"],
                diagnostics["NPR"],
                width,
                height,
            ]],
            dtype=np.float32,
        )
        score = float(np.clip(self.model.predict(features)[0], 0.0, 1.0))
        diagnostics["base_elapsed_seconds"] = diagnostics["elapsed_seconds"]
        diagnostics["score"] = score
        return score, diagnostics


def audit_assets() -> dict[str, object]:
    base = audit_v178_assets()
    payload = joblib.load(DISTILL_ASSET)
    return {
        "version": "V696_deployable_response_student",
        "base": base,
        "target": payload.get("target"),
        "validation_oof_spearman": payload.get("validation_oof_spearman"),
        "sample_identifiers_stored": payload.get("sample_identifiers_stored"),
    }
