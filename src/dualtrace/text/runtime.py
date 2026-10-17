from __future__ import annotations

import os
from pathlib import Path
import time

import joblib
import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


from dualtrace.paths import checkpoints

ASSETS = Path(os.environ.get("TV388_ASSETS", str(checkpoints() / "text")))


def ecdf(value: float, sorted_values: np.ndarray) -> float:
    left = int(np.searchsorted(sorted_values, value, side="left"))
    right = int(np.searchsorted(sorted_values, value, side="right"))
    return float(np.clip((left + right + 1) / (2.0 * len(sorted_values)), 0.0, 1.0))


class TV388Runtime:
    def __init__(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("TV388 requires an NVIDIA CUDA GPU")
        started = time.time()
        adapter = ASSETS / "tv226_adapter"
        self.tokenizer = AutoTokenizer.from_pretrained(adapter, local_files_only=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        base = AutoModelForSequenceClassification.from_pretrained(
            ASSETS / "qwen2.5_1.5b_instruct",
            local_files_only=True,
            num_labels=1,
            problem_type="regression",
            torch_dtype=torch.bfloat16,
            device_map={"": "cuda"},
        )
        base.config.pad_token_id = base.config.eos_token_id
        self.model = PeftModel.from_pretrained(base, adapter).eval()
        self.qwen_ecdf = np.load(ASSETS / "tv226_raw_ecdf.npy").astype(np.float64)
        payload = joblib.load(ASSETS / "tv388_distill.joblib")
        if payload.get("sample_identifiers_stored") is not False:
            raise RuntimeError("TV388 asset provenance check failed")
        self.vectorizer = payload["vectorizer"]
        self.ridge = payload["ridge"]
        self.lexical_ecdf = np.asarray(payload["lexical_ecdf"], np.float64)
        self.qwen_weight = float(payload["qwen_weight"])
        self.lexical_weight = float(payload["lexical_weight"])
        self.metadata = {key: value for key, value in payload.items() if key not in {"vectorizer", "ridge", "lexical_ecdf"}}
        self.load_seconds = time.time() - started

    @torch.inference_mode()
    def predict(self, text: str) -> tuple[float, dict[str, float]]:
        started = time.time()
        value = text or ""
        encoded = self.tokenizer(
            value,
            truncation=True,
            max_length=2048,
            padding=False,
            return_tensors="pt",
        )
        if encoded["input_ids"].shape[1] == 0:
            token_id = self.tokenizer.eos_token_id
            if token_id is None:
                raise RuntimeError("tokenizer has no fallback token for empty text")
            encoded = {
                "input_ids": torch.tensor([[token_id]], dtype=torch.long),
                "attention_mask": torch.ones((1, 1), dtype=torch.long),
            }
        encoded = {key: tensor.to("cuda") for key, tensor in encoded.items()}
        raw = float(self.model(**encoded, use_cache=False).logits.view(-1)[0].float().item())
        qwen_rank = ecdf(raw, self.qwen_ecdf)
        lexical_raw = float(self.ridge.predict(self.vectorizer.transform([value]))[0])
        lexical_rank = ecdf(lexical_raw, self.lexical_ecdf)
        score = float(np.clip(
            self.qwen_weight * qwen_rank + self.lexical_weight * lexical_rank,
            0.0,
            1.0,
        ))
        return score, {
            "elapsed_seconds": time.time() - started,
            "qwen_rank": qwen_rank,
            "lexical_rank": lexical_rank,
        }


def audit_assets() -> dict[str, object]:
    import json
    return json.loads((ASSETS / "manifest.json").read_text())
