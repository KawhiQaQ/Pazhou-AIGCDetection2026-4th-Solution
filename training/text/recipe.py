"""Mechanically retained TV225/TV226 training recipe; no historical evaluation imports.

Original functions are preserved except train function names to avoid collision.
This module requires CUDA. The lightweight CLI validates assets before importing it.
"""
from __future__ import annotations
import hashlib
import json
import math
import random
import re
import time
import unicodedata
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification
PUNCTUATION = set("，。！？；：、,.!?;:…—-()（）[]【】《》<>\"'“”‘’{}")
ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class TextRows(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.text = frame.text.fillna("").astype(str).tolist()
        self.target = frame.edit_target.to_numpy(np.float32)
        counts = frame.role.value_counts().to_dict()
        weight = frame.role.map({role: len(frame) / (3.0 * count) for role, count in counts.items()})
        self.weight = weight.to_numpy(np.float32)

    def __len__(self) -> int:
        return len(self.text)

    def __getitem__(self, index: int):
        return self.text[index], self.target[index], self.weight[index]

class RegressionCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        text, target, weight = zip(*rows, strict=True)
        encoded = self.tokenizer(
            list(text),
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=True,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"],
            encoded["attention_mask"],
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )

def load_model(model_dir: Path):
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        num_labels=1,
        problem_type="regression",
        dtype=torch.bfloat16,
        device_map={"": "cuda"},
    )
    model.config.pad_token_id = model.config.eos_token_id
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        modules_to_save=["score"],
        bias="none",
    )
    model = get_peft_model(model, config)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model

def pretrain(model, frame: pd.DataFrame, tokenizer, args, output: Path) -> list[dict]:
    dataset = TextRows(frame)
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
        pin_memory=True,
        collate_fn=RegressionCollator(tokenizer, args.max_length),
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    updates = math.ceil(len(loader) / args.accumulation)
    warmup = max(1, int(round(0.03 * updates)))

    def schedule(step: int) -> float:
        return min((step + 1) / warmup, 1.0)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, schedule)
    history = []
    started = time.monotonic()
    running_weighted = 0.0
    running_weight = 0.0
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        input_ids, attention, targets, weights = [value.to("cuda", non_blocking=True) for value in batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            score = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.view(-1)
            per_row = torch.nn.functional.mse_loss(score.float(), targets, reduction="none")
            loss = (per_row * weights).sum() / weights.sum() / args.accumulation
        if not torch.isfinite(loss):
            raise RuntimeError(f"TV225 non-finite loss at step {step}")
        loss.backward()
        running_weighted += float((per_row.detach() * weights).sum())
        running_weight += float(weights.sum())
        if (step + 1) % args.accumulation == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update = math.ceil((step + 1) / args.accumulation)
            if update == 1 or update % args.log_every == 0 or update == updates:
                row = {
                    "update": update,
                    "updates": updates,
                    "weighted_mse": running_weighted / max(running_weight, 1e-12),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }
                history.append(row)
                print(json.dumps(row), flush=True)
            if update % args.save_every == 0 and update < updates:
                rolling = output / "rolling_adapter"
                rolling.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(rolling)
    return history

def stable_seed(text: str, salt: str) -> int:
    digest = hashlib.sha256((salt + "\0" + text).encode("utf-8", errors="replace")).digest()
    return int.from_bytes(digest[:8], "big")

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(char for char in text if char not in ZERO_WIDTH and unicodedata.category(char) != "Cf")
    text = re.sub(r"[\t\r\f\v\u00a0\u3000]+", " ", text)
    text = re.sub(r"[ ]{2,}", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def degrade_text(text: str, salt: str = "tv200") -> str:
    """A deterministic social/OCR-like view that preserves textual meaning."""
    text = normalize_text(text)
    generator = np.random.default_rng(stable_seed(text, salt))
    output: list[str] = []
    for char in text:
        if char in PUNCTUATION and generator.random() < 0.35:
            if generator.random() < 0.25:
                output.append(" ")
            continue
        output.append(char)
        if "\u4e00" <= char <= "\u9fff" and generator.random() < 0.006:
            output.append(" ")
    result = "".join(output)
    result = re.sub(r"[ ]{2,}", " ", result)
    return result.strip()

def shuffle_sentences(text: str) -> str:
    text = str(text)
    parts = [part for part in re.findall(r".*?[。！？!?；;](?:[\"”’》）】])?|.+$", text, re.S) if part]
    if len(parts) < 3:
        return text
    generator = np.random.default_rng(stable_seed(text, "shuffle"))
    order = generator.permutation(len(parts))
    return "".join(parts[index] for index in order)

def truncate_middle(text: str) -> str:
    text = str(text)
    if len(text) < 96:
        return text
    keep = max(64, int(round(len(text) * 0.65)))
    head = keep // 2
    return text[:head] + text[-(keep - head) :]

def mix_insert(base: str, inserted: str) -> str:
    base = str(base)
    inserted = str(inserted)
    if not base:
        return inserted
    if not inserted:
        return base
    base_head = max(1, len(base) // 3)
    base_tail = max(base_head + 1, 2 * len(base) // 3)
    insert_head = max(0, len(inserted) // 4)
    insert_tail = max(insert_head + 1, 3 * len(inserted) // 4)
    return base[:base_head] + inserted[insert_head:insert_tail] + base[base_tail:]

def hybrid_stress(frame: pd.DataFrame) -> pd.DataFrame:
    positive = frame.loc[frame.label == 1].sort_values("text", key=lambda x: x.str.len()).reset_index(drop=True)
    negative = frame.loc[frame.label == 0].sort_values("text", key=lambda x: x.str.len()).reset_index(drop=True)
    size = min(len(positive), len(negative))
    positive = positive.iloc[:size]
    negative = negative.iloc[:size]
    positive_mix = [mix_insert(negative.text.iloc[index], positive.text.iloc[index]) for index in range(size)]
    negative_mix = [
        mix_insert(negative.text.iloc[index], negative.text.iloc[(index + max(1, size // 2)) % size])
        for index in range(size)
    ]
    return pd.DataFrame(
        {"text": positive_mix + negative_mix,
         "label": np.concatenate([np.ones(size, dtype=np.int8), np.zeros(size, dtype=np.int8)])}
    )

class AdaptRows(Dataset):
    def __init__(self, frame: pd.DataFrame):
        self.text = frame.text.fillna("").astype(str).tolist()
        self.target = frame.target.to_numpy(np.float32)
        self.weight = frame.weight.to_numpy(np.float32)

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index], self.target[index], self.weight[index]

class Collator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, rows):
        text, target, weight = zip(*rows, strict=True)
        encoded = self.tokenizer(
            list(text), add_special_tokens=True, truncation=True,
            max_length=self.max_length, padding=True, pad_to_multiple_of=8,
            return_tensors="pt",
        )
        return (
            encoded["input_ids"], encoded["attention_mask"],
            torch.tensor(target, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )

def load_trainable(model_dir: Path, adapter: Path):
    base = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, num_labels=1, problem_type="regression",
        dtype=torch.bfloat16, device_map={"": "cuda"},
    )
    base.config.pad_token_id = base.config.eos_token_id
    base.config.use_cache = False
    base.gradient_checkpointing_enable()
    base.enable_input_require_grads()
    model = PeftModel.from_pretrained(base, adapter, is_trainable=True)
    for parameter in model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    return model

def one_corruption(text: str) -> str:
    functions = (normalize_text, degrade_text, shuffle_sentences, truncate_middle)
    function = functions[stable_seed(str(text), "tv226-corruption") % len(functions)]
    return function(str(text))

def source_block(frame: pd.DataFrame, mass: float) -> pd.DataFrame:
    result = frame.copy()
    result["weight"] = mass / len(result)
    return result

def build_training_rows(competition: pd.DataFrame, external: pd.DataFrame, seed: int) -> pd.DataFrame:
    base = competition[["text", "label"]].rename(columns={"label": "target"}).copy()
    raw = source_block(base, 1.0)
    corrupted = base.copy()
    corrupted["text"] = corrupted.text.map(one_corruption)
    corrupted = source_block(corrupted, 0.5)
    mixed = hybrid_stress(competition[["text", "label"]]).rename(columns={"label": "target"})
    mixed = source_block(mixed, 0.5)
    replay_parts = []
    for offset, role in enumerate(("HWT", "LGT", "HLT")):
        current = external.loc[external.role == role, ["text", "edit_target"]]
        current = current.sample(n=min(800, len(current)), random_state=seed + offset)
        current = current.rename(columns={"edit_target": "target"})
        current["weight"] = 2.0 / (3.0 * len(current))
        replay_parts.append(current)
    result = pd.concat([raw, corrupted, mixed, *replay_parts], ignore_index=True)
    result["weight"] = result.weight / result.weight.mean()
    return result

def adapt_train(model, tokenizer, frame: pd.DataFrame, args) -> list[dict]:
    loader = DataLoader(
        AdaptRows(frame), batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + args.fold),
        num_workers=0, pin_memory=True, collate_fn=Collator(tokenizer, args.max_length),
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0.01)
    updates = math.ceil(len(loader) / args.accumulation)
    warmup = max(1, int(round(0.03 * updates)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, 1.0)
    )
    history = []
    weighted_sum = 0.0
    weight_sum = 0.0
    started = time.monotonic()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for step, batch in enumerate(loader):
        input_ids, attention, targets, weights = [value.to("cuda", non_blocking=True) for value in batch]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            score = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.view(-1)
            losses = torch.nn.functional.mse_loss(score.float(), targets, reduction="none")
            loss = (losses * weights).sum() / weights.sum() / args.accumulation
        if not torch.isfinite(loss):
            raise RuntimeError(f"TV226 non-finite loss at step {step}")
        loss.backward()
        weighted_sum += float((losses.detach() * weights).sum())
        weight_sum += float(weights.sum())
        if (step + 1) % args.accumulation == 0 or step + 1 == len(loader):
            torch.nn.utils.clip_grad_norm_(parameters, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update = math.ceil((step + 1) / args.accumulation)
            if update == 1 or update % 25 == 0 or update == updates:
                row = {
                    "update": update, "updates": updates,
                    "weighted_mse": weighted_sum / max(weight_sum, 1e-12),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_seconds": round(time.monotonic() - started, 2),
                }
                history.append(row)
                print(json.dumps(row), flush=True)
    return history
