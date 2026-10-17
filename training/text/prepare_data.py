#!/usr/bin/env python3
"""Audit and prepare public NLPCC-2026 Task 6 training data for binary detection.

Any group with exact/canonical/prefix/suffix overlap against competition development
or semifinal test text is excluded before labels or edit targets are consumed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / ".local/legacy"
ROLES = (("HWT", 0), ("LGT", 1), ("HLT", 1))
ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text))
    text = "".join(
        char for char in text if char not in ZERO_WIDTH and unicodedata.category(char) != "Cf"
    )
    return re.sub(r"\s+", " ", text).strip()


def canonical(text: str) -> str:
    return "".join(char.lower() for char in normalize(text) if char.isalnum())


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def signatures(text: str) -> dict[str, str]:
    normal = normalize(text)
    canon = canonical(normal)
    return {
        "exact": digest(normal),
        "canonical": digest(canon),
        "prefix256": digest(canon[:256]) if len(canon) >= 256 else "",
        "suffix256": digest(canon[-256:]) if len(canon) >= 256 else "",
    }


def reference_sets(paths: list[Path]) -> tuple[dict[str, set[str]], int]:
    values = {name: set() for name in ("exact", "canonical", "prefix256", "suffix256")}
    rows = 0
    for path in paths:
        frame = pd.read_csv(path, usecols=["text"])
        rows += len(frame)
        for text in frame.text.fillna("").astype(str):
            for name, value in signatures(text).items():
                if value:
                    values[name].add(value)
    return values, rows


def char_ngrams(text: str, n: int) -> Counter:
    chars = [char for char in normalize(text) if not char.isspace()]
    if len(chars) < n:
        return Counter()
    return Counter("".join(chars[index : index + n]) for index in range(len(chars) - n + 1))


def cosine(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    common = set(left) & set(right)
    dot = sum(left[key] * right[key] for key in common)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    return dot / (left_norm * right_norm) if left_norm and right_norm else 0.0


def edit_target(hwt: str, hlt: str) -> float:
    weighted = 0.0
    total = 0.0
    for order in range(1, 8):
        weight = float(order)
        weighted += weight * cosine(char_ngrams(hwt, order), char_ngrams(hlt, order))
        total += weight
    return max(0.0, min(1.0, 1.0 - weighted / total))


def edit_target_job(item: tuple[str, str, str]) -> tuple[str, float]:
    base_id, hwt, hlt = item
    return base_id, edit_target(hwt, hlt)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=LEGACY / "external_data/NLPCC2026-Task6/train_data.json",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=LEGACY / "external_data/NLPCC2026-Task6/filtered_binary_train.jsonl",
    )
    parser.add_argument("--competition", type=Path, default=LEGACY / "data/competition/text/text_sample_data.csv")
    parser.add_argument("--test-reference", type=Path, default=LEGACY / "data/semifinal_transfer/text/text_test.csv", help="Unlabeled reference used only to exclude duplicate text groups")
    args = parser.parse_args()
    references, reference_rows = reference_sets(
        [
            args.competition,
            args.test_reference,
        ]
    )
    groups = json.loads(args.input.read_text(encoding="utf-8-sig"))
    if not isinstance(groups, list):
        raise RuntimeError("NLPCC2026 training data is not a JSON list")
    excluded_overlap: dict[str, int] = defaultdict(int)
    staged: list[dict] = []
    group_records: dict[str, list[dict]] = {}
    target_inputs: list[tuple[str, str, str]] = []
    for index, group in enumerate(groups):
        base_id = str(group.get("ID", group.get("id", index)))
        if any(role not in group for role, _ in ROLES):
            raise RuntimeError(f"NLPCC2026 group {base_id} missing HWT/LGT/HLT")
        current = []
        group_overlap = set()
        for role, label in ROLES:
            text = str(group[role])
            sig = signatures(text)
            for name, value in sig.items():
                if value and value in references[name]:
                    group_overlap.add(name)
            current.append(
                {
                    "id": f"nlpcc26-{base_id}-{role}",
                    "base_id": base_id,
                    "role": role,
                    "label": label,
                    "text": text,
                    "canonical_hash": sig["canonical"],
                }
            )
        if group_overlap:
            for name in group_overlap:
                excluded_overlap[name] += 1
            continue
        group_records[base_id] = current
        target_inputs.append((base_id, str(group["HWT"]), str(group["HLT"])))
        staged.extend(current)

    with mp.Pool(processes=max(1, args.workers)) as pool:
        targets = dict(pool.imap_unordered(edit_target_job, target_inputs, chunksize=16))
    for base_id, records in group_records.items():
        hlt_target = targets[base_id]
        for record in records:
            record["edit_target"] = 0.0 if record["role"] == "HWT" else (
                1.0 if record["role"] == "LGT" else hlt_target
            )

    labels_by_hash: dict[str, set[int]] = defaultdict(set)
    groups_by_hash: dict[str, set[str]] = defaultdict(set)
    for record in staged:
        labels_by_hash[record["canonical_hash"]].add(int(record["label"]))
        groups_by_hash[record["canonical_hash"]].add(record["base_id"])
    ambiguous_groups = {
        base_id
        for value, labels in labels_by_hash.items()
        if len(labels) > 1
        for base_id in groups_by_hash[value]
    }
    selected: list[dict] = []
    seen: set[tuple[str, int]] = set()
    duplicate_rows = 0
    for base_id, records in group_records.items():
        if base_id in ambiguous_groups:
            continue
        for record in records:
            key = (record["canonical_hash"], int(record["label"]))
            if key in seen:
                duplicate_rows += 1
                continue
            seen.add(key)
            selected.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in selected:
            record = {name: value for name, value in record.items() if name != "canonical_hash"}
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    role_counts = Counter(record["role"] for record in selected)
    report = {
        "source": str(args.input),
        "source_groups": len(groups),
        "reference_rows": reference_rows,
        "excluded_overlap_groups_by_signature": dict(excluded_overlap),
        "excluded_ambiguous_groups": len(ambiguous_groups),
        "deduplicated_same_label_rows": duplicate_rows,
        "selected_rows": len(selected),
        "selected_role_counts": dict(role_counts),
        "semifinal_labels_read": False,
        "output": str(args.output),
    }
    report_path = args.output.with_suffix(".audit.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
