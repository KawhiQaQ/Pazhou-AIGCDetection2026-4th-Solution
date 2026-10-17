from __future__ import annotations

import hashlib
import math
from pathlib import Path
import re
import unicodedata

import imagehash
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedGroupKFold


class UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))
        self.rank = [0] * size

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root == right_root:
            return
        if self.rank[left_root] < self.rank[right_root]:
            left_root, right_root = right_root, left_root
        self.parent[right_root] = left_root
        if self.rank[left_root] == self.rank[right_root]:
            self.rank[left_root] += 1

    def labels(self, prefix: str) -> list[str]:
        roots: dict[int, int] = {}
        labels = []
        for index in range(len(self.parent)):
            root = self.find(index)
            roots.setdefault(root, len(roots))
            labels.append(f"{prefix}{roots[root]:04d}")
        return labels


def normalize_text(text: object) -> str:
    value = "" if pd.isna(text) else str(text)
    value = unicodedata.normalize("NFKC", value).lower()
    value = re.sub(r"\s+", " ", value).strip()
    return value


def text_near_duplicate_groups(
    texts: list[str], *, cosine_threshold: float = 0.93
) -> tuple[list[str], dict[str, float | int]]:
    normalized = [normalize_text(text) for text in texts]
    union_find = UnionFind(len(normalized))
    exact: dict[str, int] = {}
    for index, text in enumerate(normalized):
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if digest in exact:
            union_find.union(index, exact[digest])
        else:
            exact[digest] = index

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=2,
        max_features=50_000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(normalized)
    similarities = (matrix @ matrix.T).tocoo()
    near_pairs = 0
    for left, right, similarity in zip(similarities.row, similarities.col, similarities.data):
        if left < right and similarity >= cosine_threshold:
            union_find.union(int(left), int(right))
            near_pairs += 1

    groups = union_find.labels("text_g")
    counts = pd.Series(groups).value_counts()
    return groups, {
        "cosine_threshold": cosine_threshold,
        "near_duplicate_pairs": near_pairs,
        "n_groups": int(counts.size),
        "n_grouped_samples": int(counts[counts > 1].sum()),
        "max_group_size": int(counts.max()),
    }


def image_near_duplicate_groups(
    image_paths: list[Path], *, phash_distance: int = 4
) -> tuple[list[str], pd.DataFrame, dict[str, float | int]]:
    union_find = UnionFind(len(image_paths))
    records: list[dict[str, object]] = []
    exact: dict[str, int] = {}
    for index, path in enumerate(image_paths):
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            width, height = image.size
            perceptual_hash = int(str(imagehash.phash(image, hash_size=8)), 16)
        if digest in exact:
            union_find.union(index, exact[digest])
        else:
            exact[digest] = index
        records.append(
            {
                "width": width,
                "height": height,
                "pixels": width * height,
                "aspect_ratio": width / height,
                "file_bytes": len(payload),
                "sha256": digest,
                "phash64": f"{perceptual_hash:016x}",
                "_phash_int": perceptual_hash,
            }
        )

    near_pairs = 0
    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            aspect_delta = abs(
                math.log(float(records[left]["aspect_ratio"]) / float(records[right]["aspect_ratio"]))
            )
            if aspect_delta > 0.03:
                continue
            distance = (int(records[left]["_phash_int"]) ^ int(records[right]["_phash_int"])).bit_count()
            if distance <= phash_distance:
                union_find.union(left, right)
                near_pairs += 1

    groups = union_find.labels("image_g")
    metadata = pd.DataFrame(records).drop(columns=["_phash_int"])
    counts = pd.Series(groups).value_counts()
    return groups, metadata, {
        "phash_distance": phash_distance,
        "near_duplicate_pairs": near_pairs,
        "n_groups": int(counts.size),
        "n_grouped_samples": int(counts[counts > 1].sum()),
        "max_group_size": int(counts.max()),
    }


def assign_balanced_group_folds(
    labels: np.ndarray,
    groups: list[str],
    balance_values: np.ndarray,
    *,
    n_splits: int = 5,
    seed: int = 20260806,
) -> tuple[np.ndarray, dict[str, int]]:
    labels = np.asarray(labels, dtype=np.int8)
    values = np.asarray(balance_values, dtype=np.float64)
    if len(labels) != len(groups) or len(labels) != len(values):
        raise ValueError("labels, groups, and balance_values must have equal length")

    chosen_bins = 1
    stratification = labels.astype(str)
    for bins in (5, 4, 3, 2):
        ranked = pd.Series(values).rank(method="first")
        value_bin = pd.qcut(ranked, q=bins, labels=False, duplicates="drop").astype(int)
        candidate = pd.Series(labels).astype(str) + "_b" + value_bin.astype(str)
        if int(candidate.value_counts().min()) >= n_splits:
            chosen_bins = bins
            stratification = candidate.to_numpy()
            break

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = np.full(len(labels), -1, dtype=np.int8)
    dummy = np.zeros((len(labels), 1), dtype=np.float32)
    for fold, (_, validation_index) in enumerate(splitter.split(dummy, stratification, groups)):
        folds[validation_index] = fold
    if (folds < 0).any():
        raise RuntimeError("some samples were not assigned to a fold")
    group_audit = pd.DataFrame({"group": groups, "fold": folds}).groupby("group").fold.nunique()
    if int(group_audit.max()) != 1:
        raise RuntimeError("a duplicate group crosses validation folds")
    return folds, {"balance_bins": chosen_bins, "seed": seed, "n_splits": n_splits}

