#!/usr/bin/env python3
"""Refit the TV388 character response student; teacher predictions are pseudo-labels.

Does not retrain Qwen or recreate the historical teacher from scratch. Produces
only the character model, its ECDF, and the original Qwen ECDF in a new run.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LEGACY = ROOT / '.local/legacy'
COMPONENT = LEGACY / 'workspaces/semifinal_text/outputs/TV227_TV228_semifinal_editlens_committee'


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_csv(path: Path, columns: set[str]) -> list[str]:
    if not path.is_file():
        raise ValueError(f'Missing input: {path}')
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not columns <= set(reader.fieldnames or []):
            raise ValueError(f'{path} requires columns {sorted(columns)}')
        ids = []
        for row in reader:
            ids.append(row['id'])
            for column in columns - {'id', 'text'}:
                if not math.isfinite(float(row[column])):
                    raise ValueError(f'Nonfinite {column} in {path}')
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f'Empty or duplicate IDs in {path}')
    return ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--texts', type=Path, default=LEGACY / 'data/semifinal_transfer/text/text_test.csv')
    parser.add_argument('--teacher', type=Path, default=LEGACY / 'submissions/submission_TV388.csv')
    parser.add_argument('--components', type=Path, default=COMPONENT / 'test_predictions.csv')
    parser.add_argument('--raw-parts', type=Path, default=COMPONENT / 'tv226_parts')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/text_distill/assets')
    parser.add_argument('--check', action='store_true', help='check inputs without importing ML dependencies')
    args = parser.parse_args()
    errors = []
    id_lists = []
    for path, cols in ((args.texts, {'id','text'}), (args.teacher, {'id','score'}),
                       (args.components, {'id','TV226'})):
        try:
            id_lists.append(check_csv(path, cols))
        except (ValueError, OSError) as exc:
            errors.append(str(exc))
    if len(id_lists) == 3 and any(set(ids) != set(id_lists[0]) for ids in id_lists[1:]):
        errors.append('Text, teacher and component IDs must match exactly')
    parts = sorted(args.raw_parts.glob('part_*.npz'))
    if not parts:
        errors.append(f'Missing raw Qwen calibration NPZ parts: {args.raw_parts}')
    output = args.output.resolve()
    protected = ROOT / '.local/checkpoints'
    if output == protected or protected in output.parents:
        errors.append('Refusing to overwrite preserved checkpoints')
    if output.exists() and any(output.iterdir()):
        errors.append(f'Output is not empty: {output}')
    if args.check or errors:
        print(json.dumps({'inputs_ready': not errors, 'errors': errors,
                          'pseudo_labels_used': True, 'test_labels_used': False}, indent=2))
        if errors:
            raise SystemExit(2)
        return

    import joblib
    import numpy as np
    import pandas as pd
    from scipy.stats import rankdata, spearmanr
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import Ridge

    def ecdf(values):
        return rankdata(np.asarray(values, np.float64), method='average') / len(values)

    test = pd.read_csv(args.texts, dtype={'id': str})
    teacher = pd.read_csv(args.teacher, dtype={'id': str}).set_index('id').loc[test.id, 'score'].to_numpy(np.float64)
    component = pd.read_csv(args.components, dtype={'id': str}).set_index('id').loc[test.id]
    raw, raw_ids = [], []
    for path in parts:
        # These are trusted local historical NPZ files; object-string IDs need
        # allow_pickle=True. Never use this loader for untrusted archives.
        with np.load(path, allow_pickle=True) as part:
            raw.append(part['score'].astype(np.float64))
            raw_ids.extend(part['ids'].astype(str).tolist())
    raw = np.concatenate(raw)
    if len(raw) != len(test) or raw_ids != test.id.astype(str).tolist():
        raise RuntimeError('Qwen raw part IDs/order differ from text reference')
    if not np.isfinite(raw).all():
        raise RuntimeError('Qwen raw calibration contains nonfinite scores')
    vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(2,5), min_df=3,
                                 max_features=300_000, sublinear_tf=True,
                                 dtype=np.float32, norm='l2')
    matrix = vectorizer.fit_transform(test.text.fillna('').astype(str))
    ridge = Ridge(alpha=1.0, solver='lsqr').fit(matrix, teacher)
    lexical = ridge.predict(matrix).astype(np.float64)
    score = ecdf(0.5 * component.TV226.to_numpy(np.float64) + 0.5 * ecdf(lexical))
    if not np.isfinite(lexical).all() or not np.isfinite(score).all():
        raise RuntimeError('Nonfinite student predictions')
    agreement = float(spearmanr(score, teacher).statistic)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump({'vectorizer': vectorizer, 'ridge': ridge,
                 'lexical_ecdf': np.sort(lexical), 'qwen_weight': 0.5,
                 'lexical_weight': 0.5, 'target': 'TV388',
                 'training_spearman': agreement, 'pseudo_labels_used': True,
                 'sample_identifiers_stored': False},
                output / 'tv388_distill.joblib', compress=3)
    np.save(output / 'tv226_raw_ecdf.npy', np.sort(raw))
    manifest = {'version': 'TV388_deployable_response_student_refit', 'target': 'TV388',
                'teacher_training_rows': len(test), 'training_spearman': agreement,
                'validation_holdout_spearman': None, 'validation_recomputed': False,
                'sample_identifiers_stored': False, 'test_text_stored': False,
                'test_labels_used': False, 'pseudo_labels_used': True,
                'reference_texts_sha256': checksum(args.texts),
                'teacher_sha256': checksum(args.teacher),
                'components_sha256': checksum(args.components),
                'raw_parts_sha256': {p.name: checksum(p) for p in parts},
                'base_and_adapter_not_copied': True}
    (output / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
