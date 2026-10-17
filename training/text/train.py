#!/usr/bin/env python3
"""Portable, exact TV225 public preadaptation -> TV226 full-data adapter training.

Starts from the pretrained Qwen base (not random initialization), never from the
preserved final adapter. Outputs a new detector, NOT the full TV388 leaderboard
construction or its character response student. See README.md for boundaries.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def check_inputs(model: Path, competition: Path, external: Path) -> list[str]:
    errors = []
    for filename in ('config.json', 'tokenizer_config.json', 'tokenizer.json'):
        if not (model / filename).is_file():
            errors.append(f'Missing base-model asset: {model / filename}')
    if not list(model.glob('*.safetensors')):
        errors.append(f'Missing base-model safetensors: {model}')
    if not competition.is_file():
        errors.append(f'Missing competition training CSV: {competition}')
    else:
        with competition.open(encoding='utf-8-sig', newline='') as stream:
            reader = csv.DictReader(stream)
            if not {'id', 'text', 'label'} <= set(reader.fieldnames or []):
                errors.append('Competition CSV requires id,text,label columns')
            else:
                labels, count = set(), 0
                for row in reader:
                    labels.add(row['label']); count += 1
                if count == 0 or not labels <= {'0', '1'} or labels != {'0', '1'}:
                    errors.append('Competition labels must contain both classes 0 and 1')
    if not external.is_file():
        errors.append(f'Missing filtered NLPCC training JSONL: {external}; obtain lawful HWT/LGT/HLT training data and run prepare_data.py')
    else:
        ids, roles = set(), set()
        with external.open(encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    if not {'id', 'base_id', 'role', 'text', 'edit_target'} <= set(row):
                        raise ValueError('missing required fields')
                    if row['id'] in ids:
                        raise ValueError('duplicate id')
                    if row['role'] not in {'HWT', 'LGT', 'HLT'}:
                        raise ValueError('unknown role')
                    if not isinstance(row['text'], str) or not 0 <= float(row['edit_target']) <= 1:
                        raise ValueError('invalid text or target')
                    ids.add(row['id']); roles.add(row['role'])
                except (ValueError, TypeError, KeyError) as exc:
                    errors.append(f'External JSONL line {number}: {exc}'); break
        if roles != {'HWT', 'LGT', 'HLT'}:
            errors.append('External data must contain all HWT,LGT,HLT roles')
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path(__file__).with_name('config.json'))
    parser.add_argument('--model-dir')
    parser.add_argument('--competition')
    parser.add_argument('--external')
    parser.add_argument('--output')
    parser.add_argument('--check', action='store_true', help='validate inputs without importing torch or starting training')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    for key in ('model_dir', 'competition', 'external', 'output'):
        override = getattr(args, key)
        if override is not None:
            config[key] = override
    model, competition, external, output = [resolve(config[key]) for key in ('model_dir', 'competition', 'external', 'output')]
    errors = check_inputs(model, competition, external)
    for stage in ('pretrain', 'adapt'):
        if config[stage]['epochs'] != 1:
            errors.append('The historical recipe performs exactly one epoch at each stage')
    protected = ROOT / '.local/checkpoints'
    if output == protected or protected in output.parents:
        errors.append('Refusing to write training outputs into preserved checkpoints')
    if output.exists() and any(output.iterdir()):
        errors.append(f'Output is not empty: {output}; choose a new run directory')
    report = {'inputs_ready': not errors, 'errors': errors, 'config': config,
              'recipe': 'TV225 public preadaptation + TV226 full adaptation',
              'does_not_retrain_TV388_character_student': True}
    if args.check or errors:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if errors:
            raise SystemExit(2)
        return

    # Import only after fail-fast data checks. No historical evaluation dependency.
    import pandas as pd
    import torch
    from transformers import AutoTokenizer
    import recipe
    if not torch.cuda.is_available():
        raise RuntimeError('This original BF16 recipe requires an NVIDIA CUDA GPU')
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('The GPU must support BF16')
    external_frame = pd.read_json(external, lines=True, dtype={'id': str, 'base_id': str})
    sample = pd.read_csv(competition, dtype={'id': str})
    output.mkdir(parents=True, exist_ok=True)
    (output / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    seed = int(config['seed'])
    recipe.seed_everything(seed)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    pre_args = argparse.Namespace(**config['pretrain'], seed=seed,
                                  batch_size=config['batch_size'], accumulation=config['accumulation'],
                                  log_every=25, save_every=250)
    detector = recipe.load_model(model)
    pre_output = output / 'pretrain'
    pre_output.mkdir()
    pre_history = recipe.pretrain(detector, external_frame, tokenizer, pre_args, pre_output)
    pre_adapter = pre_output / 'adapter'
    detector.save_pretrained(pre_adapter)
    tokenizer.save_pretrained(pre_adapter)
    del detector
    torch.cuda.empty_cache()

    # Match train_text_TV226_full.py: reset seed, fold=-1, and freshly reload base.
    recipe.seed_everything(seed)
    training = recipe.build_training_rows(sample[['text', 'label']].reset_index(drop=True), external_frame, seed)
    tokenizer = AutoTokenizer.from_pretrained(pre_adapter, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    detector = recipe.load_trainable(model, pre_adapter)
    adapt_args = argparse.Namespace(**config['adapt'], seed=seed, fold=-1,
                                    batch_size=config['batch_size'], accumulation=config['accumulation'])
    history = recipe.adapt_train(detector, tokenizer, training, adapt_args)
    final_adapter = output / 'adapter'
    detector.save_pretrained(final_adapter)
    tokenizer.save_pretrained(final_adapter)
    manifest = {'recipe': report['recipe'], 'config': config,
                'competition_sha256': sha256(competition), 'external_sha256': sha256(external),
                'competition_rows': len(sample), 'external_rows': len(external_frame),
                'adaptation_rows': len(training), 'pretrain_history': pre_history,
                'adapt_history': history, 'checkpoint': str(final_adapter),
                'score_reproduction_claim': False}
    (output / 'training.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'completed': True, 'adapter': str(final_adapter)}, indent=2))


if __name__ == '__main__':
    main()
