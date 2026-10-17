#!/usr/bin/env python3
"""Infer a fresh TV226 adapter, optionally fitting/applying its own frozen ECDF."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=ROOT / '.local/checkpoints/text/qwen2.5_1.5b_instruct')
    parser.add_argument('--adapter', type=Path, default=ROOT / 'runs/text_retrain/adapter')
    parser.add_argument('--input', type=Path, required=True, help='CSV with id,text')
    parser.add_argument('--output', type=Path, required=True, help='New CSV with id,score')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--view', choices=['head-middle-tail', 'first'], default='head-middle-tail')
    calibrate = parser.add_mutually_exclusive_group()
    calibrate.add_argument('--save-calibration', type=Path, help='Fit frozen ECDF from this reference CSV; .npz output')
    calibrate.add_argument('--calibration', type=Path, help='Apply ECDF made by this script with this exact adapter')
    parser.add_argument('--check', action='store_true', help='Validate files without loading ML libraries')
    args = parser.parse_args()
    errors = []
    for directory, names in ((args.model_dir, ('config.json','model.safetensors')),
                             (args.adapter, ('adapter_config.json','adapter_model.safetensors','tokenizer.json','tokenizer_config.json'))):
        for name in names:
            if not (directory / name).is_file(): errors.append(f'Missing {directory / name}')
    if args.max_length < 1 or args.batch_size < 1:
        errors.append('Batch size and maximum length must be positive')
    texts, ids = [], []
    if not args.input.is_file():
        errors.append(f'Missing input CSV: {args.input}')
    else:
        with args.input.open(encoding='utf-8-sig', newline='') as stream:
            rows = csv.DictReader(stream)
            if not {'id','text'} <= set(rows.fieldnames or []):
                errors.append('Input requires id,text columns')
            else:
                for row in rows:
                    ids.append(row['id']); texts.append(row['text'] or '')
        if not ids or len(ids) != len(set(ids)):
            errors.append('Input IDs must be nonempty and unique')
    for output in (args.output, args.output.with_suffix(args.output.suffix+'.metadata.json'), args.save_calibration):
        if output is None: continue
        if output.exists(): errors.append(f'Refusing to overwrite {output}')
        protected = (ROOT / '.local/checkpoints').resolve()
        if output.resolve() == protected or protected in output.resolve().parents:
            errors.append('Outputs must not overwrite preserved checkpoints')
    if args.save_calibration and args.save_calibration.suffix != '.npz':
        errors.append('--save-calibration must end in .npz')
    if args.calibration and not args.calibration.is_file():
        errors.append(f'Missing calibration: {args.calibration}')
    if args.check or errors:
        print(json.dumps({'inputs_ready':not errors,'errors':errors,'rows':len(ids)}, indent=2))
        if errors: raise SystemExit(2)
        return

    import numpy as np
    import torch
    from peft import PeftModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    if not torch.cuda.is_available(): raise RuntimeError('NVIDIA CUDA GPU required')
    identity = {'adapter_sha256': digest(args.adapter/'adapter_model.safetensors'),
                'base_sha256': digest(args.model_dir/'model.safetensors'),
                'tokenizer_sha256': digest(args.adapter/'tokenizer.json'),
                'max_length': args.max_length, 'view':args.view}
    reference = None
    if args.calibration:
        with np.load(args.calibration, allow_pickle=False) as saved:
            calibration_identity = json.loads(str(saved['identity'].item()))
            if calibration_identity != identity:
                raise RuntimeError('Calibration belongs to another adapter/base/tokenizer/inference configuration')
            reference = saved['sorted_scores'].astype(np.float64)
        if reference.size == 0 or not np.isfinite(reference).all() or np.any(np.diff(reference) < 0):
            raise RuntimeError('Invalid sorted calibration scores')
    tokenizer = AutoTokenizer.from_pretrained(args.adapter, local_files_only=True)
    if tokenizer.pad_token_id is None: tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'
    base = AutoModelForSequenceClassification.from_pretrained(args.model_dir, local_files_only=True,
        num_labels=1, problem_type='regression', torch_dtype=torch.bfloat16, device_map={'':'cuda'})
    base.config.pad_token_id = base.config.eos_token_id
    model = PeftModel.from_pretrained(base, args.adapter).eval()
    windows, owners = [], []
    for owner, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        if not tokens: tokens=[tokenizer.eos_token_id]
        starts = [0]
        if args.view == 'head-middle-tail' and len(tokens) > args.max_length:
            starts = list(dict.fromkeys([0,(len(tokens)-args.max_length)//2,len(tokens)-args.max_length]))
        for start in starts:
            windows.append(tokens[start:start+args.max_length]); owners.append(owner)
    total, counts = np.zeros(len(ids),np.float64), np.zeros(len(ids),np.int64)
    with torch.inference_mode():
        for start in range(0,len(windows),args.batch_size):
            encoded = tokenizer.pad({'input_ids':windows[start:start+args.batch_size]}, padding=True,
                                     pad_to_multiple_of=8, return_tensors='pt')
            encoded = {key:value.to('cuda') for key,value in encoded.items()}
            with torch.autocast('cuda',dtype=torch.bfloat16):
                values = model(**encoded,use_cache=False).logits.view(-1).float().cpu().numpy()
            for owner,value in zip(owners[start:start+args.batch_size],values):
                total[owner]+=float(value); counts[owner]+=1
    raw = total/counts
    if not np.isfinite(raw).all(): raise RuntimeError('Nonfinite model output')
    scores = raw
    if reference is not None:
        scores = np.clip((np.searchsorted(reference,raw,side='left')+np.searchsorted(reference,raw,side='right')+1)/(2*len(reference)),0,1)
    if args.save_calibration:
        args.save_calibration.parent.mkdir(parents=True,exist_ok=True)
        np.savez(args.save_calibration, sorted_scores=np.sort(raw), identity=json.dumps(identity,sort_keys=True),
                 reference_sha256=digest(args.input))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',encoding='utf-8',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['id','score']); writer.writerows(zip(ids,scores))
    metadata={'identity':identity,'rows':len(ids),'input_sha256':digest(args.input),
              'score_type':'frozen_ECDF_rank' if reference is not None else 'raw_regression_score',
              'probability_calibration':False,'calibration_saved':str(args.save_calibration) if args.save_calibration else None}
    args.output.with_suffix(args.output.suffix+'.metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2))


if __name__ == '__main__':
    main()
