#!/usr/bin/env python3
"""CPU checkpoint-load audit; optional real Qwen forward, not CUDA runtime QA."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoints', type=Path, default=ROOT / '.local/checkpoints')
    p.add_argument('--text-forward', action='store_true', help='Load the full Qwen+LoRA on CPU and score a short example')
    p.add_argument('--report', type=Path)
    a = p.parse_args()
    import joblib
    import numpy as np
    import torch
    import timm
    from safetensors import safe_open
    from transformers import AutoTokenizer
    from dualtrace.image import runtime as image
    report = {'cuda_inference_tested': False, 'image_full_forward_tested': False}
    torch.set_num_threads(4)
    heads = a.checkpoints / 'image/heads'
    loaded = {}
    for path in sorted(heads.glob('*.joblib')):
        loaded[path.name] = type(joblib.load(path)).__name__
    report['loaded_sklearn_heads'] = loaded
    for name in ['V47_linear.pth', 'V51_linear.pth']:
        image._load_linear(heads / name)
    for name in ['V54_residual.pth', 'V56_residual.pth', 'V58_residual.pth']:
        image._load_residual(heads / name)
    for name in ['V67_multidepth.pth', 'V71_multidepth.pth', 'V76_multidepth.pth', 'V77_multidepth.pth']:
        image._load_multidepth(heads / name)
    report['strict_neural_head_loads'] = 9
    report['backbone_shapes'] = {}
    for model, filename in [('vit_pe_core_gigantic_patch14_448.fb', 'pe_core_g14.safetensors'), ('vit_pe_spatial_gigantic_patch14_448.fb', 'pe_spatial_g14.safetensors')]:
        with torch.device('meta'):
            net = timm.create_model(model, pretrained=False)
        expected = {k: tuple(v.shape) for k, v in net.state_dict().items()}
        with safe_open(a.checkpoints / 'image/backbones' / filename, framework='pt', device='cpu') as f:
            actual = {k: tuple(f.get_slice(k).get_shape()) for k in f.keys()}
        if expected != actual:
            raise RuntimeError('Backbone key/shape mismatch: ' + model)
        report['backbone_shapes'][model] = {'matched': True, 'tensors': len(actual)}
        del net
    for name in ['qwen2.5_1.5b_instruct', 'tv226_adapter']:
        AutoTokenizer.from_pretrained(a.checkpoints / 'text' / name, local_files_only=True)
    report['text_tokenizers_loaded'] = True
    if a.text_forward:
        from peft import PeftModel
        from transformers import AutoModelForSequenceClassification
        from dualtrace.text.runtime import ecdf
        assets = a.checkpoints / 'text'
        tokenizer = AutoTokenizer.from_pretrained(assets / 'tv226_adapter', local_files_only=True)
        base = AutoModelForSequenceClassification.from_pretrained(assets / 'qwen2.5_1.5b_instruct', local_files_only=True, num_labels=1, problem_type='regression', dtype=torch.bfloat16, device_map={'': 'cpu'})
        base.config.pad_token_id = base.config.eos_token_id
        net = PeftModel.from_pretrained(base, assets / 'tv226_adapter').eval()
        student = joblib.load(assets / 'tv388_distill.joblib')
        cdf = np.load(assets / 'tv226_raw_ecdf.npy')
        outputs = []
        for text in ['这是一条用于核对模型载入的文本。', '']:
            encoded = tokenizer(text, return_tensors='pt')
            if encoded['input_ids'].shape[1] == 0:
                encoded = {'input_ids': torch.tensor([[tokenizer.eos_token_id]]), 'attention_mask': torch.ones((1, 1), dtype=torch.long)}
            with torch.inference_mode():
                raw = float(net(**encoded, use_cache=False).logits.view(-1)[0].float())
            lexical = float(student['ridge'].predict(student['vectorizer'].transform([text]))[0])
            score = student['qwen_weight'] * ecdf(raw, cdf) + student['lexical_weight'] * ecdf(lexical, student['lexical_ecdf'])
            if not np.isfinite(score) or not 0 <= score <= 1:
                raise RuntimeError('Invalid CPU text score')
            outputs.append({'empty': not bool(text), 'raw': raw, 'score': score})
        report['cpu_text_forward'] = outputs
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
