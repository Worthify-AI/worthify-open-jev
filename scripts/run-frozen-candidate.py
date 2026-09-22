#!/usr/bin/env python3
"""Measure one pinned model against a frozen validation file on one visible GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import time


def main():
    from openjev_phase1.core import load_causal_model
    from openjev_phase1.direct import PROMPT_VERSION, score
    from openjev_phase1.evaluation import evaluate_predictions
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=2048)
    parser.add_argument('--expected-gpu', default='A100')
    args = parser.parse_args()
    if torch.cuda.device_count() != 1 or args.expected_gpu not in torch.cuda.get_device_name(0):
        raise ValueError('Select exactly one GPU matching --expected-gpu before loading weights')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    input_bytes = args.input.read_bytes()
    input_sha256 = hashlib.sha256(input_bytes).hexdigest()
    rows = [json.loads(line) for line in input_bytes.splitlines() if line.strip()]
    if not rows or any(row.get('split') != 'validation' for row in rows):
        raise ValueError('Candidate selection input must contain validation rows only')
    code_files = [Path(__file__), *sorted((Path(__file__).resolve().parents[1]/'src/openjev_phase1').glob('*.py'))]
    code = {'revision': subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
            'dirty': bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip()),
            'sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code_files}}
    protocol = {'quantization': 'nf4', 'max_tokens': args.max_tokens,
                'prompt_version': PROMPT_VERSION, 'dtype': 'bfloat16', 'batch_size': 1,
                'hardware_name': torch.cuda.get_device_name(0)}
    started = time.perf_counter()
    model, tokenizer, metadata = load_causal_model(args.model, args.revision, quantization='nf4', cache_dir=str(args.cache_dir))
    metadata.update(gpu=torch.cuda.get_device_name(0), python=platform.python_version(),
                    gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
                    cuda=torch.version.cuda, load_seconds=time.perf_counter()-started)
    (args.output_dir/'run-manifest.json').write_text(json.dumps({
        'model':metadata, 'code':code, 'comparison_protocol':protocol,
        'input_sha256':input_sha256,
    }, indent=2)+'\n')
    for row in rows[:3]:
        score(model, tokenizer, row, metadata, args.max_tokens)
    outputs = []
    with (args.output_dir/'predictions.jsonl').open('x') as destination:
        for index, row in enumerate(rows):
            result = score(model, tokenizer, row, metadata, args.max_tokens)
            result['warm'] = True
            outputs.append(result)
            destination.write(json.dumps(result, allow_nan=False)+'\n')
            destination.flush()
            if (index+1) % 25 == 0:
                print(json.dumps({'model':args.model,'completed':index+1,'total':len(rows)}), flush=True)
    report = evaluate_predictions(rows, outputs, bootstrap_samples=1000)
    report.update(candidate=args.model, model=metadata, code=code, comparison_protocol=protocol,
                  input_sha256=input_sha256,
                  predictions_sha256=hashlib.sha256((args.output_dir/'predictions.jsonl').read_bytes()).hexdigest())
    (args.output_dir/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'candidate':args.model,'macro_f1':report['mean_task_macro_f1'],
                      'latency':report['latency'],'report':str(args.output_dir/'report.json')}),flush=True)


if __name__ == '__main__':
    main()
