"""Evaluate the small authored diagnostic using the shared strict option alignment."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

from openjev_phase1.evaluation import _metrics, align_predictions


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def evaluate(gold, outputs):
    if not gold:
        raise ValueError('Diagnostic cannot be empty')
    aligned = align_predictions(gold, outputs)
    source = {row['id']: row for row in gold}
    by_id = {row['id']: row for row in aligned}
    by_kind = defaultdict(list)
    pairs = []
    for row in aligned:
        perturbation = source[row['id']]['perturbation']
        kind, base_id, comparison = (perturbation[key] for key in ('kind', 'base_id', 'comparison'))
        if base_id not in by_id or source[base_id]['perturbation']['kind'] != 'base':
            raise ValueError('Every diagnostic variant must reference an existing base')
        same_gold = row['gold_id'] == by_id[base_id]['gold_id']
        if comparison not in {'same_target', 'changed_target'} or same_gold != (comparison == 'same_target'):
            raise ValueError('Perturbation comparison does not match semantic gold labels')
        if kind == 'base' and base_id != row['id']:
            raise ValueError('Base rows must reference themselves')
        by_kind[kind].append(row)
        if kind != 'base' and comparison == 'same_target':
            pairs.append(row['predicted_id'] == by_id[base_id]['predicted_id'])
    metrics = _metrics(aligned, ece_bins=10)
    return {
        'schema': 'worthify-openjev-authored-diagnostic-v1',
        'n': len(aligned), 'accuracy': metrics['accuracy'], 'macro_f1': metrics['macro_f1'],
        'calibration': {key: metrics[key] for key in ('brier', 'ece', 'reliability_bins', 'probability_status')},
        'per_perturbation': {kind: _metrics(rows, ece_bins=10) for kind, rows in sorted(by_kind.items())},
        'paired_semantic_consistency': {'n': len(pairs), 'rate': sum(pairs)/len(pairs) if pairs else None},
        'limitation': 'Small authored diagnostic; not a broad capability or cyber benchmark.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gold', type=Path, default=Path('examples/worthify-robustness.jsonl'))
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output must be new')
    result = evaluate(read(args.gold), read(args.predictions))
    result['inputs'] = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                        for name, path in [('gold_sha256', args.gold), ('predictions_sha256', args.predictions)]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as destination:
        destination.write(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
