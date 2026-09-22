#!/usr/bin/env python3
"""Freeze a balanced, output-blind validation sample for every model candidate."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--per-task',type=int,default=512)
    args=parser.parse_args()
    if args.per_task<1 or args.output.exists() or args.output.with_suffix('.manifest.json').exists():
        parser.error('Use a new output path and a positive sample size')
    selected=[]
    sources={}
    for dataset in ('clinc150','wanli'):
        path=args.data_dir/f'{dataset}-validation.jsonl'
        payload=path.read_bytes()
        sources[path.name]=hashlib.sha256(payload).hexdigest()
        groups=defaultdict(list)
        for line in payload.splitlines():
            row=json.loads(line)
            if row['split']!='validation':
                raise ValueError('Only validation data is eligible')
            groups[row['gold_option_id']].append(row)
        labels=sorted(groups)
        for label in labels:
            groups[label].sort(key=lambda row:hashlib.sha256(('comparison-v1:'+row['id']).encode()).hexdigest())
        sample=[]
        while len(sample)<args.per_task and any(groups.values()):
            for label in labels:
                if groups[label] and len(sample)<args.per_task:
                    sample.append(groups[label].pop())
        if len(sample)!=args.per_task:
            raise ValueError(f'{dataset} lacks enough validation rows')
        selected.extend(sample)
    selected.sort(key=lambda row:hashlib.sha256(('interleave-v1:'+row['id']).encode()).hexdigest())
    payload=''.join(json.dumps(row,sort_keys=True)+'\n' for row in selected).encode()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_bytes(payload)
    manifest={'protocol':'comparison-validation-v1','per_task':args.per_task,'rows':len(selected),
              'source_sha256':sources,'sha256':hashlib.sha256(payload).hexdigest(),
              'selection':'round-robin semantic gold labels, stable SHA-256 row ordering; no predictions used'}
    args.output.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest))


if __name__=='__main__':
    main()
