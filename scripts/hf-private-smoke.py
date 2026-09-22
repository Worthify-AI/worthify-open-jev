#!/usr/bin/env python3
"""Verify private Hub upload/download transport without publishing model weights."""
import argparse
import hashlib
import json
from pathlib import Path

from openjev_phase1.publish import smoke_upload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-id', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--code-revision', required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = args.output / 'payload'
    bundle.mkdir()
    (bundle / 'README.md').write_text('# Worthify OpenJev private transport check\n\n'
                                    'Inert CI test payload. This is not a trained model.\n')
    (bundle / 'smoke-payload.json').write_text(json.dumps({
        'schema': 'worthify-openjev-private-transport-v1',
        'code_revision': args.code_revision,
        'purpose': 'private upload and immutable-revision download checksum verification',
    }, indent=2) + '\n')
    (bundle / 'SHA256SUMS').write_text(''.join(
        f'{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n'
        for path in sorted(bundle.iterdir())
    ))
    receipt = smoke_upload(bundle, args.repo_id)
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
