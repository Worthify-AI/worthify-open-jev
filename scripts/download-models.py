#!/usr/bin/env python3
"""Download only pinned model artifacts; reuse cached immutable blobs where available."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('manifests/gemma-models.json'))
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--reuse-cache', type=Path)
    parser.add_argument('--status-dir', type=Path, required=True)
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    args.status_dir.mkdir(parents=True, exist_ok=True)
    specs = json.loads(args.manifest.read_text())['models']
    order = ['Qwen/Qwen3.5-4B', 'google/gemma-4-12B-it', 'google/gemma-4-26B-A4B-it', 'google/gemma-4-31B-it']

    def run(model_id):
        revision = specs[model_id]['revision']
        cache_name = 'models--' + model_id.replace('/', '--')
        if args.reuse_cache:
            old_blobs = args.reuse_cache / cache_name / 'blobs'
            new_blobs = args.cache / cache_name / 'blobs'
            if old_blobs.is_dir():
                new_blobs.mkdir(parents=True, exist_ok=True)
                for source in old_blobs.iterdir():
                    if source.is_file() and not source.name.endswith('.incomplete'):
                        destination = new_blobs / source.name
                        if not destination.exists():
                            try:
                                destination.symlink_to(source.resolve())
                            except FileExistsError:
                                pass
        status = {'model': model_id, 'revision': revision, 'status': 'downloading'}
        status_path = args.status_dir / (cache_name + '.json')
        status_path.write_text(json.dumps(status, indent=2) + '\n')
        try:
            path = snapshot_download(model_id, revision=revision, cache_dir=args.cache,
                                     token=False, max_workers=2,
                                     allow_patterns=['*.json', '*.safetensors', '*.model', '*.jinja', 'LICENSE*', 'NOTICE*'])
            status.update(status='ready', snapshot=str(path))
        except Exception as exc:
            status.update(status='failed', error_type=type(exc).__name__,
                          http_status=getattr(getattr(exc, 'response', None), 'status_code', None))
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        os.replace(temporary, status_path)
        print(json.dumps(status), flush=True)
        return status['status'] == 'ready'

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(run, order))
    raise SystemExit(0 if all(results) else 1)


if __name__ == '__main__':
    main()
