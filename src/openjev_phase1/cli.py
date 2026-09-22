"""Create-only JSONL command line scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .core import load_causal_model, validate_row
from .direct import score as direct_score
from .reranker import score as reranker_score
from .serial import SerialPrefixScorer
from .shared import score_shared


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("direct", "serial", "shared", "reranker"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--adapter-revision")
    parser.add_argument("--quantization", choices=("none", "nf4"), default="none")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=0,
                        help="Discard this many direct scoring passes before measuring all input rows")
    args = parser.parse_args()
    if args.output.exists() or args.max_tokens < 1:
        parser.error("Output must be new and max-tokens must be positive")
    if args.warmup < 0 or (args.warmup and args.mode != "direct"):
        parser.error("warmup must be nonnegative and is supported in direct mode only")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    if not rows:
        parser.error("Input is empty")
    for row in rows:
        validate_row(row)
    if args.mode in {"serial", "shared"}:
        parser.error("Cached serial/shared scoring is disabled after failed equivalence checks; use --mode direct")
    model, tokenizer, metadata = load_causal_model(
        args.model,
        args.revision,
        adapter=args.adapter,
        adapter_revision=args.adapter_revision,
        quantization=args.quantization,
        cache_dir=str(args.cache_dir) if args.cache_dir else None,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.warmup):
        direct_score(model, tokenizer, rows[index % len(rows)], metadata, args.max_tokens)
    with args.output.open("x") as destination:
        if args.mode == "shared":
            results, timing = score_shared(model, tokenizer, rows, metadata, args.max_tokens)
            for result in results:
                destination.write(json.dumps({**result, "shared_timing": timing}, allow_nan=False) + "\n")
        elif args.mode == "serial":
            scorer = SerialPrefixScorer(model, tokenizer, metadata, args.max_tokens)
            for row in rows:
                destination.write(json.dumps(scorer.score(row), allow_nan=False) + "\n")
                destination.flush()
        else:
            scorer = direct_score if args.mode == "direct" else reranker_score
            for row in rows:
                result = scorer(model, tokenizer, row, metadata, args.max_tokens)
                if args.mode == "direct":
                    result["warm"] = args.warmup > 0
                destination.write(json.dumps(result, allow_nan=False) + "\n")
                destination.flush()


if __name__ == "__main__":
    main()
