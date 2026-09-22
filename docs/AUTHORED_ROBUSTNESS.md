# Authored robustness diagnostic

`examples/worthify-robustness.jsonl` is a small public, authored diagnostic. It is not a broad benchmark and is separate from the converted CLINC unseen-label test and all upstream evaluation fixtures. It contains defensive cyber triage, patch-evidence, alert-routing, and general decision examples.

The 15 records comprise three base cases with separate option-order, question-paraphrase, irrelevant-context, and missing-evidence variants. Routing rules are supplied explicitly. Missing-evidence variants deliberately change the correct answer to `insufficient`; the harness excludes those changed-target pairs from semantic-consistency scoring.

```bash
CUDA_VISIBLE_DEVICES=0 openjev-score --mode direct --model "$MODEL" --revision "$REVISION" \
  --quantization nf4 --max-tokens 2048 --warmup 3 \
  --input examples/worthify-robustness.jsonl --output /tmp/authored-robustness.predictions.jsonl
python benchmarks/evaluate_authored_robustness.py \
  --gold examples/worthify-robustness.jsonl --predictions /tmp/authored-robustness.predictions.jsonl \
  --output /tmp/authored-robustness.report.json
```

The report provides accuracy, macro F1, Brier score, ECE/reliability bins, per-perturbation metrics, and paired semantic consistency only for same-target pairs. Run the same commands with a pinned tuned adapter by adding its explicit adapter path and revision to `openjev-score`.
