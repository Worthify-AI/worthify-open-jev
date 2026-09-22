# Gemma 4 12B long-context experiment

This isolated experiment measures whether the pinned 12B model can complete
option-logit QLoRA updates at longer sequence lengths on one GPU. It does not
change the released scorer or the two 2,048-token training recipes.

The base revision is `google/gemma-4-12B-it` at
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. Its configured native context is
262,144 tokens. Increasing that configuration value does not establish support
for one million tokens.

## Generic scoring API

From an editable checkout installed with `pip install -e '.[test,train]'`, score
ordinary OpenJEV JSONL rows containing `id`, `state`, `question`, and 2–16
runtime-defined `options`:

```bash
CUDA_VISIBLE_DEVICES=<gpu-uuid> python -m experiments.long_context.score \
  --input decisions.jsonl --output scores-v1.jsonl \
  --cache-dir /path/to/hub-cache --max-tokens 262144
```

The create-only experimental command uses the pinned 12B revision with NF4
weights, BF16 computation, native analytical FlexAttention masks, no KV cache,
and no truncation. An immutable adapter can be supplied with both `--adapter`
and `--adapter-revision`. The output reports caller option IDs, logits,
conditional uncalibrated probabilities, the prediction, hashes, timing, memory,
and runtime backend metadata; `timing_scope` makes clear that per-row timers
exclude model loading, preflight, tokenization, and mask construction. Input
rows need no gold label or provenance. The
B200 receipts establish full 262,144-token forward/backward systems execution,
not generic task quality or one-million-token support.

## Procedure

Use the repository's pinned environment, installed with
`pip install -e '.[test,train]'`, and expose exactly one GPU by UUID. Outputs
must use new paths. The initial reference machine is a RunPod NVIDIA B200;
the main recipe benchmark reference remains the A100.

First compare compiled FlexAttention outputs and Q/K/V gradients against
PyTorch's math SDPA reference at a bounded length:

```bash
CUDA_VISIBLE_DEVICES=<gpu-uuid> python experiments/long_context/kernel_probe.py \
  --sequence-length 2048 --output artifacts/kernel-probe-v1.json
```

Both native attention shapes must pass: local causal attention with 256-wide
heads and a 1,024-token window, and global causal attention with 512-wide heads.
Do not proceed after nonfinite values, an unsupported kernel, or a failed
equivalence check. Preserve failures alongside successful attempts.

After the kernel check, start the full-model probe at 8,192 tokens:

```bash
CUDA_VISIBLE_DEVICES=<gpu-uuid> python experiments/long_context/pilot.py \
  --tokens 8192 --steps 2 --cache-dir /path/to/hub-cache \
  --output artifacts/context-8192-v1
```

Run longer lengths sequentially only after reviewing each result: 16,384,
32,768, 65,536, 131,072, then 262,144. Apply an external deadline to each
process and a provider-level shutdown deadline to rented hardware. A process
timeout alone does not stop GPU billing. Export results before stopping a pod.

The probe uses NF4 weights, BF16 computation, rank-16 attention LoRA,
nonreentrant gradient checkpointing, microbatch one, and only the final token's
logits. Analytical block masks avoid constructing a dense token-by-token mask.
The first optimizer step is warmup; the requested steps are measured separately.
Receipts record exact prompt length, source hashes, runtime versions, attention
backend, trainable parameters, loss, gradient checks, timing, and peak memory.

The kernel probe also requires analytical and PyTorch-generated block masks to
produce identical outputs and gradients for both attention shapes. Comparison
with math SDPA uses documented BF16 error bounds; it does not require or claim
FP32 equivalence. The shared kernel configuration uses 32-token tiles and keeps
the contiguous-block and guaranteed-safe-row hints disabled. Larger default
tiles exceeded the B200's per-block shared-memory limit for 512-wide heads.

## Optional checkpoint offloading

Before enabling CPU storage of checkpoint inputs, compare an 8K forward and
backward pass with and without offloading:

```bash
CUDA_VISIBLE_DEVICES=<gpu-uuid> python -m experiments.long_context.check_offload \
  --cache-dir /path/to/hub-cache --output artifacts/offload-check-v1
```

This loads one model, restores identical random states for both passes, performs
no optimizer updates, and compares every LoRA gradient element plus option logits
and loss. Require a successful receipt and useful GPU memory savings before
adding `--activation-offload` to a new pilot run. The context offloads checkpoint
inputs only, uses native synchronous copies, and retains 16 GiB of headroom below
the measured host cgroup limit. Offloading trades host RAM and transfer time for
GPU RAM; it does not remove the model's native context ceiling.

## Authored evidence checks

Generate deterministic fictional records with source-disjoint splits, explicit
negative evidence, absent outcomes, and early/middle/late relevant records:

```bash
python -m experiments.long_context.authored_data \
  --output artifacts/authored-v1 --sources 9 --distractors 24
CUDA_VISIBLE_DEVICES=<gpu-uuid> python -m experiments.long_context.evaluate_authored \
  --input artifacts/authored-v1/authored-long-context-test.jsonl \
  --output artifacts/authored-base-v1.predictions.jsonl \
  --summary artifacts/authored-base-v1.summary.json \
  --cache-dir /path/to/hub-cache --max-tokens 262144
```

More distractors create longer records; tokenize the resulting prompts and
freeze the files before evaluation. The evaluator never truncates. Its summary
reports actual coverage, accuracy, and macro-F1 for the authored check. These
templated records are an exploratory test rather than a public benchmark. Add
`--adapter` and `--adapter-revision` to evaluate an immutable adapter separately.

## Interpretation

The repeated synthetic state is a **memory and gradient probe**, not a training
dataset or a quality benchmark. Its arbitrary labels and losses cannot show
that the model learns useful long-context judgments. A passing receipt proves
only that this configuration completed those measured steps.

Useful long-context fine-tuning requires meaningful evidence placed at varying
distances, held-out source groups, reordered options, irrelevant-context tests,
and comparison with the frozen base. Report those separately before advertising
long-context adapter quality. Neither native context support nor this systems
probe establishes one-million-token support.

## Fixed authored training pilot

`train_authored.py` is a bounded one-epoch pilot over the existing frozen
authored JSONL. It selects exactly nine training rows using the train split
alone: one row for every semantic-label/evidence-position cell, cyclically
distributed across the three train source groups. It then reorders each row's
options deterministically with seed 42 and derives the numeric target from the
reordered semantic IDs.

The frozen plan, source-disjoint 27-row train and validation files, and output
directory are explicit inputs. The plan requires NF4/BF16 rank-16 q/k/v/o LoRA,
activation offload, one 262,144-token-or-shorter example per optimizer step,
effective batch one, learning rate 2e-4, gradient clipping at 1.0, and exactly
nine updates. This small-batch choice intentionally differs from the main
effective-batch-16 recipes. Every selected train row and all 27 validation rows
are tokenized without truncation before the first update.

```bash
CUDA_VISIBLE_DEVICES=<one-gpu-uuid> python -m experiments.long_context.train_authored \
  --plan manifests/authored-long-context-pilot-20260922.json \
  --train /path/to/authored-260k-v1/authored-long-context-train.jsonl \
  --validation /path/to/authored-260k-v1/authored-long-context-validation.jsonl \
  --output /path/to/new/authored-training-v1 \
  --cache-dir /path/to/hub-cache --max-tokens 262144 --activation-offload
```

The output is create-only. Adapter-only checkpoints are atomically saved after
every update; the ninth checkpoint is always the final model, with no validation
checkpoint selection. `steps.jsonl` records row IDs, semantic labels, prompt
hashes, token counts, finite loss/gradient evidence, clipping, timing, GPU/host
memory, and offload statistics. Final validation uses native last-position
option logits on all 27 held-out validation rows. `final-adapter/` includes the
adapter, a provenance receipt, its local revision string, and `SHA256SUMS`; pass
that receipt's `local_adapter_revision` as `--adapter-revision` when using the
authored evaluator with the local adapter path.

This pilot uses nine fictional templated examples and one seed. Its validation
is exploratory authored-data evidence, not a public benchmark, and it cannot
support claims of generalized 256K competence or any one-million-token context
extension. Evaluate matched short and long held-out test rows separately from
the training run.
