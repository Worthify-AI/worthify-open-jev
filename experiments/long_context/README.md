# Gemma 4 12B long-context experiment

This isolated experiment measures whether the pinned 12B model can complete
option-logit QLoRA updates at longer sequence lengths on one GPU. It does not
change the released scorer or the two 2,048-token training recipes.

The base revision is `google/gemma-4-12B-it` at
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`. Its configured native context is
262,144 tokens. Increasing that configuration value does not establish support
for one million tokens.

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
