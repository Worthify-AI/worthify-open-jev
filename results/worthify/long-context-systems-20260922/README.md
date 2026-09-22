# Gemma 4 12B on one B200: long-context systems measurements

These September 22, 2026 measurements use the pinned
`google/gemma-4-12B-it` revision
`707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, NF4 weights, BF16 computation,
attention LoRA rank 16, and one NVIDIA B200. The original A100 recipe
benchmarks are separate.

The complete model performed forward, backward, and optimizer updates at
8,192, 16,384, 32,768, 65,536, 131,072, and 262,144 input tokens. Each run
contains one warmup update and two measured updates. All recorded updates
passed the finite-gradient check. At 262,144 tokens, the two measured updates
averaged **580.78 seconds** and peaked at **134.38 GiB allocated** and
**160.68 GiB reserved** GPU memory.

The prompts contain repeated filler and arbitrary option targets. These are
memory and gradient measurements, not useful training examples or model-quality
results. No adapter from these probes is a release candidate. A separate authored
evidence experiment evaluates judgments and meaningful long-context updates.

## Implementation and checks

Native FlexAttention uses analytical causal block masks, a 1,024-token window
on local layers, and the model's existing global layers. It computes only the
final token's option logits. Full dense token-by-token masks are never built.
The final 256K run additionally offloads checkpoint inputs to CPU RAM, while
retaining 16 GiB below the host cgroup memory limit.

The 8K offload comparison restored identical random states and compared every
LoRA gradient element, option logits, and loss. It found zero difference and
saved 2.45 GiB of peak allocated GPU memory. This bounded equivalence check does
not constitute a proof for every input or sequence length.

The final kernel check (`kernel-evidence/kernel-v7.json`) compares both native
attention head shapes against PyTorch's math SDPA using documented BF16 error
bounds. It also requires analytical and PyTorch-generated masks to produce
identical outputs and gradients. Earlier failed attempts remain in the bundle:
they exposed incorrect sparse index strides, unsafe row hints, and oversized
512-wide-head kernel tiles. An intermediate development receipt is not a passing
kernel validation; use the final v7 receipt.

The first five length probes predate explicit random-seed recording. Their exact
pilot source is retained in `source-snapshots/pilot-initial.py`. The final 256K
run records seed 42. Treat the ladder as exploratory systems measurements,
not a controlled comparison of optimization trajectories or model quality.

## Reproduction and scope

Follow `experiments/long_context/README.md` at the source revision recorded in
`sources.json`. Use one visible GPU, the pinned environment, fresh output paths,
and the kernel and offload gates before launching long runs. The receipts retain
runtime versions, source hashes, token counts, hardware, timing, gradient checks,
and memory observations. Recompute this bundle's summary with:

```bash
python benchmarks/verify_long_context.py \
  --artifact-dir results/worthify/long-context-systems-20260922
```

Two measured updates provide a small timing sample. Reserved memory differs
from live allocated memory, and CPU offloading adds transfer overhead. This
does not establish a speed advantage, general long-context competence, or
one-million-token support. The pinned base's native context limit is 262,144.
