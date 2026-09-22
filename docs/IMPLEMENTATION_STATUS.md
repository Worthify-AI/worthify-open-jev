# Implementation status

Updated September 22, 2026. This is an implementation record, not a trained-model release.

- Dedicated GitLab project and GitHub repository created; upstream history is preserved at `53e3028363509f8533d90fe82d983770da1f6c02`. GitLab is authoritative.
- Implemented native Gemma text loading, pinned LoRA loading, one trainer with two recipes, public dataset conversion, evaluation, and checksummed Hugging Face publication.
- The current suite passes 101 tests. Original raw-result hashes and all 69 upstream summary claims still verify.
- Qwen BF16 reproduction on an RTX 3090 matches all 252 published quality decisions. Both cached shape-scoring modes reproduce their published results, including 5 and 6 disagreements with fresh scoring. Cached scoring is disabled in the Worthify runtime. See `results/worthify/upstream-reproduction-20260922.json`.
- Data and the common 1,024-row validation comparison sample are frozen; source hashes and counts are committed in `manifests/data-v1.json` and `manifests/comparison-validation-v1.json`.
- The pinned Gemma 4 12B loads and scores on one A100 40 GB with NF4/BF16. A 64-row pilot using consistent native BF16 base weights passed a fresh-process adapter reload with zero logit difference on eight validation rows. It used about 7.7 GiB and processed 1.4–1.5 training examples/second. See `results/worthify/gemma12b-pilot-bf16-20260922.json`. This corrects the earlier pilot's training-only FP32 base upcast; the earlier record remains available.
- The project owner selected 12B for both recipes on September 22, independently of the larger-model comparison. `manifests/base-selection-v1.json` records the choice. The 26B MoE exceeds 40 GB with the frozen NF4 loader; 31B work was interrupted to free both A100s for 12B training. No comparative winning-model or tuned-improvement claim is made.
- Both full training queues started September 22 at 13:07 UTC from source commit `6f1e47ef563095d83b87c57e2e6956c238b1f67b`: one seed per A100, classification followed by evidence, two epochs per recipe. Fresh adapter verification and held-out evaluation are queued after each recipe. Frozen baselines and verified private Hub uploads follow successful completion of both queues. No full-run result is claimed yet.
- Reference hardware: one A100 40 GB per run, with a second A100 available for independent experiments. Existing workloads are preserved. Use verified GPU UUIDs because CUDA numeric ordering can differ from nvidia-smi.
- Source is public at https://github.com/Worthify-AI/worthify-open-jev and matches the GitLab default branch. Repository-scoped SSH credentials and the one-way mirror script were tested locally. Installing GitLab CI variables and verifying the actual pipeline remain pending.
- No Worthify adapter has been publicly released. The launch article remains a draft pending results and downloadable adapters.

## Access dependencies

The Hugging Face organization `Worthify` is on the Team plan, which is sufficient. The authorized task-scoped login passed automated private upload, immutable download, and checksum verification at `Worthify/worthify-jev-upload-test`, commit `5c6eda33c31f721534f9a941430ab715491c81da`. The non-secret receipt is recorded in `results/worthify/hf-private-transport-20260922.json`. Planned public adapters are `Worthify/worthify-jev-classification` and `Worthify/worthify-jev-evidence`. GitLab API credentials are expired; SSH pushes work, but CI secret and mirror configuration requires restored API/browser access. Local training and publication have the required Hub access; the actual GitLab CI transport job remains unverified.

## Schedule

| Window | Deliverable |
|---|---|
| September 22–23 | Repository, licensing inventory, private Hub transport proof, source mirror |
| September 24–25 | Qwen reproduction, common Gemma validation comparison, frozen datasets |
| September 28–30 | Classification and evidence recipes, both seeds |
| October 1–5 | Final benchmarks, clean-install verification, public adapters, tutorials, live website article |
| October 6–7 | Access or compatibility contingency |

Early full-data classification rates are about 1.5 examples/second; the evidence pilot measured about 1.46 examples/second. Together they estimate 12–16 hours for both recipes per seed, with the seeds running in parallel, including validation and evaluation. Initial training/evaluation completion is expected by September 23, subject to successful full runs. October 5 remains the public-launch target, conditional on final verification and CI access. No improvement or completed launch is promised. No cloud spending is planned.
