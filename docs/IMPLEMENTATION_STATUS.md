# Implementation status

Updated September 22, 2026. This is an implementation record, not a trained-model release.

- Dedicated GitLab project and GitHub repository created; upstream history is preserved at `53e3028363509f8533d90fe82d983770da1f6c02`. GitLab is authoritative.
- Implemented native Gemma text loading, pinned LoRA loading, one trainer with two recipes, public dataset conversion, evaluation, and checksummed Hugging Face publication.
- The current suite passes 95 tests. Original raw-result hashes and all 69 upstream summary claims still verify.
- Qwen BF16 reproduction on an RTX 3090 matches all 252 published quality decisions. Both cached shape-scoring modes reproduce their published results, including 5 and 6 disagreements with fresh scoring. Cached scoring is disabled in the Worthify runtime. See `results/worthify/upstream-reproduction-20260922.json`.
- Data and the common 1,024-row validation comparison sample are frozen; source hashes and counts are committed in `manifests/data-v1.json` and `manifests/comparison-validation-v1.json`.
- The pinned Gemma 4 12B loads and scores on one A100 40 GB with NF4/BF16. A 64-row pilot using consistent native BF16 base weights passed a fresh-process adapter reload with zero logit difference on eight validation rows. It used about 7.7 GiB and processed 1.4–1.5 training examples/second. See `results/worthify/gemma12b-pilot-bf16-20260922.json`. This corrects the earlier pilot's training-only FP32 base upcast; the earlier record remains available. Final three-candidate selection is pending; no winning base or tuned improvement is claimed.
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

The corrected 12B pilot extrapolates to roughly eight training hours for two WANLI epochs on one A100, plus evaluation. Both seeds can run independently on the two A100s. Larger candidates need their own timing after selection. October 5 remains plausible, conditional on full runs and CI access; no improvement or completed launch is promised. No cloud spending is planned.
