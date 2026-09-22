# Dataset and evaluation contract

OpenJev converts two attributed public datasets into the same decision row used by the scorers. Third-party source snapshots and derived rows stay outside the repository. The committed recipes contain immutable URLs, revisions, SHA-256 hashes, license identifiers, split rules, and caps.

Each row contains `id`, `task`, `state`, `question`, `options`, `gold_option_id`, `group_id`, `split`, and `source`. Options use semantic IDs rather than answer letters. `gold_option_id` and source metadata are never rendered by `direct_messages`; only state, question, and option descriptions enter the prompt. Source text is marked untrusted, is preserved without truncation, and must be nonempty and at most 20,000 characters. Scoring continues to enforce its token limit without truncation.

CLINC150 uses the authors' `data_full.json` snapshot at commit `828f8093932c8fe6ca7936c3d2e52903b1c523de` (CC BY 3.0; Larson et al., *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction*). The converter deterministically reserves 15 intent labels for validation and 15 different labels for the unseen-label test slice. Held-out label IDs and descriptions are also excluded from earlier-split distractor options, so the labels are actually unseen rather than merely absent as gold answers. Official test records can only enter test. Normalized duplicate utterances stay in one split. Rows rotate across 2, 4, 8, and 16 options; each in-domain row contains its correct intent, while OOS rows use `none_of_above`. The preparation output includes separate `clinc150-test-unseen-label.jsonl` and `clinc150-test-seen-label.jsonl` files so those claims are not conflated.

WANLI uses the authors' training snapshot at revision `61c95318fd71c55b6ba355d76253254615f387ec` (CC BY 4.0; Liu et al., *WANLI: Worker and AI Collaboration for Natural Language Inference Dataset Creation*). Pair IDs and normalized premises define connected components. Whole components enter exactly one split, then deterministic balanced selection caps training at 20,000 rows and validation/test at 1,500 rows each. The official WANLI test set never enters training or internal evaluation. Preparation resolves the repository's frozen external-test selection against the pinned official test snapshot and removes every training component connected to a selected row by pair ID or normalized premise. This includes transitive connections through bridge rows. Normalization catches punctuation, case, and Unicode-form variants; it does not claim to detect semantic paraphrases.

Fetch the pinned training snapshots into a new ignored directory, and fetch the already governed WANLI evaluation snapshot through the benchmark fetcher:

```bash
python -m openjev_phase1.datasets fetch --output data/sources
python benchmarks/fetch_sources.py --output data/evaluation-sources
python -m openjev_phase1.datasets prepare \
  --clinc-source data/sources/clinc150-data_full.json \
  --wanli-source data/sources/wanli-train.jsonl \
  --wanli-exclusions benchmarks/manifests/source-selection.jsonl \
  --wanli-test-source data/evaluation-sources/wanli-test.jsonl \
  --output data/derived/openjev-data-v1
```

Both commands are create-only. Preparation verifies source hashes and writes a manifest containing source metadata, exclusion-manifest hashes, row counts, and every derived file hash. Train files are capped independently by dataset. Validation and test remain separate; model choice uses validation only.

Predictions must contain each gold ID exactly once with `option_ids` and a complete probability vector. Evaluation joins by ID, reorders probabilities by semantic option ID, and rejects missing, unknown, duplicate, nonfinite, or option-mismatched results. It reports accuracy, macro F1, multiclass Brier score, ten-bin ECE, a source-group bootstrap stratified by task, warm per-row median and p95 latency where present, and peak memory where present. Probability metrics are explicitly uncalibrated.

```bash
python -m openjev_phase1.evaluation \
  --gold data/derived/openjev-data-v1/clinc150-validation.jsonl \
  --predictions runs/model-seed-42/validation.predictions.jsonl \
  --base-model-id google/gemma-4-12B-it --base-model-revision PINNED_COMMIT_SHA40 \
  --adapter-sha256 ADAPTER_SAFETENSORS_SHA256 \
  --seed 42 --output runs/model-seed-42/validation.report.json
```

Every CLI report records `seed`, normalized `base_model` ID/revision coordinates, and `adapter_sha256` for release packaging. `evaluate_seeds` keeps a complete report for every seed. `select_candidate` requires both task reports, averages their validation macro F1, and chooses the fastest warm median per-row candidate within one absolute percentage point of the best validation mean. It rejects test-bearing selection records. Test results are generated only after that choice is frozen.

`build_robustness_examples` creates deterministic option-order and criterion-wrapper variants only for project-owned examples. It preserves the semantic gold ID and records the base ID and output-blind perturbation type.
