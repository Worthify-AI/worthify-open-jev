# Draft: OpenJev Phase 1 training framework

OpenJev Phase 1 has a reproducible framework for preparing governed classification and evidence data, training a LoRA adapter with two fixed seeds, and evaluating predictions with provenance checks. It pins base-model revisions, keeps training and test splits separate, and selects final adapters on validation evidence.

This is not a model release. The results table is pending completed two-seed runs and reviewed evidence. This draft makes no trained-adapter, benchmark-result, or quality claim.

CLINC150 and WANLI are fetched at pinned revisions, while the governed WANLI upstream-test selection is excluded from internal splits. Labels remain outside prompts. Evaluation requires complete option probabilities and verifies the base-model identity, adapter hash, and seed for every prediction.

Any future release will first be staged privately and hash-verified. A public adapter would carry its Apache-2.0 card and license, pinned base identity, aggregate evidence for both seeds, and linked training and evaluation manifests. Until then, this document remains a draft and is not a publication announcement.
