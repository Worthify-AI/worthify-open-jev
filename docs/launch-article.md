# DRAFT: Worthify OpenJev brings small, typed decisions to open models

> **Draft status:** This article is waiting for the completed two-seed training runs, reviewed final measurements, and public Hugging Face adapters. It does not announce a model release.

Software often needs a small decision from a large amount of text: route a request, choose the next workflow step, or decide whether a record supports a claim. A chat model can produce an explanation, but many applications only need a typed choice that code can use.

Worthify OpenJev explores that narrower job with open models. It extends the open-source OpenJev project, which reproduces the public interface pattern of TypeSafe's Jev service. We supply a state, a question or criterion, and between 2 and 16 described options at runtime. The model scores those options directly at its next-token position and returns their conditional probabilities. It does not generate an answer sentence or require a parser to recover the choice.

The options do not have to come from a fixed taxonomy baked into the model. One request might route a customer message among account access, billing, and delivery. Another might ask whether supplied records support, contradict, or fail to establish a statement. Applications can add an explicit “insufficient evidence” or “none of the above” option when the workflow needs one.

Consider a defensive security review. The state could contain a gateway's maintenance record and a vendor advisory. The question asks whether the records establish that the gateway received the fixed release. The available choices are *supported*, *contradicted*, and *insufficient*. OpenJev returns one score for each choice. The surrounding application still owns the policy and the action; the model supplies a bounded judgment about the text it received.

## Two reproducible training recipes

The Worthify extension adds separate LoRA recipes for classification and routing, and for evidence judgments. The classification recipe converts attributed CLINC150 data into runtime candidate sets. The evidence recipe converts attributed WANLI records into supported, contradicted, and insufficient judgments. Each recipe trains twice with fixed seeds, holds related examples apart across data splits, and chooses its release adapter from validation results before examining held-out test performance.

LoRA keeps the trained artifact small and reusable while the base model remains a separate, pinned dependency. Worthify's source additions use the MIT license. The candidate Gemma bases use Apache 2.0, as will any released derived adapters, with the required attribution and fine-tuning notice. CLINC150 and WANLI retain their CC BY 3.0 and CC BY 4.0 terms. The release package excludes base weights, source data, row-level predictions, caches, credentials, and private company records.

## Measuring the adapter honestly

We will report the frozen base and the tuned adapter side by side on the same held-out inputs. That comparison matters because an adapter is useful only if the measured change survives a fixed evaluation. The final report will include both seeds for each recipe, the validation-selected seed, provenance hashes, and separate task metrics instead of one blended score.

The groundwork has passed two limited checks. A pinned Qwen baseline reproduction matched all 252 published quality decisions, which verifies the evaluation path. A 64-row Gemma 4 12B training pilot loaded, trained, exported, and reloaded its adapter successfully. The pilot tested compatibility and timing; it did not select the winning base, measure full training, or establish a quality improvement.

Both recipes use Gemma 4 12B at a pinned revision. This first release focuses on that base and does not claim it won a completed comparison against larger models. The four full training runs are in progress. Cached prefix and shared-state scoring are disabled for Worthify release measurements because equivalence testing found decision changes against fresh direct scoring. Release results will use fresh direct scoring. Returned probabilities are conditional on the supplied options and are not calibrated confidence values; deployment teams must validate and calibrate them for their own workload and decision costs.

## Release evidence to insert after review

Before publication, replace this block from the checksummed release manifests and reviewed aggregate reports:

- **Base:** `google/gemma-4-12B-it @ 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`
- **Classification:** `frozen [METRIC]; tuned seed 42 [METRIC]; tuned seed 43 [METRIC]; selected seed [42/43] by validation macro-F1`
- **Evidence judgments:** `frozen [METRIC]; tuned seed 42 [METRIC]; tuned seed 43 [METRIC]; selected seed [42/43] by validation macro-F1`
- **Public adapters:** `Worthify/[CLASSIFICATION REPO] @ [40-CHAR COMMIT]`; `Worthify/[EVIDENCE REPO] @ [40-CHAR COMMIT]`
- **Reviewed limitations:** `[FINAL TASK-SPECIFIC LIMITATIONS AND CALIBRATION NOTES]`

This work does not reproduce Jev's undisclosed model or training. Worthify does not use Jev outputs as training labels. The project studies a public interface pattern with open code, pinned open-model revisions, attributed public datasets, and evidence that readers can inspect. Once the two recipes, two seeds, immutable public adapter commits, and final review are complete, this draft can become the release article.
