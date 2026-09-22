# Gemma LoRA recipes

Train two separate adapters on the same pinned base: classification/routing and evidence judgments. Each recipe has two seeds and its own test report. These are reproducible experiments; improvement is not assumed.

## Environment and model

```bash
python3 -m venv .venv
. .venv/bin/activate
pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
pip install -e '.[test,train]'
export WORK="$PWD/runs/tutorial-v1"
export CACHE="$WORK/hf-cache"
export MODEL='google/gemma-4-12B-it'
export REVISION='707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7'
export GPU=0
# On shared hosts, prefer an available GPU UUID from nvidia-smi.
mkdir -p "$WORK"
```

This 12B revision is an example, not a winning-model claim. For the project release, use the base selected by the frozen validation comparison. Change `MODEL` and `REVISION` together. Expose exactly one available GPU; CUDA numeric ordering can differ from nvidia-smi, so use a verified GPU UUID on a shared host; do not stop other workloads. Use a new `WORK` directory for each experiment.

## Fetch and freeze data

```bash
python -m openjev_phase1.datasets fetch --output "$WORK/source-snapshots"
python benchmarks/fetch_sources.py --output "$WORK/evaluation-sources"
python -m openjev_phase1.datasets prepare \
  --clinc-source "$WORK/source-snapshots/clinc150-data_full.json" \
  --wanli-source "$WORK/source-snapshots/wanli-train.jsonl" \
  --wanli-exclusions benchmarks/manifests/source-selection.jsonl \
  --wanli-test-source "$WORK/evaluation-sources/wanli-test.jsonl" \
  --output "$WORK/datasets" --seed 291607
```

CLINC150 supplies classification rows; WANLI supplies evidence rows. The converter records source licenses, revisions, checksums, split IDs, and group separation. Classification uses runtime sets of 2–16 choices; results are not the original 150-way CLINC benchmark. Unseen labels are withheld from training option descriptions as well as training targets. WANLI components related to the upstream test material are excluded from training.

## Train each recipe twice

```bash
for RECIPE in classification evidence; do
  case "$RECIPE" in
    classification) DATASET=clinc150 ;;
    evidence) DATASET=wanli ;;
  esac
  for SEED in 42 43; do
    CUDA_VISIBLE_DEVICES="$GPU" python -m openjev_phase1.training train \
      --recipe "recipes/$RECIPE.json" \
      --train "$WORK/datasets/$DATASET-train.jsonl" \
      --validation "$WORK/datasets/$DATASET-validation.jsonl" \
      --output "$WORK/runs/$RECIPE/seed$SEED" \
      --model "$MODEL" --revision "$REVISION" --seed "$SEED" \
      --quantization nf4 --cache-dir "$CACHE" --max-tokens 2048 \
      --effective-batch-size 16 --epochs 2
    CUDA_VISIBLE_DEVICES="$GPU" python scripts/verify-adapter-reload.py \
      --run-dir "$WORK/runs/$RECIPE/seed$SEED" \
      --validation "$WORK/datasets/$DATASET-validation.jsonl" \
      --cache-dir "$CACHE" \
      --output "$WORK/runs/$RECIPE/seed$SEED/fresh-reload-verification.json"
  done
done
```

Defaults are attention-only rank-16 LoRA, NF4/BF16, learning rate `2e-4`, and two epochs. Loss is computed only over the declared option logits at the final prompt position. The gold target is recomputed after option shuffling. Each run writes resumable epoch checkpoints, a validation-selected `final-adapter/`, and a manifest. Independent recipes/seeds can run on different GPUs with separate output paths.

The two recipe files validate task and source-dataset identity. Their hashes are recorded with each run. Omit `--recipe` when substituting your own labeled JSONL; the same default hyperparameters still apply. Progress records include processed examples and elapsed training time. The separate verification command loads the base and exported adapter in a fresh process, then compares their option logits against the saved training-process reference on eight validation rows. Both training and inference preserve the native BF16 base parameters, with BF16 computation inside NF4 layers and FP32 trainable LoRA parameters.

## Evaluate and package both adapters

```bash
for RECIPE in classification evidence; do
  case "$RECIPE" in
    classification) DATASET=clinc150 ;;
    evidence) DATASET=wanli ;;
  esac
  for SEED in 42 43; do
    CUDA_VISIBLE_DEVICES="$GPU" openjev-score --mode direct \
      --model "$MODEL" --revision "$REVISION" --quantization nf4 \
      --adapter "$WORK/runs/$RECIPE/seed$SEED/final-adapter" \
      --adapter-revision local-final-adapter \
      --input "$WORK/datasets/$DATASET-test.jsonl" \
      --output "$WORK/$RECIPE-predictions-seed$SEED.jsonl" \
      --cache-dir "$CACHE" --max-tokens 2048 --warmup 3
    ADAPTER_SHA256=$(sha256sum "$WORK/runs/$RECIPE/seed$SEED/final-adapter/adapter_model.safetensors" | awk '{print $1}')
    python -m openjev_phase1.evaluation \
      --gold "$WORK/datasets/$DATASET-test.jsonl" \
      --predictions "$WORK/$RECIPE-predictions-seed$SEED.jsonl" \
      --output "$WORK/$RECIPE-metrics-seed$SEED.json" \
      --seed "$SEED" --bootstrap-samples 1000 \
      --base-model-id "$MODEL" --base-model-revision "$REVISION" \
      --adapter-sha256 "$ADAPTER_SHA256"
  done
  SELECTED_SEED=$(python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); scores={s:json.loads((p/f"seed{s}"/"manifest.json").read_text())["best_validation_macro_f1"] for s in (42,43)}; print(min(scores,key=lambda s:(-scores[s],s)))' "$WORK/runs/$RECIPE")
  python -m openjev_phase1.publish package --recipe "$RECIPE" \
    --run-dir "$WORK/runs/$RECIPE/seed42" --run-dir "$WORK/runs/$RECIPE/seed43" \
    --evaluation "$WORK/$RECIPE-metrics-seed42.json" \
    --evaluation "$WORK/$RECIPE-metrics-seed43.json" \
    --selected-seed "$SELECTED_SEED" --output "$WORK/release-$RECIPE"
  python -m openjev_phase1.publish validate --artifact-dir "$WORK/release-$RECIPE"
done
```

Local adapters require an explicit revision marker; remote adapters require their immutable 40-character Hugging Face commit. Packaging verifies prediction provenance and adapter hashes. Seed selection uses validation performance only. Public release requires both recipes, both seed reports, baseline comparisons, and the checks in [RELEASE.md](RELEASE.md).

## Use your own data

Create separate training, validation, and test JSONL files. Each line follows this schema:

```json
{"id":"custom-001","task":"classification","group_id":"customer-17","split":"train","state":"Customer asks to change a delivery address.","question":"Which request type applies?","options":[{"id":"address_change","description":"Change a delivery address"},{"id":"cancel","description":"Cancel an order"}],"gold_option_id":"address_change"}
```

IDs and semantic option IDs must be stable. Keep related records and normalized evidence out of different splits. Supply 2–16 unique choices and a matching gold ID. Gold labels and provenance fields never enter the prompt. Large taxonomies need a separately evaluated routing hierarchy; do not silently truncate choices. Include an explicit insufficient-evidence or none-of-the-above option when the task needs one. Scores are conditional on the supplied choices and are uncalibrated.

Publish only data you have rights to redistribute. Keep private company records, credentials, model caches, and unlicensed evaluation material outside the repository.
