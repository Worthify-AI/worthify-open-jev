"""Shared input validation, prompts, model loading, and numeric helpers."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

LETTERS = "ABCDEFGHIJKLMNOP"
DIRECT_SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)


def validate_row(row: dict) -> None:
    required = {"id", "state", "question", "options"}
    if not required <= row.keys():
        raise ValueError(f"Row is missing fields: {sorted(required - row.keys())}")
    if not all(isinstance(row[key], str) and row[key] for key in ("id", "question")):
        raise ValueError("id and question must be nonempty strings")
    state = row["state"]
    if not isinstance(state, (str, dict, list)) or not state:
        raise ValueError("state must be a nonempty string, object, or array")
    try:
        json.dumps(state, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("state must be finite JSON-compatible data") from error
    options = row["options"]
    if not isinstance(options, list) or not 2 <= len(options) <= len(LETTERS):
        raise ValueError("options must contain 2-16 entries")
    ids = []
    for option in options:
        if not isinstance(option, dict) or not isinstance(option.get("id"), str) or not isinstance(option.get("description"), str):
            raise ValueError("Each option needs string id and description fields")
        ids.append(option["id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Option IDs must be unique")


def direct_messages(row: dict) -> list[dict]:
    validate_row(row)
    payload = {
        "evidence": row["state"],
        "criterion": row["question"],
        "options": [
            {"letter": LETTERS[index], "description": option["description"]}
            for index, option in enumerate(row["options"])
        ],
    }
    return [
        {"role": "system", "content": DIRECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def softmax(values: list[float]) -> list[float]:
    if len(values) < 2 or any(not math.isfinite(value) for value in values):
        raise ValueError("Need at least two finite scores")
    maximum = max(values)
    weights = [math.exp(value - maximum) for value in values]
    total = sum(weights)
    return [weight / total for weight in weights]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _pinned_source(source: str, revision: str, kind: str) -> bool:
    """Validate a local manifest or immutable Hub commit and return local status."""
    local = Path(source).exists()
    if local and not Path(source).is_dir():
        raise ValueError(f"Local {kind} must be a checkpoint directory")
    if not local and not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
        raise ValueError(f"Remote {kind} require a pinned 40-character commit revision")
    if local and not revision:
        raise ValueError(f"Local {kind} require an explicit manifest/revision string")
    return local


def _native_model_class(transformers, model_type: str):
    """Select the text architecture explicitly; never coerce Unified into Gemma 4."""
    if model_type in {"qwen3_5", "qwen3_5_text"}:
        name, label = "Qwen3_5ForCausalLM", "Qwen3.5"
    elif model_type in {"gemma4_unified", "gemma4_unified_text"}:
        name, label = "Gemma4UnifiedForCausalLM", "text-only Gemma4 Unified"
    elif model_type in {"gemma4", "gemma4_text"}:
        name, label = "Gemma4ForCausalLM", "text-only Gemma4"
    else:
        return transformers.AutoModelForCausalLM
    cls = getattr(transformers, name, None)
    if cls is None:
        raise RuntimeError(f"Installed transformers lacks the native {label} model")
    return cls


def _gemma_text_checkpoint(source: str, revision: str, cls, config, *, local: bool,
                           cache_dir: str | None = None) -> dict:
    """Check pinned tensor headers against the native text model before GPU allocation.

    Google's full checkpoints nest text tensors under model.language_model, while
    the native causal classes expect model. Transformers 5.17 does not provide
    this conversion for either Gemma 4 architecture.
    """
    import torch
    from safetensors import safe_open
    from transformers.initialization import no_init_weights

    def checkpoint_file(name: str, *, optional: bool = False):
        if local:
            path = Path(source) / name
            if not path.is_file():
                if optional:
                    return None
                raise ValueError(f"Gemma checkpoint is missing {name}")
            return path
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import EntryNotFoundError

        try:
            return Path(hf_hub_download(source, name, revision=revision, cache_dir=cache_dir))
        except EntryNotFoundError:
            if optional:
                return None
            raise

    index = checkpoint_file("model.safetensors.index.json", optional=True)
    if index is None:
        filenames = ["model.safetensors"]
    else:
        weight_map = json.loads(index.read_text())["weight_map"]
        filenames = sorted(set(weight_map.values()))
        if any(Path(name).name != name or not name.endswith(".safetensors") for name in filenames):
            raise ValueError("Checkpoint index contains invalid shard paths")
    shapes = {}
    for filename in filenames:
        with safe_open(checkpoint_file(filename), framework="pt", device="cpu") as weights:
            for key in weights.keys():
                if key in shapes:
                    raise ValueError(f"Duplicate checkpoint tensor: {key}")
                shapes[key] = tuple(weights.get_slice(key).get_shape())
    key_mapping = {r"^model\.language_model\.": "model."} if any(
        key.startswith("model.language_model.") for key in shapes
    ) else {}
    mapped = {}
    for key, shape in shapes.items():
        target = key
        for pattern, replacement in key_mapping.items():
            target = re.sub(pattern, replacement, target)
        if target in mapped:
            raise ValueError(f"Checkpoint key conversion collides at {target}")
        mapped[target] = shape
    # Meta tensors provide names and shapes without allocating model weights.
    with no_init_weights(), torch.device("meta"):
        skeleton = cls(config)
    expected = {key: tuple(tensor.shape) for key, tensor in skeleton.state_dict().items()}
    ties = skeleton.all_tied_weights_keys
    missing = sorted(key for key in expected if key not in mapped and not (
        key in ties and ties[key] in mapped and mapped[ties[key]] == expected[key]
    ))
    mismatched = sorted(key for key in expected.keys() & mapped.keys() if expected[key] != mapped[key])
    del skeleton
    if missing or mismatched:
        raise RuntimeError(f"Gemma text checkpoint failed preflight: missing={missing[:12]}, mismatched={mismatched[:12]}")
    return {"key_mapping": key_mapping} if key_mapping else {}


def load_causal_model(
    source: str,
    revision: str,
    *,
    adapter: str | None = None,
    adapter_revision: str | None = None,
    quantization: str = "none",
    cache_dir: str | None = None,
):
    """Load a pinned, native text-only causal model on one visible CUDA device.

    The returned model deliberately has caching disabled for Gemma 4 unless a caller
    has explicitly certified that model/cache combination.
    """
    import torch
    import transformers

    if quantization not in {"none", "nf4"}:
        raise ValueError("quantization must be 'none' or 'nf4'")
    local = _pinned_source(source, revision, "models")
    adapter_local = False
    if adapter:
        adapter_local = _pinned_source(adapter, adapter_revision or "", "adapters")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("Expose exactly one CUDA GPU, for example with CUDA_VISIBLE_DEVICES")
    common = {
        "revision": None if local else revision,
        "local_files_only": local,
        "trust_remote_code": False,
        "cache_dir": cache_dir,
    }
    config = transformers.AutoConfig.from_pretrained(source, **common)
    is_gemma = config.model_type in {"gemma4", "gemma4_text", "gemma4_unified", "gemma4_unified_text"}
    tokenizer = transformers.AutoTokenizer.from_pretrained(source, **common)
    cls = _native_model_class(transformers, config.model_type)
    if config.model_type in {"qwen3_5", "qwen3_5_text"}:
        config = config.get_text_config()
    elif is_gemma:
        config = config.get_text_config() if hasattr(config, "get_text_config") else config
    loading_kwargs = {}
    if is_gemma:
        loading_kwargs.update(_gemma_text_checkpoint(source, revision, cls, config, local=local, cache_dir=cache_dir))
    if quantization == "nf4":
        bits = getattr(transformers, "BitsAndBytesConfig", None)
        if bits is None:
            raise RuntimeError("NF4 requested but transformers BitsAndBytesConfig is unavailable")
        loading_kwargs["quantization_config"] = bits(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model, loading = cls.from_pretrained(
        source,
        config=config,
        dtype=torch.bfloat16,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True,
        output_loading_info=True,
        **common,
        **loading_kwargs,
    )
    if any(loading.get(key) for key in ("missing_keys", "mismatched_keys", "error_msgs")):
        raise RuntimeError(f"Checkpoint did not load completely: {loading}")
    if is_gemma:
        model.config.use_cache = False
    if adapter:
        try:
            import peft
        except ImportError as error:
            raise RuntimeError("Adapter loading requires peft") from error
        adapter_common = {
            "revision": None if adapter_local else adapter_revision,
            "local_files_only": adapter_local,
            "cache_dir": cache_dir,
        }
        model = peft.PeftModel.from_pretrained(model, adapter, is_trainable=False, **adapter_common)
    model.eval()
    metadata = {
        "source": source,
        "revision": revision,
        "dtype": "bfloat16",
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "quantization": quantization,
        "use_cache": bool(getattr(model.config, "use_cache", False)),
    }
    if quantization == "nf4":
        import bitsandbytes

        metadata["bitsandbytes_version"] = bitsandbytes.__version__
    if adapter:
        import peft

        if adapter_local:
            adapter_weights = Path(adapter) / "adapter_model.safetensors"
            training_manifest_path = Path(adapter) / "training-manifest.json"
            if not training_manifest_path.exists():
                training_manifest_path = Path(adapter).parent / "manifest.json"
        else:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.errors import EntryNotFoundError

            adapter_weights = Path(hf_hub_download(adapter, "adapter_model.safetensors", revision=adapter_revision, cache_dir=cache_dir))
            try:
                training_manifest_path = Path(hf_hub_download(adapter, "training-manifest.json", revision=adapter_revision, cache_dir=cache_dir))
            except EntryNotFoundError:
                training_manifest_path = None
        if not adapter_weights.is_file():
            raise ValueError("Adapters must provide adapter_model.safetensors")
        adapter_hash = hashlib.sha256()
        with adapter_weights.open("rb") as weights:
            for block in iter(lambda: weights.read(1024 * 1024), b""):
                adapter_hash.update(block)
        training_seed = None
        if training_manifest_path is not None and training_manifest_path.is_file():
            training_manifest = json.loads(training_manifest_path.read_text())
            trained_base = training_manifest.get("model", {})
            if trained_base.get("source") != source or trained_base.get("revision") != revision:
                raise ValueError("Adapter training manifest does not match the selected base model revision")
            training_seed = training_manifest.get("seed")
            if not isinstance(training_seed, int) or isinstance(training_seed, bool):
                raise ValueError("Adapter training manifest must declare an integer seed")
        metadata.update({
            "adapter": adapter,
            "adapter_revision": adapter_revision,
            "adapter_sha256": adapter_hash.hexdigest(),
            "training_seed": training_seed,
            "peft_version": peft.__version__,
        })
    return model, tokenizer, metadata
