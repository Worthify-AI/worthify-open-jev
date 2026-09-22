"""Bounded single-GPU long-context memory/gradient probe for pinned Gemma 4.

This is deliberately a systems probe.  Its repeated synthetic state is not a
training dataset, and its loss is not a model-quality metric.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import os
import resource
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

from openjev_phase1.core import direct_messages, load_causal_model
from openjev_phase1.direct import _forward, _slot_ids
from openjev_phase1.training import _configure_lora

try:
    from experiments.long_context.kernel_probe import gemma4_flex_kernel_options
except ModuleNotFoundError as error:
    if error.name not in {"experiments", "experiments.long_context"}:
        raise
    from kernel_probe import gemma4_flex_kernel_options


MODEL = "google/gemma-4-12B-it"
REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
MAX_NATIVE_CONTEXT = 262_144
FILLER_FRAGMENTS = (" x", " z", " 0", " memory")
RECEIPT_NAME = "probe-receipt.json"
FLEX_KERNEL_OPTIONS = gemma4_flex_kernel_options()


class ProbeTerminated(RuntimeError):
    """Raised when an external timeout asks the probe to terminate cleanly."""


def validate_token_length(tokens: int) -> int:
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 1:
        raise ValueError("--tokens must be one positive exact token length")
    if tokens > MAX_NATIVE_CONTEXT:
        raise ValueError(
            f"--tokens exceeds the pinned native context limit of {MAX_NATIVE_CONTEXT}"
        )
    return tokens


def _probe_row(state: str) -> dict:
    return {
        "id": "synthetic-memory-gradient-probe",
        "state": state,
        "question": "For this synthetic systems probe, select one authored option label.",
        "options": [
            {"id": "synthetic-a", "description": "authored probe label one"},
            {"id": "synthetic-b", "description": "authored probe label two"},
        ],
    }


def _render_probe_prompt(tokenizer, repeats: int, filler: str) -> str:
    state = (
        "MEMORY/GRADIENT PROBE. Synthetic token-space expansion; not evidence, "
        "not a training dataset, and not a quality metric. Repeated filler follows:"
        + filler * repeats
    )
    return tokenizer.apply_chat_template(
        direct_messages(_probe_row(state)),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def build_probe_encoding(tokenizer, tokens: int) -> dict:
    """Build one exact-length prompt while preserving the runtime option boundary."""
    validate_token_length(tokens)
    base_prompt = _render_probe_prompt(tokenizer, 0, FILLER_FRAGMENTS[0])
    base_ids = tokenizer.encode(base_prompt, add_special_tokens=False)
    if tokens < len(base_ids):
        raise ValueError(
            f"--tokens={tokens} is shorter than the synthetic prompt shell ({len(base_ids)} tokens)"
        )

    selected = None
    for filler in FILLER_FRAGMENTS:
        standalone = tokenizer.encode(filler, add_special_tokens=False)
        lengths = [
            len(tokenizer.encode(_render_probe_prompt(tokenizer, count, filler), add_special_tokens=False))
            for count in (0, 1, 2, 3)
        ]
        if (
            len(standalone) == 1
            and tokenizer.encode(tokenizer.decode(standalone), add_special_tokens=False) == standalone
            # The first insertion can change tokenization at the state boundary;
            # require stable subsequent increments and correct the final length
            # below. Gemma's real tokenizer adds two tokens for the first " x".
            and lengths[2] - lengths[1] == lengths[3] - lengths[2] == 1
        ):
            selected = filler
            break
    if selected is None:
        raise RuntimeError("Tokenizer has no verified one-token repeated probe filler")

    repeats = tokens - len(base_ids)
    prompt = ""
    ids: list[int] = []
    for _ in range(8):
        if repeats < 0:
            break
        prompt = _render_probe_prompt(tokenizer, repeats, selected)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) == tokens:
            break
        repeats += tokens - len(ids)
    if len(ids) != tokens:
        raise RuntimeError(f"Could not construct exactly {tokens} prompt tokens; got {len(ids)}")

    slots = _slot_ids(tokenizer, 2)
    for letter, slot in zip(("A", "B"), slots):
        if tokenizer.encode(prompt + letter, add_special_tokens=False) != ids + [slot]:
            raise RuntimeError(f"Synthetic prompt boundary does not preserve one-token option {letter}")
    return {
        "input_ids": ids,
        "option_token_ids": slots,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "filler_token_id": tokenizer.encode(selected, add_special_tokens=False)[0],
        "filler_repetitions": repeats,
        "actual_tokens": len(ids),
    }


def _runtime_identity(root: Path) -> dict:
    versions = {"python": sys.version.split()[0]}
    for package in ("torch", "transformers", "peft", "bitsandbytes", "triton"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"

    paths = [
        Path(__file__),
        root / "experiments" / "long_context" / "kernel_probe.py",
        root / "src" / "openjev_phase1" / "core.py",
        root / "src" / "openjev_phase1" / "direct.py",
        root / "src" / "openjev_phase1" / "training.py",
    ]
    hashes = {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.is_file()
    }

    def git(*args: str) -> str:
        try:
            return subprocess.run(
                ("git", *args), cwd=root, check=False, capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            return ""

    return {
        "git_head": git("rev-parse", "HEAD") or "unavailable",
        "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "source_sha256s": hashes,
        "versions": versions,
    }


def _rss_bytes() -> dict:
    # Linux reports ru_maxrss in KiB.  /proc gives a useful current reading too.
    current = None
    try:
        fields = {}
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith(("VmRSS:", "VmHWM:")):
                key, value, _unit = line.split()
                fields[key.rstrip(":")] = int(value) * 1024
        current = fields.get("VmRSS")
        high_water = fields.get("VmHWM")
    except (OSError, ValueError):
        high_water = None
    rusage = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
    return {"current": current, "high_water": high_water or rusage}


def _host_memory_limit_bytes() -> int | None:
    """Return the active cgroup memory ceiling without guessing from host RAM."""
    for path in (Path("/sys/fs/cgroup/memory.max"), Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")):
        try:
            value = path.read_text().strip()
            if value != "max":
                return int(value)
        except (OSError, ValueError):
            continue
    return None


def _nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            ("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"),
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ",".join(versions) or None


def _model_config(model):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    config = base.config
    return config.get_text_config() if hasattr(config, "get_text_config") else config


def _select_and_verify_flex(model) -> dict:
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    setter = getattr(base, "set_attn_implementation", None)
    if not callable(setter):
        raise RuntimeError("Loaded native model does not support set_attn_implementation")
    setter("flex_attention")
    config = _model_config(model)
    backends = Counter()
    for module in base.modules():
        module_config = getattr(module, "config", None)
        if module.__class__.__name__.endswith("Attention") and module_config is not None:
            backends[str(getattr(module_config, "_attn_implementation", None))] += 1
    if not backends or set(backends) != {"flex_attention"}:
        raise RuntimeError(f"Attention backend is not uniformly flex_attention: {dict(backends)}")
    layer_types = list(getattr(config, "layer_types", []))
    if int(getattr(config, "num_hidden_layers", -1)) != 48 or len(layer_types) != 48:
        raise RuntimeError("Pinned Gemma 4 probe requires the native 48-layer text model")
    base_forward = inspect.signature(base.forward).parameters
    if "logits_to_keep" not in base_forward:
        raise RuntimeError("Native forward lacks logits_to_keep; refusing an L-by-vocabulary allocation")
    return {
        "attention_module_backends": dict(sorted(backends.items())),
        "layer_type_distribution": dict(sorted(Counter(layer_types).items())),
        "num_hidden_layers": len(layer_types),
        "configured_backend": str(getattr(config, "_attn_implementation", None)),
    }


def _prepare_masks(model, sequence_length: int, device) -> tuple[dict, dict]:
    from torch.nn.attention.flex_attention import BlockMask

    try:
        from experiments.long_context.kernel_probe import build_analytical_causal_block_mask
    except ModuleNotFoundError as error:
        # Direct ``python experiments/long_context/pilot.py`` execution puts the
        # script directory, rather than the repository root, on sys.path.
        if error.name not in {"experiments", "experiments.long_context"}:
            raise
        from kernel_probe import build_analytical_causal_block_mask

    config = _model_config(model)
    layer_types = set(getattr(config, "layer_types", []))
    unsupported = layer_types - {"full_attention", "sliding_attention"}
    if unsupported:
        raise RuntimeError(f"Unsupported native attention layer types: {sorted(unsupported)}")
    masks = {}
    if "full_attention" in layer_types:
        masks["full_attention"] = build_analytical_causal_block_mask(
            sequence_length, device=device
        )
    if "sliding_attention" in layer_types:
        window = getattr(config, "sliding_window", None)
        if not isinstance(window, int) or window < 1:
            raise RuntimeError("Sliding layers require one positive native sliding_window")
        masks["sliding_attention"] = build_analytical_causal_block_mask(
            sequence_length, window_size=window, device=device
        )
    if set(masks) != layer_types or any(not isinstance(mask, BlockMask) for mask in masks.values()):
        raise RuntimeError("Mask preparation returned a dense or unsupported mask")
    inventory = {
        name: {
            "class": type(mask).__name__,
            "shape": list(mask.shape),
            "block_size": list(mask.BLOCK_SIZE),
        }
        for name, mask in sorted(masks.items())
    }
    return masks, inventory


def _prepare_existing_adapter(model) -> str:
    configs = getattr(model, "peft_config", None)
    if not configs or len(configs) != 1:
        raise RuntimeError("Pinned input adapter must expose exactly one PEFT configuration")
    name, config = next(iter(configs.items()))
    targets = set(config.target_modules or ())
    if config.r != 16 or targets != {"q_proj", "k_proj", "v_proj", "o_proj"}:
        raise RuntimeError("Pinned input adapter must be rank-16 q/k/v/o LoRA")
    model.set_adapter(name)
    for parameter_name, parameter in model.named_parameters():
        parameter.requires_grad_(".lora_" in parameter_name)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    return importlib.metadata.version("peft")


def _write_receipt(output: Path, receipt: dict) -> None:
    (output / RECEIPT_NAME).write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")


def run_probe(args) -> dict:
    import torch

    validate_token_length(args.tokens)
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    if args.output.exists():
        raise ValueError("--output must be a new create-only directory")
    if bool(args.adapter) != bool(args.adapter_revision):
        raise ValueError("--adapter and --adapter-revision must be provided together")
    args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    receipt = {
        "schema": "openjev-long-context-memory-gradient-probe-v1",
        "status": "running",
        "support_claim": False,
        "probe_kind": "synthetic MEMORY/GRADIENT PROBE",
        "limitations": [
            "Synthetic token-space expansion is not a training dataset.",
            "Authored synthetic option labels are not meaningful targets or a quality metric.",
            "A completed run is only an observed single-GPU systems measurement at this exact configuration.",
        ],
        "requested": {
            "tokens": args.tokens,
            "warmup_steps": 1,
            "measured_steps": args.steps,
            "model": MODEL,
            "revision": REVISION,
            "quantization": args.quantization,
            "adapter": args.adapter,
            "adapter_revision": args.adapter_revision,
        },
        "runtime": _runtime_identity(root),
        "started_unix": time.time(),
    }
    _write_receipt(args.output, receipt)

    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
        if torch._dynamo.config.suppress_errors:
            raise RuntimeError("Compiler fallback must not be enabled for the probe")
        device = torch.device("cuda:0")
        properties = torch.cuda.get_device_properties(device)
        receipt["hardware"] = {
            "name": properties.name,
            "total_memory_bytes": properties.total_memory,
            "compute_capability": [properties.major, properties.minor],
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": _nvidia_driver_version(),
            "visible_device_count": torch.cuda.device_count(),
            "host_cgroup_memory_limit_bytes": _host_memory_limit_bytes(),
        }

        load_started = time.perf_counter()
        model, tokenizer, metadata = load_causal_model(
            MODEL,
            REVISION,
            adapter=args.adapter,
            adapter_revision=args.adapter_revision,
            quantization=args.quantization,
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        backend = _select_and_verify_flex(model)
        model.config.use_cache = False
        if args.adapter:
            peft_version = _prepare_existing_adapter(model)
        else:
            model, peft_version = _configure_lora(model, args.quantization)
        model.train()
        config = _model_config(model)
        if args.tokens > int(getattr(config, "max_position_embeddings", MAX_NATIVE_CONTEXT)):
            raise RuntimeError("Requested length exceeds the loaded model's native context")
        encoding = build_probe_encoding(tokenizer, args.tokens)
        input_ids = torch.tensor([encoding.pop("input_ids")], dtype=torch.long, device=device)
        position_ids = torch.arange(args.tokens, dtype=torch.long, device=device).unsqueeze(0)
        masks, mask_inventory = _prepare_masks(model, args.tokens, device)
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or any(".lora_" not in name for name, _ in trainable):
            raise RuntimeError("Trainable parameters must be restricted to LoRA adapters")
        receipt["model"] = {
            **metadata,
            "peft_version": peft_version,
            "use_cache": bool(model.config.use_cache),
            "backend": backend,
            "flex_kernel_options": FLEX_KERNEL_OPTIONS,
            "masks": mask_inventory,
            "total_parameter_count": sum(p.numel() for p in model.parameters()),
            "trainable_parameter_count": sum(p.numel() for _, p in trainable),
            "parameter_dtype_distribution": dict(sorted(Counter(
                str(parameter.dtype) for parameter in model.parameters()
            ).items())),
            "trainable_dtype_distribution": dict(sorted(Counter(
                str(parameter.dtype) for _, parameter in trainable
            ).items())),
            "gradient_checkpointing": bool(getattr(model, "is_gradient_checkpointing", False)),
        }
        receipt["workload"] = encoding
        receipt["load_seconds"] = time.perf_counter() - load_started
        receipt["memory_after_load"] = {
            "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
            "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
            "host_rss_bytes": _rss_bytes(),
        }
        _write_receipt(args.output, receipt)

        optimizer = torch.optim.AdamW((p for _, p in trainable), lr=2e-4)
        slots = torch.tensor(encoding["option_token_ids"], dtype=torch.long, device=device)

        def one_step(index: int) -> dict:
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            vocabulary = _forward(
                model,
                {"input_ids": input_ids, "attention_mask": masks, "position_ids": position_ids,
                 "kernel_options": FLEX_KERNEL_OPTIONS},
            )[0].float()
            selected_logits = vocabulary.index_select(0, slots)
            label = torch.tensor([index % 2], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(selected_logits.unsqueeze(0), label)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError("Probe produced a nonfinite option loss")
            torch.cuda.synchronize(device)
            forward_seconds = time.perf_counter() - started

            started = time.perf_counter()
            loss.backward()
            torch.cuda.synchronize(device)
            backward_seconds = time.perf_counter() - started
            finite = True
            nonzero_tensors = 0
            nonzero_elements = 0
            gradient_elements = 0
            for _name, parameter in trainable:
                gradient = parameter.grad
                if gradient is None:
                    finite = False
                    continue
                finite = finite and bool(torch.isfinite(gradient).all().item())
                count = int(torch.count_nonzero(gradient).item())
                nonzero_tensors += int(count > 0)
                nonzero_elements += count
                gradient_elements += gradient.numel()

            if not finite or nonzero_tensors == 0:
                raise RuntimeError("Probe produced missing, nonfinite, or entirely zero LoRA gradients")
            started = time.perf_counter()
            optimizer.step()
            torch.cuda.synchronize(device)
            optimizer_seconds = time.perf_counter() - started
            return {
                "index": index,
                "synthetic_label_index": index % 2,
                "loss": float(loss.detach().item()),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "optimizer_seconds": optimizer_seconds,
                "total_fwd_bwd_optim_seconds": forward_seconds + backward_seconds + optimizer_seconds,
                "gradients_finite": finite,
                "gradient_tensor_count": len(trainable),
                "nonzero_gradient_tensor_count": nonzero_tensors,
                "gradient_element_count": gradient_elements,
                "nonzero_gradient_element_count": nonzero_elements,
                "cuda_allocated_bytes": torch.cuda.memory_allocated(device),
                "cuda_reserved_bytes": torch.cuda.memory_reserved(device),
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "host_rss_bytes": _rss_bytes(),
            }

        receipt["warmup"] = one_step(-1)
        _write_receipt(args.output, receipt)
        torch.cuda.reset_peak_memory_stats(device)
        measured = []
        receipt["steps"] = measured
        for index in range(args.steps):
            step = one_step(index)
            measured.append(step)
            _write_receipt(args.output, receipt)
            print(json.dumps({"status": "measured", "tokens": args.tokens, **step}), flush=True)
        receipt["measured_totals"] = {
            key: sum(step[key] for step in measured)
            for key in ("forward_seconds", "backward_seconds", "optimizer_seconds", "total_fwd_bwd_optim_seconds")
        }
        receipt["measured_peak"] = {
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
            "host_rss_bytes": _rss_bytes(),
        }
        receipt["status"] = "completed"
        receipt["finished_unix"] = time.time()
        _write_receipt(args.output, receipt)
        return receipt
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["support_claim"] = False
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)[:1000]}
        receipt["finished_unix"] = time.time()
        receipt["failure_memory"] = {"host_rss_bytes": _rss_bytes()}
        if torch.cuda.is_available():
            receipt["failure_memory"].update(
                cuda_allocated_bytes=torch.cuda.memory_allocated(),
                cuda_reserved_bytes=torch.cuda.memory_reserved(),
                cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            )
        _write_receipt(args.output, receipt)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, required=True, help="one exact actual prompt-token length")
    parser.add_argument("--steps", type=int, default=2, help="measured steps after one warmup")
    parser.add_argument("--output", type=Path, required=True, help="new create-only receipt directory")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--quantization", choices=("nf4", "none"), default="nf4")
    parser.add_argument("--adapter", help="optional pinned classifier adapter source")
    parser.add_argument("--adapter-revision", help="required immutable revision for --adapter")
    return parser


def main() -> None:
    signal.signal(signal.SIGTERM, lambda _signum, _frame: (_ for _ in ()).throw(
        ProbeTerminated("External timeout terminated the probe")
    ))
    args = _parser().parse_args()
    try:
        run_probe(args)
    except BaseException as error:
        print(f"probe failed: {type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
