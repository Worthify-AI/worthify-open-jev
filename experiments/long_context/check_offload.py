"""One-load numerical check of checkpoint offload on the pinned 8K systems probe.

Run with exactly one visible CUDA GPU. No optimizer updates occur. A shared
warmup excludes initial compilation from both measured passes; restoring CPU
and CUDA RNG states makes LoRA dropout identical. Every LoRA gradient element,
the two option logits, and loss must agree at rtol=1e-4, atol=1e-6. These small
float32 comparison tolerances allow kernel reduction noise, not a quality claim.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import signal
import time

import torch

try:
    from experiments.long_context import pilot
    from experiments.long_context.activation_offload import checkpoint_activation_offload
except ModuleNotFoundError as error:
    if error.name not in {"experiments", "experiments.long_context"}:
        raise
    import pilot
    from activation_offload import checkpoint_activation_offload


RTOL = 1e-4
ATOL = 1e-6
TOKENS = 8192


def _compare(expected, actual):
    expected, actual = expected.float(), actual.float()
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    difference = (actual - expected).abs()
    close = torch.isclose(actual, expected, rtol=RTOL, atol=ATOL)
    return {
        "passed": finite and bool(close.all()),
        "elements": expected.numel(),
        "mismatched_elements": int((~close).sum()),
        "max_abs_diff": float(difference.max()) if finite else None,
        "baseline_max_abs": float(expected.abs().max()) if finite else None,
    }


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    receipt_path = args.output / "offload-comparison.json"
    root = Path(__file__).resolve().parents[2]
    receipt = {
        "schema": "openjev-checkpoint-offload-comparison-v1",
        "status": "running",
        "support_claim": False,
        "model": pilot.MODEL,
        "revision": pilot.REVISION,
        "quantization": "nf4",
        "tokens": TOKENS,
        "seed": args.seed,
        "optimizer_updates": 0,
        "rtol": RTOL,
        "atol": ATOL,
        "runtime": pilot._runtime_identity(root),
        "source_sha256s": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__), Path(__file__).with_name("activation_offload.py"))
        },
        "started_unix": time.time(),
    }

    def write():
        receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n")

    write()
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA GPU with CUDA_VISIBLE_DEVICES")
        if torch._dynamo.config.suppress_errors:
            raise RuntimeError("Compiler fallback must not be enabled")
        device = torch.device("cuda:0")
        receipt["hardware"] = {
            "name": torch.cuda.get_device_properties(device).name,
            "total_memory_bytes": torch.cuda.get_device_properties(device).total_memory,
            "host_cgroup_limit_bytes": pilot._host_memory_limit_bytes(),
        }
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        started = time.perf_counter()
        model, tokenizer, metadata = pilot.load_causal_model(
            pilot.MODEL, pilot.REVISION, quantization="nf4",
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
        )
        receipt["backend"] = pilot._select_and_verify_flex(model)
        model.config.use_cache = False
        model, peft_version = pilot._configure_lora(model, "nf4")
        model.train()
        receipt["load_seconds"] = time.perf_counter() - started
        receipt["model_metadata"] = metadata
        receipt["peft_version"] = peft_version
        trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
        if not trainable or any(".lora_" not in name for name, _ in trainable):
            raise RuntimeError("Only LoRA parameters may be trainable")
        encoding = pilot.build_probe_encoding(tokenizer, TOKENS)
        inputs = {
            "input_ids": torch.tensor([encoding.pop("input_ids")], dtype=torch.long, device=device),
            "position_ids": torch.arange(TOKENS, dtype=torch.long, device=device).unsqueeze(0),
            "kernel_options": pilot.FLEX_KERNEL_OPTIONS,
        }
        inputs["attention_mask"], receipt["masks"] = pilot._prepare_masks(model, TOKENS, device)
        receipt["workload"] = encoding
        slots = torch.tensor(encoding["option_token_ids"], dtype=torch.long, device=device)
        label = torch.tensor([0], dtype=torch.long, device=device)
        write()

        def one_pass(enabled, collect=True):
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            context = checkpoint_activation_offload(model) if enabled else nullcontext(None)
            started = time.perf_counter()
            with context as stats:
                vocabulary = pilot._forward(model, inputs)[0].float()
                logits = vocabulary.index_select(0, slots)
                loss = torch.nn.functional.cross_entropy(logits.unsqueeze(0), label)
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("Nonfinite loss")
                torch.cuda.synchronize(device)
                forward_seconds = time.perf_counter() - started
                started = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize(device)
                backward_seconds = time.perf_counter() - started
            values = {"logits": logits.detach().cpu(), "loss": loss.detach().cpu()}
            gradients = {}
            nonzero = 0
            for name, parameter in trainable:
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError(f"Missing/nonfinite gradient: {name}")
                nonzero += int(torch.count_nonzero(parameter.grad))
                if collect:
                    gradients[name] = parameter.grad.detach().cpu().clone()
            if not nonzero:
                raise RuntimeError("All LoRA gradient elements are zero")
            measured = {
                "option_logits": values["logits"].tolist(),
                "loss": float(values["loss"]),
                "forward_seconds": forward_seconds,
                "backward_seconds": backward_seconds,
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
                "gradient_tensor_count": len(trainable),
                "nonzero_gradient_elements": nonzero,
                "offload": stats,
                "host_rss_bytes": pilot._rss_bytes(),
            }
            return measured, values, gradients

        receipt["warmup"], _, _ = one_pass(False, collect=False)
        write()
        cpu_rng = torch.get_rng_state().clone()
        cuda_rng = torch.cuda.get_rng_state(device).clone()
        receipt["baseline"], baseline_values, baseline_grads = one_pass(False)
        write()
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng, device)
        receipt["offload"], offload_values, offload_grads = one_pass(True)
        comparisons = {key: _compare(value, offload_values[key]) for key, value in baseline_values.items()}
        comparisons["gradients"] = {
            name: _compare(value, offload_grads[name]) for name, value in baseline_grads.items()
        }
        passed = all(comparisons[key]["passed"] for key in ("logits", "loss")) and all(
            value["passed"] for value in comparisons["gradients"].values()
        )
        receipt["comparison"] = comparisons
        if receipt["offload"]["offload"]["offloaded_bytes"] == 0:
            raise RuntimeError("Offload context copied no checkpoint activations")
        receipt["numerically_equivalent"] = passed
        if not passed:
            raise RuntimeError("Baseline/offload numerical comparison exceeded tolerance")
        receipt["status"] = "completed"
        receipt["finished_unix"] = time.time()
        write()
        print(json.dumps({"status": "completed", "numerically_equivalent": True, "receipt": str(receipt_path)}), flush=True)
        return receipt
    except BaseException as error:
        receipt["status"] = "failed"
        receipt["failure"] = {"type": type(error).__name__, "message": str(error)[:1000]}
        receipt["finished_unix"] = time.time()
        write()
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new create-only directory")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, lambda _signum, _frame: (_ for _ in ()).throw(
        pilot.ProbeTerminated("External timeout terminated offload check")
    ))
    run(args)


if __name__ == "__main__":
    main()
