#!/usr/bin/env python3
"""Bounded FlexAttention preflight for the Gemma 4 12B long-context pilot.

The probe deliberately stops at 2,048 tokens because its SDPA math reference
uses a dense attention mask.  The analytical BlockMask builder is separate: it
does not materialize a token-by-token mask and is suitable for the pilot's long
sequence masks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F


MAX_REFERENCE_SEQUENCE_LENGTH = 2_048
DEFAULT_BLOCK_SIZE = 128


def causal_mask_mod(window_size: int | None = None) -> Callable:
    """Return a causal mask predicate, optionally limited to a trailing window."""
    if window_size is not None and window_size <= 0:
        raise ValueError("window_size must be positive")

    if window_size is None:

        def causal(b, h, q_idx, kv_idx):
            return kv_idx <= q_idx

        return causal

    def local_causal(b, h, q_idx, kv_idx):
        return (kv_idx <= q_idx) & (kv_idx > q_idx - window_size)

    return local_causal


def gemma4_flex_kernel_options() -> dict[str, int | bool | str]:
    """Kernel options validated for both Gemma 4 local and global head shapes.

    On Torch 2.10/B200, ``ROWS_GUARANTEED_SAFE=True`` produced nonfinite local
    output at L=2048 with compact custom metadata, despite every semantic row
    containing its causal diagonal.  Partial local boundary blocks can also be
    separated by full blocks.  Keep both hints false.  The 32x32 tiles avoid a
    D=512 default-autotune configuration that exceeded B200 shared memory.
    """
    return {
        "BACKEND": "TRITON",
        "ROWS_GUARANTEED_SAFE": False,
        "BLOCKS_ARE_CONTIGUOUS": False,
        "fwd_BLOCK_M": 32,
        "fwd_BLOCK_N": 32,
        "bwd_BLOCK_M1": 32,
        "bwd_BLOCK_N1": 32,
        "bwd_BLOCK_M2": 32,
        "bwd_BLOCK_N2": 32,
        "num_warps": 4,
        "num_stages": 2,
    }


def _pack_rows(
    rows: list[list[int]], device: torch.device, *, width: int | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack ordered sparse rows without ever constructing a dense token mask."""
    required_width = max(1, max((len(row) for row in rows), default=0))
    if width is None:
        width = required_width
    elif width < required_width:
        raise ValueError("packed row width is smaller than an active row")
    indices = torch.zeros((len(rows), width), dtype=torch.int32)
    counts = torch.tensor([len(row) for row in rows], dtype=torch.int32)
    for row_idx, row in enumerate(rows):
        if row:
            indices[row_idx, : len(row)] = torch.tensor(row, dtype=torch.int32)
    return counts[None, None].to(device), indices[None, None].to(device)


def _transpose_rows(rows: list[list[int]], column_count: int) -> list[list[int]]:
    transposed: list[list[int]] = [[] for _ in range(column_count)]
    for row_idx, columns in enumerate(rows):
        for column_idx in columns:
            transposed[column_idx].append(row_idx)
    return transposed


def build_analytical_causal_block_mask(
    sequence_length: int,
    *,
    window_size: int | None = None,
    device: torch.device | str,
    block_size: int = DEFAULT_BLOCK_SIZE,
):
    """Build a causal ``BlockMask`` from block intervals rather than an L² mask.

    Batch and head dimensions are one and therefore broadcast.  Partial blocks
    retain the exact token predicate; full blocks bypass it.  Local masks use
    O((L / block_size) * (window_size / block_size)) metadata.  Full causal
    masks necessarily retain a triangular O((L / block_size)²) block index,
    still a block_size² reduction from a dense token mask.
    """
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if window_size is not None and window_size <= 0:
        raise ValueError("window_size must be positive")

    from torch.nn.attention.flex_attention import BlockMask

    target_device = torch.device(device)
    block_count = math.ceil(sequence_length / block_size)
    full_rows: list[list[int]] = []
    partial_rows: list[list[int]] = []

    for q_block in range(block_count):
        q_start = q_block * block_size
        q_end = min(sequence_length, (q_block + 1) * block_size) - 1
        full_lower = 0 if window_size is None else max(0, (q_end - window_size) // block_size + 1)
        full_upper = min(q_block - 1, block_count - 1)
        full_rows.append(list(range(full_lower, full_upper + 1)) if full_lower <= full_upper else [])

        partial = [q_block]  # The causal diagonal always needs the token predicate.
        if window_size is not None:
            # Blocks between the earliest key visible to q_start and the first
            # block that is valid for every query are active but partial.  A
            # non-block-aligned window can produce two such boundary blocks.
            active_lower = max(0, (q_start - window_size + 1) // block_size)
            partial = list(range(active_lower, full_lower)) + partial
        partial_rows.append(partial)

    q_full_rows = _transpose_rows(full_rows, block_count)
    q_partial_rows = _transpose_rows(partial_rows, block_count)
    # Torch 2.10's FlexAttention template indexes full and partial arrays with
    # shared stride-derived offsets.  All four index arrays therefore need the
    # same packed width even though their active counts differ.
    packed_width = max(
        1,
        *(len(row) for rows in (partial_rows, full_rows, q_partial_rows, q_full_rows) for row in rows),
    )
    kv_num, kv_idx = _pack_rows(partial_rows, target_device, width=packed_width)
    full_kv_num, full_kv_idx = _pack_rows(full_rows, target_device, width=packed_width)
    q_num, q_idx = _pack_rows(q_partial_rows, target_device, width=packed_width)
    full_q_num, full_q_idx = _pack_rows(q_full_rows, target_device, width=packed_width)

    return BlockMask(
        seq_lengths=(sequence_length, sequence_length),
        kv_num_blocks=kv_num,
        kv_indices=kv_idx,
        full_kv_num_blocks=full_kv_num,
        full_kv_indices=full_kv_idx,
        q_num_blocks=q_num,
        q_indices=q_idx,
        full_q_num_blocks=full_q_num,
        full_q_indices=full_q_idx,
        BLOCK_SIZE=(block_size, block_size),
        mask_mod=causal_mask_mod(window_size),
    )


def _tensor_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool | int]:
    delta = (actual.float() - reference.float()).abs()
    denominator = reference.float().abs().clamp_min(1e-5)
    rms_error = delta.square().mean().sqrt()
    reference_rms = reference.float().square().mean().sqrt()
    return {
        "all_finite": bool(torch.isfinite(actual).all().item()),
        "nonzero_count": int(torch.count_nonzero(actual).item()),
        "max_abs_error": float(delta.max().item()),
        "mean_abs_error": float(delta.mean().item()),
        "rms_error": float(rms_error.item()),
        "reference_rms": float(reference_rms.item()),
        "normalized_rms_error": float((rms_error / reference_rms.clamp_min(1e-8)).item()),
        "max_abs_error_over_reference_rms": float(
            (delta.max() / reference_rms.clamp_min(1e-8)).item()
        ),
        "max_relative_error_above_1e-5": float((delta / denominator).max().item()),
    }


def _check_tensor(
    name: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
    *,
    max_abs_error: float | None,
    max_normalized_rms_error: float,
    max_abs_error_over_reference_rms: float | None = None,
) -> dict:
    metrics = _tensor_metrics(actual, reference)
    metrics["within_tolerance"] = bool(
        metrics["all_finite"]
        and metrics["nonzero_count"] > 0
        and (max_abs_error is None or metrics["max_abs_error"] <= max_abs_error)
        and metrics["normalized_rms_error"] <= max_normalized_rms_error
        and (
            max_abs_error_over_reference_rms is None
            or metrics["max_abs_error_over_reference_rms"] <= max_abs_error_over_reference_rms
        )
    )
    metrics["name"] = name
    return metrics


def _run_case(
    *,
    name: str,
    sequence_length: int,
    query_heads: int,
    kv_heads: int,
    head_dim: int,
    window_size: int | None,
    seed: int,
) -> dict:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.flex_attention import flex_attention

    device = torch.device("cuda")
    generator = torch.Generator(device=device).manual_seed(seed)
    shape_q = (1, query_heads, sequence_length, head_dim)
    shape_kv = (1, kv_heads, sequence_length, head_dim)
    q0 = torch.randn(shape_q, device=device, dtype=torch.bfloat16, generator=generator)
    k0 = torch.randn(shape_kv, device=device, dtype=torch.bfloat16, generator=generator)
    v0 = torch.randn(shape_kv, device=device, dtype=torch.bfloat16, generator=generator)
    # Gemma 4 applies RMSNorm to Q and K and then passes scaling=1.0 to
    # Transformers' attention interface.  Reproduce those kernel inputs.
    q0 = (q0.float() * q0.float().square().mean(-1, keepdim=True).add(1e-6).rsqrt()).to(torch.bfloat16)
    k0 = (k0.float() * k0.float().square().mean(-1, keepdim=True).add(1e-6).rsqrt()).to(torch.bfloat16)
    cotangent = torch.randn(shape_q, device=device, dtype=torch.bfloat16, generator=generator)

    mask = build_analytical_causal_block_mask(
        sequence_length, window_size=window_size, device=device, block_size=DEFAULT_BLOCK_SIZE
    )
    dense_positions = torch.arange(sequence_length, device=device)
    dense_mask = dense_positions[None, :] <= dense_positions[:, None]
    if window_size is not None:
        dense_mask &= dense_positions[None, :] > dense_positions[:, None] - window_size

    q_ref, k_ref, v_ref = (x.detach().clone().requires_grad_(True) for x in (q0, k0, v0))
    with sdpa_kernel(SDPBackend.MATH):
        reference = F.scaled_dot_product_attention(
            q_ref, k_ref, v_ref, attn_mask=dense_mask, dropout_p=0.0, scale=1.0, enable_gqa=True
        )
    (reference.float() * cotangent.float()).sum().backward()

    q_flex, k_flex, v_flex = (x.detach().clone().requires_grad_(True) for x in (q0, k0, v0))
    if torch._dynamo.config.suppress_errors:
        raise RuntimeError("torch._dynamo.config.suppress_errors must be false; fallback would invalidate the probe")
    compiled_flex = torch.compile(flex_attention, fullgraph=True, dynamic=False)
    # Default D=512 autotuning exceeds B200 shared-memory resources.  Use one
    # conservative option set for both layer patterns so model.forward can pass
    # a single validated dictionary through Transformers' attention wrapper.
    kernel_options = gemma4_flex_kernel_options()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    actual = compiled_flex(
        q_flex,
        k_flex,
        v_flex,
        block_mask=mask,
        scale=1.0,
        enable_gqa=True,
        kernel_options=kernel_options,
    )
    (actual.float() * cotangent.float()).sum().backward()
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    # Pointwise relative error is ill-conditioned for near-zero BF16 gradients.
    # Bound both the worst absolute deviation and aggregate RMS error normalized
    # by the reference RMS instead.  The limits allow BF16 fused-reduction order
    # differences while rejecting material numerical disagreement.
    tolerances = {
        "output": {"max_abs_error": 0.05, "max_normalized_rms_error": 0.002},
        "gradient": {
            "max_abs_error": None,
            "max_normalized_rms_error": 0.02,
            "max_abs_error_over_reference_rms": 0.2,
        },
    }
    checks = {
        "output": _check_tensor("output", actual, reference, **tolerances["output"]),
        "q_gradient": _check_tensor("q gradient", q_flex.grad, q_ref.grad, **tolerances["gradient"]),
        "k_gradient": _check_tensor("k gradient", k_flex.grad, k_ref.grad, **tolerances["gradient"]),
        "v_gradient": _check_tensor("v gradient", v_flex.grad, v_ref.grad, **tolerances["gradient"]),
    }
    status = "pass" if all(check["within_tolerance"] for check in checks.values()) else "fail"
    return {
        "name": name,
        "status": status,
        "shape": {
            "batch": 1,
            "sequence_length": sequence_length,
            "query_heads": query_heads,
            "kv_heads": kv_heads,
            "head_dim": head_dim,
            "window_size": window_size,
        },
        "dtype": "bfloat16",
        "implementation": "torch.compile(fullgraph=True, dynamic=False) flex_attention",
        "kernel_options": kernel_options,
        "reference": "scaled_dot_product_attention SDPBackend.MATH with dense boolean mask",
        "attention_scale": 1.0,
        "tolerances": tolerances,
        "checks": checks,
        "compiled_forward_backward_seconds": elapsed,
        "peak_cuda_bytes_flex_phase": int(torch.cuda.max_memory_allocated(device)),
    }


def _run_mask_equivalence(sequence_length: int, seed: int) -> dict:
    """Require analytical and canonical masks to produce identical Flex results."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    device = torch.device("cuda")
    compiled_flex = torch.compile(flex_attention, fullgraph=True, dynamic=False)
    options = gemma4_flex_kernel_options()
    patterns = []
    for offset, (name, query_heads, kv_heads, head_dim, window_size) in enumerate(
        (
            ("local_causal_gqa", 16, 8, 256, 1_024),
            ("global_causal_gqa", 16, 1, 512, None),
        )
    ):
        generator = torch.Generator(device=device).manual_seed(seed + offset)
        q0 = torch.randn(
            (1, query_heads, sequence_length, head_dim), device=device, dtype=torch.bfloat16, generator=generator
        )
        k0 = torch.randn(
            (1, kv_heads, sequence_length, head_dim), device=device, dtype=torch.bfloat16, generator=generator
        )
        v0 = torch.randn(k0.shape, device=device, dtype=torch.bfloat16, generator=generator)
        cotangent = torch.randn(q0.shape, device=device, dtype=torch.bfloat16, generator=generator)
        analytical = build_analytical_causal_block_mask(
            sequence_length, window_size=window_size, device=device
        )
        canonical = create_block_mask(
            causal_mask_mod(window_size), 1, None, sequence_length, sequence_length, device
        )
        results = []
        for block_mask in (analytical, canonical):
            q, k, v = (tensor.detach().clone().requires_grad_(True) for tensor in (q0, k0, v0))
            output = compiled_flex(
                q, k, v, block_mask=block_mask, enable_gqa=True, scale=1.0, kernel_options=options
            )
            (output.float() * cotangent.float()).sum().backward()
            results.append((output, q.grad, k.grad, v.grad))
        torch.cuda.synchronize(device)
        value_names = ("output", "q_gradient", "k_gradient", "v_gradient")
        checks = {
            value_name: _tensor_metrics(analytical_value, canonical_value)
            for value_name, analytical_value, canonical_value in zip(
                value_names, results[0], results[1], strict=True
            )
        }
        exact = all(check["max_abs_error"] == 0.0 for check in checks.values())
        patterns.append(
            {
                "name": name,
                "status": "pass" if exact else "fail",
                "query_heads": query_heads,
                "kv_heads": kv_heads,
                "head_dim": head_dim,
                "window_size": window_size,
                "checks": checks,
            }
        )
    return {
        "status": "pass" if all(pattern["status"] == "pass" for pattern in patterns) else "fail",
        "sequence_length": sequence_length,
        "criterion": "bitwise-identical output and q/k/v gradients",
        "patterns": patterns,
        "kernel_options": options,
    }


def _metadata() -> dict:
    try:
        import transformers

        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    cuda_available = torch.cuda.is_available()
    cuda = {
        "available": cuda_available,
        "runtime_version": torch.version.cuda,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
    }
    if cuda_available:
        properties = torch.cuda.get_device_properties(0)
        cuda.update(
            {
                "device_index": 0,
                "device_name": properties.name,
                "compute_capability": list(torch.cuda.get_device_capability(0)),
                "total_memory_bytes": properties.total_memory,
            }
        )
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "transformers": transformers_version,
        "cuda": cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON result path (must not already exist)")
    parser.add_argument("--sequence-length", type=int, default=MAX_REFERENCE_SEQUENCE_LENGTH)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    if not 1 <= args.sequence_length <= MAX_REFERENCE_SEQUENCE_LENGTH:
        parser.error(f"--sequence-length must be between 1 and {MAX_REFERENCE_SEQUENCE_LENGTH}")
    if args.output.exists():
        parser.error(f"refusing to overwrite existing result: {args.output}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = {
        "schema_version": 1,
        "probe": "gemma4-12b-long-context-flex-attention",
        "metadata": _metadata(),
        "reference_sequence_length_cap": MAX_REFERENCE_SEQUENCE_LENGTH,
        "mask_builder": {
            "kind": "analytical BlockMask",
            "transformers_native_mask_warning": (
                "Transformers 5.17.0 flex_attention_mask calls create_block_mask/create_mask, "
                "which materializes a token-level B*H*Q*KV boolean mask."
            ),
            "stride_requirement": (
                "Torch 2.10 shares stride-derived offsets across partial/full KV and Q index arrays; "
                "all analytical index arrays are padded to identical shapes and strides."
            ),
        },
        "numerical_acceptance": {
            "rationale": (
                "BF16 fused and SDPA-math reductions may differ in order. Near-zero pointwise relative "
                "errors are ill-conditioned, so acceptance bounds worst absolute error and RMS error "
                "normalized by reference RMS; this does not claim FP32 equivalence."
            ),
            "output": {"max_abs_error": 0.05, "max_normalized_rms_error": 0.002},
            "gradient": {
                "max_abs_error": "unbounded; gradient scale differs by head pattern",
                "max_normalized_rms_error": 0.02,
                "max_abs_error_over_reference_rms": 0.2,
            },
        },
        "cases": [],
    }
    exit_code = 0
    if not torch.cuda.is_available():
        result["status"] = "unavailable"
        result["error"] = "CUDA is required for the compiled native FlexAttention probe"
        exit_code = 2
    else:
        try:
            result["mask_equivalence"] = _run_mask_equivalence(args.sequence_length, args.seed + 99)
            result["cases"].append(
                _run_case(
                    name="local_causal_gqa",
                    sequence_length=args.sequence_length,
                    query_heads=16,
                    kv_heads=8,
                    head_dim=256,
                    window_size=1_024,
                    seed=args.seed,
                )
            )
            result["cases"].append(
                _run_case(
                    name="global_causal_gqa",
                    sequence_length=args.sequence_length,
                    query_heads=16,
                    kv_heads=1,
                    head_dim=512,
                    window_size=None,
                    seed=args.seed + 1,
                )
            )
            result["status"] = "pass" if (
                result["mask_equivalence"]["status"] == "pass"
                and all(case["status"] == "pass" for case in result["cases"])
            ) else "fail"
            if result["status"] == "fail":
                exit_code = 1
        except Exception as exc:
            result["status"] = "fail"
            result["error"] = f"{type(exc).__name__}: {exc}"
            exit_code = 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
