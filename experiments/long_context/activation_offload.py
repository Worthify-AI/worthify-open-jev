"""Opt-in CPU storage of nonreentrant checkpoint inputs for the systems probe.

Scope the native saved-tensor hook around *checkpoint calls*, not the model's
whole forward (which would also copy frozen weights). Torch's inner checkpoint
hook owns the layer intermediates. In pinned Gemma4Unified, hidden_states is the
only positional layer argument; RoPE, BlockMask and shared KV tensors are kwargs
captured by Transformers' callable and are therefore not copied per layer.

Pageable, synchronous copies deliberately avoid pinned-host allocator retention.
This trades transfer time and host RAM for GPU RAM; it does not offload live
layer workspaces, shared KV states, or tensors retained by other references.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import partial, wraps
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint


DEFAULT_HOST_RESERVE_BYTES = 16 * 1024**3


def _host_memory_state() -> tuple[int, int]:
    """Require a finite, readable cgroup ceiling; do not guess from host RAM."""
    for current_path, limit_path in (
        ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory.max"),
        (
            "/sys/fs/cgroup/memory/memory.usage_in_bytes",
            "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        ),
    ):
        try:
            current = int(Path(current_path).read_text().strip())
            limit = int(Path(limit_path).read_text().strip())
        except (OSError, ValueError):
            continue
        if 0 <= current < limit < 2**60:
            return current, limit
    raise RuntimeError("Activation offload requires readable finite cgroup memory usage/limit")


def _check_host_budget(copy_bytes: int, reserve_bytes: int) -> tuple[int, int]:
    current, limit = _host_memory_state()
    if current + copy_bytes + reserve_bytes >= limit:
        raise RuntimeError(
            "Activation offload host budget exhausted: "
            f"current={current}, next_copy={copy_bytes}, reserve={reserve_bytes}, limit={limit}"
        )
    return current, limit


class _GuardedSaveOnCPU(torch.autograd.graph.saved_tensors_hooks):
    def __init__(self, reserve_bytes: int, stats: dict):
        native = torch.autograd.graph.save_on_cpu(pin_memory=False)

        def pack(tensor):
            if tensor.device.type == "cuda" and tensor.numel():
                copy_bytes = tensor.numel() * tensor.element_size()
                current, limit = _check_host_budget(copy_bytes, reserve_bytes)
                packed = native.pack_hook(tensor)
                stats["offloaded_tensors"] += 1
                stats["offloaded_bytes"] += copy_bytes
                stats["host_cgroup_limit_bytes"] = limit
                stats["max_pre_copy_cgroup_bytes"] = max(
                    stats["max_pre_copy_cgroup_bytes"], current
                )
                return packed
            stats["passthrough_tensors"] += 1
            return tensor.device, tensor.detach()

        super().__init__(pack, native.unpack_hook)


@contextmanager
def checkpoint_activation_offload(model, *, host_reserve_bytes=DEFAULT_HOST_RESERVE_BYTES):
    """Temporarily offload checkpoint boundaries; create a fresh context each step.

    Use ``with checkpoint_activation_offload(model) as stats:`` around the
    forward/loss/backward region, after enabling ``use_reentrant=False``.
    The returned counters are cumulative transfers, not live memory measurements.
    The host check is conservative (includes cgroup page cache), not a reservation
    against concurrent processes. Leave ample reserve for backward and the OS.
    This single-process probe context is not safe for concurrent model execution.
    """
    from transformers.modeling_layers import GradientCheckpointingLayer

    if isinstance(host_reserve_bytes, bool) or not isinstance(host_reserve_bytes, int) or host_reserve_bytes < 1:
        raise ValueError("host_reserve_bytes must be a positive integer")
    current, limit = _check_host_budget(0, host_reserve_bytes)
    originals = []
    for module in model.modules():
        if not isinstance(module, GradientCheckpointingLayer) or not module.gradient_checkpointing:
            continue
        original = getattr(module, "_gradient_checkpointing_func", None)
        if (
            not isinstance(original, partial)
            or original.func is not checkpoint
            or original.keywords.get("use_reentrant") is not False
        ):
            raise RuntimeError("Activation offload requires native nonreentrant checkpointing")
        originals.append((module, original))
    if not originals:
        raise RuntimeError("Activation offload found no enabled checkpoint layers")

    stats = {
        "mechanism": "native-save-on-cpu-checkpoint-inputs",
        "pin_memory": False,
        "wrapped_layers": len(originals),
        "offloaded_tensors": 0,
        "offloaded_bytes": 0,
        "passthrough_tensors": 0,
        "host_reserve_bytes": host_reserve_bytes,
        "host_cgroup_limit_bytes": limit,
        "max_pre_copy_cgroup_bytes": current,
    }

    def wrap(original):
        @wraps(original)
        def run(*args, **kwargs):
            with _GuardedSaveOnCPU(host_reserve_bytes, stats):
                return original(*args, **kwargs)
        return run

    try:
        for module, original in originals:
            module._gradient_checkpointing_func = wrap(original)
        yield stats
    finally:
        for module, original in originals:
            module._gradient_checkpointing_func = original
