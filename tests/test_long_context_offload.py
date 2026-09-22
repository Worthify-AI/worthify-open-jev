from functools import partial

import pytest
import torch
from torch.utils.checkpoint import checkpoint
from transformers.modeling_layers import GradientCheckpointingLayer

from experiments.long_context import activation_offload as offload


class ToyLayer(GradientCheckpointingLayer):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(5, 5, dtype=torch.float64))
        self.gradient_checkpointing = True
        self._gradient_checkpointing_func = partial(checkpoint, use_reentrant=False)
        self.calls = 0

    def forward(self, hidden, *, shared):
        self.calls += 1
        return torch.sin(torch.nn.functional.dropout(hidden, p=0.2, training=True) @ self.weight + shared)


def test_checkpoint_input_scope_gradient_equivalence_and_restoration(monkeypatch):
    monkeypatch.setattr(offload, "_host_memory_state", lambda: (100, 10000))
    torch.manual_seed(7)
    layers = torch.nn.ModuleList([ToyLayer(), ToyLayer()])
    originals = [layer._gradient_checkpointing_func for layer in layers]
    source = torch.randn(3, 5, dtype=torch.float64)
    shared = torch.randn(3, 5, dtype=torch.float64, requires_grad=True)

    def run():
        hidden = source.clone().requires_grad_()
        initial = hidden
        for layer in layers:
            hidden = layer(hidden, shared=shared)
        loss = hidden.square().sum()
        loss.backward()
        return loss.detach(), initial.grad.clone(), shared.grad.clone(), [p.grad.clone() for p in layers.parameters()]

    rng_state = torch.get_rng_state()
    expected = run()
    layers.zero_grad(set_to_none=True)
    shared.grad = None
    torch.set_rng_state(rng_state)
    with offload.checkpoint_activation_offload(layers, host_reserve_bytes=10) as stats:
        actual = run()
    for lhs, rhs in zip(expected[:3], actual[:3]):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
    for lhs, rhs in zip(expected[3], actual[3]):
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)
    assert stats["wrapped_layers"] == 2
    # Each boundary saves the zero-sized dummy and hidden state only. Neither
    # the shared kwarg nor recomputation intermediates reach our outer hook.
    assert stats["passthrough_tensors"] == 4
    assert stats["offloaded_bytes"] == 0  # CPU test does not claim a CUDA transfer.
    assert all(layer.calls == 4 for layer in layers)
    assert [layer._gradient_checkpointing_func for layer in layers] == originals


def test_restores_checkpoint_function_after_failure(monkeypatch):
    monkeypatch.setattr(offload, "_host_memory_state", lambda: (100, 10000))
    model = ToyLayer()
    original = model._gradient_checkpointing_func
    with pytest.raises(ValueError, match="forward failed"):
        with offload.checkpoint_activation_offload(model, host_reserve_bytes=10):
            raise ValueError("forward failed")
    assert model._gradient_checkpointing_func is original


def test_host_budget_refuses_next_allocation_at_reserve(monkeypatch):
    monkeypatch.setattr(offload, "_host_memory_state", lambda: (800, 1000))
    assert offload._check_host_budget(99, 100) == (800, 1000)
    with pytest.raises(RuntimeError, match="host budget exhausted"):
        offload._check_host_budget(100, 100)


def test_unknown_host_limit_fails_closed(monkeypatch):
    monkeypatch.setattr(offload.Path, "read_text", lambda self: "max")
    with pytest.raises(RuntimeError, match="finite cgroup"):
        offload._host_memory_state()


def test_reentrant_checkpointing_rejected_without_mutation(monkeypatch):
    monkeypatch.setattr(offload, "_host_memory_state", lambda: (100, 10000))
    model = ToyLayer()
    model._gradient_checkpointing_func = partial(checkpoint, use_reentrant=True)
    original = model._gradient_checkpointing_func
    with pytest.raises(RuntimeError, match="nonreentrant"):
        with offload.checkpoint_activation_offload(model, host_reserve_bytes=10):
            pass
    assert model._gradient_checkpointing_func is original


def test_numerical_gate_checks_all_elements_and_rejects_nonfinite():
    from experiments.long_context.check_offload import _compare

    expected = torch.tensor([0.0, 1.0, -2.0])
    assert _compare(expected, expected + 1e-7)["passed"]
    bad = expected.clone()
    bad[0] = 2e-6
    result = _compare(expected, bad)
    assert not result["passed"]
    assert result["mismatched_elements"] == 1
    assert not _compare(expected, torch.tensor([0.0, float("nan"), -2.0]))["passed"]
