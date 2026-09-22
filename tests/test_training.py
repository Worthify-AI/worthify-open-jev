import torch
import pytest
import json
import random
from types import SimpleNamespace

from openjev_phase1 import training


ROW = {
    "id": "row-1", "state": "evidence", "question": "choose", "gold_option_id": "b",
    "options": [{"id": "a", "description": "first"}, {"id": "b", "description": "second"}],
}


def test_permutation_is_deterministic_and_moves_gold():
    first, label = training.randomized_row(ROW, seed=42, epoch=1)
    again, again_label = training.randomized_row(ROW, seed=42, epoch=1)
    assert first["options"] == again["options"]
    assert label == again_label
    assert first["options"][label]["id"] == "b"
    assert ROW["options"][1]["id"] == "b"  # caller's row is not mutated


def test_training_gold_never_enters_prompt():
    shuffled, _ = training.randomized_row(ROW, seed=42, epoch=0)
    # direct_messages, used by encode_prompt, permits unrelated fields but renders none.
    from openjev_phase1.core import direct_messages

    assert "gold_option_id" not in str(direct_messages(shuffled))


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 0


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))

    def forward(self, input_ids, attention_mask, **kwargs):
        logits = torch.zeros((input_ids.shape[0], 1, 20), device=input_ids.device)
        logits[:, :, 7] = self.weight * 3
        logits[:, :, 9] = self.weight
        return type("Output", (), {"logits": logits})()


def test_batch_loss_gathers_only_declared_slot_logits(monkeypatch):
    def encoded(_, row, __):
        return [1, 2], [7, 9] if row["id"] == "one" else [9, 7], "hash"

    monkeypatch.setattr(training, "encode_prompt", encoded)
    model = _Model()
    rows = [({"id": "one"}, 0), ({"id": "two"}, 1)]
    loss, logits, labels = training.batch_loss(model, _Tokenizer(), rows, 20)
    assert loss.item() < 0.2
    assert logits.argmax(1).tolist() == [0, 1]
    assert labels.tolist() == [0, 1]
    loss.backward()
    assert model.weight.grad is not None


def test_macro_f1_penalizes_missing_class():
    assert training.macro_f1([0, 0], [0, 1], classes=2) == 1 / 3


def test_batch_loss_handles_variable_lengths_and_option_counts(monkeypatch):
    monkeypatch.setattr(training, "encode_prompt", lambda _, row, __: (
        [1] * row["length"], row["slots"], "hash"))

    class PositionModel(_Model):
        def forward(self, input_ids, attention_mask, position_ids, **kwargs):
            assert input_ids.tolist() == [[0, 0, 1], [1, 1, 1]]
            assert attention_mask.tolist() == [[0, 0, 1], [1, 1, 1]]
            assert position_ids.tolist() == [[0, 0, 0], [0, 1, 2]]
            return super().forward(input_ids, attention_mask, **kwargs)

    loss, logits, _ = training.batch_loss(PositionModel(), _Tokenizer(), [
        ({"length": 1, "slots": [7, 9]}, 0), ({"length": 3, "slots": [9, 7, 8]}, 1),
    ], 20)
    assert torch.isneginf(logits[0, 2])
    assert logits.argmax(1).tolist() == [0, 1]
    assert torch.isfinite(loss)


def test_validation_f1_uses_semantic_ids_across_option_orders(monkeypatch):
    reordered = {**ROW, "id": "row-2", "options": list(reversed(ROW["options"]))}

    def loss(_, __, examples, ___):
        row, target = examples[0]
        selected = torch.tensor([[0., 1.]]) if row["id"] == "row-1" else torch.tensor([[1., 0.]])
        return torch.tensor(0.), selected, torch.tensor([target])

    monkeypatch.setattr(training, "batch_loss", loss)
    assert training.evaluate(_Model(), _Tokenizer(), [ROW, reordered], 20) == 1.
    assert training.macro_f1(["a", "a"], ["a", "b"], ["a", "b"]) == 1 / 3


@pytest.mark.parametrize("change", [
    {"id": ROW["id"], "state": "different"},
    {"id": "new", "state": "  EVIDENCE  "},
    {"id": "new", "state": "different", "group_id": "same"},
])
def test_training_rejects_split_leakage(change):
    train = {**ROW, "split": "train", "group_id": "same"}
    validation = {**ROW, "split": "validation", **change}
    with pytest.raises(ValueError, match="leakage"):
        training._validate_splits([train], [validation])


class _CheckpointModel(_Model):
    def save_pretrained(self, path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.weight.detach().clone(), path / "weight.pt")

    def delete_adapter(self, name):
        pass

    def load_adapter(self, path, **kwargs):
        from pathlib import Path
        self.weight = torch.nn.Parameter(torch.load(Path(path) / "weight.pt", weights_only=True))

    def set_adapter(self, name):
        pass


def test_resume_restores_optimizer_and_rng_and_checks_training_spec(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model = _CheckpointModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    (model.weight ** 2).backward()
    optimizer.step()
    checkpoint = tmp_path / "checkpoint"
    training._save_checkpoint(checkpoint, model, optimizer, 0, .7,
                              best_epoch=0, best_checkpoint=checkpoint, training_spec={"seed": 42})
    expected_random, expected_torch = random.random(), torch.rand(3)
    state = training._resume_checkpoint(checkpoint, model, {"seed": 42})
    restored_optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    restored_optimizer.load_state_dict(state["optimizer"])
    assert random.random() == expected_random
    assert torch.equal(torch.rand(3), expected_torch)
    assert restored_optimizer.state[model.weight]["step"] == 1
    assert state["best_epoch"] == 0
    with pytest.raises(ValueError, match="must match"):
        training._resume_checkpoint(checkpoint, model, {"seed": 43})


def test_training_exports_best_adapter_and_saves_latest_resume_state(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    train_path, validation_path = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train_path.write_text(json.dumps(ROW) + "\n")
    validation_path.write_text(json.dumps({**ROW, "id": "validation", "state": "other"}) + "\n")
    model = _CheckpointModel()
    monkeypatch.setattr(training, "load_causal_model", lambda *args, **kwargs: (model, _Tokenizer(), {}))
    monkeypatch.setattr(training, "_configure_lora", lambda model, quantization: (model, "test"))
    provenance = {"git_head": "b" * 40, "git_dirty": False,
                  "source_sha256s": {"src/openjev_phase1/training.py": "c" * 64},
                  "runtime_versions": {"python": "3.12.0", "torch": "test",
                                       "transformers": "test", "peft": "test", "bitsandbytes": "test"}}
    monkeypatch.setattr(training, "_source_provenance", lambda: provenance)
    monkeypatch.setattr(training, "batch_loss", lambda model, *args: (
        model.weight ** 2, torch.stack([model.weight, model.weight + 1]).view(1, 2), torch.tensor([0])))
    scores = iter([.9, .5])
    monkeypatch.setattr(training, "evaluate", lambda *args: next(scores))
    output = tmp_path / "output"
    args = SimpleNamespace(train=train_path, validation=validation_path, output=output,
                           seed=42, model="test", revision="a" * 40, quantization="none",
                           cache_dir=None, resume=None, epochs=2, max_tokens=20, effective_batch_size=1)
    manifest = training.train(args)
    assert manifest["best_epoch"] == 0
    first = torch.load(output / "checkpoint-epoch-00/adapter/weight.pt", weights_only=True)
    last = torch.load(output / "checkpoint-epoch-01/adapter/weight.pt", weights_only=True)
    final = torch.load(output / "final-adapter/weight.pt", weights_only=True)
    assert torch.equal(final, first)
    assert not torch.equal(first, last)
    state = torch.load(output / "checkpoint-epoch-01/state.pt", weights_only=False)
    assert state["epoch"] == 1 and state["best_epoch"] == 0
    assert manifest["adapter_reload_verification"]["max_abs_logit_error"] == 0.
    assert manifest["source_provenance"] == provenance
    assert state["training_spec"]["source_provenance"] == provenance
    reference = json.loads((output / "final-adapter/reload-reference.json").read_text())
    assert reference["schema"] == "openjev-phase1-adapter-reload-reference-v1"
    assert reference["max_tokens"] == 20 and reference["tolerance"] == 1e-4
    assert reference["rows"][0]["id"] == "validation"
    assert reference["rows"][0]["option_ids"] == ["a", "b"]
    assert torch.isfinite(torch.tensor(reference["rows"][0]["logits"])).all()
    assert manifest["fresh_reload_reference"]["path"] == "final-adapter/reload-reference.json"


def test_recipe_defaults_preserve_explicit_choice_and_reject_data_mismatch(tmp_path):
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps({
        "version": "openjev-training-recipe-v1", "task": "classification", "dataset": "CLINC150",
        "seeds": [42, 43], "training": training.APPROVED_DEFAULTS,
    }))
    recipe = training.load_recipe(recipe_path)
    args = SimpleNamespace(seed=None, quantization=None, max_tokens=1024,
                           effective_batch_size=None, epochs=None)
    training.apply_recipe_defaults(args, recipe)
    assert args.seed == 42 and args.max_tokens == 1024 and args.effective_batch_size == 16
    matching = {**ROW, "task": "classification", "source": {"dataset": "CLINC150"}}
    training._validate_recipe_rows([matching], recipe)
    with pytest.raises(ValueError, match="dataset"):
        training._validate_recipe_rows([{**matching, "source": {"dataset": "WANLI"}}], recipe)


def test_reload_rejects_nonfinite_declared_logits(tmp_path, monkeypatch):
    model = _CheckpointModel()
    model.save_pretrained(tmp_path)
    responses = iter([torch.tensor([[1., 2.]]), torch.tensor([[1., float("nan")]])])
    monkeypatch.setattr(training, "batch_loss", lambda *args: (None, next(responses), None))
    with pytest.raises(RuntimeError, match="nonfinite"):
        training._adapter_reload_error(model, _Tokenizer(), [ROW], 20, tmp_path)


def test_reload_sets_new_modules_to_evaluation_mode(tmp_path, monkeypatch):
    class ReloadModel(_CheckpointModel):
        def load_adapter(self, *args, **kwargs):
            super().load_adapter(*args, **kwargs)
            self.add_module("new_dropout", torch.nn.Dropout(.9))

    model = ReloadModel()
    model.save_pretrained(tmp_path)
    def batch(*args):
        assert all(not child.training for child in model.modules())
        return None, torch.tensor([[1., 2.]]), None
    monkeypatch.setattr(training, "batch_loss", batch)
    assert training._adapter_reload_error(model, _Tokenizer(), [ROW], 20, tmp_path) == 0.


def test_native_unified_lora_preserves_bfloat16_base_and_backward_with_frozen_embeddings(monkeypatch):
    pytest.importorskip("peft")
    from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig

    config = Gemma4UnifiedTextConfig(
        vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
        num_global_key_value_heads=1, max_position_embeddings=64,
        layer_types=["sliding_attention", "full_attention"],
    )
    base = Gemma4UnifiedForCausalLM(config).to(dtype=torch.bfloat16)
    assert {parameter.dtype for parameter in base.parameters()} == {torch.bfloat16}
    model, _ = training._configure_lora(base, "nf4")
    base_parameters = [parameter for name, parameter in model.named_parameters() if ".lora_" not in name]
    assert base_parameters and {parameter.dtype for parameter in base_parameters} == {torch.bfloat16}
    assert all(not parameter.requires_grad for parameter in base_parameters)
    monkeypatch.setattr(training, "encode_prompt", lambda _, row, __: (row["tokens"], [7, 9], "hash"))
    examples = [({"tokens": [2, 3]}, 0), ({"tokens": [2, 4, 5, 6]}, 1)]
    model.eval()
    with torch.no_grad():
        _, batched, _ = training.batch_loss(model, _Tokenizer(), examples, 20)
        for index, example in enumerate(examples):
            _, single, _ = training.batch_loss(model, _Tokenizer(), [example], 20)
            torch.testing.assert_close(batched[index], single[0], rtol=1e-4, atol=1e-5)
    model.train()
    loss, _, _ = training.batch_loss(model, _Tokenizer(), examples, 20)
    loss.backward()
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for _, parameter in trainable)
    assert any(torch.count_nonzero(parameter.grad) for _, parameter in trainable)
    assert base.get_input_embeddings().weight.grad is None
