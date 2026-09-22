import pytest
from types import SimpleNamespace

from openjev_phase1.core import _native_model_class, _pinned_source


def test_remote_model_revision_must_be_full_commit():
    with pytest.raises(ValueError, match="pinned 40-character"):
        _pinned_source("org/model", "main", "models")


def test_local_source_requires_manifest_revision(tmp_path):
    with pytest.raises(ValueError, match="explicit manifest"):
        _pinned_source(str(tmp_path), "", "models")


def test_local_source_accepts_manifest_revision(tmp_path):
    assert _pinned_source(str(tmp_path), "dataset-sha256:abc", "models") is True


@pytest.mark.parametrize("model_type", ["gemma4_unified", "gemma4_unified_text"])
def test_unified_gemma_uses_its_native_text_class(model_type):
    unified = object()
    transformers = SimpleNamespace(AutoModelForCausalLM=object(), Gemma4UnifiedForCausalLM=unified)
    assert _native_model_class(transformers, model_type) is unified


def test_unified_gemma_refuses_standard_gemma_fallback():
    transformers = SimpleNamespace(AutoModelForCausalLM=object(), Gemma4ForCausalLM=object())
    with pytest.raises(RuntimeError, match="Unified"):
        _native_model_class(transformers, "gemma4_unified")


def test_local_checkpoint_must_be_directory(tmp_path):
    source = tmp_path / "weights.bin"
    source.touch()
    with pytest.raises(ValueError, match="directory"):
        _pinned_source(str(source), "manifest:abc", "models")


@pytest.mark.parametrize("model_type", ["gemma4", "gemma4_unified", "gemma4_unified_text"])
def test_loader_uses_native_text_config_and_pinned_revision(monkeypatch, model_type):
    import sys
    import torch
    from openjev_phase1.core import load_causal_model

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr("openjev_phase1.core._gemma_text_checkpoint", lambda *a, **kw: {})
    text_config = SimpleNamespace(model_type=model_type + "_text", use_cache=True)
    config = SimpleNamespace(model_type=model_type, get_text_config=lambda: text_config)
    calls = []

    class Model:
        @staticmethod
        def from_pretrained(source, **kwargs):
            calls.append((source, kwargs))
            return SimpleNamespace(config=text_config, eval=lambda: None), {}

    def get_config(source, **kwargs):
        assert kwargs["revision"] == "a" * 40
        assert kwargs["trust_remote_code"] is False
        return config

    fake = SimpleNamespace(__version__="test", AutoModelForCausalLM=object(),
                           Gemma4ForCausalLM=Model, Gemma4UnifiedForCausalLM=Model,
                           AutoConfig=SimpleNamespace(from_pretrained=get_config),
                           AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **kw: object()))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    _, _, metadata = load_causal_model("org/checkpoint", "a" * 40)
    assert calls[0][1]["config"] is text_config
    assert calls[0][1]["revision"] == "a" * 40
    assert calls[0][1]["output_loading_info"] is True
    assert metadata["use_cache"] is False


@pytest.mark.parametrize("architecture", ["Gemma4", "Gemma4Unified"])
def test_full_gemma_checkpoint_loads_native_text_weights_and_tied_head(tmp_path, architecture):
    import torch
    import transformers
    from openjev_phase1.core import _gemma_text_checkpoint

    settings = dict(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                    num_attention_heads=2, num_key_value_heads=1, head_dim=16, global_head_dim=16,
                    num_global_key_value_heads=1, max_position_embeddings=64,
                    layer_types=["sliding_attention", "full_attention"])
    if architecture == "Gemma4":
        settings.update(hidden_size_per_layer_input=0, vocab_size_per_layer_input=32)
    text_config = getattr(transformers, architecture + "TextConfig")(**settings)
    full_config = getattr(transformers, architecture + "Config")(
        text_config=text_config, vision_config=None, audio_config=None)
    original = getattr(transformers, architecture + "ForConditionalGeneration")(full_config).eval()
    original.save_pretrained(tmp_path)
    text_class = getattr(transformers, architecture + "ForCausalLM")
    options = _gemma_text_checkpoint(str(tmp_path), "test", text_class, text_config, local=True)
    assert options == {"key_mapping": {r"^model\.language_model\.": "model."}}
    converted, loading = text_class.from_pretrained(tmp_path, config=text_config,
                                                     output_loading_info=True, **options)
    assert not any(loading.values())
    assert converted.lm_head.weight is converted.model.embed_tokens.weight
    ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        expected = original(input_ids=ids, use_cache=False).logits
        actual = converted(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    # A missing real text tensor must fail during header/meta preflight, before
    # from_pretrained could initialize missing weights on a GPU.
    from safetensors.torch import load_file, save_file
    path = tmp_path / "model.safetensors"
    tensors = load_file(path)
    tensors.pop("model.language_model.layers.0.self_attn.q_proj.weight")
    save_file(tensors, path, metadata={"format": "pt"})
    with pytest.raises(RuntimeError, match="failed preflight.*missing"):
        _gemma_text_checkpoint(str(tmp_path), "test", text_class, text_config, local=True)


def test_text_only_gemma_checkpoint_needs_no_prefix_conversion(tmp_path):
    from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig
    from openjev_phase1.core import _gemma_text_checkpoint

    config = Gemma4UnifiedTextConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
                                     num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
                                     head_dim=16, global_head_dim=16, num_global_key_value_heads=1)
    Gemma4UnifiedForCausalLM(config).save_pretrained(tmp_path)
    assert _gemma_text_checkpoint(str(tmp_path), "test", Gemma4UnifiedForCausalLM, config, local=True) == {}
