import pytest

from experiments.long_context import pilot


class CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 0

    def apply_chat_template(self, messages, **_kwargs):
        return "\n".join(message["content"] for message in messages) + "\nassistant:"

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [ord(character) for character in text]

    def decode(self, token):
        if isinstance(token, list):
            return "".join(chr(value) for value in token)
        return chr(token)


def test_synthetic_probe_has_one_exact_actual_token_length(monkeypatch):
    monkeypatch.setattr(pilot, "FILLER_FRAGMENTS", ("x",))
    tokenizer = CharacterTokenizer()
    shell = len(tokenizer.encode(pilot._render_probe_prompt(tokenizer, 0, "x")))
    encoded = pilot.build_probe_encoding(tokenizer, shell + 37)
    assert encoded["actual_tokens"] == shell + 37
    assert encoded["filler_repetitions"] == 37
    assert encoded["option_token_ids"] == [ord("A"), ord("B")]
    assert len(encoded["prompt_sha256"]) == 64


def test_synthetic_probe_rejects_changed_option_boundary(monkeypatch):
    monkeypatch.setattr(pilot, "FILLER_FRAGMENTS", ("x",))

    class BoundaryTokenizer(CharacterTokenizer):
        def encode(self, text, add_special_tokens=False):
            encoded = super().encode(text, add_special_tokens=add_special_tokens)
            if text.endswith("assistant:A"):
                return encoded[:-2] + [999]
            return encoded

    tokenizer = BoundaryTokenizer()
    shell = len(tokenizer.encode(pilot._render_probe_prompt(tokenizer, 0, "x")))
    with pytest.raises(RuntimeError, match="option A"):
        pilot.build_probe_encoding(tokenizer, shell + 4)


def test_probe_corrects_the_first_filler_boundary_change(monkeypatch):
    monkeypatch.setattr(pilot, "FILLER_FRAGMENTS", ("x",))

    class FirstInsertionTokenizer(CharacterTokenizer):
        def encode(self, text, add_special_tokens=False):
            encoded = super().encode(text, add_special_tokens=add_special_tokens)
            if "Repeated filler follows:x" in text:
                encoded.insert(0, 999)
            return encoded

    tokenizer = FirstInsertionTokenizer()
    shell = len(tokenizer.encode(pilot._render_probe_prompt(tokenizer, 0, "x")))
    encoded = pilot.build_probe_encoding(tokenizer, shell + 37)
    assert encoded["actual_tokens"] == shell + 37
    assert encoded["filler_repetitions"] == 36


@pytest.mark.parametrize("tokens", [0, -1, pilot.MAX_NATIVE_CONTEXT + 1])
def test_token_preflight_rejects_invalid_or_unsupported_lengths(tokens):
    with pytest.raises(ValueError, match="tokens"):
        pilot.validate_token_length(tokens)
