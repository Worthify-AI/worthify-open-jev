import pytest

from openjev_phase1.shared import _suffix_layout, score_shared


def test_suffix_padding_follows_real_tokens():
    layout, ends = _suffix_layout([[3, 4], [5]], 7, 0)
    assert layout["input_ids"] == [[3, 4], [5, 0]]
    assert layout["attention_mask"] == [[1] * 9, [1] * 8 + [0]]
    assert layout["position_ids"] == [[7, 8], [7, 0]]
    assert ends == [1, 0]


def test_empty_suffix_is_rejected():
    with pytest.raises(ValueError):
        _suffix_layout([[1], []], 7, 0)


def test_cached_shared_is_rejected_before_model_or_gpu_access():
    class Model:
        def parameters(self):
            raise AssertionError("cache guard must run before model access")

    with pytest.raises(RuntimeError, match="6/777"):
        score_shared(Model(), object(), [], {})
