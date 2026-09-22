import math

import pytest
import torch
from torch.nn.attention.flex_attention import create_block_mask

from experiments.long_context.kernel_probe import (
    MAX_REFERENCE_SEQUENCE_LENGTH,
    _parse_args,
    build_analytical_causal_block_mask,
    causal_mask_mod,
    gemma4_flex_kernel_options,
)


def _row_sets(counts, indices):
    counts = counts[0, 0].cpu().tolist()
    indices = indices[0, 0].cpu().tolist()
    return [{int(value) for value in row[:count]} for count, row in zip(counts, indices, strict=True)]


def _transpose_sets(rows):
    transposed = [set() for _ in rows]
    for row_idx, columns in enumerate(rows):
        for column_idx in columns:
            transposed[column_idx].add(row_idx)
    return transposed


def _materialize_tokens(mask, length, window, block_size=128):
    partial = _row_sets(mask.kv_num_blocks, mask.kv_indices)
    full = _row_sets(mask.full_kv_num_blocks, mask.full_kv_indices)
    result = torch.zeros((length, length), dtype=torch.bool)
    for q_idx in range(length):
        q_block = q_idx // block_size
        for kv_idx in range(length):
            kv_block = kv_idx // block_size
            if kv_block in full[q_block]:
                result[q_idx, kv_idx] = True
            elif kv_block in partial[q_block]:
                result[q_idx, kv_idx] = kv_idx <= q_idx and (window is None or kv_idx > q_idx - window)
    return result


@pytest.mark.parametrize(
    ("length", "window"),
    [(1, None), (127, None), (129, None), (385, None), (129, 1), (257, 128), (385, 129), (777, 300)],
)
def test_analytical_block_mask_matches_canonical_small_reference(length, window):
    block_size = 128
    analytical = build_analytical_causal_block_mask(
        length, window_size=window, device="cpu", block_size=block_size
    )
    canonical = create_block_mask(
        causal_mask_mod(window),
        B=1,
        H=None,
        Q_LEN=length,
        KV_LEN=length,
        device="cpu",
        BLOCK_SIZE=block_size,
    )

    assert analytical.seq_lengths == canonical.seq_lengths == (length, length)
    assert analytical.BLOCK_SIZE == canonical.BLOCK_SIZE == (block_size, block_size)
    positions = torch.arange(length)
    expected = positions[None, :] <= positions[:, None]
    if window is not None:
        expected &= positions[None, :] > positions[:, None] - window
    assert torch.equal(_materialize_tokens(analytical, length, window, block_size), expected)

    # Every canonical active token is preserved, while analytical metadata may
    # classify an incomplete final block more efficiently than create_block_mask.
    assert torch.equal(_materialize_tokens(canonical, length, window, block_size), expected)

    kv_partial = _row_sets(analytical.kv_num_blocks, analytical.kv_indices)
    kv_full = _row_sets(analytical.full_kv_num_blocks, analytical.full_kv_indices)
    assert _row_sets(analytical.q_num_blocks, analytical.q_indices) == _transpose_sets(kv_partial)
    assert _row_sets(analytical.full_q_num_blocks, analytical.full_q_indices) == _transpose_sets(kv_full)


def test_local_builder_metadata_scales_by_window_blocks():
    length = 16_384
    window = 1_024
    block_size = 128
    mask = build_analytical_causal_block_mask(
        length, window_size=window, device="cpu", block_size=block_size
    )
    expected_blocks = math.ceil(length / block_size)
    assert mask.kv_indices.shape[-2] == expected_blocks
    assert int(mask.kv_num_blocks.max()) <= 2
    assert mask.full_kv_indices.shape[-1] <= math.ceil(window / block_size)
    assert int(mask.q_num_blocks.max()) <= 2
    assert mask.full_q_indices.shape[-1] <= math.ceil(window / block_size)
    index_tensors = (mask.kv_indices, mask.full_kv_indices, mask.q_indices, mask.full_q_indices)
    assert len({tensor.shape for tensor in index_tensors}) == 1
    assert len({tensor.stride() for tensor in index_tensors}) == 1


def test_builder_broadcasts_batch_and_heads():
    mask = build_analytical_causal_block_mask(257, window_size=128, device="cpu")
    assert mask.kv_num_blocks.shape[:2] == (1, 1)
    assert mask.kv_indices.shape[:2] == (1, 1)


def test_shared_gemma4_kernel_options_cover_forward_and_backward():
    options = gemma4_flex_kernel_options()
    assert options == {
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


@pytest.mark.parametrize("length", [0, -1])
def test_builder_rejects_nonpositive_length(length):
    with pytest.raises(ValueError, match="sequence_length"):
        build_analytical_causal_block_mask(length, device="cpu")


def test_cli_is_bounded_and_create_only(tmp_path):
    with pytest.raises(SystemExit):
        _parse_args(["--output", str(tmp_path / "too-long.json"), "--sequence-length", str(MAX_REFERENCE_SEQUENCE_LENGTH + 1)])

    existing = tmp_path / "existing.json"
    existing.write_text("keep", encoding="utf-8")
    with pytest.raises(SystemExit):
        _parse_args(["--output", str(existing)])
    assert existing.read_text(encoding="utf-8") == "keep"
