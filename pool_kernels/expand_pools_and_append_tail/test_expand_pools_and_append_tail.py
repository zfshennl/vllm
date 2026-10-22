# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the standalone ``expand_pools_and_append_tail`` kernel.

The fused Triton kernel must replicate ``expand_pools_to_tokens`` +
``append_tail_to_topk`` (identity path) exactly: a selected pool id expands to
its ``pool_size`` consecutive tokens, invalid ids (-1) stay -1, and each row's
trailing incomplete pool is appended after the history columns, anchored at
the row's own sequence length.

Includes the non-uniform decode batch scenario that motivated the fused
kernel (padded ``[batch, next_n]`` rows): a row whose seq_len is derived from
another request's positions drops its own tail and anchors foreign token ids.
"""

import pytest
import torch

from expand_pools_and_append_tail import expand_pools_and_append_tail

try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    _NPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _NPU_AVAILABLE, reason="NPU required")

KPOOL = 4
TOPK_TOKENS = 16
SELECT_K = TOPK_TOKENS // KPOOL


def _expand_append_reference(pool_ids, seq_lens, pool_size):
    """Pure-torch reference: expand_pools_to_tokens + append_tail_to_topk
    (identity path, no page_table / topk_offsets)."""
    rows, n_groups = pool_ids.shape
    topk = n_groups * pool_size
    out_cols = topk + pool_size - 1
    device = pool_ids.device

    offsets = torch.arange(pool_size, device=device, dtype=torch.int64)
    token_ids = (pool_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets).reshape(
        rows, topk
    )
    valid = (pool_ids >= 0).unsqueeze(-1).expand(-1, -1, pool_size).reshape(rows, topk)
    expanded = torch.where(valid, token_ids, torch.full_like(token_ids, -1)).to(
        torch.int32
    )

    if pool_size == 1:
        return expanded

    cols = torch.arange(out_cols, device=device)[None, :]
    is_history = cols < topk
    tail_off = cols - topk
    pool_len = (seq_lens // pool_size).to(torch.int32)
    tail_start = pool_len * pool_size
    tail_count = seq_lens.to(torch.int32) - tail_start
    is_tail = (tail_off >= 0) & (tail_off < tail_count[:, None])

    safe_hist = torch.minimum(cols, torch.full_like(cols, topk - 1)).expand(
        rows, out_cols
    )
    history_val = torch.gather(expanded, 1, safe_hist)
    tail_val = (tail_start[:, None] + tail_off).to(torch.int32)

    out = torch.where(is_history, history_val, torch.full_like(history_val, -1))
    out = torch.where(is_tail, tail_val, out)
    return out


@pytest.mark.parametrize("pool_size", [1, 2, 4, 16])
@pytest.mark.parametrize("n_groups", [1, 4])
@pytest.mark.parametrize("pid_dtype", [torch.int32, torch.int64])
def test_matches_expand_append_pair(pool_size, n_groups, pid_dtype):
    """The fused kernel equals the pure-torch expand+append pair for random
    selections (including invalid pool ids) and random sequence lengths."""
    g = torch.Generator(device="npu").manual_seed(0)
    rows = 8
    pool_ids = torch.randint(-1, 64, (rows, n_groups), generator=g, device="npu")
    pool_ids = pool_ids.to(pid_dtype)
    topk = n_groups * pool_size
    seq_lens = torch.randint(
        0, 4 * topk + pool_size, (rows,), generator=g, device="npu", dtype=torch.int32
    )

    out = expand_pools_and_append_tail(pool_ids, seq_lens, pool_size)
    assert out.dtype == torch.int32
    assert out.shape == (rows, topk + pool_size - 1)
    ref = _expand_append_reference(pool_ids, seq_lens, pool_size)
    assert torch.equal(out, ref)


def test_invalid_pool_ids_and_empty_tail():
    """Invalid ids (-1) expand to -1 columns; a seq_len that is a multiple of
    the pool size leaves the tail columns empty."""
    pool_size = 4
    pool_ids = torch.tensor([[3, -1], [0, 1]], dtype=torch.int64, device="npu")
    seq_lens = torch.tensor([8, 6], dtype=torch.int32, device="npu")
    out = expand_pools_and_append_tail(pool_ids, seq_lens, pool_size)
    assert out.tolist() == [
        [12, 13, 14, 15, -1, -1, -1, -1, -1, -1, -1],
        [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, -1],
    ]


def make_non_uniform_batch():
    """3 requests: plain decode (1 token), MTP verify (4), adaptive verify (3).

    Flat decode positions (production layout: decode tokens first, then any
    prefill tokens of the same batch):
      req0: [30]           (context len 30)
      req1: [100..103]     (verify at context len 100)
      req2: [7, 8, 9]      (verify at context len 7)
    Followed by 4 prefill tokens at positions 555..558 so the flat tensor is
    at least ``n = 3 * 4`` long, as it is in a real mixed batch.
    """
    decode_lens = torch.tensor([1, 4, 3], dtype=torch.int64)
    per_req_positions = [[30], [100, 101, 102, 103], [7, 8, 9]]
    flat_decode = [p for req in per_req_positions for p in req]
    positions = torch.tensor(flat_decode + [555, 556, 557, 558], dtype=torch.int64)
    return decode_lens, per_req_positions, positions


def expected_row_seq_lens(per_req_positions, batch_size, next_n):
    """Row (b, t) -> pos + 1 for real rows, 0 for pad rows."""
    out = torch.zeros(batch_size * next_n, dtype=torch.int32)
    for b, req in enumerate(per_req_positions):
        for t, pos in enumerate(req):
            out[b * next_n + t] = pos + 1
    return out


def test_tail_anchors_at_each_rows_own_length():
    """Padded non-uniform decode rows: the per-row mapping keeps each request's
    trailing incomplete pool; the legacy flat ``positions[:n] + 1`` mapping
    drops real tail tokens and anchors foreign (prefill) token ids."""
    decode_lens, per_req, positions = make_non_uniform_batch()
    batch_size, next_n = decode_lens.shape[0], int(decode_lens.max())
    n = batch_size * next_n
    fixed = expected_row_seq_lens(per_req, batch_size, next_n).npu()
    legacy = (positions[:n].to(torch.int32) + 1).npu()

    # Identity pool selection: pools 0..SELECT_K-1 -> tokens 0..TOPK_TOKENS-1.
    pool_ids = (
        torch.arange(SELECT_K, dtype=torch.int64)
        .expand(n, SELECT_K)
        .contiguous()
        .npu()
    )

    fixed_out = expand_pools_and_append_tail(pool_ids, fixed, KPOOL)
    legacy_out = expand_pools_and_append_tail(pool_ids, legacy, KPOOL)
    assert torch.equal(fixed_out, _expand_append_reference(pool_ids, fixed, KPOOL))

    fixed_tail = fixed_out[:, TOPK_TOKENS:]
    legacy_tail = legacy_out[:, TOPK_TOKENS:]

    # Fixed: row (1, 0) (seq 101) keeps its single tail token 100; row (2, 2)
    # (seq 10) keeps its full trailing pool [8, 9].
    assert fixed_tail[4, 0].item() == 100
    assert fixed_tail[10, :2].tolist() == [8, 9]

    # Legacy: row (1, 0) loses its tail token (anchored at 104, count 0)...
    assert legacy_tail[4, 0].item() == -1
    # ...and row (2, 2) reads prefill position 557 -> tail indices 556/557,
    # way past req2's 10-token sequence -> out-of-bounds block-table reads.
    assert legacy_tail[10, :2].tolist() == [556, 557]

    assert not torch.equal(legacy_tail, fixed_tail)
