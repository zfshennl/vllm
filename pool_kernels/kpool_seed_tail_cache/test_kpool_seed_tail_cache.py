# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the standalone ``kpool_seed_tail_cache`` kernel.

``kpool_seed_tail_cache`` copies each prefill token's raw K + gate score into
its request's paged tail ring. Token ``i`` belongs to the tail iff the token
``kpool`` slots ahead belongs to a different tail block (request boundary) or
does not exist (batch end / padding, slot < 0).

Covers multi-request tail selection against an independent pure-torch
reference, the negative-ahead (padding) case, a randomized fuzz pass, and the
padded block-stride regression (in production the tail aliases a larger
indexer allocation, so the kernel must honor ``tail_kv_cache.stride(0)``).
"""

import pytest
import torch

from kpool_seed_tail_cache import kpool_seed_tail_cache

try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    _NPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _NPU_AVAILABLE, reason="NPU required")

HEAD_DIM = 128


def _seed_reference(tail, key, score, tslot, kpool):
    """Mirror of the kernel: token ``i`` is seeded iff ``tslot[i] >= 0`` and
    (``i + kpool`` is past the batch, or ``tslot[i+kpool] < 0``, or its tail
    block differs from token ``i``'s)."""
    tail = tail.clone()
    n = tslot.shape[0]
    ring = tail.shape[2]
    tslot_cpu = tslot.cpu().tolist()
    key_cpu = key.cpu()
    score_cpu = score.cpu()
    for i in range(n):
        t = tslot_cpu[i]
        if t < 0:
            continue
        blk = t // ring
        if i + kpool < n:
            ahead = tslot_cpu[i + kpool]
            if ahead >= 0 and ahead // ring == blk:
                continue
        ring_slot = t % ring
        tail[blk, 0, ring_slot] = key_cpu[i]
        tail[blk, 1, ring_slot] = score_cpu[i]
    return tail


def _make_batch(lens, blocks, ring, n_pad=0):
    """Flat prefill batch: per-request positions from 0, tail slots
    ``block * ring + pos % ring``, followed by ``n_pad`` padding tokens (-1)."""
    tslot = []
    for length, blk in zip(lens, blocks):
        tslot.extend(blk * ring + p % ring for p in range(length))
    tslot.extend([-1] * n_pad)
    n = len(tslot)
    tslot_t = torch.tensor(tslot, dtype=torch.int32, device="npu")
    key = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    score = torch.randn(n, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    return tslot_t, key, score


def _run_seed(tail, key, score, tslot, kpool):
    kpool_seed_tail_cache(tail, key, score, tslot, kpool, HEAD_DIM)
    torch.npu.synchronize()
    return tail


@pytest.mark.parametrize("kpool", [4, 8])
@pytest.mark.parametrize("ring_pools", [1, 2])
def test_seeds_only_each_request_tail_tokens(kpool, ring_pools):
    """Tokens whose kpool-ahead neighbor stays in the same request must not be
    seeded; exactly the last min(len, kpool) tokens of each request land in
    the ring at ``pos % ring``."""
    ring = ring_pools * kpool
    lens = [5, 4, 7, 2]
    blocks = [0, 2, 5, 6]
    tail = torch.zeros(
        8, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device="npu"
    )
    tslot, key, score = _make_batch(lens, blocks, ring, n_pad=3)

    out = _run_seed(tail, key, score, tslot, kpool)
    ref = _seed_reference(tail, key, score, tslot, kpool)
    assert torch.equal(out, ref)

    for length, blk in zip(lens, blocks):
        n_seeded = int((out[blk, 0].abs().sum(dim=-1) > 0).sum().item())
        assert n_seeded == min(length, kpool)


def test_padding_ahead_neighbor_seeds_tail():
    """A token whose kpool-ahead slot is padding (-1) is a tail token; the
    padding tokens themselves are never seeded."""
    kpool, ring = 4, 8
    tail = torch.zeros(2, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    # tokens 0..3 of block 0, then two padding tokens
    tslot = torch.tensor([0, 1, 2, 3, -1, -1], dtype=torch.int32, device="npu")
    key = torch.randn(6, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    score = torch.randn(6, HEAD_DIM, dtype=torch.bfloat16, device="npu")

    out = _run_seed(tail, key, score, tslot, kpool)
    ref = _seed_reference(tail, key, score, tslot, kpool)
    assert torch.equal(out, ref)

    # tokens 0 (ahead is padding) and 1..3 (past the batch) are all seeded
    for i in range(4):
        assert torch.equal(out[0, 0, i % ring], key[i])
        assert torch.equal(out[0, 1, i % ring], score[i])
    # ring slots beyond the seeded tokens stay untouched
    assert torch.equal(out[0, :, 4:], torch.zeros_like(out[0, :, 4:]))


@pytest.mark.parametrize("seed", range(10))
def test_seed_matches_reference_fuzz(seed):
    """Random request lengths, ring sizes, and padding counts."""
    g = torch.Generator(device="npu").manual_seed(seed)
    n_req = int(torch.randint(1, 6, (1,), generator=g, device="npu").item())
    kpool = 4
    ring = kpool * int(torch.randint(1, 3, (1,), generator=g, device="npu").item())
    lens = [
        int(torch.randint(1, 2 * kpool + 2, (1,), generator=g, device="npu").item())
        for _ in range(n_req)
    ]
    blocks = list(range(n_req))
    n_pad = int(torch.randint(0, 4, (1,), generator=g, device="npu").item())

    tail = torch.zeros(
        n_req + 2, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device="npu"
    )
    tslot, key, score = _make_batch(lens, blocks, ring, n_pad)

    out = _run_seed(tail, key, score, tslot, kpool)
    ref = _seed_reference(tail, key, score, tslot, kpool)
    assert torch.equal(out, ref)


def test_seed_honors_padded_tail_block_stride():
    """The tail shares a padded indexer allocation in production: each tail
    block starts at its indexer block stride (larger than a dense tail block).
    A kernel that addresses blocks densely writes into an unrelated indexer
    block and leaves the request's tail block untouched. Regression for the
    NVIDIA kernel's dense-addressing bug; also checks the untouched padding
    region of the backing buffer stays at its sentinel."""
    kpool = 4
    num_blocks = 6
    logical_block_elems = 2 * kpool * HEAD_DIM
    padded_block_elems = logical_block_elems + 256
    sentinel = -123.0
    backing = torch.full(
        (num_blocks * padded_block_elems,),
        sentinel,
        dtype=torch.bfloat16,
        device="npu",
    )
    tail = torch.as_strided(
        backing,
        size=(num_blocks, 2, kpool, HEAD_DIM),
        stride=(padded_block_elems, kpool * HEAD_DIM, HEAD_DIM, 1),
    )

    block = 3
    ring_slot = 2
    key = torch.arange(HEAD_DIM, dtype=torch.bfloat16, device="npu").unsqueeze(0)
    score = (key + 256).to(torch.bfloat16)
    tail_slot = torch.tensor(
        [block * kpool + ring_slot], dtype=torch.int32, device="npu"
    )

    kpool_seed_tail_cache(tail, key, score, tail_slot, kpool, HEAD_DIM)
    torch.npu.synchronize()

    assert torch.equal(tail[block, 0, ring_slot], key[0])
    assert torch.equal(tail[block, 1, ring_slot], score[0])

    compact_offset = (block * 2 * kpool + ring_slot) * HEAD_DIM
    assert torch.all(backing[compact_offset : compact_offset + HEAD_DIM] == sentinel)
