# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone extract of vLLM's ``kpool_seed_tail_cache`` Triton kernel.

Source: ``vllm/models/glm5next/nvidia/ops/kpool_compress.py`` (NVIDIA variant).
Depends only on ``torch`` and ``triton`` — no vLLM imports.

Seeds each request's incomplete pool into its paged tail cache during prefill.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# The GLM-5.3-Flash indexer head dimension is fixed at 128.
INDEX_HEAD_DIM = 128


@triton.jit
def _kpool_tail_seed_kernel(
    key_ptr,
    score_ptr,
    tslot_ptr,
    tail_ptr,
    n_tokens,
    TAIL_BLOCK_ELEMS: tl.constexpr,
    KPOOL_HEAD: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    KPOOL: tl.constexpr,
    RING: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Copy token ``i``'s raw K + gate into its request's tail ring.

    Token ``i`` is among its request's last KPOOL tokens iff the token KPOOL
    ahead belongs to a different tail block (or is past the batch / padding,
    slot < 0). ``tslot = block * RING + pos % RING``; the destination is
    ``tail[block, {0:K, 1:score}, pos % RING, :]``.
    """
    i = tl.program_id(0)
    t = tl.load(tslot_ptr + i).to(tl.int64)
    if t < 0:
        return
    blk = t // RING  # t >= 0 here, so trunc == floor
    ahead = tl.load(tslot_ptr + i + KPOOL, mask=i + KPOOL < n_tokens, other=-1).to(
        tl.int64
    )
    # Match the torch semantics exactly: a negative ahead slot floors to a
    # block id that differs from every real block -> token is in the tail.
    # Only divide non-negative slots (Triton int div truncates, torch floors).
    if ahead >= 0 and ahead // RING == blk:
        return
    offs = tl.arange(0, BLOCK_D)
    m = offs < HEAD_DIM
    block_base = blk * TAIL_BLOCK_ELEMS
    base = block_base + (t % RING) * HEAD_DIM
    k = tl.load(key_ptr + i * HEAD_DIM + offs, mask=m)
    s = tl.load(score_ptr + i * HEAD_DIM + offs, mask=m)
    tl.store(tail_ptr + base + offs, k, mask=m)
    tl.store(
        tail_ptr + block_base + KPOOL_HEAD + (t % RING) * HEAD_DIM + offs, s, mask=m
    )


def kpool_seed_tail_cache(
    tail_kv_cache: torch.Tensor,
    key: torch.Tensor,
    gate_score: torch.Tensor,
    tslot: torch.Tensor,
    kpool: int,
    head_dim: int = INDEX_HEAD_DIM,
) -> None:
    """Seed the paged tail cache from a prefill batch (see the kernel)."""
    assert tail_kv_cache.dtype == torch.bfloat16
    assert tail_kv_cache.ndim == 4 and tail_kv_cache.shape[1] == 2
    assert tail_kv_cache.stride(3) == 1 and tail_kv_cache.stride(2) == head_dim
    assert key.dtype == torch.bfloat16
    n = tslot.shape[0]
    if n == 0:
        return
    _kpool_tail_seed_kernel[(n,)](
        key,
        gate_score,
        tslot,
        tail_kv_cache,
        n,
        TAIL_BLOCK_ELEMS=tail_kv_cache.stride(0),
        KPOOL_HEAD=tail_kv_cache.stride(1),
        HEAD_DIM=head_dim,
        KPOOL=kpool,
        RING=tail_kv_cache.shape[2],
        BLOCK_D=triton.next_power_of_2(head_dim),
    )
