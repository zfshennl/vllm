# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the standalone batched kpool decode-update kernel.

Validates ``kpool_decode_update_and_maybe_write_cache_batched`` against an
independent pure-torch reference that replicates the per-request, in-position
order semantics: stash each token into a paged tail ring; on pool completion
(``pos % pool_size == pool_size-1``) softmax(gate+ape)-weighted sum + Hadamard-128
+ per-vector fp8 absmax quant + write to the indexer K cache. Covers
no-completion, completion-at-end, completion-mid-batch, non-uniform padding,
plain decode, the MX-scale (packed UE8M0) write path, plus a randomized fuzz
pass.

The kernel iterates each request's ``next_n`` tokens in position order inside
one program (grid = num_requests) to preserve the pool-completion
read-after-stash dependency; the reference mirrors that ordering.

``test_decode_writer_matches_prefill_writer_reference`` and
``test_rejected_draft_redo_needs_ring_slots`` mirror the production invariant
that the decode writer and the prefill writer produce identical cache bytes.
The prefill writer (``kpool_compress_and_write_cache``) is not part of this
standalone extract, so the prefill side is its pure-torch reference; beware
that a reference-based test can drift to match a buggy kernel — that is
exactly how the stash-gating bug (intra-pool tokens never entering the tail
ring, because the stash was gated on the pool-granular ``slot_mapping``)
stayed green upstream.
"""

import math

import pytest
import torch

from kpool_decode_update_and_maybe_write_cache_batched import (
    kpool_decode_update_and_maybe_write_cache_batched,
)

try:
    import torch_npu  # noqa: F401

    _NPU_AVAILABLE = torch.npu.is_available()
except ImportError:
    _NPU_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _NPU_AVAILABLE, reason="NPU required")

HEAD_DIM = 128
POOL_SIZE = 16
PAGE_SIZE = 64
NUM_BLOCKS = 32
ROUND_SCALE = True
FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max


def _make_caches():
    kv = torch.zeros(
        NUM_BLOCKS, PAGE_SIZE, HEAD_DIM + 4, dtype=torch.uint8, device="npu"
    )
    tail = torch.zeros(
        NUM_BLOCKS, 2, POOL_SIZE, HEAD_DIM, dtype=torch.bfloat16, device="npu"
    )
    return kv, tail


def _tail_slot_for(blocks, pos):
    """tail_slot = block*POOL + pos%POOL; each request owns a distinct tail block."""
    blk = torch.tensor(blocks, device=pos.device, dtype=torch.int32).unsqueeze(1)
    return (blk * POOL_SIZE + pos % POOL_SIZE).to(torch.int32)


def _seed_prior(tail, blocks, n_prior, seed=42):
    if n_prior <= 0:
        return
    g = torch.Generator(device=tail.device).manual_seed(seed)
    prior_k = torch.randn(
        len(blocks),
        n_prior,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=tail.device,
        generator=g,
    )
    prior_s = torch.randn(
        len(blocks),
        n_prior,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=tail.device,
        generator=g,
    )
    for i, blk in enumerate(blocks):
        tail[blk, 0, :n_prior, :] = prior_k[i]
        tail[blk, 1, :n_prior, :] = prior_s[i]


def _hadamard128_torch(x: torch.Tensor) -> torch.Tensor:
    """Reference Hadamard-128 on the last dim (must be 128)."""
    n = x.shape[-1]
    assert n == 128
    h = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float32, device=x.device)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], dim=1), torch.cat([h, -h], dim=1)], dim=0)
    h = h / math.sqrt(n)
    return x @ h


def _quantize_pool(pool_ks, pool_scores, round_scale):
    """softmax(pool_scores)-weighted sum of pool_ks -> Hadamard-128 -> bf16 ->
    per-vector fp8 absmax quant. Returns (fp8 K [D], scale bytes uint8[4])."""
    max_score = pool_scores.max(dim=0).values
    prob = torch.exp(pool_scores - max_score)
    acc = (pool_ks * prob).sum(dim=0)
    x = (acc / prob.sum(dim=0)).to(torch.bfloat16).to(torch.float32)
    x = _hadamard128_torch(x).to(torch.bfloat16).to(torch.float32)
    absmax = torch.clamp(x.abs().max(), min=1e-4)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(absmax / FP8_MAX)))
    else:
        scale = absmax / FP8_MAX
    quantized = torch.clamp(x / scale, -FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    scale_bytes = scale.detach().reshape(1).view(torch.uint8)
    return quantized, scale_bytes


def _write_pool_bytes(kv_flat, quantized, scale_bytes, cache_loc):
    """Write K and scale at the separated-layout offsets for a flat slot."""
    page_bytes = PAGE_SIZE * (HEAD_DIM + 4)
    k_region = HEAD_DIM * PAGE_SIZE
    page_base = (cache_loc // PAGE_SIZE) * page_bytes
    loc_tok = cache_loc % PAGE_SIZE
    k_off = page_base + loc_tok * HEAD_DIM + torch.arange(HEAD_DIM)
    s_off = page_base + k_region + loc_tok * 4
    kv_flat[k_off] = quantized.view(torch.uint8)
    kv_flat[s_off : s_off + 4] = scale_bytes


def _torch_reference(
    kv: torch.Tensor,
    tail: torch.Tensor,
    tail_slot: torch.Tensor,
    key: torch.Tensor,
    score: torch.Tensor,
    ape: torch.Tensor,
    slot_map: torch.Tensor,
    pos: torch.Tensor,
    mx_scale: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent reference for the batched decode-update kernel.

    For each request, iterate its next_n tokens in order. On each token:
      - if pos%POOL == POOL-1 and pos_valid: compress the pool (slots
        [pool_start..pool_start+POOL-1], current token via is_current) and write
        fp8 K + scale to kv_cache at cache_loc.
      - always stash the current token's K/score into tail[block, pos%POOL].
    """
    kv = kv.clone()
    tail = tail.clone()
    B, next_n = pos.shape
    # The indexer K cache is [num_blocks, PAGE_SIZE, HEAD_DIM+4] uint8 but the
    # kernels interpret each page as [HEAD_DIM*PAGE_SIZE bytes of K (token-major)
    # | 4*PAGE_SIZE bytes of scale (token-major)]. Operate on a flat byte
    # view so the reference writes K and scale at the exact offsets the kernel
    # uses (page_base + tok*HEAD_DIM for K; page_base + HEAD_DIM*PAGE_SIZE +
    # tok*4 for scale).
    tail_slot_cpu = tail_slot.cpu().tolist()
    slot_map_cpu = slot_map.cpu().tolist()
    pos_cpu = pos.cpu().tolist()
    key_cpu = key.float().cpu()
    score_cpu = score.float().cpu()
    ape_cpu = ape.cpu()
    tail_cpu = tail.float().cpu()
    kv_flat = kv.view(torch.uint8).reshape(-1).cpu()
    round_eff = ROUND_SCALE or mx_scale

    for b in range(B):
        for t in range(next_n):
            cache_loc = slot_map_cpu[b][t]
            p = pos_cpu[b][t]
            pos_valid = cache_loc >= 0 and p >= 0
            safe_pos = max(p, 0)
            slot = safe_pos % POOL_SIZE
            phys_slot = safe_pos % POOL_SIZE
            # Per-token block derivation (a leading invalid sentinel must not
            # poison the base for the rest of the request); clamped like the
            # kernel so an invalid entry can't form a negative base.
            block = max(tail_slot_cpu[b][t], 0) // POOL_SIZE

            cur_key = key_cpu[b, t]
            cur_score = score_cpu[b, t]

            if pos_valid and slot == POOL_SIZE - 1:
                pool_logical_start = safe_pos - slot
                pool_ks = []
                pool_scores = []
                for ps in range(POOL_SIZE):
                    is_current = ps == slot
                    phys = (pool_logical_start + ps) % POOL_SIZE
                    pool_ks.append(cur_key if is_current else tail_cpu[block, 0, phys])
                    pool_scores.append(
                        cur_score if is_current else tail_cpu[block, 1, phys]
                    )
                quantized, scale_bytes = _quantize_pool(
                    torch.stack(pool_ks),
                    torch.stack(pool_scores) + ape_cpu,
                    round_eff,
                )
                if mx_scale:
                    # The power-of-two scale's exponent as 4 equal UE8M0 bytes.
                    ue8m0 = scale_bytes.view(torch.int32) >> 23
                    scale_bytes = ue8m0.to(torch.uint8).repeat(4)
                _write_pool_bytes(kv_flat, quantized, scale_bytes, cache_loc)

            # stash -- gated on the TOKEN-granular tail slot, not on pos_valid.
            # pos_valid keys off the pool-granular cache_loc, which is -1 for
            # every token that is not the pool's last, so gating the stash on it
            # would drop all intra-pool tokens.
            if p >= 0 and tail_slot_cpu[b][t] >= 0:
                tail_cpu[block, 0, phys_slot] = cur_key
                tail_cpu[block, 1, phys_slot] = cur_score

    kv_out = kv_flat.view(NUM_BLOCKS, PAGE_SIZE, HEAD_DIM + 4).to(device="npu")
    return kv_out, tail_cpu.to(torch.bfloat16).to(device="npu")


def _assert_eq(r_ref, r_kern, k_byte_tol=0):
    kv_ref, tail_ref = r_ref
    kv_kern, tail_kern = r_kern
    # Each page holds the fp8 K bytes followed by the scales.
    k_region = PAGE_SIZE * HEAD_DIM
    pages_ref = kv_ref.view(kv_ref.shape[0], -1)
    pages_kern = kv_kern.view(kv_kern.shape[0], -1)
    k_diff = (pages_ref[:, :k_region].int() - pages_kern[:, :k_region].int()).abs()
    assert k_diff.max().item() <= k_byte_tol, (
        f"kv_cache K differs: max diff {k_diff.max().item()}"
    )
    assert torch.equal(pages_ref[:, k_region:], pages_kern[:, k_region:]), (
        "kv_cache scales differ"
    )
    assert torch.equal(tail_ref, tail_kern), (
        "tail_kv_cache differs: max diff "
        f"{(tail_ref.float() - tail_kern.float()).abs().max().item()}"
    )


def _cache_pool_bytes(kv_cache: torch.Tensor, pool_slot: int) -> torch.Tensor:
    """Read a logical pool's K and scale bytes from the cache layout."""
    page_size = kv_cache.shape[1]
    head_dim = kv_cache.shape[2] - 4
    page_idx, token_offset = divmod(pool_slot, page_size)
    flat = kv_cache[page_idx].reshape(-1)
    dims = torch.arange(head_dim, device=kv_cache.device)
    k_offsets = token_offset * head_dim + dims
    scale_offset = page_size * head_dim + 4 * token_offset
    return torch.cat((flat[k_offsets], flat[scale_offset : scale_offset + 4]))


def _pool_matches(kv_ref, kv_kern, pool_slot, k_byte_tol=1) -> bool:
    """Pool-by-pool comparison. The K bytes may differ by at most
    ``k_byte_tol`` per byte (the torch reference's fp32 reduction order can
    flip one fp8 step at a bf16 rounding midpoint) and the scales must agree
    up to one doubling (a power-of-two boundary flip for the same reason)."""
    ref = _cache_pool_bytes(kv_ref, pool_slot)
    kern = _cache_pool_bytes(kv_kern, pool_slot)
    k_diff = (ref[:HEAD_DIM].int() - kern[:HEAD_DIM].int()).abs().max().item()
    if k_diff > k_byte_tol:
        return False
    scale_ref = ref[HEAD_DIM:].view(torch.float32).item()
    scale_kern = kern[HEAD_DIM:].view(torch.float32).item()
    return scale_ref == scale_kern or max(
        scale_ref / scale_kern, scale_kern / scale_ref
    ) <= 2.0


def _prefill_writer_reference(
    kv: torch.Tensor,
    slot_k: torch.Tensor,
    slot_score: torch.Tensor,
    ape: torch.Tensor,
    loc: torch.Tensor,
    round_scale: bool,
) -> torch.Tensor:
    """Pure-torch stand-in for the prefill writer ``kpool_compress_and_write_cache``
    (not part of this extract): per pool, softmax(slot_score+ape)-weighted sum
    -> Hadamard-128 -> fp8 absmax quant -> write K + scale at ``loc``."""
    kv_flat = kv.view(torch.uint8).reshape(-1).cpu()
    ape_cpu = ape.float().cpu()
    for p in range(slot_k.shape[0]):
        quantized, scale_bytes = _quantize_pool(
            slot_k[p].float().cpu(), slot_score[p].float().cpu() + ape_cpu, round_scale
        )
        _write_pool_bytes(kv_flat, quantized, scale_bytes, int(loc[p].item()))
    return kv_flat.view(kv.shape).to(device="npu")


def _run_kernel(kv, tail, tail_slot, key, score, ape, slot_map, pos, mx_scale=False):
    kv = kv.clone()
    tail = tail.clone()
    kpool_decode_update_and_maybe_write_cache_batched(
        kv,
        tail,
        tail_slot,
        key,
        score,
        ape,
        slot_map,
        pos,
        POOL_SIZE,
        HEAD_DIM,
        round_scale=ROUND_SCALE,
        mx_scale=mx_scale,
    )
    return kv, tail


@pytest.mark.parametrize("pool_size", [4, 16])
@pytest.mark.parametrize("ring_pools", [1, 2])
def test_decode_writer_matches_prefill_writer_reference(pool_size, ring_pools):
    """The decode writer must produce (near-)identical cache entries to the
    prefill writer — here its pure-torch reference — the invariant that
    actually matters in production. A decode-only reference can drift to match
    a buggy kernel; the prefill path cross-checks it."""
    ring = ring_pools * pool_size
    n_pools, page, nblk = 8, 64, 4
    n_tok = n_pools * pool_size
    dev = "npu"
    torch.manual_seed(0)
    k = torch.randn(n_tok, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    score = torch.randn(n_tok, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    ape = torch.randn(pool_size, HEAD_DIM, dtype=torch.float32, device=dev)

    kv_prefill = torch.zeros(nblk, page, HEAD_DIM + 4, dtype=torch.uint8, device=dev)
    kv_prefill = _prefill_writer_reference(
        kv_prefill,
        k.view(n_pools, pool_size, HEAD_DIM),
        score.view(n_pools, pool_size, HEAD_DIM),
        ape,
        torch.arange(n_pools, dtype=torch.int64, device=dev),
        ROUND_SCALE,
    )

    # One request owning tail block 0, fed one token per decode step.
    kv_decode = torch.zeros_like(kv_prefill)
    tail = torch.zeros(nblk, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    for t in range(n_tok):
        completes = t % pool_size == pool_size - 1
        kpool_decode_update_and_maybe_write_cache_batched(
            kv_decode,
            tail,
            # token-granular: every token has a valid tail slot
            torch.tensor([[t % ring]], dtype=torch.int32, device=dev),
            k[t].view(1, 1, HEAD_DIM),
            score[t].view(1, 1, HEAD_DIM),
            ape,
            # pool-granular: only the pool's last token carries a cache slot
            torch.tensor(
                [[t // pool_size if completes else -1]], dtype=torch.int32, device=dev
            ),
            torch.tensor([[t]], dtype=torch.int32, device=dev),
            pool_size,
            HEAD_DIM,
            round_scale=ROUND_SCALE,
        )

    differing = [
        p for p in range(n_pools) if not _pool_matches(kv_prefill, kv_decode, p)
    ]
    assert not differing, (
        f"decode-written pools differ from the prefill writer reference: "
        f"{len(differing)}/{n_pools} (pool_size={pool_size}, first={differing[:5]})"
    )


@pytest.mark.parametrize("ring_pools", [1, 2])
def test_rejected_draft_redo_needs_ring_slots(ring_pools):
    """With a one-pool ring, the drafts behind a rejected pool-completing draft
    overwrote the pool's earlier keys, so its redo compressed wrong keys."""
    pool, spec, page, nblk = 4, 3, 64, 2
    ring = ring_pools * pool
    dev = "npu"
    torch.manual_seed(1)
    n_tok = 3 * pool
    k = torch.randn(n_tok, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    score = torch.randn(n_tok, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    ape = torch.randn(pool, HEAD_DIM, dtype=torch.float32, device=dev)
    kv_ref = torch.zeros(nblk, page, HEAD_DIM + 4, dtype=torch.uint8, device=dev)
    kv_ref = _prefill_writer_reference(
        kv_ref,
        k.view(3, pool, HEAD_DIM),
        score.view(3, pool, HEAD_DIM),
        ape,
        torch.arange(3, dtype=torch.int64, device=dev),
        ROUND_SCALE,
    )

    kv = torch.zeros_like(kv_ref)
    tail = torch.zeros(nblk, 2, ring, HEAD_DIM, dtype=torch.bfloat16, device=dev)

    def step(positions, keys, scores):
        pos = torch.tensor([positions], dtype=torch.int32, device=dev)
        slots = [(p // pool) if p % pool == pool - 1 else -1 for p in positions]
        kpool_decode_update_and_maybe_write_cache_batched(
            kv,
            tail,
            pos % ring,
            keys.view(1, -1, HEAD_DIM),
            scores.view(1, -1, HEAD_DIM),
            ape,
            torch.tensor([slots], dtype=torch.int32, device=dev),
            pos,
            pool,
            HEAD_DIM,
            round_scale=ROUND_SCALE,
        )

    for t in range(7):
        step([t], k[t], score[t])
    # Control: verified token 7 completes pool 1 before drafts 8..10 are stashed.
    drafts = torch.randn(spec, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    draft_scores = torch.randn(spec, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    step(
        [7, 8, 9, 10],
        torch.cat([k[7:8], drafts]),
        torch.cat([score[7:8], draft_scores]),
    )
    step([8, 9, 10, 11], k[8:12], score[8:12])  # all drafts rejected
    for p in (1, 2):
        assert _pool_matches(kv_ref, kv, p), p

    # Draft 7 completes pool 1 and is rejected. With a one-pool ring, drafts
    # 8 and 9 overwrite the slots of positions 4 and 5, which are read by the
    # redo of 7.
    kv.zero_()
    tail.zero_()
    for t in range(6):
        step([t], k[t], score[t])
    step(
        [6, 7, 8, 9],
        torch.cat([k[6:7], drafts]),
        torch.cat([score[6:7], draft_scores]),
    )
    step([7, 8, 9, 10], k[7:11], score[7:11])
    pool1_ok = _pool_matches(kv_ref, kv, 1)
    if ring_pools == 1:
        assert not pool1_ok, "expected a one-pool ring to corrupt pool 1"
    else:
        assert pool1_ok


def test_leading_invalid_tail_slot():
    """A request whose FIRST token carries an invalid (-1) tail slot while a
    later token is a real pool completion.

    The tail block must be derived per token, not from token 0: a leading
    invalid sentinel would otherwise poison the base address for the whole
    request (out-of-bounds tail reads on the completion).
    """
    torch.manual_seed(0)
    B, next_n, blocks = 2, 4, [3, 5]
    # req 0: token 0 invalid (pos -1), tokens 1..3 valid, completion at pos 15
    # req 1: all valid, no completion
    pos = torch.tensor(
        [[-1, 13, 14, 15], [4, 5, 6, 7]], dtype=torch.int32, device="npu"
    )
    safe_pos = torch.where(pos >= 0, pos, 0)
    tail_slot = _tail_slot_for(blocks, safe_pos)
    # leading invalid entry carries the -1 sentinel, as the scatter path emits
    tail_slot[0, 0] = -1
    slot_map = torch.full((B, next_n), -1, dtype=torch.int32, device="npu")
    slot_map[0, 3] = 15  # req 0 completes its pool on the last verify token

    key = torch.randn(B, next_n, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    score = torch.randn(B, next_n, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    ape = torch.randn(POOL_SIZE, HEAD_DIM, dtype=torch.float32, device="npu")

    kv, tail = _make_caches()
    _seed_prior(tail, blocks, 13)
    r_ref = _torch_reference(kv, tail, tail_slot, key, score, ape, slot_map, pos)
    r_kern = _run_kernel(kv, tail, tail_slot, key, score, ape, slot_map, pos)
    _assert_eq(r_ref, r_kern)


@pytest.mark.parametrize("mx_scale", [False, True])
@pytest.mark.parametrize(
    "case_id",
    [
        "no_completion",
        "completion_at_end",
        "completion_mid_batch",
        "non_uniform_padding",
        "plain_decode",
    ],
)
def test_batched_matches_reference(case_id, mx_scale):
    torch.manual_seed(0)
    if case_id == "no_completion":
        B, next_n, blocks = 3, 4, [0, 1, 2]
        pos = (
            torch.arange(next_n, device="npu", dtype=torch.int32)
            .unsqueeze(0)
            .expand(B, -1)
            .contiguous()
        )
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32, device="npu")
        n_prior = 0
    elif case_id == "completion_at_end":
        B, next_n, blocks = 2, 4, [0, 1]
        pos = torch.tensor(
            [[12, 13, 14, 15], [12, 13, 14, 15]], dtype=torch.int32, device="npu"
        )
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32, device="npu")
        slot_map[:, 3] = torch.tensor(
            [15, PAGE_SIZE + 15], dtype=torch.int32, device="npu"
        )
        n_prior = POOL_SIZE - next_n
    elif case_id == "completion_mid_batch":
        B, next_n, blocks = 3, 4, [0, 1, 2]
        pos = torch.tensor([[13, 14, 15, 16]] * B, dtype=torch.int32, device="npu")
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32, device="npu")
        slot_map[:, 2] = torch.tensor(
            [15, PAGE_SIZE + 15, 2 * PAGE_SIZE + 15], dtype=torch.int32, device="npu"
        )
        n_prior = 13
    elif case_id == "non_uniform_padding":
        B, next_n, blocks = 2, 4, [0, 1]
        pos = torch.tensor(
            [[12, 13, 14, 15], [12, 13, -1, -1]], dtype=torch.int32, device="npu"
        )
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32, device="npu")
        slot_map[0, 3] = 15
        n_prior = POOL_SIZE - 4
    else:  # plain_decode
        B, next_n, blocks = 4, 1, [0, 1, 2, 3]
        pos = torch.tensor([[5], [6], [7], [8]], dtype=torch.int32, device="npu")
        slot_map = torch.full((B, next_n), -1, dtype=torch.int32, device="npu")
        n_prior = 0

    if case_id == "non_uniform_padding":
        safe_pos = torch.where(pos >= 0, pos, 0)
        tail_slot = torch.where(pos >= 0, _tail_slot_for(blocks, safe_pos), 0)
    else:
        tail_slot = _tail_slot_for(blocks, pos)

    key = torch.randn(B, next_n, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    score = torch.randn(B, next_n, HEAD_DIM, dtype=torch.bfloat16, device="npu")
    ape = torch.randn(POOL_SIZE, HEAD_DIM, dtype=torch.float32, device="npu")

    kv, tail = _make_caches()
    _seed_prior(tail, blocks, n_prior)
    r_ref = _torch_reference(
        kv, tail, tail_slot, key, score, ape, slot_map, pos, mx_scale=mx_scale
    )
    r_kern = _run_kernel(
        kv, tail, tail_slot, key, score, ape, slot_map, pos, mx_scale=mx_scale
    )
    _assert_eq(r_ref, r_kern)


@pytest.mark.parametrize("seed", list(range(20)))
def test_batched_matches_reference_fuzz(seed):
    """Random B / next_n / start positions; covers 0, 1, and multi completion."""
    g = torch.Generator(device="npu").manual_seed(seed)
    B = int(torch.randint(1, 6, (1,), generator=g, device="npu").item())
    next_n = int(torch.randint(1, 8, (1,), generator=g, device="npu").item())
    blocks = list(range(B))

    starts = torch.randint(0, 33, (B,), generator=g, device="npu", dtype=torch.int32)
    pos = starts.unsqueeze(1) + torch.arange(
        next_n, device="npu", dtype=torch.int32
    ).unsqueeze(0)
    tail_slot = _tail_slot_for(blocks, pos)

    is_completion = pos % POOL_SIZE == POOL_SIZE - 1
    blk = torch.tensor(blocks, device="npu", dtype=torch.int32).unsqueeze(1)
    pool_slot = blk * PAGE_SIZE + (POOL_SIZE - 1)
    slot_map = torch.where(is_completion, pool_slot, torch.full_like(pos, -1))

    key = torch.randn(
        B, next_n, HEAD_DIM, dtype=torch.bfloat16, device="npu", generator=g
    )
    score = torch.randn(
        B, next_n, HEAD_DIM, dtype=torch.bfloat16, device="npu", generator=g
    )
    ape = torch.randn(
        POOL_SIZE, HEAD_DIM, dtype=torch.float32, device="npu", generator=g
    )

    kv, tail = _make_caches()
    prior_g = torch.Generator(device="npu").manual_seed(seed + 1000)
    for b in range(B):
        n_prior = int(starts[b].item()) % POOL_SIZE
        if n_prior > 0:
            pk = torch.randn(
                n_prior,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device="npu",
                generator=prior_g,
            )
            ps = torch.randn(
                n_prior,
                HEAD_DIM,
                dtype=torch.bfloat16,
                device="npu",
                generator=prior_g,
            )
            tail[blocks[b], 0, :n_prior, :] = pk
            tail[blocks[b], 1, :n_prior, :] = ps

    r_ref = _torch_reference(kv, tail, tail_slot, key, score, ape, slot_map, pos)
    r_kern = _run_kernel(kv, tail, tail_slot, key, score, ape, slot_map, pos)
    # Random inputs can land the pooled value on a bf16 rounding midpoint,
    # where the kernel and the CPU reference may round apart by one fp8 step.
    _assert_eq(r_ref, r_kern, k_byte_tol=1)
