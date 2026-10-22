# pool_kernels

Standalone extracts of three GLM-5.3-Flash sparse-indexer Triton kernels from
vLLM (`vllm/models/glm5next/nvidia/ops/kpool_compress.py`), with their unit
tests. Every file depends only on `torch` and `triton` — no vLLM imports.
Target device: **Ascend NPU** (`device="npu"`, `torch_npu`); the kernel
implementations themselves are unchanged from the NVIDIA-path source.

Each operator lives in its own subdirectory together with its tests:

| Kernel | Directory |
| --- | --- |
| `expand_pools_and_append_tail` | `expand_pools_and_append_tail/` |
| `kpool_seed_tail_cache` | `kpool_seed_tail_cache/` |
| `kpool_decode_update_and_maybe_write_cache_batched` | `kpool_decode_update_and_maybe_write_cache_batched/` |

Differences vs. the in-tree kernels:

- `from vllm.triton_utils import tl, triton` is replaced with plain
  `triton` / `triton.language` imports.
- `pack_ue8m0_x4` is defined locally (was `vllm.v1.attention.ops.common`).
- The decode kernel takes an explicit `mx_scale: bool = False` argument
  instead of calling `vllm.utils.deep_gemm.mqa_logits_use_mx_scales()`
  (True on SM100); the extracted kernels keep the contiguous
  (non-preshuffled) cache layout of the NVIDIA path.
- The decode tests' writer-consistency cases compare against a pure-torch
  reference of the prefill writer `kpool_compress_and_write_cache`, which is
  not extracted.

Run (NPU device required; needs `torch` + `torch_npu`):

```
pytest pool_kernels/          # all three operators
pytest pool_kernels/<op>/     # a single operator
```
