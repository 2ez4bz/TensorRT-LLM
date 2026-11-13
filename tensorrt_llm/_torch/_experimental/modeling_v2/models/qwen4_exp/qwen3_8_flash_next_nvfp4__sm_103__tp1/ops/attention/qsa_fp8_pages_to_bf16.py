# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# bridging_gaps: FP8 analogue of qsa_nvfp4_pages_to_bf16 (same table/offset semantics; e4m3 -> bf16 is exact).
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8_pages_to_bf16_kernel(
    k_cache,
    v_cache,
    block_table,
    slots,
    counts,
    offsets,
    table,
    k_out,
    v_out,
    num_pages,
    listed,
    s0,
    s1,
    s2,
    bt_stride,
    tb_stride,
    o0,
    o1,
    o2,
    HEADS: tl.constexpr,
    TPB: tl.constexpr,
    D: tl.constexpr,
    BLOCK: tl.constexpr,
):
    r = tl.program_id(0)
    j = tl.program_id(1)
    cnt = tl.load(counts + r)
    base = tl.load(offsets + r)
    dst = base + j
    if (j < cnt) & (dst < listed):
        slot = tl.load(slots + r).to(tl.int64)
        src = tl.load(block_table + slot * bt_stride + j).to(tl.int64)
        resident = (src >= 0) & (src < num_pages)
        tl.store(
            table + slot * tb_stride + j, tl.where(resident, dst, -1).to(table.dtype.element_ty)
        )
        src_safe = tl.where(resident, src, 0)
        dst64 = dst.to(tl.int64)
        for off in tl.range(0, HEADS * TPB * D, BLOCK):
            idx = off + tl.arange(0, BLOCK)
            h = idx // (TPB * D)
            t = (idx // D) % TPB
            d = idx % D
            src_off = src_safe * s0 + h * s1 + t * s2 + d
            dst_off = dst64 * o0 + h * o1 + t * o2 + d
            k = tl.load(k_cache + src_off).to(tl.float32)
            v = tl.load(v_cache + src_off).to(tl.float32)
            k = tl.where(resident, k, 0.0)
            v = tl.where(resident, v, 0.0)
            tl.store(k_out + dst_off, k.to(tl.bfloat16))
            tl.store(v_out + dst_off, v.to(tl.bfloat16))


def qsa_fp8_pages_to_bf16(
    k_cache, v_cache, block_table, slots, counts, offsets, table, k_out, v_out
):
    """k/v_cache [pages, Hkv, tpb, D] float8_e4m3fn (read at face value, as the per-row kernel does);
    k/v_out [listed, Hkv, tpb, D] bf16 contiguous; table int tensor of block_table's shape (all -1 before)."""
    requests, listed = slots.shape[0], k_out.shape[0]
    if requests and listed:
        assert (
            k_cache.stride(3) == 1
            and v_cache.stride() == k_cache.stride()
            and k_out.is_contiguous()
        )
        _fp8_pages_to_bf16_kernel[(requests, block_table.shape[1])](
            k_cache,
            v_cache,
            block_table,
            slots,
            counts,
            offsets,
            table,
            k_out,
            v_out,
            k_cache.shape[0],
            listed,
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            block_table.stride(0),
            table.stride(0),
            k_out.stride(0),
            k_out.stride(1),
            k_out.stride(2),
            HEADS=k_cache.shape[1],
            TPB=k_cache.shape[2],
            D=k_cache.shape[3],
            BLOCK=2048,
            num_warps=8,
        )
    return k_out, v_out, table


def reference(k_cache, v_cache, block_table, slots, counts, offsets, table, k_out, v_out):
    k_ref, v_ref, t_ref = k_out.clone(), v_out.clone(), table.clone()
    for r in range(slots.shape[0]):
        slot, base = int(slots[r]), int(offsets[r])
        for j in range(min(int(counts[r]), block_table.shape[1])):
            dst = base + j
            if dst >= k_out.shape[0]:
                break
            src = int(block_table[slot, j])
            res = 0 <= src < k_cache.shape[0]
            t_ref[slot, j] = dst if res else -1
            k_ref[dst] = k_cache[src].to(torch.bfloat16) if res else 0
            v_ref[dst] = v_cache[src].to(torch.bfloat16) if res else 0
    return k_ref, v_ref, t_ref
