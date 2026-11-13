<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
# Qwen3.8 Flash Next modeling_v2 reproduction

Base commit: `d782229575f93d9c5eca406e7d15562699d1f75f`.
Base image: `nvcr.io/nvidia/tensorrt-llm/release:1.4.0rc3.dev202611060007`,
ARM64 digest `sha256:df0605c87f10f3c7549abc76933f9bc901a489dbdf0ec1d6e0269ab763b69b14`.
Use a TensorRT-LLM installation built from this branch, including its Python sources.
Do not reapply the original startup overlays on top of these changes.

## Serve

Model: `nvidia/Qwen3.8-Flash-Next-NVFP4`, revision
`fc694b54fb0174e0913e6adf86691ef85a4ead47`.
One Rubin SM107 GPU, TP1. Keep persistent FlashInfer and Triton caches warm.
Set `TRITON_CACHE_DIR` to a writable persistent directory before launching.

```bash
bash qwen38-repro/serve.sh /path/to/model
```

The launcher and `llm_args.yaml` retain the measured launch settings, including
FP8 KV, BF16 recurrent state, MTP3, stream interval 4 and padded CUDA graph sizes.
The source changes include the 14-file baseline overlay plus `mv2_sm107`,
`mv2_rowtile`, and `mv2_combo`: SM107 routing, FP8 context scratch/row-tile
attention, Rubin CuTe MoE, and the corresponding FC1 weight layout.

## Recorded performance

Historical measurements, not new measurements of this branch:

| Variant | Requests | Output tok/s/GPU | P90 TTFT (s) | P50 output tok/s/user |
| --- | ---: | ---: | ---: | ---: |
| Full-length combo | 10,240 | 13,365.78 | 2.664 | 27.22 |
| Combo2 | 2,560 | 13,385.53 | 12.892 | 27.97 |
| Natural-acceptance qualification (combo) | 512 | 13,691.03 | 23.267 | 47.96 |

Full-length combo was +35.7% over its same-wave control (9,847 tok/s/GPU).
Combo2 adds the FC2 B-reuse tactic and finalize epilogue patch; its measured
+0.6% over the same-wave combo reference is within observed run variation.
The full-length combo and combo2 source changes are separate commits.

Performance workload: concurrency 512, exactly 8,192 input / 1,024 output tokens,
10 warmup requests followed by 10,240 measured requests (2,560 for combo2).
Streaming completions, greedy sampling, `ignore_eos: true`, server token counts,
client connection reuse disabled, direct Service IP. Use the same frozen request
dataset/tokenizer when comparing results. Short qualification runs and their
synchronized-start TTFT are not comparable to the full-length protocol.

The launcher forces accepted drafts to 1.635858, excluding the bonus token.
This is a calibrated synthetic performance proxy, not an accuracy measurement.
For natural acceptance, unset `TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS` and
launch the serve command directly with the remaining environment settings.
Full accuracy validation on SM107 remains outstanding.
