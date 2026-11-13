#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
MODEL_PATH=${1:?Usage: bash qwen38-repro/serve.sh MODEL_PATH [serve options...]}
shift
export TRTLLM_QWEN4_EXP_PLE_HOST_OFFLOAD=0
export TRTLLM_MODELING_V2_PLE_HOST_OFFLOAD=0
export TRTLLM_MV2_FP8_ROWTILE=1
export TRTLLM_GDN_FUSED_CONV=1
export TLLM_SPEC_DECODE_FORCE_NUM_ACCEPTED_TOKENS=1.635858
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TLLM_LOG_LEVEL=INFO
exec python3 -m tensorrt_llm.commands.serve "$MODEL_PATH" \
    --host 0.0.0.0 --port 8000 --reasoning_parser qwen3_5 --tool_parser qwen3 \
    --config "$(dirname "$0")/llm_args.yaml" "$@"
