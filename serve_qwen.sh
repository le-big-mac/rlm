#!/bin/bash
# Serve Qwen3.5-27B-FP8 via vLLM on a single A40 GPU
#
# Usage:
#   ./serve_qwen.sh          # uses GPU 0, port 8000
#   ./serve_qwen.sh 2 8001   # uses GPU 2, port 8001
#
# Prerequisites:
#   conda activate rlm-qwen

set -euo pipefail

GPU_ID="${1:-0}"
PORT="${2:-8000}"
MODEL="Qwen/Qwen3.5-27B-FP8"

echo "Starting vLLM server on GPU ${GPU_ID}, port ${PORT}"
echo "Model: ${MODEL}"

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --host 0.0.0.0 \
    --port "${PORT}" \
    --tensor-parallel-size 1 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.90 \
    --dtype auto \
    --enforce-eager \
    --trust-remote-code \
    --disable-log-stats \
    --served-model-name qwen3.5-27b
