# Dockerfile for running rlm with vLLM-served Qwen3.5 and GLM-4.7 models
# on H200 GPUs with FP8 support.
#
# Build:
#   docker build -t rlm-vllm .
#
# Run vLLM server (Qwen3.5-27B-FP8, single GPU):
#   docker run --gpus '"device=0"' -p 8000:8000 rlm-vllm \
#     vllm serve Qwen/Qwen3.5-27B-FP8 \
#       --host 0.0.0.0 --port 8000 \
#       --tensor-parallel-size 1 \
#       --max-model-len 8192 \
#       --gpu-memory-utilization 0.90 \
#       --dtype auto --enforce-eager --trust-remote-code \
#       --served-model-name qwen3.5-27b
#
# Run vLLM server (GLM-4.7-FP8, 4 GPUs):
#   docker run --gpus all -p 8000:8000 rlm-vllm \
#     vllm serve zai-org/GLM-4.7-FP8 \
#       --host 0.0.0.0 --port 8000 \
#       --tensor-parallel-size 4 \
#       --max-model-len 8192 \
#       --gpu-memory-utilization 0.90 \
#       --dtype auto --trust-remote-code \
#       --tool-call-parser glm47 \
#       --reasoning-parser glm45 \
#       --enable-auto-tool-choice \
#       --speculative-config.method mtp \
#       --speculative-config.num_speculative_tokens 1 \
#       --served-model-name glm-4.7
#
# Run rlm scripts (client side):
#   docker run --gpus none --network host rlm-vllm \
#     python examples/qwen_vllm.py

FROM vllm/vllm-openai:nightly

# Store model weights on the large /data disk
ENV HF_HOME=/data/huggingface

# The base image doesn't include git, which we need to install
# transformers from source.
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Install transformers from source — required for both Qwen3.5 and
# GLM-4.7 (glm4_moe_lite arch is not yet in a stable release).
RUN pip install --upgrade huggingface_hub \
    && pip install --force-reinstall --no-deps \
    git+https://github.com/huggingface/transformers.git@main

# Install the rlm package
COPY pyproject.toml README.md LICENSE MANIFEST.IN /app/
COPY rlm/ /app/rlm/
COPY examples/ /app/examples/
WORKDIR /app
RUN pip install .

# Default: drop into a shell so the user can choose what to run
CMD ["/bin/bash"]
