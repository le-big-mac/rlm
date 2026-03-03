"""
Example: Using RLM with a local Qwen3.5-27B-FP8 model served via vLLM.

Prerequisites:
  1. Start the vLLM server:
       conda activate rlm-qwen
       ./serve_qwen.sh        # GPU 0, port 8000

  2. Run this script:
       conda activate rlm-qwen
       python examples/qwen_vllm.py
"""

from rlm import RLM
from rlm.logger import RLMLogger

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "qwen3.5-27b"  # must match --served-model-name in serve_qwen.sh

logger = RLMLogger(log_dir="./logs")

rlm = RLM(
    backend="vllm",
    backend_kwargs={
        "model_name": MODEL_NAME,
        "base_url": VLLM_BASE_URL,
        "api_key": "not-used",  # vLLM doesn't require an API key
    },
    environment="local",
    environment_kwargs={},
    max_depth=1,
    logger=logger,
    verbose=True,
)

result = rlm.completion("Print me the first 5 powers of two, each on a newline.")
print(result)
