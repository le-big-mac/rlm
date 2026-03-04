# RLM Docker Setup

Run RLM with local vLLM-served models (Qwen3.5-27B, GLM-4.7) on GPU machines.

## Prerequisites

- NVIDIA GPUs (tested on 4x H200 NVL)
- Docker with NVIDIA Container Toolkit
- HuggingFace token (for gated model access / faster downloads)

## Quick Start

```bash
# 1. Create .env with your HF token
echo "HF_TOKEN=hf_your_token_here" > .env

# 2. Build the image
docker compose build

# 3. Start the Qwen model server (GPUs 0,1)
docker compose up qwen

# 4. In another terminal, run an RLM example
docker compose run --rm rlm python3 examples/qwen_vllm.py 0

# 5. Start the visualizer to view traces
docker compose up visualizer
# Access at http://localhost:3000
```

## Services

### `qwen` — Qwen3.5-27B-FP8 on GPUs 0,1

```bash
docker compose up qwen       # Start serving
docker compose down qwen     # Stop
```

- Port: 8000
- Tensor parallel across 2 GPUs
- 262144 max context length (native max)
- FP8 KV cache + prefix caching enabled

### `glm` — GLM-4.7-FP8 on all 4 GPUs

```bash
docker compose up glm
```

- Port: 8000 (cannot run simultaneously with qwen)
- Tensor parallel across 4 GPUs
- Speculative decoding (MTP) enabled

### `visualizer` — Next.js trajectory viewer

```bash
docker compose up visualizer
```

- Port: 3000
- Automatically picks up `.jsonl` log files from `./logs/`
- Use SSH port forwarding to access remotely: `ssh -L 3000:localhost:3000 user@host`

### `rlm` — Interactive shell for running RLM scripts

```bash
# Interactive shell
docker compose run --rm rlm bash

# Run a specific example
docker compose run --rm rlm python3 examples/qwen_vllm.py 0

# List available questions
docker compose run --rm rlm python3 examples/qwen_vllm.py --list
```

## Examples

### `examples/qwen_vllm.py`

Runs RLM on math benchmark questions from `data/math/easy_all.json`.

```bash
# Run question at index 0
docker compose run --rm rlm python3 examples/qwen_vllm.py 0

# Run question at index 5
docker compose run --rm rlm python3 examples/qwen_vllm.py 5

# List all questions
docker compose run --rm rlm python3 examples/qwen_vllm.py --list
```

Logs are written to `./logs/` and appear in the visualizer automatically.

## GPU Management

### Hold unused GPUs

To reserve GPUs so others can't use them:

```bash
docker run -d --name hold-gpus --gpus '"device=2,3"' \
  --entrypoint python3 rlm-vllm -c \
  "import torch; [torch.empty(int(135e9/4), dtype=torch.uint8, device=f'cuda:{i}') for i in range(torch.cuda.device_count())]; import time; time.sleep(1e9)"
```

Release them:

```bash
docker stop hold-gpus && docker rm hold-gpus
```

## Volume Mounts

| Host Path | Container Path | Purpose |
|-----------|---------------|---------|
| `/data` | `/data` | Model weights (`HF_HOME=/data/huggingface`) |
| `./logs` | `/app/logs` | RLM trajectory logs |
| `./examples` | `/app/examples` | Example scripts (live-mounted) |
| `./data` | `/app/data` | Math benchmark data |

## Useful Commands

```bash
# Check GPU usage
nvidia-smi

# View model server logs
docker compose logs -f qwen

# Check if model is ready
curl http://localhost:8000/health

# Check throughput metrics
curl -s http://localhost:8000/metrics | grep "vllm:num_requests"

# Stop everything
docker compose down
```
