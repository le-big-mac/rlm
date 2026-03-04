"""
Example: Using RLM with a local Qwen3.5-27B-FP8 model served via vLLM.

Usage:
  python examples/qwen_vllm.py                  # first question (index 0)
  python examples/qwen_vllm.py 5                # question at index 5
  python examples/qwen_vllm.py --list           # list all question IDs
"""

import argparse
import json
import sys

from rlm import RLM
from rlm.logger import RLMLogger

VLLM_BASE_URL = "http://localhost:8000/v1"
MODEL_NAME = "qwen3.5-27b"
DATA_PATH = "data/math/easy_all.json"


def load_question(index: int) -> dict:
    with open(DATA_PATH) as f:
        questions = json.load(f)["questions"]
    if index < 0 or index >= len(questions):
        print(f"Error: index {index} out of range (0-{len(questions) - 1})")
        sys.exit(1)
    return questions[index]


def list_questions():
    with open(DATA_PATH) as f:
        questions = json.load(f)["questions"]
    for i, q in enumerate(questions):
        preview = q["prompt"][:80].replace("\n", " ")
        print(f"[{i}] id={q['question_id']}: {preview}...")


def main():
    parser = argparse.ArgumentParser(description="Run RLM on a math question")
    parser.add_argument("index", nargs="?", type=int, default=0,
                        help="Question index in easy.json (default: 0)")
    parser.add_argument("--list", action="store_true",
                        help="List all available questions")
    args = parser.parse_args()

    if args.list:
        list_questions()
        return

    question = load_question(args.index)
    print(f"Question ID: {question['question_id']}")
    print(f"Expected answer: {question['answer']}")
    print()

    logger = RLMLogger(log_dir="./logs")

    rlm = RLM(
        backend="vllm",
        backend_kwargs={
            "model_name": MODEL_NAME,
            "base_url": VLLM_BASE_URL,
            "api_key": "not-used",
        },
        environment="local",
        environment_kwargs={},
        max_depth=2,
        logger=logger,
        verbose=True,
    )

    result = rlm.completion(question["prompt"], root_prompt="You are an orchestrator model. Your only job is to plan how to approach the problem, and delegate via subcalls. Code you write should only be for interacting with the prompt, planning, and delegation. Whenver you feel it is necessary to run code for another purpose, delegate it to a sub rlm_query. Do not overfill your own context window.")
    print(result)


if __name__ == "__main__":
    main()
