"""Hold GPUs 1, 2, 3 by allocating a small tensor on each. Ctrl+C to release."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

import torch
import signal
import sys

tensors = []
for i in range(3):
    t = torch.zeros(1, device=f"cuda:{i}")
    tensors.append(t)
    print(f"Holding GPU {i + 1} (logical cuda:{i})")

print("GPUs held. Press Ctrl+C to release.")
signal.pause()
