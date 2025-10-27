import torch
import cupy as cp
import time

# --- Test PyTorch ---
print("PyTorch test:")
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
    # simple GPU tensor test
    x = torch.rand((5000, 5000), device="cuda")
    y = torch.mm(x, x)
    print("Matrix multiply done on:", y.device)
else:
    print("CUDA not detected.")

# --- Test cuPy ---
print("\ncuPy test:")
props = cp.cuda.runtime.getDeviceProperties(0)
print(f"Device: {props['name'].decode()}")

# simple GPU array test
a = cp.random.rand(5000, 5000)
b = cp.dot(a, a)
print("Matrix multiply done on GPU:", b.device)

# --- Timing Comparison ---
print("\nPerformance comparison (CPU vs GPU)")

import numpy as np
cpu_a = np.random.rand(2000, 2000)
cpu_b = np.random.rand(2000, 2000)

start = time.time()
np.dot(cpu_a, cpu_b)
print("CPU time:", time.time() - start, "seconds")

start = time.time()
cp.dot(a[:2000,:2000], a[:2000,:2000])
cp.cuda.Device(0).synchronize()
print("GPU time:", time.time() - start, "seconds")
