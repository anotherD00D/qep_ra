import cupy as cp, numpy as np, time

size = 10000
a = cp.random.rand(size, size)
b = cp.random.rand(size, size)

start = time.time()
cp.dot(a, b)
cp.cuda.Device(0).synchronize()
print("GPU time:", time.time() - start, "seconds")

a_cpu = np.random.rand(size, size)
b_cpu = np.random.rand(size, size)
start = time.time()
np.dot(a_cpu, b_cpu)
print("CPU time:", time.time() - start, "seconds")