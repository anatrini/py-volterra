import time
import numpy as np
from volterra import VolterraKernel2, VolterraProcessor2, LowRankEngine, DirectNumpyEngine


def benchmark_engine(engine, N=512, signal_len=48000):
    h1 = np.random.randn(N) * 0.01
    h2 = np.random.randn(N, N) * 0.01
    h2 = 0.5 * (h2 + h2.T)
    kernel = VolterraKernel2(h1, h2)
    
    proc = VolterraProcessor2(kernel, engine=engine)
    x = np.random.randn(signal_len)
    
    start = time.perf_counter()
    y = proc.process(x, block_size=512)
    elapsed = time.perf_counter() - start
    
    rt_factor = (signal_len / 48000) / elapsed
    return rt_factor


print("Dense engine:", benchmark_engine(DirectNumpyEngine()), "× real-time")
print("Low-rank engine:", benchmark_engine(LowRankEngine(energy=0.999)), "× real-time")