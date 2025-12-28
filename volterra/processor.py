from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter
from volterra.kernels import VolterraKernel2, ArrayF
from volterra.engines import Volterra2Engine, DirectNumpyEngine


@dataclass
class VolterraProcessor2:
    """
    Block-based processor with correct history management.
    Extensible to 3rd order: just add kernel3 + engine3 to the sum.
    """
    kernel: VolterraKernel2
    engine: Volterra2Engine = DirectNumpyEngine()
    sample_rate: int = 48000
    
    def __post_init__(self):
        self.reset()
    
    def reset(self):
        N = self.kernel.N
        self._x_hist = np.zeros(N - 1, dtype=np.float64)
        self._zi1 = np.zeros(N - 1, dtype=np.float64)  # lfilter state for h1
    
    def process_block(self, x_block: ArrayF) -> ArrayF:
        """
        Process one block with correct causal FIR handling.
        Council consensus: this state management is correct.
        """
        x_block = np.asarray(x_block, dtype=np.float64)
        N = self.kernel.N
        B = len(x_block)
        
        # Linear part (efficient streaming FIR)
        y1, self._zi1 = lfilter(self.kernel.h1, [1.0], x_block, zi=self._zi1)
        
        # Nonlinear 2nd-order part
        x_ext = np.concatenate([self._x_hist, x_block])  # (N-1+B,)
        y2 = self.engine.process_block(x_ext, N, B, self.kernel)
        
        # Update history for next block
        self._x_hist = x_ext[-(N-1):].copy()
        
        return y1 + y2
    
    def process(self, x: ArrayF, block_size: int = 512) -> ArrayF:
        """Offline processing of complete signal."""
        x = np.asarray(x, dtype=np.float64)
        y = np.empty_like(x)
        self.reset()
        
        for i in range(0, len(x), block_size):
            xb = x[i:i+block_size]
            y[i:i+len(xb)] = self.process_block(xb)
        
        return y