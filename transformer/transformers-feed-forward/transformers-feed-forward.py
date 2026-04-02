import numpy as np

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Apply position-wise feed-forward network.
    """
    # Your code here

    out = np.matmul(x,W1)+b1
    mask = out > 0
    out = np.where(mask,out,0)
    out = np.matmul(out,W2)+b2
    return out