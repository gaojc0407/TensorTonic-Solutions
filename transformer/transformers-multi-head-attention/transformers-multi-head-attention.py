import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    batch, seq_len, dim = Q.shape
    dhead = dim // num_heads
    Q, K, V = np.matmul(Q,W_q),np.matmul(K,W_k),np.matmul(V,W_v)
    Q = Q.reshape(batch,seq_len,num_heads,dhead).transpose(0,2,1,3)
    K = K.reshape(batch,seq_len,num_heads,dhead).transpose(0,2,1,3)
    V = V.reshape(batch,seq_len,num_heads,dhead).transpose(0,2,1,3)
    scores = np.matmul(Q,K.transpose(0,1,3,2))/np.sqrt(dhead)
    scores = softmax(scores)
    output = np.matmul(scores,V).transpose(0,2,1,3).reshape(batch,seq_len,-1)
    return output