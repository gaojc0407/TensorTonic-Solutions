import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply layer normalization.
    """
    
    # Your code here
    miu = np.mean(x,axis=-1,keepdims=True)
    sigma2 = np.var(x,axis=-1,keepdims=True)
    out = gamma * (x-miu)/np.sqrt(sigma2+eps) + beta
    return out

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Multi-head attention.
    """
    # Your code here
    Q,K,V = np.matmul(Q,W_q),np.matmul(K,W_k),np.matmul(V,W_v)
    batch,seq_len,dim = Q.shape
    dhead = dim // num_heads
    Q = Q.reshape(batch,seq_len,num_heads,dhead).transpose(0,2,1,3)
    K = K.reshape(batch,seq_len,num_heads,dhead).transpose(0,2,1,3)
    V = V.reshape(batch,seq_len,num_heads,dhead).transpose(0,2,1,3)
    scores = np.matmul(Q,K.transpose(0,1,3,2)) / np.sqrt(dhead)
    scores = softmax(scores)
    out = np.matmul(scores,V).transpose(0,2,1,3).reshape(batch,seq_len,-1)
    return out

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Position-wise feed-forward network.
    """
    # Your code here
    x = np.matmul(x,W1)+b1
    mask = x > 0
    x = np.where(mask,x,0)
    return np.matmul(x,W2)+b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Complete encoder block: MHA + FFN with residuals and layer norms.
    """
    # Your code here

    attn = multi_head_attention(x,x,x,W_q,W_k,W_v,W_o,num_heads)
    x = x + attn
    x = layer_norm(x,gamma1,beta1)
    x = feed_forward(x,W1,b1,W2,b2) + x
    x = layer_norm(x,gamma2,beta2)
    return x 