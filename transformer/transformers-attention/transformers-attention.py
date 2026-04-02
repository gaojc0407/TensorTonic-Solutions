import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    batch_size,seq_len,dim = Q.shape
    scores = torch.bmm(Q,K.transpose(1,2))
    scores = scores/torch.sqrt(torch.tensor(dim))
    scores = F.softmax(scores,dim=-1)
    output = torch.bmm(scores,V)
    return output