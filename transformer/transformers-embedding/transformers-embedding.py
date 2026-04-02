import torch
import torch.nn as nn
import math

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embeding:
    """
    创建一个嵌入层。
    根据要求，权重初始化需要考虑缩放（通常PyTorch默认的正态分布已符合要求，
    但部分严格场景需手动设置，这里先按标准实现）。
    """
    embedding = nn.Embedding(vocab_size, d_model)
    # 注意：PyTorch默认使用均值0、标准差1的正态分布初始化，
    # 这通常符合Xavier/标准正态缩放的要求。
    return embedding

def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
    """
    将Token索引转换为缩放后的嵌入向量。
    
    步骤：
    1. 通过embedding层查找对应的向量 (Lookup)
    2. 将结果乘以 sqrt(d_model) (Scaling)
    """
    # 1. 查找嵌入向量
    embeddings = embedding(tokens) 
    
    # 2. 应用缩放因子 (关键步骤！)
    # 根据Transformer论文，乘以 sqrt(d_model)
    scaled_embeddings = embeddings * math.sqrt(d_model)
    
    return scaled_embeddings