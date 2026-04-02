import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    
    Args:
        seq_length: Length of the sequence
        d_model: Dimensionality of the model
    
    Returns:
        Positional encoding matrix of shape (seq_length, d_model)
    """
    # 1. 生成位置索引 [seq_length, 1]
    # reshape 为列向量是为了利用广播机制与行向量计算
    position = np.arange(seq_length).reshape(-1, 1) 
    
    # 2. 生成维度索引 [d_model]
    # 这里使用了 log 变换来加速计算并保证数值稳定性
    # np.arange(0, d_model, 2) 生成 0, 2, 4, ... 等偶数索引
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    # 3. 初始化结果矩阵
    pe = np.zeros((seq_length, d_model))
    
    # 4. 填充偶数维度 (Sine)
    # np.sin(position * div_term) 的形状会自动广播为 (seq_length, d_model/2)
    pe[:, 0::2] = np.sin(position * div_term) # 0::2 表示从索引0开始每隔2个取值
    
    # 5. 填充奇数维度 (Cosine)
    pe[:, 1::2] = np.cos(position * div_term) # 1::2 表示从索引1开始每隔2个取值
    
    return pe