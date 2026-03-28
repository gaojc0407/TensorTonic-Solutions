import numpy as np

def sigmoid(x):
    """
    向量化Sigmoid函数。
    """
    # 1. 将输入转换为NumPy数组，并确保数据类型为float
    x = np.asarray(x, dtype=float)
    
    # 2. 计算Sigmoid函数
    # 使用 np.exp 计算 e^(-x)，并代入公式
    s = 1 / (1 + np.exp(-x))
    
    return s