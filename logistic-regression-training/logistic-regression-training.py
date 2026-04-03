import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # 获取数据维度
    N, D = X.shape
    
    # 初始化权重和偏置
    w = np.zeros(D)  # 权重向量，形状(D,)
    b = 0.0          # 偏置项
    
    # 梯度下降训练循环
    for step in range(steps):
        # 前向传播：计算预测概率
        z = X @ w + b  # Xw + b
        p = _sigmoid(z)  # σ(Xw + b)
        
        # 计算损失（可选，用于监控训练过程）
        # 二元交叉熵损失
        # loss = -np.mean(y * np.log(p + 1e-15) + (1 - y) * np.log(1 - p + 1e-15))
        
        # 反向传播：计算梯度
        # 梯度公式：∇w = X^T(p - y)/N, ∇b = mean(p - y)
        error = p - y
        dw = X.T @ error / N  # 权重梯度
        db = np.mean(error)   # 偏置梯度
        
        # 参数更新
        w = w - lr * dw
        b = b - lr * db
    
    return w, b
