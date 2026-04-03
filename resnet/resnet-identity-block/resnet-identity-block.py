import numpy as np

def relu(x):
    return np.maximum(0, x)

class IdentityBlock:
    """
    Identity Block: F(x) + x
    Used when input and output dimensions match.
    """
    def __init__(self, channels: int):
        self.channels = channels
        # Simplified: using dense layers instead of conv for demo
        self.W1 = np.random.randn(channels, channels) * 0.01
        self.W2 = np.random.randn(channels, channels) * 0.01
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass: y = ReLU(W2 @ ReLU(W1 @ x)) + x
        """
        # 保存原始输入
        identity = x
        
        # 获取输入形状
        shape = x.shape
        
        # 根据题目示例，输入应该是4D (batch, channels, height, width)
        # 但为了安全，我们检查形状
        if len(shape) == 4:
            batch, channels, h, w = shape
            # 重塑为 (batch, channels, h*w)
            x_reshaped = x.reshape(batch, channels, -1)
            
            # 应用变换
            # 注意：W1 和 W2 的形状是 (channels, channels)
            # 我们需要在channels维度上进行矩阵乘法
            out1 = np.einsum('bci,ij->bcj', x_reshaped, self.W1)
            out1_relu = relu(out1)
            
            out2 = np.einsum('bci,ij->bcj', out1_relu, self.W2)
            out2_relu = relu(out2)
            
            # 恢复形状
            output = out2_relu.reshape(batch, channels, h, w) + identity
        else:
            # 如果输入不是4D，使用更通用的方法
            # 假设最后一个维度是channels
            out1 = x @ self.W1
            out1_relu = relu(out1)
            out2 = out1_relu @ self.W2
            out2_relu = relu(out2)
            output = out2_relu + identity
        
        return output

