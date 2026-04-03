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

        if x.ndim == 2:

            fx = x @ self.W1      
            fx = relu(fx)         
            fx = fx @ self.W2     
            
            # 跳跃连接: F(x) + x
            # 激活输出: ReLU(F(x) + x)
            return relu(fx)+x
        
        # 情况 2: 四维输入 (Batch, Channels, Height, Width) - 典型的图像数据
        elif x.ndim == 4:
            bs, ch, h, w = x.shape

            x_flat = x.transpose(0, 2, 3, 1).reshape(-1, ch)
            

            fx = x_flat @ self.W1
            fx = relu(fx)
            fx = fx @ self.W2
            
            y = relu(fx)+x_flat
            
            # 将结果恢复为原始的四维形状
            # (B*H*W, C) -> (B, H, W, C) -> (B, C, H, W)
            y = y.reshape(bs, h, w, ch).transpose(0, 3, 1, 2)
            return y
        
        else:
            raise ValueError(f"Unsupported input shape: {x.shape}")
