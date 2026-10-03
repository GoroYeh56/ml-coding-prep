import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model=512, max_len=1000):
        super().__init__()
        assert d_model % 2 == 0, "d_model must be an even number"
        # Initialize an array of sequence positions (ks)
        positions = torch.arange(max_len, dtype=torch.float32).reshape(-1,1) # (max_len, 1)

        # Compute the scaling terms (denominators: n^(2*i/d))
        # Compute it in log space for numerical stability. 
        # 2i: [0, 2, 4, ...]
        dims = torch.arange(0,d_model,2,dtype=torch.float32)
        denoms = torch.exp( dims/d_model * -math.log(10000))

        # Initialize the positional encoding matrix (S, d_model)
        PE = torch.zeros(max_len, d_model)

        # Compute the PE matrix:
        PE[:,0::2] = torch.sin(positions * denoms)
        PE[:,1::2] = torch.cos(positions * denoms)

        # Register as a buffer so it's saved with model state dict 
        # but NOT treated as learnable parameters
        self.register_buffer('PE', PE.reshape(1,max_len,d_model))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """X: input embedding (B, S, d_model)"""
        B, S, d_model = X.shape
        return X + self.PE[:, :S,:] # Slice PE to input X's sequence length


# Test
max_len=100
d_model=512
pe = PositionalEncoding(d_model, max_len)
X = torch.rand(2, 20, d_model)
print(f"Apply PE: {pe.forward(X).shape}")
