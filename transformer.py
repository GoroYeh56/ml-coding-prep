"""Notes:
1. Post-LN v.s. Pre-LN
    # Post-LN (Original Transformer)
    X = self.ln1(X + self.dropout(self.mhsa(X, mask)))

    # Pre-LN (Modern Standard / LLaMA / GPT-2+)
    X = X + self.dropout(self.mhsa(self.ln1(X), mask))
"""
from mhsa import MultiHeadSelfAttn
import torch
import torch.nn as nn

# LayerNorm Block
class LayerNorm(nn.Module):
    def __init__(self, embed_dim, eps=1e-4):
        super().__init__()
        """
            Use nn.Parameter (no 's')
            gamma (scale) -> 1s, beta (shift) -> 0s
        """
        # Initialize learnable parameters:
        self.gamma = nn.Parameter(torch.ones(embed_dim))
        self.beta = nn.Parameter(torch.zeros(embed_dim))
        self.eps = eps # Avoid division by zero.

    def forward(self, X:torch.Tensor)->torch.Tensor:
        """Normalize the last dimension (embed_dim)
        Args:
            X: (B, L, embed_dim)
        Operation:
            (X - Xmean)/ (Xvar + eps) **0.5
            Use torch.var(X, unbiased=False) to remove the Bessel's correction
                (Use population variance: dividing by N)
        Return: 
            Xnorm: (B, L, embed_dim)
        """
        B, L, embed_dim = X.shape
        Xmean = torch.mean(X, dim=-1, keepdim=True) # (B, L, 1)
        Xvar = torch.var(X, unbiased=False, dim=-1, keepdim=True)   # (B, L, 1)
        Xnorm = (X - Xmean) / torch.sqrt(Xvar+self.eps) # (B, L, embed_dim)
        return self.gamma * Xnorm + self.beta

# --- Unit Test LayerNorm ---
if __name__ == "__main__":
    B, L, embed_dim = 2, 3, 16
    X = torch.randn(B, L, embed_dim)

    custom_ln = LayerNorm(embed_dim, eps=1e-5)
    official_ln = nn.LayerNorm(embed_dim, eps=1e-5)

    diff = torch.max(torch.abs(custom_ln(X) - official_ln(X)))
    print(f"Max difference from official nn.LayerNorm: {diff.item():.8e}")
    assert diff < 1e-6, "Implementation mismatch!"
    print("LayerNorm validation passed!")

class Encoder(nn.Module):
    def __init__(self, num_heads, embed_dim, feedforward_dim, dropout=0.3):
        super().__init__()
        self.ln1 = LayerNorm(embed_dim)
        self.mhsa = MultiHeadSelfAttn(num_heads, embed_dim)
        self.ln2 = LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, mask=None):
        """Use Pre-LN with dropout
        Args:
            X: Input tensor (B, L, embed_dim)
            mask: optional mask (L, L)
        """
        # 1. Multi-Head Self-Attention branch with Pre-LN & Residual
        X = X + self.dropout(self.mhsa(self.ln1(X), mask=mask))
        # 2. Feed-Forward branch with Pre-LN & Residual
        X = X + self.dropout(self.ff(self.ln2(X)))
        return X

    def forward_postln(self, X, mask=None):
        """Use Post-LN with dropout
        Args:
            X: Input tensor (B, L, embed_dim)
            mask: optional mask (L, L)
        """
        X = self.ln1(X + self.dropout(self.mhsa(X, mask)))
        X = self.ln2(X + self.dropout(self.ff(X)))
        return X


# --- Unit Test ---
if __name__ == "__main__":
    B, L, embed_dim, num_heads, feedforward_dim = 2, 4, 16, 8, 64
    X = torch.randn(B, L, embed_dim)
    
    encoder_layer = Encoder(num_heads, embed_dim, feedforward_dim)
    # Test1: no mask
    out = encoder_layer(X)
    assert out.shape == (B, L, embed_dim)
    print(f"Output shape: {out.shape}")  # (2, 4, 16)

    # Test2: with mask  
    mask = torch.full((L,L), float('-inf'))
    mask = torch.triu(mask, diagonal=1)
    out_masked = encoder_layer(X, mask=mask)
    print("Encoder Layer forward pass successful!")
    