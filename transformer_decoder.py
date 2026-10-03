import torch
import torch.nn as nn
from mhsa import scaled_dot_product_attention, MultiHeadSelfAttn
from transformer import LayerNorm

class MultiHeadCrossAttention(nn.Module):
    """Multi-Head Cross-Attention where Query comes from Decoder (tgt)

    and Key/Value come from Encoder (memory).
    """

    def __init__(self, num_heads, embed_dim):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads

        # Separate projections since Q comes from tgt, K & V come from memory
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.kv_linear = nn.Linear(embed_dim, 2 * embed_dim)  # Fused K and V
        self.linear = nn.Linear(embed_dim, embed_dim)  # Output projection

    def forward(self, x, memory, mask=None):
        """Args:

            x: Target tensor (B, L_tgt, embed_dim)
            memory: Encoder output tensor (B, L_src, embed_dim)
            mask: Optional cross-attention mask (B, 1, L_tgt, L_src) or
            (L_tgt, L_src)
        Returns:
            out: (B, L_tgt, embed_dim)
        """
        B, L_tgt, _ = x.shape
        _, L_src, _ = memory.shape

        # 1. Project Query from x, and Key/Value from memory
        q = self.q_linear(x)  # (B, L_tgt, embed_dim)
        kv = self.kv_linear(memory)  # (B, L_src, 2 * embed_dim)
        k, v = torch.chunk(kv, 2, dim=-1)

        # 2. Reshape into (B, num_heads, L, head_dim)
        q = q.reshape(B, L_tgt, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, L_src, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L_src, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Scaled Dot-Product Attention & Output Projection
        attn_out = scaled_dot_product_attention(q, k, v, mask=mask)
        out = attn_out.transpose(1, 2).reshape(B, L_tgt, self.embed_dim)
        return self.linear(out)


class DecoderLayer(nn.Module):
    """Pre-LN Transformer Decoder Layer."""

    def __init__(self, num_heads, embed_dim, feedforward_dim, dropout=0.3):
        super().__init__()
        # Sub-layer 1: Self-Attention
        self.ln1 = LayerNorm(embed_dim)
        self.self_attn = MultiHeadSelfAttn(num_heads, embed_dim)

        # Sub-layer 2: Cross-Attention
        self.ln2 = LayerNorm(embed_dim)
        self.cross_attn = MultiHeadCrossAttention(num_heads, embed_dim)

        # Sub-layer 3: Feed-Forward Network
        self.ln3 = LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embed_dim),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """Pre-LN Decoder Forward Pass.

        Args:
            tgt: Target sequence tensor (B, L_tgt, embed_dim)
            memory: Encoder output tensor (B, L_src, embed_dim)
            tgt_mask: Causal mask for self-attention (L_tgt, L_tgt)
            memory_mask: Optional mask for cross-attention (L_tgt, L_src)
        """
        # 1. Masked Self-Attention Branch (Pre-LN)
        tgt = tgt + self.dropout(self.self_attn(self.ln1(tgt), mask=tgt_mask))

        # 2. Encoder-Decoder Cross-Attention Branch (Pre-LN)
        tgt = tgt + self.dropout(
            self.cross_attn(self.ln2(tgt), memory, mask=memory_mask)
        )

        # 3. Feed-Forward Branch (Pre-LN)
        tgt = tgt + self.dropout(self.ff(self.ln3(tgt)))

        return tgt


# --- Verification & Unit Test ---
if __name__ == "__main__":
    B, L_tgt, L_src, embed_dim, num_heads, feedforward_dim = 2, 4, 6, 16, 8, 64

    tgt = torch.randn(B, L_tgt, embed_dim)  # Target input (Decoder)
    memory = torch.randn(B, L_src, embed_dim)  # Encoder output

    decoder = DecoderLayer(num_heads, embed_dim, feedforward_dim)

    # Causal mask for target self-attention
    tgt_mask = torch.full((L_tgt, L_tgt), float("-inf"))
    tgt_mask = torch.triu(tgt_mask, diagonal=1)

    out = decoder(tgt, memory, tgt_mask=tgt_mask)

    print(f"Decoder Output shape: {out.shape}")  # (B, L_tgt, embed_dim)
    assert out.shape == (B, L_tgt, embed_dim)
    print("Decoder Layer forward pass successful!")