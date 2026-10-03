"""
Implement Multihead Self Attention from scratch

1. Stable softmax: subtract the max logit
2. QKV layer: project together, then chunk

3. Follow-up:
LayerNorm.
Encoder layer (Pre-LN style)
"""

import torch 
import torch.nn as nn

def stable_softmax(logits:torch.Tensor, dim:int=-1, eps:float=1e-9)->torch.Tensor:
    """
    logits: (N, num_heads, S, head_dim)
    Robust Softmax handling NaN, +Inf, -Inf, and fully-masked logits.
    """
    cleaned_logits = torch.nan_to_num(logits, nan=-1e9, posinf=1e9, neginf=-1e9)
    max_logits, _ = torch.max(cleaned_logits, dim=-1, keepdim=True) # (N, num_heads, S, 1)
    exp_logits = torch.exp(cleaned_logits - max_logits) # (N, num_heads, S, head_dim)
    sum_logits = torch.sum(exp_logits, dim=-1, keepdim=True) # (N, num_heads, S, 1)
    return exp_logits / (sum_logits + eps) # Avoid division by zero. (N, num_heads, S, S)

def scaled_dot_product_attention(q, k, v, mask=None):
    """
        q, k, v: each shape (N, num_heads, S, head_dim)
        d_k = q.shape[-1] head_dim
        scaled = (q @ k.T) / sqrt(d_k) (N, num_heads, S, S)
        attn_score = stable_softmax(scaled+mask) @ v # (N, num_heads, S head_dim)
        return attn_score
    """
    d_k = q.shape[-1] # scalar
    scaled = (q @ k.transpose(-1, -2)) / d_k ** 0.5 
    if mask is not None:
        scaled = scaled + mask # Avoid gradients being overwrittent during backward
    attn_score = stable_softmax(scaled) @ v # (N, num_heads, S, head_dim)
    return attn_score


class MultiHeadSelfAttn(nn.Module):
    def __init__(self, num_heads, embed_dim):
        super().__init__()
        self.num_heads = num_heads
        assert embed_dim % num_heads == 0 , f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        self.head_dim = embed_dim//num_heads
        self.qkv_layer = nn.Linear(embed_dim, 3*embed_dim)
        self.out_proj_layer = nn.Linear(embed_dim, embed_dim)
    
    def forward(self, X, mask=None):
        """ 
        X: (N, S, D) batch size, sequence length, input_dim
        mask: (L, L) causal mask (upper triangular should be 0)
        Return:
        out: (N, S, D)
        """
        N, S, D = X.shape
        qkv = self.qkv_layer(X) # (N, S, 3*embed_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1) # (N, S, embed_dim)
        q = q.reshape(N, S, self.num_heads, self.head_dim).permute(0,2,1,3) # (N, num_heads, S, head_dim)
        k = k.reshape(N, S, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = v.reshape(N, S, self.num_heads, self.head_dim).permute(0,2,1,3)
        attention_score = scaled_dot_product_attention(q, k, v, mask) # (N, num_heads, S, head_dim)
        attention_score = attention_score.permute(0,2,1,3).reshape(N, S, -1)
        out = self.out_proj_layer(attention_score) # (N, S, embed_dim)
        return out

# ----- Test -------
if __name__ == "__main__":
    N = 4
    S = 30
    num_heads = 8
    embed_dim = 512
    mhsa = MultiHeadSelfAttn(num_heads, embed_dim)

    X = torch.rand(N, S, embed_dim)
    mask = torch.full((S,S), float('-inf'))
    mask = torch.triu(mask, diagonal=1)
    attn_score = mhsa(X, mask)
    assert attn_score.shape == (N, S, embed_dim), \
        f"Wrong output shape! Expect ({N},{S},{embed_dim}) but got {attn_score.shape}"
    print(f"attn_score.shape: {attn_score.shape}") # expect: (N, S, embed_dim) = (4, 30 512)

    # Test stable softmax with nan logits
    import torch.nn.functional as F
    logits = torch.tensor([float('nan'), 100, 0, float('-inf')])
    print(f"stale_softmax: {stable_softmax(logits)} v.s. F.softmax {F.softmax(logits, dim=-1)}") # (nan, 0, 0)
    print(f"Predicted class: {torch.argmax(stable_softmax(logits))} v.s. {torch.argmax(F.softmax(logits, dim=-1))}")
    # Expect result:
    # F.softmax:       tensor([nan, nan, nan, nan])
    # stable_softmax:  tensor([0., 1., 0., 0.])
