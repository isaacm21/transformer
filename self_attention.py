import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, n_heads, embedding_dim, dropout=None):
        super().__init__()
        self.n_heads = n_heads
        self.embedding_dim = embedding_dim
        self.dropout = nn.Dropout(dropout) if dropout else None
        self.head_dim = self.embedding_dim // self.n_heads

        self.Q = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.K = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.V = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.mlp_out = nn.Linear(self.embedding_dim, self.embedding_dim)

    def forward(self, q, k, v, mask=None):
        # expects q, k, v each to be of shape (B, seq_len, embed_dim)
        batch_size = q.shape[0]
        seq_len = q.shape[1]

        # (B, seq_len, embed_dim) -> (B, seq_len, embed_dim)
        q_ = self.Q(q)
        k_ = self.K(k)
        v_ = self.V(v)

        # (B, seq_len, embed_dim) -> (B, n_heads, seq_len, head_dim)
        q_ = q_.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        q_ = torch.permute(q_, (0, 2, 1, 3))
        k_ = k_.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        k_ = torch.permute(k_, (0, 2, 1, 3))
        v_ = v_.reshape(batch_size, seq_len, self.n_heads, self.head_dim)
        v_ = torch.permute(v_, (0, 2, 1, 3))

        # attn passed to further layers, attn_score can be used for visualization
        attn_score = torch.matmul(q_, k_.transpose(-2, -1)) / (self.head_dim ** .5)
        if mask is not None:
            attn_score = attn_score.masked_fill(mask == 0, -1e20)
        attn_score = F.softmax(attn_score, dim=-1)  # attn will be of shape (B, n_heads, seq_len, seq_len)
        if self.dropout:
            self.dropout(attn_score)
        attn = torch.matmul(attn_score, v_)  # (B, n_heads, seq_len, seq_len) -> (B, n_heads, seq_len, head_dim)
        attn = torch.permute(attn, (0, 2, 1, 3))  # (B, n_heads, seq_len, head_dim) -> (B, seq_len, n_heads, head_dim)
        attn = attn.reshape(batch_size, seq_len, self.embedding_dim)  # (B, seq_len, n_heads, head_dim) -> (B, seq_len, embed_dim)
        attn = self.mlp_out(attn)
        return attn, attn_score


if __name__ == "__main__":
    # test script
    n_heads = 4
    seq_len = 100
    embed_dim = 256
    batch_size = 64

    self_attention = SelfAttention(n_heads=n_heads, embedding_dim=embed_dim)
    src = torch.rand((batch_size, seq_len, embed_dim))
    attn, attn_score = self_attention(q=src, v=src, k=src)
    print(attn.shape)
    print(attn_score.shape)
