import torch
import torch.nn as nn
from self_attention import SelfAttention


class TransformerBlock(nn.Module):
    def __init__(self, n_heads, embedding_dim, attn_dropout=None, mlp_dropout=None):
        super().__init__()
        self.norm_layer1 = nn.LayerNorm(embedding_dim)
        self.norm_layer2 = nn.LayerNorm(embedding_dim)
        self.self_attention = SelfAttention(n_heads=n_heads,
                                            embedding_dim=embedding_dim,
                                            dropout=attn_dropout)
        hidden_layer_dim = int(4 * embedding_dim)  # from paper
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, hidden_layer_dim),
                                 nn.ReLU(),
                                 nn.Linear(hidden_layer_dim, embedding_dim))
        self.dropout = nn.Dropout(mlp_dropout) if mlp_dropout else None

    def forward(self, q, k, v, mask=None):
        attn, attn_score = self.self_attention(q=q, k=k, v=v, mask=mask)
        x = self.norm_layer1(attn + q)
        if self.dropout:
            x = self.dropout(x)
        x = x + self.mlp(x)
        x = self.norm_layer2(x)
        return x


if __name__ == "__main__":
    # test script
    n_heads = 4
    embedding_dim = 256
    attn_dropout = 0.1
    mlp_dropout = 0.1
    batch_size = 64
    seq_len = 100

    transformer_block = TransformerBlock(n_heads=n_heads,
                                         embedding_dim=embedding_dim,
                                         attn_dropout=attn_dropout,
                                         mlp_dropout=mlp_dropout)

    src = torch.rand((batch_size, seq_len, embedding_dim))
    out = transformer_block(q=src, k=src, v=src, mask=None)
    print(out.shape)
