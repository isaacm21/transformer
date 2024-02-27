import torch
import math
import torch.nn as nn


class PositionalEmbedder(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

    def forward(self):
        position = torch.arange(0, self.seq_len)[:, None]
        scale_term = torch.exp(torch.arange(0, self.embed_dim, 2) * (-math.log(10000.0) / self.embed_dim))
        pos_embedding = torch.zeros(self.seq_len, self.embed_dim)
        pos_embedding[:, 0::2] = torch.sin(position * scale_term)
        pos_embedding[:, 1::2] = torch.cos(position * scale_term)
        return pos_embedding


if __name__ == "__main__":
    seq_len = 100
    embed_dim = 256

    positional_embedder = PositionalEmbedder(seq_len=seq_len, embed_dim=embed_dim)
    pos_embed = positional_embedder()
    print(pos_embed.shape)

    # TODO heatmap of pos_embed
