import torch
import torch.nn as nn
from postional_embedder import PositionalEmbedder
from transformer_block import TransformerBlock


class TransformerEncoder(nn.Module):
    def __init__(self, max_seq_len, embedding_dim, vocab_size, n_heads, n_blocks, enc_dropout=None, attn_dropout=None,
                 mlp_dropout=None):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.word_embedder = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedder = nn.Embedding(max_seq_len, embedding_dim)
        # self.pos_embedder = PositionalEmbedder(seq_len=max_seq_len, embed_dim=embedding_dim)
        self.transformer_block = TransformerBlock(n_heads=n_heads,
                                                  embedding_dim=embedding_dim,
                                                  attn_dropout=attn_dropout,
                                                  mlp_dropout=mlp_dropout)
        self.n_blocks = n_blocks
        self.enc_dropout = nn.Dropout(enc_dropout) if enc_dropout else None

    def forward(self, sequence, mask=None):
        # TODO should positions be max_seq_len or seq_len?
        # TODO how to deal with seq_len < max_seq_len. will the mask take care of it?

        batch, seq_len = sequence.shape
        positions = torch.arange(0, self.max_seq_len)[None, :].expand(batch, max_seq_len)
        word_embeddings = self.word_embedder(sequence)
        pos_embeddings = self.pos_embedder(positions)
        x = word_embeddings + pos_embeddings
        if self.enc_dropout:
            x = self.enc_dropout(x)
        for _ in range(self.n_blocks):
            x = self.transformer_block(q=x, k=x, v=x, mask=mask)
        return x


if __name__ == "__main__":
    # test script
    max_seq_len = 100
    embedding_dim = 256
    vocab_size = 1000
    n_heads = 4
    n_blocks = 2
    enc_dropout = 0.1
    attn_dropout = 0.1
    mlp_dropout = 0.1

    batch_size = 64
    padding_len = 25

    encoder = TransformerEncoder(max_seq_len=max_seq_len,
                                 embedding_dim=embedding_dim,
                                 vocab_size=vocab_size,
                                 n_heads=n_heads,
                                 n_blocks=n_blocks,
                                 enc_dropout=enc_dropout,
                                 attn_dropout=attn_dropout,
                                 mlp_dropout=mlp_dropout)

    padding = torch.zeros((batch_size, padding_len))
    sequence = torch.arange(1, max_seq_len + 1 - padding_len)[None, :].expand(batch_size, max_seq_len - padding_len)
    padded_sequence = torch.concat((sequence, padding), dim=-1).to(torch.long)
    mask = torch.where(padded_sequence == 0, 0, 1)

    # TODO fix masking. use AND or OR operator to create it?

    out = encoder(padded_sequence)
    print(out.shape)
