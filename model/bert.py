import torch
import torch.nn as nn


class MiniBERT(nn.Module):
    def __init__(self, vocab_size, embed_dim=32, num_heads=2,
                 hidden_dim=64, num_layers=1, seq_len=8):
        super().__init__()
        self.seq_len = seq_len
        self.embed_dim = embed_dim

        self.token_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.pos_embeddings = nn.Embedding(seq_len, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

    def encode(self, input_ids):
        batch_size, seq_len = input_ids.shape
        positions = (
            torch.arange(seq_len, device=input_ids.device)
            .unsqueeze(0)
            .expand(batch_size, -1)
        )
        x = self.token_embeddings(input_ids) + self.pos_embeddings(positions)
        return self.encoder(x)

    def forward(self, input_ids):
        return self.lm_head(self.encode(input_ids))


def build_model(vocab_size, embed_dim, num_heads, hidden_dim, num_layers, seq_len):
    return MiniBERT(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        seq_len=seq_len,
    )
