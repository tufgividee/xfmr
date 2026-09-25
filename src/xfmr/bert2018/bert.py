import torch
from torch import nn

from .modules import MultiHeadAttention, PositionWiseFeedForward


class BertEmbeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, dropout):
        super().__init__()

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_len, d_model)
        self.type_embeddings = nn.Embedding(2, d_model)

        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.register_buffer(
            "position_ids",
            torch.arange(max_len),
            persistent=False,
        )

    def forward(self, input_ids, token_type_ids):
        # input_ids: [batch_size, seq_len]
        batch_size, seq_len = input_ids.shape

        position_ids = self.position_ids[:seq_len]

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        type_embeddings = self.type_embeddings(token_type_ids)

        embeddings = (
            token_embeddings
            + position_embeddings
            + type_embeddings
        )

        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        # Self-attention
        attn_output = self.self_attn(x, x, x, mask)

        x = self.norm1(
            x + self.dropout(attn_output)
        )

        # Feed-forward
        ff_output = self.feed_forward(x)

        x = self.norm2(
            x + self.dropout(ff_output)
        )

        return x


class Encoder(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        num_layers,
        dropout,
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    d_model,
                    num_heads,
                    d_ff,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)

        return x


class BertModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        dropout,
        max_len=512,
    ):
        super().__init__()

        self.embeddings = BertEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            max_len=max_len,
            dropout=dropout,
        )

        self.encoder = Encoder(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            num_layers=num_layers,
            dropout=dropout,
        )

        # MLM
        self.mlm_transform = nn.Linear(
            d_model,
            d_model,
        )

        self.mlm_activation = nn.GELU()

        self.mlm_layer_norm = nn.LayerNorm(d_model)

        self.mlm_decoder = nn.Linear(
            d_model,
            vocab_size,
            bias=False,
        )

        # Tie MLM decoder weights to token embeddings
        self.mlm_decoder.weight = self.embeddings.token_embeddings.weight

        # NSP
        self.nsp_classifier = nn.Linear(
            d_model,
            2,
        )

    def make_attention_mask(self, attention_mask):
        # [B, L] → [B, 1, 1, L]
        return attention_mask.bool().unsqueeze(1).unsqueeze(2)

    def forward(
        self,
        input_ids,
        token_type_ids,
        attention_mask,
    ):
        # ---------------------------------------------------------
        # Embeddings
        # ---------------------------------------------------------
        x = self.embeddings(
            input_ids,
            token_type_ids,
        )

        # ---------------------------------------------------------
        # Encoder
        # ---------------------------------------------------------
        mask = self.make_attention_mask(attention_mask)

        hidden_states = self.encoder(
            x,
            mask,
        )

        # ---------------------------------------------------------
        # MLM head
        # ---------------------------------------------------------
        mlm_hidden = self.mlm_transform(hidden_states)
        mlm_hidden = self.mlm_activation(mlm_hidden)
        mlm_hidden = self.mlm_layer_norm(mlm_hidden)

        mlm_logits = self.mlm_decoder(mlm_hidden)

        # ---------------------------------------------------------
        # NSP head
        # ---------------------------------------------------------
        cls_hidden = hidden_states[:, 0, :]

        nsp_logits = self.nsp_classifier(cls_hidden)

        return mlm_logits, nsp_logits