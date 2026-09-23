import torch
from torch import nn

from xfmr.xfmr2017.transformer import PositionalEncoding


class TorchTransformer(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        tgt_vocab_size,
        d_model,
        num_heads,
        num_layers,
        d_ff,
        dropout,
        src_pad_idx,
        tgt_pad_idx,
    ):
        super().__init__()

        self.d_model = d_model
        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx

        self.encoder_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
        )

        self.decoder_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
        )

        self.embedding_scale = d_model**0.5

        self.positional_encoding = PositionalEncoding(
            d_model
        )

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )

        self.fc_out = nn.Linear(
            d_model,
            tgt_vocab_size,
        )

        # Match the custom Transformer.
        self.fc_out.weight = self.decoder_embedding.weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, src, tgt):
        src_key_padding_mask = (
            src == self.src_pad_idx
        )

        tgt_key_padding_mask = (
            tgt == self.tgt_pad_idx
        )

        src = self.encoder_embedding(src)
        src = src * self.embedding_scale
        src = self.dropout(self.positional_encoding(src))

        tgt = self.decoder_embedding(tgt)
        tgt = tgt * self.embedding_scale
        tgt = self.dropout(self.positional_encoding(tgt))

        # tgt_mask = nn.Transformer.generate_square_subsequent_mask(
        #     tgt.size(1),
        #     device=tgt.device,
        # )

        tgt_mask = torch.triu(
            torch.ones(
                tgt.size(1),
                tgt.size(1),
                dtype=torch.bool,
                device=tgt.device,
            ),
            diagonal=1,
        )

        output = self.transformer(
            src,
            tgt,
            tgt_mask=tgt_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        return self.fc_out(output)