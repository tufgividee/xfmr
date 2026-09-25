import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # q, k, v are (batch_size, num_heads, sequence_length, d_k)
        # (batch, heads, seq, seq)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) \
            / (self.d_k**0.5)
        if mask is not None:
            # Apply mask by filling masked positions with a very small number
            # to make their softmax probabilities close to zero.
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, v) # (batch, heads, seq, seq) x (batch, heads, seq, _d_k)
        return output # (batch, heads, seq, d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections
        # query: (batch, seq, d_model), q: (batch, seq, d_model),
        # .view(***) -1 is autocalc -> seq, d_model splits into num_heads x d_k
        # .transpose(1, 2) swaps -> (batch, heads, seq, d_k)
        # multiplication occurs only on the last two dimensions of the tensors.
        q = self.w_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output = self.scaled_dot_product_attention(q, k, v, mask)

        # Concatenate and apply final linear layer
        # recontructs original input dimension
        # and .view basically is the concat in the paper for multihead
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        output = self.w_o(attn_output) # (batch, seq, d_model)
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))