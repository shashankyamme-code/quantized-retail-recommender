"""Compact causal Transformer for next-item prediction."""

from __future__ import annotations

import torch
from torch import nn


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads.")
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        batch, sequence_length, width = hidden.shape
        shape = (batch, sequence_length, self.n_heads, self.head_dim)
        query = self.query(hidden).view(shape).transpose(1, 2)
        key = self.key(hidden).view(shape).transpose(1, 2)
        value = self.value(hidden).view(shape).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_dim**0.5
        causal_mask = torch.triu(
            torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=hidden.device),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        scores = scores.masked_fill(
            padding_mask[:, None, None, :],
            torch.finfo(scores.dtype).min,
        )
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.matmul(weights, value).transpose(1, 2).contiguous()
        attended = attended.view(batch, sequence_length, width)
        return self.output(attended)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, dropout)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.dropout(self.attention(self.attention_norm(hidden), padding_mask))
        return hidden + self.feed_forward(self.feed_forward_norm(hidden))


class RetailLanguageModel(nn.Module):
    """A small, scratch-trained autoregressive product language model."""

    def __init__(
        self,
        vocab_size: int,
        max_length: int = 12,
        d_model: int = 48,
        n_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_length, d_model)
        self.layers = nn.ModuleList(
            TransformerBlock(d_model, n_heads, dropout) for _ in range(num_layers)
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, tokens: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        if tokens.ndim != 2 or tokens.shape[1] > self.max_length:
            raise ValueError(f"tokens must have shape (batch, <= {self.max_length}).")
        if lengths is None:
            lengths = tokens.ne(0).sum(dim=1)
        lengths = lengths.clamp(min=1, max=tokens.shape[1])
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        padding_mask = tokens.eq(0)
        for layer in self.layers:
            hidden = layer(hidden, padding_mask)
        hidden = self.final_norm(hidden)
        final_hidden = hidden[torch.arange(tokens.shape[0], device=tokens.device), lengths - 1]
        return self.output_projection(final_hidden)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
