"""Mask-free Transformer blocks shared by the encoder and decoder."""

import math
import torch
import torch.nn as nn

from src.models.normalization import build_norm


class _DropPath(nn.Module):
    def __init__(self, probability=0.0):
        super().__init__()
        self.probability = float(probability)

    def forward(self, value):
        if not self.training or self.probability == 0.0:
            return value
        keep = 1.0 - self.probability
        shape = (value.shape[0],) + (1,) * (value.ndim - 1)
        return value * value.new_empty(shape).bernoulli_(keep) / keep


class _Attention(nn.Module):
    def __init__(self, dim, num_heads, qkv_bias, attention_dropout, dropout):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dimension must be divisible by heads")
        self.num_heads = int(num_heads)
        self.head_dim = int(dim) // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=bool(qkv_bias))
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output = nn.Linear(dim, dim)
        self.output_dropout = nn.Dropout(dropout)
        self.capture_diagnostics = False
        self.last_model_diagnostics = {}

    def forward(self, tokens, token_mask=None):
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch, length, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        logits = (query @ key.transpose(-2, -1)) * self.scale
        qk_diagnostic = logits.detach() if self.capture_diagnostics else None
        if token_mask is not None:
            logits = logits.masked_fill(
                ~token_mask[:, None, None], torch.finfo(logits.dtype).min
            )
        probabilities = logits.softmax(dim=-1)
        if self.capture_diagnostics:
            from src.utils.model_diagnostics import attention_diagnostics
            valid = token_mask if token_mask is not None else torch.ones(
                batch, length, dtype=torch.bool, device=tokens.device)
            self.last_model_diagnostics = attention_diagnostics(
                qk_diagnostic, None, probabilities, valid, sample_limit=4)
        weights = self.attention_dropout(probabilities)
        output = (weights @ value).transpose(1, 2).reshape(batch, length, dim)
        output = self.output_dropout(self.output(output))
        return output if token_mask is None else output * token_mask.unsqueeze(-1)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio,
        qkv_bias,
        dropout,
        attention_dropout,
        norm_epsilon,
        norm_type="rms_norm",
        drop_path=0.0,
    ):
        super().__init__()
        self.attention_norm = build_norm(dim, norm_type, norm_epsilon)
        self.attention = _Attention(
            dim, num_heads, qkv_bias, attention_dropout, dropout
        )
        self.mlp_norm = build_norm(dim, norm_type, norm_epsilon)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.drop_path = _DropPath(drop_path)

    def forward(self, tokens, token_mask=None):
        tokens = tokens + self.drop_path(
            self.attention(self.attention_norm(tokens), token_mask)
        )
        tokens = tokens + self.drop_path(self.mlp(self.mlp_norm(tokens)))
        return tokens if token_mask is None else tokens * token_mask.unsqueeze(-1)


def initialize_historical_transformer(module, init_std=0.02):
    """Initialization used by the archived D192 decoder."""
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=float(init_std))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


def rescale_historical_decoder_blocks(blocks):
    """Apply the archived depth-dependent residual-branch rescaling."""
    for index, block in enumerate(blocks, start=1):
        scale = math.sqrt(2.0 * index)
        block.attention.output.weight.data.div_(scale)
        block.mlp[3].weight.data.div_(scale)
