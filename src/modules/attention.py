"""Attention 수식과 [B,C,T,D]의 축 변환. Residual은 encoder/decoder에 있다."""

import torch
from torch import nn


class MaskedAttention(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim)
        self.output = nn.Linear(dim, dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.capture = False
        self.diagnostics = {}

    def forward(self, tokens, valid):
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(0)
        scores = (query @ key.transpose(-2, -1)) * self.scale
        qk = scores.detach()
        key_mask = valid[:, None, None]
        scores = scores.masked_fill(~key_mask, -torch.inf)
        maximum = scores.amax(dim=-1, keepdim=True)
        # 보이는 key가 없는 행도 있다. 원본의 zero-row softmax 수식이다.
        maximum = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
        weights = torch.exp(scores - maximum)
        weights = torch.where(key_mask, weights, torch.zeros_like(weights))
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        if self.capture:
            self.diagnostics = attention_diagnostics(qk, weights, valid)
            self.diagnostics["attention/total_axis_rows"] = batch
        weights = self.attention_dropout(weights)
        output = (weights @ value).transpose(1, 2).reshape(batch, length, dim)
        return self.output(output) * valid.unsqueeze(-1)


class SpatialAttention(MaskedAttention):
    """같은 시점의 채널끼리 attention을 계산한다."""

    @staticmethod
    def to_rows(grid, valid):
        batch, channels, patches, dim = grid.shape
        rows = grid.permute(0, 2, 1, 3).reshape(batch * patches, channels, dim)
        mask = valid.permute(0, 2, 1).reshape(batch * patches, channels)
        return rows, mask

    @staticmethod
    def to_grid(rows, shape):
        batch, channels, patches, dim = shape
        return rows.reshape(batch, patches, channels, dim).permute(0, 2, 1, 3)


class TemporalAttention(MaskedAttention):
    """같은 채널의 시점끼리 attention을 계산한다."""

    @staticmethod
    def to_rows(grid, valid):
        batch, channels, patches, dim = grid.shape
        return grid.reshape(batch * channels, patches, dim), valid.reshape(batch * channels, patches)

    @staticmethod
    def to_grid(rows, shape):
        return rows.reshape(shape)


class DecoderAttention(nn.Module):
    """복원 sequence는 모든 위치가 유효하다. 출력 dropout도 유지한다."""

    def __init__(self, dim, heads, attention_dropout, dropout):
        super().__init__()
        self.num_heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, 3 * dim, bias=True)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.output = nn.Linear(dim, dim)
        self.output_dropout = nn.Dropout(dropout)
        self.capture = False
        self.diagnostics = {}

    def forward(self, tokens):
        batch, length, dim = tokens.shape
        qkv = self.qkv(tokens).reshape(batch, length, 3, self.num_heads, self.head_dim)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        logits = (query @ key.transpose(-2, -1)) * self.scale
        probabilities = logits.softmax(dim=-1)
        if self.capture:
            valid = torch.ones(batch, length, dtype=torch.bool, device=tokens.device)
            self.diagnostics = attention_diagnostics(logits, probabilities, valid, sample_limit=4)
            self.diagnostics["attention/total_axis_rows"] = batch
        weights = self.attention_dropout(probabilities)
        output = (weights @ value).transpose(1, 2).reshape(batch, length, dim)
        return self.output_dropout(self.output(output))


@torch.no_grad()
def attention_diagnostics(qk, probabilities, valid_mask, sample_limit=32):
    """Per-head pre-dropout attention on evenly spaced axis rows, valid pairs only."""
    indices = torch.linspace(0, qk.shape[0] - 1, min(sample_limit, qk.shape[0]),
                             device=qk.device).long()
    qk = qk[indices].detach().float()
    probabilities = probabilities[indices].detach().float()
    valid = valid_mask[indices]
    bias_values = torch.zeros_like(qk)  # GR9-1에는 attention bias가 없다.
    keys = valid[:, None, None, :]
    queries = valid[:, None, :]
    pairs = queries.unsqueeze(-1) & keys
    counts = keys.sum(-1).clamp_min(1)
    pair_count = pairs.sum().clamp_min(1)
    query_count = queries.sum().clamp_min(1)
    q_centered = qk - (qk * keys).sum(-1, keepdim=True) / counts.unsqueeze(-1)
    b_centered = bias_values - (bias_values * keys).sum(-1, keepdim=True) / counts.unsqueeze(-1)
    q_rms = ((q_centered.square() * pairs).sum((0, 2, 3)) / pair_count).sqrt()
    b_rms = ((b_centered.square() * pairs).sum((0, 2, 3)) / pair_count).sqrt()
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(-1)
    diagonal = probabilities.diagonal(dim1=-2, dim2=-1)
    output = {
        'attention/bias_present': qk.new_tensor(0.0),
        'attention/sampled_axis_rows': qk.new_tensor(qk.shape[0]),
        'attention/valid_query_count': queries.sum().float(),
        'attention/sequence_length': qk.new_tensor(qk.shape[-1]),
    }
    for head in range(qk.shape[1]):
        prefix = f'attention/head{head}'
        output.update({
            f'{prefix}/qk_row_centered_rms': q_rms[head],
            f'{prefix}/bias_row_centered_rms': b_rms[head],
            f'{prefix}/bias_to_qk_rms': b_rms[head] / q_rms[head].clamp_min(1e-12),
            f'{prefix}/bias_to_qk_defined': q_rms[head].gt(1e-12).float(),
            f'{prefix}/entropy': (entropy[:, head] * valid).sum() / query_count,
            f'{prefix}/self_attention_mass': (diagonal[:, head] * valid).sum() / query_count,
        })
    return output
