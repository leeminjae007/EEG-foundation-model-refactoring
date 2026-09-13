"""독립된 S→T와 T→S 경로를 3번 실행하고 단계마다 합친다."""

import torch
from torch import nn

from src.modules.attention import SpatialAttention, TemporalAttention
from src.modules.normalization import RMSNorm


class EncoderBlock(nn.Module):
    def __init__(self, config, axis):
        super().__init__()
        dim = config["embed_dim"]
        hidden = int(dim * config["mlp_ratio"])
        self.attention_norm = RMSNorm(dim, config["norm_epsilon"])
        if axis == "spatial":
            self.attention = SpatialAttention(dim, config["spatial_heads"], config["attention_dropout"])
        else:
            self.attention = TemporalAttention(dim, config["temporal_heads"], config["attention_dropout"])
        self.mlp_norm = RMSNorm(dim, config["norm_epsilon"])
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(config["dropout"]),
            nn.Linear(hidden, dim),
            nn.Dropout(config["dropout"]),
        )

    def forward(self, grid, visible):
        tokens, valid = self.attention.to_rows(grid, visible)
        mask = valid.unsqueeze(-1)  # True: attention에 참여하는 context token.
        tokens = tokens * mask
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(normalized, valid)
        normalized = self.mlp_norm(tokens)
        tokens = tokens + self.mlp(normalized)
        tokens = tokens * mask
        return self.attention.to_grid(tokens, grid.shape)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 생성 순서도 원본과 같다. 경로/단계 사이에 block 가중치를 공유하지 않는다.
        self.s2t_spatial = nn.ModuleList()
        self.s2t_temporal = nn.ModuleList()
        self.t2s_temporal = nn.ModuleList()
        self.t2s_spatial = nn.ModuleList()
        for _ in range(3):
            self.s2t_spatial.append(EncoderBlock(config, "spatial"))
        for _ in range(3):
            self.s2t_temporal.append(EncoderBlock(config, "temporal"))
        for _ in range(3):
            self.t2s_temporal.append(EncoderBlock(config, "temporal"))
        for _ in range(3):
            self.t2s_spatial.append(EncoderBlock(config, "spatial"))
        self.fusion_gates = nn.Parameter(torch.zeros(3, config["embed_dim"]))
        self.output_norm = RMSNorm(config["embed_dim"], config["norm_epsilon"])

    def forward(self, tokens, visible):
        mask = visible.unsqueeze(-1)
        fused = tokens * mask
        for stage in range(3):
            spatial_first = self.s2t_spatial[stage](fused, visible)
            s2t = self.s2t_temporal[stage](spatial_first, visible)

            temporal_first = self.t2s_temporal[stage](fused, visible)
            t2s = self.t2s_spatial[stage](temporal_first, visible)

            gate = self.fusion_gates[stage].sigmoid().view(1, 1, 1, -1)
            fused = (gate * s2t + (1.0 - gate) * t2s) * mask
        return self.output_norm(fused) * mask
