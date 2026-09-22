"""Target 블록마다 context를 반복하고 [context, mask+PE] sequence를 복원한다."""

import torch
from torch import nn

from src.modules.attention import DecoderAttention
from src.modules.masking import gather_context, gather_targets
from src.modules.normalization import RMSNorm
from src.modules.position_embedding import PositionEmbedding


class DecoderBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        dim = config["decoder_dim"]
        hidden = int(dim * config["decoder_mlp_ratio"])
        self.attention_norm = RMSNorm(dim, config["norm_epsilon"])
        self.attention = DecoderAttention(dim, config["decoder_heads"],
                                          config["attention_dropout"], config["decoder_dropout"])
        self.mlp_norm = RMSNorm(dim, config["norm_epsilon"])
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(config["decoder_dropout"]),
            nn.Linear(hidden, dim),
            nn.Dropout(config["decoder_dropout"]),
        )

    def forward(self, tokens):
        normalized = self.attention_norm(tokens)
        tokens = tokens + self.attention(normalized)
        normalized = self.mlp_norm(tokens)
        return tokens + self.mlp(normalized)


class Decoder(nn.Module):
    def __init__(self, config, coordinates):
        super().__init__()
        decoder = config["mae"]
        position = config["position"]
        dim = decoder["decoder_dim"]
        self.position = PositionEmbedding(position["decoder_spatial_dim"],
                                          position["decoder_temporal_dim"],
                                          position["temporal_max_period"], position["rms_epsilon"])
        self.context_projection = nn.Linear(config["encoder"]["embed_dim"], dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.blocks = nn.ModuleList()
        for _ in range(decoder["decoder_depth"]):
            self.blocks.append(DecoderBlock(decoder))
        self.norm = RMSNorm(dim, decoder["norm_epsilon"])
        self.output_projection = nn.Linear(dim, config["patch_encoder"]["patch_samples"])
        self.register_buffer("default_channel_coordinates", coordinates.detach().float().clone())

    def forward(self, context_grid, masks):
        batch, channels, patches, dim = context_grid.shape
        positions = self.position(self.default_channel_coordinates, batch, patches, context_grid.dtype)
        projected = self.context_projection(context_grid) + positions
        context = gather_context(projected, masks["context_mask"])
        target_positions = gather_targets(positions, masks["target_blocks"])
        blocks = masks["target_blocks"].shape[1]
        target_count = int(masks["target_blocks"][0, 0].sum())
        context_count = context.shape[1]
        width = projected.shape[-1]
        # [B,blocks*targets,D] → [B*blocks,targets,D]. Target는 raw 값 대신 mask token을 받는다.
        targets = target_positions.reshape(batch * blocks, target_count, width) + self.mask_token
        context = context[:, None].expand(-1, blocks, -1, -1)
        context = context.reshape(batch * blocks, context_count, width)
        sequence = torch.cat((context, targets), dim=1)
        for block in self.blocks:
            sequence = block(sequence)
        targets = self.norm(sequence)[:, context_count:]
        prediction = self.output_projection(targets)
        return prediction.reshape(batch, blocks * target_count, prediction.shape[-1])
