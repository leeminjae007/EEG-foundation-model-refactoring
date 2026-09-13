"""입력부터 복원/분류까지의 처리 순서. 저장과 학습은 training/에 있다."""

import torch
from torch import nn

from src.data.electrode_geometry import resolve_channel_coordinates
from src.decoder import Decoder
from src.encoder import Encoder
from src.modules.masking import make_masks
from src.modules.geometry_masking import physical_channel_coordinates
from src.modules.position_embedding import PositionEmbedding
from src.modules.tokenizer import Tokenizer


class EEGEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        position = config["position"]
        self.tokenizer = Tokenizer(config["patch_encoder"])
        self.position = PositionEmbedding(position["encoder_spatial_dim"],
                                          position["encoder_temporal_dim"],
                                          position["temporal_max_period"], position["rms_epsilon"])
        self.encoder = Encoder(config["encoder"])
        coordinates, valid = resolve_channel_coordinates(tuple(config["data"]["channel_names"]))
        self.register_buffer("default_channel_coordinates", coordinates)

    def forward(self, signals, coordinates, channel_valid, visible):
        # signals [B,C,L], coordinates [B,C,3], masks [B,C]와 [B,C,T].
        signals = signals.masked_fill(~channel_valid[:, :, None], 0.0)
        tokens = self.tokenizer(signals)
        batch, channels, patches, dim = tokens.shape
        positions = self.position(coordinates, batch, patches, tokens.dtype)
        tokens = tokens + positions
        tokens = tokens * channel_valid[:, :, None, None]
        visible = visible & channel_valid[:, :, None]
        # Encoder 입구에서 숨긴 token을 0으로 만들고 attention에서 제외한다.
        return self.encoder(tokens, visible)


def initialize_weights(module):
    # 원본 factory와 같은 재초기화. Bias, norm, mask token은 기본값을 유지한다.
    if isinstance(module, (nn.Linear, nn.Conv1d)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")


class PretrainModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self.mask_config = config["masking"]
        self.patch_samples = config["patch_encoder"]["patch_samples"]
        # 원본처럼 encoder를 먼저 device에 올린 뒤 decoder를 생성한다.
        self.backbone = EEGEncoder(config).to(device)
        coordinates = self.backbone.default_channel_coordinates
        self.decoder = Decoder(config, coordinates).to(device)
        self.backbone.apply(initialize_weights)
        self.decoder.apply(initialize_weights)
        # 마스킹만 실제 거리로 계산한다. PE 좌표와 checkpoint tensor는 그대로 둔다.
        self.mask_coordinates = self.backbone.default_channel_coordinates
        if self.mask_config.get("distance_metric") == "euclidean_m":
            self.mask_coordinates = physical_channel_coordinates(config["data"]["channel_names"])

    def forward(self, signals, masks=None):
        batch, channels, length = signals.shape
        patches = length // self.patch_samples
        if masks is None:
            masks = make_masks(batch, channels, patches, self.mask_config, signals.device,
                               self.mask_coordinates)
        coordinates = self.backbone.default_channel_coordinates[None].expand(batch, -1, -1)
        valid = torch.ones(batch, channels, dtype=torch.bool, device=signals.device)
        context = self.backbone(signals, coordinates, valid, masks["context_mask"])
        prediction = self.decoder(context, masks)
        return prediction, masks


class FinetuneModel(nn.Module):
    def __init__(self, backbone, channels, patches, outputs, dropout, hidden_tokens):
        super().__init__()
        self.backbone = backbone
        dim = backbone.tokenizer.embed_dim
        self.outputs = outputs
        self.head = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(channels * patches * dim, hidden_tokens * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_tokens * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, outputs),
        )

    def forward(self, signals, coordinates, channel_valid):
        patches = signals.shape[-1] // self.backbone.tokenizer.patch_samples
        visible = channel_valid[:, :, None].expand(-1, -1, patches)
        features = self.backbone(signals, coordinates, channel_valid, visible)
        features = features * visible.unsqueeze(-1)
        logits = self.head(features.contiguous())
        if self.outputs == 1:
            logits = logits.squeeze(-1)
        return logits


class SleepModel(nn.Module):
    """ISRUC는 20개 sleep epoch를 한 sequence로 분류하는 별도 head를 쓴다."""

    def __init__(self, backbone, channels, patches, dropout):
        super().__init__()
        self.backbone = backbone
        dim = backbone.tokenizer.embed_dim
        self.head = nn.ModuleDict({
            "epoch_projection": nn.Sequential(nn.Linear(channels * patches * dim, 512), nn.GELU()),
            "sequence_encoder": nn.TransformerEncoder(
                nn.TransformerEncoderLayer(d_model=512, nhead=4, dim_feedforward=2048,
                                           dropout=dropout, batch_first=True,
                                           activation="gelu", norm_first=True),
                num_layers=1, enable_nested_tensor=False),
            "classifier": nn.Linear(512, 5),
        })

    def forward(self, signals, coordinates, channel_valid):
        batch, sequence, channels, length = signals.shape
        signals = signals.reshape(batch * sequence, channels, length)
        coordinates = coordinates[:, None].expand(-1, sequence, -1, -1)
        coordinates = coordinates.reshape(batch * sequence, channels, 3)
        channel_valid = channel_valid[:, None].expand(-1, sequence, -1)
        channel_valid = channel_valid.reshape(batch * sequence, channels)
        patches = length // self.backbone.tokenizer.patch_samples
        visible = channel_valid[:, :, None].expand(-1, -1, patches)
        features = self.backbone(signals, coordinates, channel_valid, visible)
        features = features * visible.unsqueeze(-1)
        features = features.contiguous().view(batch * sequence, -1)
        epochs = self.head["epoch_projection"](features).reshape(batch, sequence, 512)
        sequence_features = self.head["sequence_encoder"](epochs)
        return self.head["classifier"](sequence_features)
