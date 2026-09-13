"""EEG [B,C,L] → 시간/주파수 토큰 [B,C,T,200]."""

import torch
from torch import nn


class Tokenizer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.patch_samples = config["patch_samples"]
        self.embed_dim = config["embed_dim"]
        layers = []
        input_channels = 1
        for output_channels, kernel, stride in zip(
            config["time_conv_channels"],
            config["time_kernel_sizes"],
            config["time_strides"],
        ):
            layers.append(nn.Conv1d(input_channels, output_channels, kernel,
                                    stride=stride, padding=kernel // 2))
            layers.append(nn.GroupNorm(config["time_group_norm_groups"], output_channels))
            layers.append(nn.GELU())
            input_channels = output_channels
        self.time_convolutions = nn.Sequential(*layers)
        with torch.no_grad():
            sample = torch.zeros(1, 1, self.patch_samples)
            output_size = self.time_convolutions(sample).numel()
        self.time_projection = nn.Linear(output_size, config["time_feature_dim"])
        self.frequency_mlp = nn.Sequential(
            nn.Linear(config["frequency_bins"], config["frequency_hidden_dim"]),
            nn.GELU(),
            nn.Linear(config["frequency_hidden_dim"], config["frequency_feature_dim"]),
        )
        self.output_dropout = nn.Dropout(config["dropout"])

    def forward(self, signals):
        batch, channels, length = signals.shape
        patch_count = length // self.patch_samples
        # [B,C,L] → [B*C*T,1,P]. 각 패치를 독립적으로 처리한다.
        patches = signals.reshape(batch * channels * patch_count, 1, self.patch_samples)
        time_features = self.time_convolutions(patches).flatten(start_dim=1)
        time_features = self.time_projection(time_features)
        spectrum = torch.fft.rfft(patches[:, 0], n=self.patch_samples,
                                  dim=-1, norm="forward")
        frequency_input = torch.log1p(spectrum.abs())
        frequency_features = self.frequency_mlp(frequency_input)
        tokens = torch.cat((time_features, frequency_features), dim=-1)
        tokens = self.output_dropout(tokens)
        return tokens.reshape(batch, channels, patch_count, self.embed_dim)
