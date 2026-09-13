"""Dataset heads with explicit full/frozen/random transfer controls."""

import torch.nn as nn

from src.models.finetune.scalp_backbone import DownstreamClassifier


class TaskModel(nn.Module):
    def __init__(
        self,
        encoder,
        embed_dim,
        num_outputs,
        head_dropout,
        pooling,
        num_latents,
        num_patches,
        head_hidden_tokens=None,
        head_activation="gelu",
        transfer_mode="full",
    ):
        super().__init__()
        if transfer_mode not in {"full", "frozen", "random"}:
            raise ValueError("unsupported transfer mode")
        self.transfer_mode = transfer_mode
        self.num_outputs = int(num_outputs)
        self.classifier = DownstreamClassifier(
            encoder=encoder,
            embed_dim=embed_dim,
            num_outputs=self.num_outputs,
            dropout=head_dropout,
            pooling=pooling,
            num_latents=num_latents,
            num_patches=num_patches,
            head_hidden_tokens=head_hidden_tokens,
            head_activation=head_activation,
        )
        if transfer_mode == "frozen":
            self.encoder.requires_grad_(False)
            self.encoder.eval()

    def train(self, mode=True):
        super().train(mode)
        # A frozen representation must not retain stochastic encoder dropout.
        # The nonlinear all-patch head remains trainable, with its own dropout.
        if self.transfer_mode == "frozen":
            self.encoder.eval()
        return self

    @property
    def encoder(self):
        return self.classifier.encoder

    @property
    def head(self):
        return self.classifier.head

    def forward(
        self,
        x,
        channel_coordinates,
        channel_region_ids,
        channel_validity,
    ):
        logits = self.classifier(
            x,
            channel_coordinates,
            channel_region_ids,
            channel_validity,
        )
        return logits.squeeze(-1) if self.num_outputs == 1 else logits
