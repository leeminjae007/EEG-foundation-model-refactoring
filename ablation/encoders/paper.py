"""Call the authors' encoder forward methods; adapt only their token interfaces."""

from functools import partial
import torch
from torch import nn

from ablation.montage import region_order
from ablation.sources import upstream


class TokenInput(nn.Module):
    def forward(self, tokens, mask=None):
        return tokens


class FlatTokenInput(nn.Module):
    def forward(self, tokens, **kwargs):
        return tokens.flatten(1, 2)


class PaperEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        settings = config["ablation"]
        self.name = settings["encoder"]
        dim = config["encoder"]["embed_dim"]
        if dim != 200:
            raise ValueError("Paper presets use the original base width 200")
        depth = settings.get("depth", 12)
        if self.name == "labram":
            module = upstream("labram", "modeling_finetune.py")
            self.core = module.NeuralTransformer(
                EEG_size=config["data"]["num_patches"] * 200, patch_size=200,
                embed_dim=200, depth=depth, num_heads=10, mlp_ratio=4,
                qkv_bias=True, qk_norm=partial(nn.LayerNorm, eps=1e-6),
                norm_layer=partial(nn.LayerNorm, eps=1e-6), init_values=0.1,
                num_classes=0, use_abs_pos_emb=False)
            self.core.patch_embed = FlatTokenInput()
            self.core.time_embed = None
        elif self.name == "cbramod":
            module = upstream("cbramod", "models/cbramod.py")
            self.core = module.CBraMod(n_layer=depth)
            self.core.patch_embedding = TokenInput()
            self.core.proj_out = nn.Identity()
        elif self.name == "csbrain":
            module = upstream("csbrain", "models/CSBrain.py")
            regions, order = region_order(config["data"]["channel_names"])
            self.core = module.CSBrain(n_layer=depth, brain_regions=regions, sorted_indices=order)
            self.core.patch_embedding = TokenInput()
            self.core.proj_out = nn.Identity()
            self.set_channels(config["data"]["channel_names"], "pretrain")
        else:
            raise ValueError(self.name)

    def set_channels(self, names, dataset):
        if self.name != "csbrain":
            return
        module = upstream("csbrain", "models/CSBrain.py")
        layer_module = upstream("csbrain", "models/CSBrain_transformerlayer.py")
        regions, order = region_order(names, dataset)
        self.core.sorted_indices = order
        self.core.brain_regions = regions
        self.core.area_config = module.generate_area_config(sorted(regions))
        self.active_indices = sorted(order)
        self.inverse_order = [order.index(i) for i in self.active_indices]
        for layer in self.core.encoder.layers:
            builder = layer_module.RegionAttentionMaskBuilder(len(order), self.core.area_config)
            layer.area_config = self.core.area_config
            layer.mask_builder = builder
            layer.region_attn_mask = builder.get_mask()
            layer.region_indices_dict = builder.get_region_indices()
        # DDP must not wait for gradients from regions absent in this montage.
        for key, block in self.core.BrainEmbedEEGLayer.region_blocks.items():
            block.requires_grad_(key in self.core.area_config)

    def forward(self, tokens, visible):
        if self.name == "labram":
            output = self.core(tokens, return_patch_tokens=True)
            return output.reshape_as(tokens)
        output = self.core(tokens.contiguous())
        if self.name == "csbrain":
            output = output[:, self.inverse_order]
            if len(self.active_indices) != tokens.shape[1]:
                indices = torch.tensor(self.active_indices, device=output.device)
                output = output.new_zeros(tokens.shape).index_copy(1, indices, output)
        return output
