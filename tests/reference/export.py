"""Run the frozen oracle in its own interpreter. Never import the live project."""

from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.models.factory import build_models
from src.utils.masking import build_masking_policy
from src.utils.reconstruction import raw_patch_grid, gather_target_blocks
from src.utils.reconstruction_loss import reconstruction_losses
from src.datasets.pretraining_dataset import PretrainingDataset

torch.set_num_threads(2)
root = Path(__file__).resolve().parents[2]
checkpoint = torch.load(root / "tests/reference/checkpoint-epoch-0040.pth", map_location="cpu")
config = checkpoint["resolved_config"]
torch.manual_seed(918)
encoder, decoder = build_models(config, torch.device("cpu"))
initial = {"context_encoder": encoder.state_dict(), "decoder": decoder.state_dict()}
torch.save(initial, root / "outputs/oracle_initial.pth")
encoder.load_state_dict(checkpoint["context_encoder"], strict=True)
decoder.load_state_dict(checkpoint["decoder"], strict=True)
dataset = PretrainingDataset(config["resolved"]["dataset_dir"], 19, 30, 200)
signals = torch.stack((dataset[0][0], dataset[1][0])) / config["data"]["value_scale"]
dataset.close()
torch.manual_seed(47)
valid = torch.ones(2, 19, 30, dtype=torch.bool)
masks = build_masking_policy(config["masking"])(2, 19, 30, valid)
target = gather_target_blocks(raw_patch_grid(signals, 200), masks["target_blocks"])


def run():
    context = encoder(signals, visible_mask=masks["context_mask"])
    prediction = decoder(context, masks["context_mask"], masks["target_mask"],
                         target_blocks=masks["target_blocks"])
    loss = reconstruction_losses(prediction, target, masks["target_token_valid"],
                                 waveform_loss="smooth_l1", smooth_l1_beta=0.1)["total_loss"]
    return context, prediction, loss


intermediates = {}


def record(name):
    def hook(module, args, result):
        intermediates[name] = result.detach().clone()
    return hook


handles = [encoder.patch_encoder.register_forward_hook(record("tokenizer")),
           encoder.positional_encoder.register_forward_hook(record("position_added")),
           decoder.register_forward_hook(record("decoder"))]
for name, block in encoder.context_encoder.named_modules():
    if name.endswith(".block"):
        handles.append(block.register_forward_hook(record("encoder." + name)))
for index, block in enumerate(decoder.blocks):
    handles.append(block.register_forward_hook(record("decoder.block" + str(index))))
encoder.eval()
decoder.eval()
with torch.no_grad():
    context, prediction, loss = run()
eval_values = dict(intermediates)
for handle in handles:
    handle.remove()
encoder.train()
decoder.train()
torch.manual_seed(629)
rng = torch.get_rng_state().clone()
train_context, train_prediction, train_loss = run()
train_loss.backward()
gradients = {}
for prefix, model in (("context_encoder", encoder), ("decoder", decoder)):
    for name, parameter in model.named_parameters():
        gradients[prefix + "." + name] = parameter.grad.detach().clone()
torch.save({"signals": signals, "masks": masks, "target": target,
            "eval": eval_values, "context": context, "prediction": prediction,
            "loss": loss, "train_prediction": train_prediction.detach(),
            "train_loss": train_loss.detach(), "gradients": gradients,
            "rng_before_train": rng, "rng_after_train": torch.get_rng_state(),
            "dataset_fingerprint": dataset.fingerprint}, root / "outputs/oracle.pth")
print("oracle exported", float(loss), float(train_loss), flush=True)
