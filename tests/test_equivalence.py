"""동일 가중치·실제 EEG·고정 mask에서 중간값, loss, dropout RNG, gradient를 비교한다."""

import json
import os
import subprocess
import sys

import pytest
import torch

from scripts.convert_checkpoint import convert
from src.encoder import EncoderBlock
from src.model import PretrainModel
from src.modules.loss import reconstruction_loss
from src.modules.masking import make_masks, gather_targets
from src.training.checkpoint import save_checkpoint, load_checkpoint, rng_state
from src.training.runtime import ROOT, set_paths

set_paths()
torch.set_num_threads(2)


def difference(actual, expected):
    return float((actual.float() - expected.float()).abs().max())


@pytest.fixture(scope="module")
def reference():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    subprocess.run([sys.executable, str(ROOT / "tests/reference/export.py")],
                   cwd=ROOT / "tests/reference", env=env, check=True)
    return torch.load(ROOT / "outputs/oracle.pth", map_location="cpu")


def test_pretrain_equivalence(reference):
    checkpoint = torch.load(ROOT / "tests/reference/checkpoint-epoch-0040.pth", map_location="cpu")
    config = checkpoint["resolved_config"]
    device = torch.device("cpu")
    torch.manual_seed(918)
    model = PretrainModel(config, device)
    initial = torch.load(ROOT / "outputs/oracle_initial.pth", map_location="cpu")
    expected_initial, _ = convert(initial)
    initialization_difference = 0.0
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, expected_initial[key], rtol=0, atol=0)
        initialization_difference = max(initialization_difference, difference(tensor, expected_initial[key]))
    state, mapping = convert(checkpoint)
    model.load_state_dict(state, strict=True)
    torch.manual_seed(47)
    masks = make_masks(2, 19, 30, config["masking"], device)
    for key, tensor in masks.items():
        assert torch.equal(tensor, reference["masks"][key]), key
    signals = reference["signals"]
    target = gather_targets(signals.unfold(-1, 200, 200), masks["target_blocks"])
    torch.testing.assert_close(target, reference["target"], rtol=0, atol=0)
    intermediate = {}

    def record(name):
        def hook(module, args, output):
            intermediate[name] = output.detach().clone()
        return hook

    handles = [model.backbone.tokenizer.register_forward_hook(record("tokenizer")),
               model.decoder.register_forward_hook(record("decoder"))]
    for name, module in model.backbone.encoder.named_modules():
        if isinstance(module, EncoderBlock):
            def row_hook(block, args, output, key="encoder." + name + ".block"):
                rows, _ = block.attention.to_rows(output, args[1])
                intermediate[key] = rows.detach().clone()
            handles.append(module.register_forward_hook(row_hook))
    for index, block in enumerate(model.decoder.blocks):
        handles.append(block.register_forward_hook(record("decoder.block" + str(index))))
    model.eval()
    with torch.no_grad():
        prediction, _ = model(signals, masks)
        loss = reconstruction_loss(prediction, target, masks["target_token_valid"], 0.1)
        positions = model.backbone.position(model.backbone.default_channel_coordinates[None].expand(2, -1, -1), 2, 30, torch.float32)
        intermediate["position_added"] = intermediate["tokenizer"] + positions
    differences = {}
    for key, expected in reference["eval"].items():
        actual = intermediate[key]
        differences[key] = difference(actual, expected)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6, msg=key)
    torch.testing.assert_close(loss, reference["loss"], rtol=1e-6, atol=1e-7)
    for handle in handles:
        handle.remove()
    model.train()
    torch.set_rng_state(reference["rng_before_train"])
    prediction, _ = model(signals, masks)
    train_loss = reconstruction_loss(prediction, target, masks["target_token_valid"], 0.1)
    train_loss.backward()
    # Seed 비교에 그치지 않고 동일 dropout 호출 순서와 실제 RNG 종료 상태까지 검사한다.
    assert torch.equal(torch.get_rng_state(), reference["rng_after_train"])
    torch.testing.assert_close(prediction, reference["train_prediction"], rtol=1e-5, atol=2e-6)
    gradients = dict(model.named_parameters())
    gradient_difference = 0.0
    for old, expected in reference["gradients"].items():
        key = mapping[old]
        actual = gradients[key].grad
        gradient_difference = max(gradient_difference, difference(actual, expected))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6, msg=key)
    report = {"device": "cpu", "dtype": "float32", "torch": torch.__version__,
              "parameter_tensors": len(gradients), "state_tensors": len(state),
              "initialization_max_abs": initialization_difference,
              "eval_intermediate_max_abs": differences,
              "eval_loss_abs": abs(float(loss - reference["loss"])),
              "train_prediction_max_abs": difference(prediction, reference["train_prediction"]),
              "train_loss_abs": abs(float(train_loss - reference["train_loss"])),
              "gradient_max_abs": gradient_difference, "dropout_final_rng_equal": True,
              "tolerance": {"atol": 2e-6, "rtol": 1e-5},
              "real_data_shape": list(signals.shape), "dataset": reference["dataset_fingerprint"]}
    (ROOT / "outputs/equivalence.json").write_text(json.dumps(report, indent=2))


def test_checkpoint_roundtrip(reference):
    original = torch.load(ROOT / "tests/reference/checkpoint-epoch-0040.pth", map_location="cpu")
    model = PretrainModel(original["resolved_config"], torch.device("cpu"))
    model.load_state_dict(convert(original)[0], strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)
    model.train()
    prediction, masks = model(reference["signals"], reference["masks"])
    loss = reconstruction_loss(prediction, reference["target"], masks["target_token_valid"], 0.1)
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    path = ROOT / "outputs/checkpoint-roundtrip.pth"
    states = [rng_state(torch.device("cpu"))]
    save_checkpoint(path, model, optimizer, scheduler, 1, original["resolved_config"], states, {"step": 1})
    expected = {}
    for name, value in model.state_dict().items():
        expected[name] = value.clone()
    with torch.no_grad():
        next(model.parameters()).add_(1.0)
    epoch, extra = load_checkpoint(path, model, optimizer, scheduler, torch.device("cpu"), 0)
    assert epoch == 1 and extra["step"] == 1
    assert scheduler.last_epoch == 1
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key]), key
    assert torch.equal(torch.get_rng_state(), states[0]["torch"])
