"""Data-free forward/backward/optimizer and checkpoint checks for every preset."""

import argparse
import json
from ablation.bootstrap import ROOT, ensure_data_imports


def check(config, device):
    import io
    import torch
    from ablation.models import build_pretrain, parameter_report
    from src.modules.loss import reconstruction_loss
    from src.modules.masking import gather_targets

    torch.manual_seed(config["seed"])
    model = build_pretrain(config, device)
    channels = len(config["data"]["channel_names"])
    # Keep all original encoder layers; shorten only the synthetic recording.
    patches = 6
    signals = torch.randn(2, channels, patches * config["patch_encoder"]["patch_samples"], device=device)
    context = torch.rand(2, channels * patches, device=device).argsort(-1)[:, :channels * patches // 2]
    visible = torch.zeros(2, channels * patches, dtype=torch.bool, device=device).scatter_(1, context, True)
    visible = visible.reshape(2, channels, patches)
    target_mask = ~visible
    masks = {"context_mask": visible, "target_mask": target_mask, "target_blocks": target_mask[:, None],
             "target_token_valid": torch.ones(2, channels * patches // 2, dtype=torch.bool, device=device)}
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    prediction, _ = model(signals, masks)
    targets = gather_targets(signals.unfold(-1, model.patch_samples, model.patch_samples), masks["target_blocks"])
    loss = reconstruction_loss(prediction, targets, masks["target_token_valid"], 0.1)
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]
    if missing:
        raise AssertionError("Trainable parameters unused by backward (DDP would fail): " + str(missing))
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    model.eval()
    with torch.no_grad():
        expected, _ = model(signals, masks)
        perturbed = signals.unfold(-1, model.patch_samples, model.patch_samples).clone()
        perturbed[target_mask] = torch.randn_like(perturbed[target_mask]) * 100
        actual, _ = model(perturbed.flatten(2), masks)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0,
                                   msg="Hidden target signals leaked into prediction")
    buffer = io.BytesIO()
    torch.save({"model": model.state_dict(), "config": config}, buffer)
    buffer.seek(0)
    saved = torch.load(buffer, map_location=device, weights_only=False)
    restored = build_pretrain(saved["config"], device)
    restored.load_state_dict(saved["model"], strict=True)
    restored.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(signals, masks)[0], expected, rtol=0, atol=0)
    return {"name": config["ablation"]["name"], "loss": float(loss.detach()),
            "gradient_norm": float(norm), "prediction_shape": list(prediction.shape),
            "target_leakage": False, "strict_checkpoint_roundtrip": True,
            "parameters": parameter_report(model)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Omit to check all 10 presets")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", default="outputs/ablation/synthetic_checks.json")
    args = parser.parse_args()
    from src.training.runtime import set_paths
    set_paths()
    data_source = ensure_data_imports()
    import torch
    from ablation.config import load_config, resolve_ablation
    from ablation.sources import verify_sources

    torch.set_num_threads(2)
    source_report = verify_sources()
    paths = [ROOT / args.config] if args.config else sorted((ROOT / "ablation/configs").glob("*.yaml"))
    results = []
    for path in paths:
        result = check(resolve_ablation(load_config(path)), torch.device(args.device))
        results.append(result)
        print(json.dumps(result), flush=True)
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"torch": torch.__version__, "device": args.device, "synthetic": True,
                                 "data_source": data_source, "sources": source_report,
                                 "results": results}, indent=2), encoding="utf-8")
    print("Saved " + str(output), flush=True)


if __name__ == "__main__":
    main()
