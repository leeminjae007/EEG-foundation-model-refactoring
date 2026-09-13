"""Sampled, detached model diagnostics without changing training RNG or outputs."""

import hashlib
import json
from pathlib import Path

import torch


def set_model_diagnostics(model, enabled):
    for module in model.modules():
        if hasattr(module, 'capture_diagnostics'):
            module.capture_diagnostics = bool(enabled)
            if not enabled:
                module.last_model_diagnostics = {}


def collect_model_diagnostics(model):
    output = {}
    for name, module in model.named_modules():
        for key, value in getattr(module, 'last_model_diagnostics', {}).items():
            output[f'model/{name}/{key}'] = value
    return output


@torch.no_grad()
def attention_diagnostics(qk, bias, probabilities, valid_mask, sample_limit=32):
    """Per-head pre-dropout attention on evenly spaced axis rows, valid pairs only."""
    indices = torch.linspace(0, qk.shape[0] - 1, min(sample_limit, qk.shape[0]),
                             device=qk.device).long()
    qk = qk[indices].detach().float()
    probabilities = probabilities[indices].detach().float()
    valid = valid_mask[indices]
    if bias is None:
        bias_values = torch.zeros_like(qk)
    else:
        if bias.ndim == 3:
            bias = bias.unsqueeze(0)
        bias_values = (bias if bias.shape[0] == 1 else bias[indices]).detach().float().expand_as(qk)
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
        'attention/bias_present': qk.new_tensor(float(bias is not None)),
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


@torch.no_grad()
def gate_diagnostics(model, include_gradients=True):
    output = {}
    terms = ('fusion_gates', 'spatial_alpha', 'temporal_beta', 'temporal_rel_scale',
             'degree_weights', 'spatial_gate', 'temporal_gate')
    for name, parameter in model.named_parameters():
        if not any(term in name for term in terms):
            continue
        # Scalar views must not change when optimizer.step mutates parameters.
        value = parameter.detach().float().clone()
        prefix = f'gate/{name}'
        output[f'{prefix}/value_rms'] = value.square().mean().sqrt()
        output[f'{prefix}/value_mean'] = value.mean()
        if parameter.numel() <= 16:
            for index, item in enumerate(value.reshape(-1)):
                output[f'{prefix}/value_{index}'] = item
        if include_gradients:
            output[f'{prefix}/gradient_present'] = value.new_tensor(float(parameter.grad is not None))
            if parameter.grad is not None:
                # clip_grad_norm_ mutates gradients in place after this snapshot.
                gradient = parameter.grad.detach().float().clone()
                output[f'{prefix}/gradient_rms_pre_clip'] = gradient.square().mean().sqrt()
                if parameter.numel() <= 16:
                    for index, item in enumerate(gradient.reshape(-1)):
                        output[f'{prefix}/gradient_{index}_pre_clip'] = item
    return output


def initialization_fingerprints(models):
    """Hash actual post-factory tensors, including new-module names and shapes."""
    tensors = {}
    for prefix, model in models.items():
        for name, value in model.state_dict().items():
            tensor = value.detach().cpu().contiguous()
            payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
            tensors[f'{prefix}.{name}'] = {
                'shape': list(tensor.shape), 'dtype': str(tensor.dtype),
                'sha256': hashlib.sha256(payload).hexdigest(),
            }
    return {'schema_version': 1, 'tensors': tensors}


def compare_initialization(actual, reference):
    a, b = actual['tensors'], reference['tensors']
    shared = sorted(set(a) & set(b))
    comparable = [name for name in shared if a[name]['shape'] == b[name]['shape']]
    different = [name for name in comparable if a[name] != b[name]]
    return {
        'reference_available': True,
        'common_shape_matched_tensors': len(comparable),
        'all_common_tensors_equal': bool(comparable) and not different,
        'different_common_tensors': different,
        'shape_mismatched_tensors': [name for name in shared if name not in comparable],
        'new_tensors': sorted(set(a) - set(b)),
        'reference_only_tensors': sorted(set(b) - set(a)),
    }


def write_initialization_audit(models, path, reference_path=None, require_match=False, reference_report=None):
    report = initialization_fingerprints(models)
    report['comparison'] = {'reference_available': False,
                            'reason': 'No historical step-zero reference configured; hashes retained for cross-run comparison.'}
    if reference_path or reference_report is not None:
        reference = reference_report if reference_report is not None else json.loads(Path(reference_path).read_text())
        report['comparison'] = compare_initialization(report, reference)
        report['reference_path'] = str(reference_path) if reference_path else 'same-device factory replay with identical initial RNG'
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    if require_match and not report['comparison'].get('all_common_tensors_equal', False):
        raise ValueError(f'Common initialization mismatch; see {path}')
    return report
