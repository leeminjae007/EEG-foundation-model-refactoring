"""동결 원본에서 GR9-1 가중치 + 기본 geometry 마스크의 forward/backward를 실행한다."""
from pathlib import Path
import sys
import torch
import yaml

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / 'tests/reference'))
from src.models.factory import build_models
from src.utils.masking import build_masking_policy
from src.utils.reconstruction import gather_target_blocks
from src.utils.reconstruction_loss import reconstruction_losses
from src.datasets.pretraining_dataset import PretrainingDataset

torch.set_num_threads(2)
checkpoint = torch.load(root / 'tests/reference/checkpoint-epoch-0040.pth', map_location='cpu')
config = checkpoint['resolved_config']
encoder, decoder = build_models(config, torch.device('cpu'))
encoder.load_state_dict(checkpoint['context_encoder'], strict=True)
decoder.load_state_dict(checkpoint['decoder'], strict=True)
dataset = PretrainingDataset(config['resolved']['dataset_dir'], 19, 30, 200)
signals = torch.stack([dataset[0][0], dataset[1][0]]) / config['data']['value_scale']
dataset.close()
masks = torch.load(root / 'outputs/geometry_reve_masks.pth', map_location='cpu')
target = gather_target_blocks(signals.unfold(-1, 200, 200), masks['target_blocks'])
encoder.train()
decoder.train()
torch.manual_seed(629)
rng = torch.get_rng_state().clone()
context = encoder(signals, visible_mask=masks['context_mask'])
prediction = decoder(context, masks['context_mask'], masks['target_mask'], target_blocks=masks['target_blocks'])
loss = reconstruction_losses(prediction, target, masks['target_token_valid'],
                             waveform_loss='smooth_l1', smooth_l1_beta=0.1)['total_loss']
loss.backward()
gradients = {}
for prefix, model in [('context_encoder', encoder), ('decoder', decoder)]:
    for name, parameter in model.named_parameters():
        gradients[prefix + '.' + name] = parameter.grad.detach().clone()
torch.save({'signals': signals, 'masks': masks, 'target': target,
            'prediction': prediction.detach(), 'loss': loss.detach(), 'gradients': gradients,
            'rng_before': rng, 'rng_after': torch.get_rng_state()},
           root / 'outputs/geometry_reve_oracle.pth')
