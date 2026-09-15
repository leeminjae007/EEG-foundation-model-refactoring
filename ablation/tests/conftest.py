from ablation.bootstrap import ensure_data_imports

ensure_data_imports()

import torch

torch.set_num_threads(2)
