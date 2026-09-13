"""실행 경로와 실제 사용하는 CPU 검증 / CUDA 분산 학습 설정."""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def set_paths():
    # 소스를 import하기 전에 호출한다. 사용자 cache와 기존 run에 쓰지 않는다.
    paths = {"XDG_CACHE_HOME": "cache/xdg", "MPLCONFIGDIR": "cache/matplotlib",
             "TORCH_HOME": "cache/torch", "MNE_DONTWRITE_HOME": "true",
             "NUMBA_CACHE_DIR": "cache/numba", "TMPDIR": "tmp",
             "WANDB_DIR": "outputs/wandb", "WANDB_CACHE_DIR": "cache/wandb",
             "WANDB_DATA_DIR": "cache/wandb-data", "PIP_CACHE_DIR": "cache/pip"}
    for key, relative in paths.items():
        if key == "MNE_DONTWRITE_HOME":
            os.environ[key] = relative
        elif key == "TMPDIR" and os.environ.get("TMPDIR"):
            # Slurm launches may provide a short node-local path required by
            # multiprocessing AF_UNIX sockets; do not replace it with ROOT/tmp.
            Path(os.environ[key]).mkdir(parents=True, exist_ok=True)
        else:
            path = ROOT / relative
            path.mkdir(parents=True, exist_ok=True)
            os.environ[key] = str(path)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["PHYSIONET_MI_OUTPUT"] = str(ROOT / "processed/physionet-mi/processed_average")


def setup(device_name, distributed, seed, deterministic, benchmark, seed_by_rank):
    import random
    import numpy as np
    import torch
    import torch.distributed as dist

    rank = 0
    world = 1
    device = torch.device(device_name)
    if distributed:
        device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
        torch.cuda.set_device(device)
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world = dist.get_world_size()
    if seed_by_rank:
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark
    return device, rank, world
