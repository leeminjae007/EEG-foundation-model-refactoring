#!/usr/bin/env bash
#SBATCH --job-name=a100-cuda-probe
#SBATCH --account=system
#SBATCH --partition=a100_short,a100_long
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/ablation/a100-cuda-probe-%j.out
#SBATCH --error=logs/ablation/a100-cuda-probe-%j.err
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source scripts/activate.sh
srun --ntasks=1 --gpus-per-task=a100:1 --gpu-bind=single:1 bash -c '
  echo "node=$SLURMD_NODENAME visible=$CUDA_VISIBLE_DEVICES"
  nvidia-smi --query-gpu=index,uuid,compute_mode,memory.used --format=csv,noheader
  python -c "import torch; print(\"torch\", torch.__version__, \"count\", torch.cuda.device_count(), \"name\", torch.cuda.get_device_name(0)); x=torch.empty((1024,1024),device=\"cuda:0\"); print(\"allocated\",x.numel())"
'
