#!/bin/bash
#SBATCH -A naiss2025-22-1584-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J train_minmax_32
#SBATCH -o train_minmax_32-%j.out
#SBATCH -e train_minmax_32-%j.err


set -euo pipefail

# ==============================
# Environment
# ==============================

module purge
module load GPU/Miniforge/26.3.2-2-eb
conda activate wetlands311
# ==============================
# Debug information
# ==============================

echo "========================================"
echo "Job ID:       $SLURM_JOB_ID"
echo "Node:         $(hostname)"
echo "Python:       $(which python)"
python --version

echo "GPU:"
nvidia-smi

python - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "========================================"


# ==============================
# Paths
# ==============================

DATA_ROOT=/nobackup/proj/flash/deep-wetlands-data-2025/personal/xuzu

OUTPUT_DIR=${DATA_ROOT}/Orebro_lan/2020/zscore_model_more_observations_16

mkdir -p "${OUTPUT_DIR}"

srun python train_predict.py \
    --images "${DATA_ROOT}/Orebro_lan/2020/train_data" \
    --output-dir "${OUTPUT_DIR}" \
    --channels 16,16 \
    --normalization zscore \
    --kernel-size 3 \
    --epochs 20 \
    --batch-size 16384 \
    --lr 5e-4 \
    --num-workers 8 \
    --use-recency-input \
    --half-life-days 12 \
    --use-wandb \
    --wandb-run-name Orebro_2020_zscore_16_more_observations

OUTPUT_DIR=${DATA_ROOT}/Orebro_lan/2018/zscore_model_more_observations_16

mkdir -p "${OUTPUT_DIR}"

srun python train_predict.py \
    --images "${DATA_ROOT}/Orebro_lan/2018/train_data" \
    --output-dir "${OUTPUT_DIR}" \
    --channels 16,16 \
    --normalization zscore \
    --kernel-size 3 \
    --epochs 20 \
    --batch-size 16384 \
    --lr 5e-4 \
    --num-workers 8 \
    --use-recency-input \
    --half-life-days 12 \
    --use-wandb \
    --wandb-run-name Orebro_2018_zscore_16_more_observations
