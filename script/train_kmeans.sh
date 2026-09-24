#!/bin/bash
#SBATCH -A naiss2025-22-1584-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 01:00:00
#SBATCH -J train_kmeans
#SBATCH -o train_kmeans-%j.out
#SBATCH -e train_kmeans-%j.err
set -euo pipefail
module purge
module load GPU/Miniforge/26.3.2-2-eb
conda activate wetlands311
DATA_ROOT=/nobackup/proj/flash/deep-wetlands-data-2025/personal/xuzu/Orebro_lan/2018
N_CLUSTERS=2

python extract_feature_mem.py \
    --images ${DATA_ROOT}/train_data/ \
    --checkpoint ${DATA_ROOT}/minmax_model_32/best_tcn_encoder.pt \
    --output-dir ${DATA_ROOT}/minmax_model_32/all_time_features \
    --batch-size 131072 \
    --use-recency-input \
    --half-life-days 12 \
    --normalization minmax

python3 train_global_kmeans.py \
  --features ${DATA_ROOT}/minmax_model_32/all_time_features/features.npy \
  --output-dir ${DATA_ROOT}/minmax_model_32/global_kmeans_k${N_CLUSTERS}   \
  --k ${N_CLUSTERS}   \
  --samples-per-time 10000000   \
  --sample-block-size 65536   \
  --batch-size 65536   --epochs 3   --seed 42

python3 assign_water_semantics_from_otsu.py   \
  --feature-dir   ${DATA_ROOT}/minmax_model_32/all_time_features   \
  --kmeans-dir   ${DATA_ROOT}/minmax_model_32/global_kmeans_k${N_CLUSTERS}   \
  --output-dir   ${DATA_ROOT}/minmax_model_32/global_kmeans_k${N_CLUSTERS}/otsu_semantics   \
  --gaussian-sigma 1.0   \
  --selection-metric mean_date_iou   
  # --save-preview