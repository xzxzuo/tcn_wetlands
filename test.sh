#!/bin/bash
set -euo pipefail

PROJECT_DIR="/cephyr/users/xuzu/Alvis/DeepWetlands/tcn_wetlands"
TESTDATA_ROOT="/mimer/NOBACKUP/groups/deep-wetlands-data-2025/xzuo/test_data"
TRAINING_AREA="orebro"
TRAINING_YEAR=2020
DIM=64
CHECKPOINT_DIR="gaussian_h6_${DIM}"
CHECKPOINT="/mimer/NOBACKUP/groups/deep-wetlands-data-2025/xzuo/Orebro_lan/${TRAINING_YEAR}/${CHECKPOINT_DIR}/best_tcn_encoder.pt"

AREAS=(
  "hjalstaviken"
  "hornborgasjon"
  # "svartadalen"
)

YEARS=(
  "2018"
  "2019"
  # "2020"
  # "2021"
  # "2022"
)

K_LIST=(
  2
  3
  4
)

EXTRACT_BATCH_SIZE=4096
NUM_WORKERS=2
FEATURE_MODE="last"
HALF_LIFE_DAYS=6

CLUSTER_BATCH_SIZE=8192
FIT_EPOCHS=5

cd "$PROJECT_DIR"

echo "Project dir: $PROJECT_DIR"
echo "Data root: $TESTDATA_ROOT"
echo "Checkpoint: $CHECKPOINT"
echo

for AREA in "${AREAS[@]}"; do
  for YEAR in "${YEARS[@]}"; do

    YEAR_DIR="${TESTDATA_ROOT}/${AREA}/${AREA}_${YEAR}"

    if [ ! -d "$YEAR_DIR" ]; then
      echo "[Skip] Missing year dir: $YEAR_DIR"
      continue
    fi

    echo "========================================"
    echo "Area: $AREA"
    echo "Year: $YEAR"
    echo "Year dir: $YEAR_DIR"
    echo "========================================"

    for DATE_DIR in "$YEAR_DIR"/*/; do
      [ -d "$DATE_DIR" ] || continue

      DATE_NAME=$(basename "$DATE_DIR")

      if [[ ! "$DATE_NAME" =~ ^[0-9]{4}$ ]]; then
        echo "[Skip] Not a date folder: $DATE_DIR"
        continue
      fi

      DATE_ID="${YEAR}${DATE_NAME}"
      OUTPUT_BASE="${TESTDATA_ROOT}/${AREA}/${DATE_ID}"

      echo
      echo "---------- Processing ${AREA} ${YEAR} ${DATE_NAME} ----------"

      # Collect sar images
      mapfile -t IMAGES < <(
        find "$DATE_DIR" -maxdepth 1 -type f \( -name "*.tif" -o -name "*.tiff" \) | sort
      )

      if [ "${#IMAGES[@]}" -eq 0 ]; then
        echo "[Skip] No tif images found in $DATE_DIR"
        continue
      fi

      echo "Found ${#IMAGES[@]} tif images."

      FEATURE_DIR="${OUTPUT_BASE}_extracted_features/gaussian_h${HALF_LIFE_DAYS}_on_${TRAINING_AREA}${TRAINING_YEAR}${DIM}"
      mkdir -p "$FEATURE_DIR"

      if [ -f "${FEATURE_DIR}/features.npy" ] && [ -f "${FEATURE_DIR}/coords.npy" ] && [ -f "${FEATURE_DIR}/metadata.npz" ]; then
        echo "[Skip extract] Features already exist: $FEATURE_DIR"
      else
        echo "[Extract] Saving features to: $FEATURE_DIR"

        python extract_feature_mem.py \
          --images "${IMAGES[@]}" \
          --checkpoint "$CHECKPOINT" \
          --output-dir "$FEATURE_DIR" \
          --batch-size "$EXTRACT_BATCH_SIZE" \
          --num-workers "$NUM_WORKERS" \
          --feature-mode "$FEATURE_MODE" \
          --use-recency-input \
          --half-life-days "$HALF_LIFE_DAYS"
      fi

      for K in "${K_LIST[@]}"; do
        CLUSTER_DIR="${OUTPUT_BASE}/k${K}_h${HALF_LIFE_DAYS}_on_${TRAINING_AREA}${TRAINING_YEAR}${DIM}"

        if [ -f "${CLUSTER_DIR}/cluster_map.npy" ]; then
          echo "[Skip cluster] K=${K} already exists: $CLUSTER_DIR"
          continue
        fi

        echo "[Cluster] K=${K}, output: $CLUSTER_DIR"

        python cluster_features_memmap.py \
          --feature-dir "$FEATURE_DIR" \
          --output-dir "$CLUSTER_DIR" \
          --n-clusters "$K" \
          --batch-size "$CLUSTER_BATCH_SIZE" \
          --fit-epochs "$FIT_EPOCHS" \
          --standardize \
          --save-tif
      done

      echo "Finished ${AREA} ${YEAR} ${DATE_NAME}"
    done
  done
done

echo
echo "All tests finished."