#!/bin/bash
#SBATCH -A naiss2025-22-1584-gpu
#SBATCH -p gpu
#SBATCH --gpus=1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -J run_test_all_configs
#SBATCH -o run_test_all_configs-%j.out
#SBATCH -e run_test_all_configs-%j.err
set -euo pipefail
shopt -s nullglob
module purge
module load GPU/Miniforge/26.3.2-2-eb
conda activate wetlands311
# ============================================================
# Fixed configuration and configuration grid
# ============================================================

DATA_ROOT=/nobackup/proj/flash/deep-wetlands-data-2025/personal/xuzu
TEST_DATA_ROOT=${DATA_ROOT}/test_data

# All final inference results are collected here for one-time download.
# Feature files remain beside the test data as intermediate results.
INFERENCE_ROOT=${DATA_ROOT}/inference_outputs_minmax_32

TRAIN_AREA=Orebro_lan

# Run all six training configurations in this order:
#   2018-d16, 2018-d32, 2018-d64,
#   2020-d16, 2020-d32, 2020-d64
TRAIN_YEARS=(2018 2020)
DIMS=(32)

N_CLUSTERS=2
HALF_LIFE_DAYS=6
BATCH_SIZE=8196
CHUNK_SIZE=262144
AREA_FILTER=${1:-}
YEAR_FILTER=${2:-}

# These variables are set for each TRAIN_YEAR/DIM combination below.
TRAIN_YEAR=
DIM=
TRAIN_ROOT=
TRAIN_MEAN=
TRAIN_STD=
MODEL_ROOT=
CHECKPOINT=
KMEANS_DIR=

set_training_configuration() {
    TRAIN_YEAR=$1
    DIM=$2

    TRAIN_ROOT=${DATA_ROOT}/${TRAIN_AREA}/${TRAIN_YEAR}
    TRAIN_MEAN=./${TRAIN_YEAR}/same_kmeans/global_sar_mean.npy
    TRAIN_STD=./${TRAIN_YEAR}/same_kmeans/global_sar_std.npy
    MODEL_ROOT=${TRAIN_ROOT}/minmax_model_${DIM}
    CHECKPOINT=${MODEL_ROOT}/best_tcn_encoder.pt
    KMEANS_DIR=${MODEL_ROOT}/global_kmeans_k${N_CLUSTERS}
}

# ============================================================
# Run one normalization model
# ============================================================
run_model() {
    local normalization=$1
    local checkpoint=$2
    local kmeans_dir=$3
    local image_dir=$4
    local reference_tif=$5
    local test_root=$6
    local date_id=$7
    local test_area=$8

    local feature_dir
    local output_dir
    local semantics_json

    feature_dir=${test_root}/mosaic/${date_id}_${TRAIN_YEAR}_${normalization}_h${HALF_LIFE_DAYS}_d${DIM}
    output_dir=${INFERENCE_ROOT}/${test_area}_${date_id}_train${TRAIN_YEAR}_d${DIM}_k${N_CLUSTERS}_${normalization}
    semantics_json=${kmeans_dir}/otsu_semantics/water_cluster_semantics.json

    echo
    echo "------------------------------------------------------------"
    echo "Normalization: ${normalization}"
    echo "Image input:   ${image_dir}"
    echo "Feature dir:   ${feature_dir}"
    echo "Output dir:    ${output_dir}"
    echo "------------------------------------------------------------"

    if [[ ! -f "${checkpoint}" ]]; then
        echo "ERROR: checkpoint not found:"
        echo "  ${checkpoint}"
        return 1
    fi

    if [[ ! -d "${kmeans_dir}" ]]; then
        echo "ERROR: KMeans directory not found:"
        echo "  ${kmeans_dir}"
        return 1
    fi

    if [[ ! -f "${semantics_json}" ]]; then
        echo "ERROR: semantics JSON not found:"
        echo "  ${semantics_json}"
        return 1
    fi

    mkdir -p "${feature_dir}" "${output_dir}"

    python3 extract_feature_mem_old.py \
        --images "${image_dir}" \
        --output-dir "${feature_dir}" \
        --checkpoint "${checkpoint}" \
        --train-mean "${TRAIN_MEAN}" \
        --train-std "${TRAIN_STD}" \
        --normalization "${normalization}" \
        --feature-mode last \
        --batch-size "${BATCH_SIZE}" \
        --use-recency-input \
        --half-life-days "${HALF_LIFE_DAYS}"

    python3 predict_final_with_trained_kmeans.py \
        --feature-dir "${feature_dir}" \
        --kmeans-dir "${kmeans_dir}" \
        --semantics-json "${semantics_json}" \
        --output-dir "${output_dir}" \
        --reference-tif "${reference_tif}" \
        --chunk-size "${CHUNK_SIZE}" \
        --save-tif \
        --save-preview

    echo "Finished: ${date_id}, ${normalization}"
}

# ============================================================
# Process one date directory
# ============================================================
process_date_directory() {
    local image_dir=$1

    local date_folder
    local year_dir
    local year_folder
    local area_dir
    local test_area
    local test_year
    local test_month
    local test_date
    local date_id
    local reference_pattern
    local reference_files
    local reference_tif
    local date_mmdd

    date_folder=$(basename "${image_dir}")
    year_dir=$(dirname "${image_dir}")
    year_folder=$(basename "${year_dir}")
    area_dir=$(dirname "${year_dir}")
    test_area=$(basename "${area_dir}")

    # Only accept folders such as 0401
    if [[ "${date_folder}" =~ ^([0-9]{4})$ ]]; then
        date_mmdd=${BASH_REMATCH[1]}
    else
        return
    fi

    # Extract the year from hjalstaviken_2019
    test_year=${year_folder##*_}

    if [[ ! "${test_year}" =~ ^[0-9]{4}$ ]]; then
        echo "Skipping invalid year directory: ${year_dir}"
        return
    fi

    # Confirm that the parent directory follows <area>_<year>
    if [[ "${year_folder}" != "${test_area}_${test_year}" ]]; then
        echo "Skipping unexpected directory structure: ${image_dir}"
        return
    fi

    # Apply optional filters
    if [[ -n "${AREA_FILTER}" && "${test_area}" != "${AREA_FILTER}" ]]; then
        return
    fi

    if [[ -n "${YEAR_FILTER}" && "${test_year}" != "${YEAR_FILTER}" ]]; then
        return
    fi

    test_month=${date_mmdd:0:2}
    test_date=${date_mmdd:2:2}
    date_id=${test_year}${test_month}${test_date}

    # Automatically find the reference SAR image
    reference_pattern="${image_dir}"/*_mosaic_"${test_year}-${test_month}-${test_date}"_sar_VH.tif
    # reference_pattern="${image_dir}"/*_"${test_year}-${test_month}-${test_date}".tif
    reference_files=(${reference_pattern})

    if [[ ${#reference_files[@]} -eq 0 ]]; then
        echo
        echo "WARNING: reference SAR image not found, skipping:"
        echo "  ${image_dir}"
        return
    fi

    if [[ ${#reference_files[@]} -gt 1 ]]; then
        echo
        echo "WARNING: multiple reference images found, skipping:"
        printf "  %s\n" "${reference_files[@]}"
        return
    fi

    reference_tif=${reference_files[0]}

    echo
    echo "============================================================"
    echo "Area:          ${test_area}"
    echo "Year:          ${test_year}"
    echo "Date:          ${test_year}-${test_month}-${test_date}"
    echo "Train config:  train${TRAIN_YEAR}_d${DIM}"
    echo "Image dir:     ${image_dir}"
    echo "Reference TIF: ${reference_tif}"
    echo "============================================================"

    # run_model \
    #     "train_zscore" \
    #     "${CHECKPOINT}" \
    #     "${KMEANS_DIR}" \
    #     "${image_dir}" \
    #     "${reference_tif}" \
    #     "${year_dir}" \
    #     "${date_id}" \
    #     "${test_area}"

    # Z-score model
    # run_model \
    #     "zscore" \
    #     "${CHECKPOINT}" \
    #     "${KMEANS_DIR}" \
    #     "${image_dir}" \
    #     "${reference_tif}" \
    #     "${year_dir}" \
    #     "${date_id}" \
    #     "${test_area}"

    # # Min-max model
    run_model \
        "minmax" \
        "${TRAIN_ROOT}/minmax_model_32/best_tcn_encoder.pt" \
        "${TRAIN_ROOT}/minmax_model_32/global_kmeans_k${N_CLUSTERS}" \
        "${image_dir}" \
        "${reference_tif}" \
        "${year_dir}" \
        "${date_id}" \
        "${test_area}"
}

# ============================================================
# Validate common files and all six training configurations
# ============================================================

if [[ ! -d "${TEST_DATA_ROOT}" ]]; then
    echo "ERROR: test data root not found:"
    echo "  ${TEST_DATA_ROOT}"
    exit 1
fi

mkdir -p "${INFERENCE_ROOT}"

for train_year in "${TRAIN_YEARS[@]}"; do
    for dim in "${DIMS[@]}"; do
        set_training_configuration "${train_year}" "${dim}"

        echo "Validating train${TRAIN_YEAR}_d${DIM} ..."

        if [[ ! -f "${TRAIN_MEAN}" ]]; then
            echo "ERROR: training mean not found:"
            echo "  ${TRAIN_MEAN}"
            exit 1
        fi

        if [[ ! -f "${TRAIN_STD}" ]]; then
            echo "ERROR: training std not found:"
            echo "  ${TRAIN_STD}"
            exit 1
        fi

        if [[ ! -f "${CHECKPOINT}" ]]; then
            echo "ERROR: checkpoint not found:"
            echo "  ${CHECKPOINT}"
            exit 1
        fi

        if [[ ! -d "${KMEANS_DIR}" ]]; then
            echo "ERROR: KMeans directory not found:"
            echo "  ${KMEANS_DIR}"
            exit 1
        fi

        if [[ ! -f "${KMEANS_DIR}/otsu_semantics/water_cluster_semantics.json" ]]; then
            echo "ERROR: semantics JSON not found:"
            echo "  ${KMEANS_DIR}/otsu_semantics/water_cluster_semantics.json"
            exit 1
        fi
    done
done

# ============================================================
# Automatically discover all <area>/<area_year>/<MMDD> folders
# ============================================================

configuration_count=0

for train_year in "${TRAIN_YEARS[@]}"; do
    for dim in "${DIMS[@]}"; do
        set_training_configuration "${train_year}" "${dim}"

        echo
        echo "############################################################"
        echo "Starting configuration: train${TRAIN_YEAR}_d${DIM}"
        echo "Checkpoint:             ${CHECKPOINT}"
        echo "KMeans:                 ${KMEANS_DIR}"
        echo "############################################################"

        found_count=0

        while IFS= read -r -d '' image_dir; do
            date_folder=$(basename "${image_dir}")

            if [[ "${date_folder}" =~ ^[0-9]{4}$ ]]; then
                process_date_directory "${image_dir}"
                found_count=$((found_count + 1))
            fi
        done < <(
            find "${TEST_DATA_ROOT}" \
                -mindepth 3 \
                -maxdepth 3 \
                -type d \
                -print0 |
            sort -z
        )

        if [[ ${found_count} -eq 0 ]]; then
            echo "No MMDD test directories were found under:"
            echo "  ${TEST_DATA_ROOT}"
            exit 1
        fi

        configuration_count=$((configuration_count + 1))
        echo "Completed configuration: train${TRAIN_YEAR}_d${DIM}"
    done
done

echo
echo "============================================================"
echo "All matching test directories have been processed."
echo "Completed configurations: ${configuration_count}"
echo "All final inference results: ${INFERENCE_ROOT}"
echo "============================================================"
