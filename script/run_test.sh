#!/bin/bash
set -euo pipefail
shopt -s nullglob

# ============================================================
# Fixed configuration
# ============================================================

DATA_ROOT=/mimer/NOBACKUP/groups/deep-wetlands-data-2025/xzuo
TEST_DATA_ROOT=${DATA_ROOT}/test_data

TRAIN_AREA=Orebro_lan
TRAIN_YEAR=2020
TRAIN_ROOT=${DATA_ROOT}/${TRAIN_AREA}/${TRAIN_YEAR}

N_CLUSTERS=2
HALF_LIFE_DAYS=6
BATCH_SIZE=8196
CHUNK_SIZE=262144
DIM=64
TRAIN_MEAN=2020/march_to_june/global_sar_mean.npy
TRAIN_STD=2020/march_to_june/global_sar_std.npy
AREA_FILTER=${1:-}
YEAR_FILTER=${2:-}
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

    local feature_dir
    local output_dir
    local semantics_json

    feature_dir=${test_root}/mosaic/${date_id}_${TRAIN_YEAR}_${normalization}_h${HALF_LIFE_DAYS}_d${DIM}
    output_dir=${test_root}/mosaic/${date_id}/${TRAIN_YEAR}_d${DIM}/prediction_k${N_CLUSTERS}_${normalization}
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

    # Only accept folders such as 0401_no_mosaic
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
    echo "Image dir:     ${image_dir}"
    echo "Reference TIF: ${reference_tif}"
    echo "============================================================"

    run_model \
        "train_zscore" \
        "${TRAIN_ROOT}/zscore_model_more_64/best_tcn_encoder.pt" \
        "${TRAIN_ROOT}/zscore_model_more_64/global_kmeans_k${N_CLUSTERS}" \
        "${image_dir}" \
        "${reference_tif}" \
        "${year_dir}" \
        "${date_id}"
    # Z-score model
    run_model \
        "zscore" \
        "${TRAIN_ROOT}/zscore_model_more_64/best_tcn_encoder.pt" \
        "${TRAIN_ROOT}/zscore_model_more_64/global_kmeans_k${N_CLUSTERS}" \
        "${image_dir}" \
        "${reference_tif}" \
        "${year_dir}" \
        "${date_id}"

    # # Min-max model
    # run_model \
    #     "minmax" \
    #     "${TRAIN_ROOT}/training_from_march_to_june/best_tcn_encoder.pt" \
    #     "${TRAIN_ROOT}/training_from_march_to_june/global_kmeans_k${N_CLUSTERS}" \
    #     "${image_dir}" \
    #     "${reference_tif}" \
    #     "${year_dir}" \
    #     "${date_id}"
}


# ============================================================
# Validate common files
# ============================================================

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

if [[ ! -d "${TEST_DATA_ROOT}" ]]; then
    echo "ERROR: test data root not found:"
    echo "  ${TEST_DATA_ROOT}"
    exit 1
fi


# ============================================================
# Automatically discover all <area>/<area_year>/<MMDD> folders
# ============================================================

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

echo
echo "============================================================"
echo "All matching test directories have been processed."
echo "Discovered date directories: ${found_count}"
echo "============================================================"