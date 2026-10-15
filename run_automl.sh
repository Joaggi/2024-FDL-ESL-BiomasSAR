#!/bin/bash

# Function to run the Python script with given settings
run_script() {
    TIME_LIMIT=$1
    MODEL_PRESET=$2
    TRAIN_SUBSET=$3
    OUTPUT_PATH=$4
    POLARIMETRY_UNION=$5
    INPUT_DATA_FOLDER=$6
    GEO_SPLIT=$7
    TARGET=$8
    BAND=$9
    STANDARIZATION=${10}

    python run_automl.py \
        --time_limit "$TIME_LIMIT" \
        --model_preset "$MODEL_PRESET" \
        --train_subset_prop "$TRAIN_SUBSET" \
        --output_path "$OUTPUT_PATH" \
        --pol_union "$POLARIMETRY_UNION" \
        --input_folder "$INPUT_DATA_FOLDER" \
        --geo_name "$GEO_SPLIT" \
        --label "$TARGET" \
        --band_opt "$BAND" \
        --standarization "$STANDARIZATION"
}

# Define different settings for the experiments
TIME_LIMIT=15000
OUTPUT_PATH="/home/takataka_ju/data/output_lbi_lidar_swath_standard" #for classification
#OUTPUT_PATH="/home/laura93schade/local/output" #for lidar
INPUT_DATA_FOLDER="/home/takataka_ju/data/mldata_lbi_lidar" #for classification
#INPUT_DATA_FOLDER="/home/laura93schade/local/mldata" #for lidar
GEO_SPLITS=('swath')

# MODEL_PRESETS=("medium_quality"  "best_quality")
MODEL_PRESETS=("best_quality") # medium_quality or best_quality
# List of train subset proportions to test
TRAIN_SUBSETS=(1)
POLARIMETRY_UNIONs=('no')
TARGET='lidar' # lidar or class
BAND='L_Bi' # P, L_Mono, L_Bi, PL
STANDARIZATION='standard' # standard or minmax

# Run experiments with different settings
for MODEL_PRESET in "${MODEL_PRESETS[@]}"; do
    for TRAIN_SUBSET in "${TRAIN_SUBSETS[@]}"; do
        for GEO_SPLIT in "${GEO_SPLITS[@]}"; do
            for POLARIMETRY_UNION in "${POLARIMETRY_UNIONs[@]}"; do
                echo "Running script with MODEL_PRESET=$MODEL_PRESET, TRAIN_SUBSET=$TRAIN_SUBSET, GEO_SPLIT=$GEO_SPLIT, POLARIMETRY_UNION=$POLARIMETRY_UNION BAND=$BAND STANDARIZATION=$STANDARIZATION"
                run_script $TIME_LIMIT $MODEL_PRESET $TRAIN_SUBSET $OUTPUT_PATH $POLARIMETRY_UNION $INPUT_DATA_FOLDER $GEO_SPLIT $TARGET $BAND $STANDARIZATION
            done
        done
    done
done
