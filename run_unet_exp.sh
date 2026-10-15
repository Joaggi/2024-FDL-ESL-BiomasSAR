#!/bin/bash

# Function to run the Python script with given settings
run_script() {
    local parent_path=$1
    local pol_ch="$2"
    local test_quadrant=$3
    local num_epochs=$4
    local lr=$5
    local batch_size=$6
    local patch_size=$7
    local stride=$8
    local model_name=$9
    local output_path="${10}"
    local project_name="${11}"
    local scaler_type="${12}"
    local batchnorm="${13}"
    local dropout_rate="${14}" 
    local loss_fn="${15}"
    
    python run_unet_GPU.py \
        --parent_path "$parent_path" \
        --pol_ch "$pol_ch" \
        --test_quadrant "$test_quadrant" \
        --num_epochs "$num_epochs" \
        --lr "$lr" \
        --batch_size "$batch_size" \
        --patch_size "$patch_size" \
        --stride "$stride" \
        --model_name "$model_name" \
        --output_path "$output_path" \
        --project_name "$project_name" \
        --scaler_type "$scaler_type" \
        --batchnorm "$batchnorm" \
        --dropout_rate "$dropout_rate" \
        --loss_fn "$loss_fn" 
}

# Set variables 
PARENT_PATH='/home/grace.bcolverd/esl_dev/tomosar-matlab'
POL_CHANNELS=('HV' "['HV','HH','VV']" 'VV' 'HV')
TEST_QUADRANTS=(3)
# TEST_QUADRANTS=(2 1 0)
NUM_EPOCHS=50
LR=0.0001
BATCH_SIZE=4
PATCH_SIZE="(16,16,36)"
STRIDE="(8,8,36)"
MODEL_NAMES=('simple_unet2')
OUTPUT_PATH="/home/grace.bcolverd/local/unet_models/2"
PROJECT_NAME="3dUnet_sweeps_simple_unet"
SCALER_TYPES=('minmax' 'standard')
BATCHNORM='yes'
DROPOUT_RATE=0
LOSSFN='mse'

# Run experiments with different settings
for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for TEST_QUAD in "${TEST_QUADRANTS[@]}"; do
        for POL_CH in "${POL_CHANNELS[@]}"; do
            for SCALER_TYPE in "${SCALER_TYPES[@]}"; do
                echo "Running script with MODEL_NAME=$MODEL_NAME, POL_CH=$POL_CH, TEST_QUAD=$TEST_QUAD, NUM_EPOCHS=$NUM_EPOCHS, LR=$LR, BATCH_SIZE=$BATCH_SIZE, PATCH_SIZE=$PATCH_SIZE, STRIDE=$STRIDE, OUTPUT_PATH=$OUTPUT_PATH, PROJECT_NAME=$PROJECT_NAME, SCALER_TYPE=$SCALER_TYPE"
                run_script "$PARENT_PATH" "$POL_CH" "$TEST_QUAD" "$NUM_EPOCHS" "$LR" "$BATCH_SIZE" "$PATCH_SIZE" "$STRIDE" "$MODEL_NAME" "$OUTPUT_PATH" "$PROJECT_NAME" "$SCALER_TYPE" "$BATCHNORM" "$DROPOUT_RATE" "$LOSSFN"
            done
        done
    done
done
