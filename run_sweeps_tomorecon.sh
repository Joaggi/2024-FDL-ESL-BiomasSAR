#!/bin/bash

# Directory containing the config files
CONFIG_DIR="/home/grace.bcolverd/2024-ESL-BiomassSAR/src/sweep_configs/tomo_sweeps"

# Number of runs for each sweep
NUM_RUNS=100

# Loop through all YAML files in the config directory
for config_file in "$CONFIG_DIR"/*.yaml; do
    if [ -f "$config_file" ]; then
        echo "Running sweep with config: $config_file"
        python train_tomo_recon.py --project_name tomo_recon_height --num_runs $NUM_RUNS --config "$config_file"
        echo "Completed sweep with config: $config_file"
        echo "------------------------------------"
    fi
done

echo "All sweeps completed."