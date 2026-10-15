#!/bin/bash

# Set variables
OUTPUT_PATH='/home/takataka_ju/data/mldata_lbi_lidar/'
#PARENT_PATH='/home/grace.bcolverd/esl_dev/tomosar-matlab'
PARENT_PATH='/home/takataka_ju/data/tomosar-matlab/'
#CLASSIFICATION_PARENT_PATH='/home/laura93schade/data/tree_species/'
CLASSIFICATION_PARENT_PATH='/home/takataka_ju/data/tree_species/'
LABEL='lidar' # lidar or class
BAND_OPT='L_Bi' # P, L_Mono, L_Bi, PL
CROP_LIDAR='yes'

python generate_tabular_data.py --output_path=$OUTPUT_PATH --parent_path=$PARENT_PATH --classification_parent_path=$CLASSIFICATION_PARENT_PATH --label=$LABEL --band_opt=$BAND_OPT --crop_lidar=$CROP_LIDAR
