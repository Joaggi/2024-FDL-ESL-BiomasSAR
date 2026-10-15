import numpy as np
from fdlbiomass.utils import load_biomass_data, merge_headings, crop_to_non_nan, split_into_quadrants
from fdlbiomass.dataloader_fns import check_pol_channels
from fdlbiomass.flatten_UTM_cube import lidar_cube_to_table
from src.fdlbiomass.flatten_UTM_cube import generate_flattened_intensity_and_lidar, convert_mldata_union_polarimetry
import os
import pandas as pd

def generate_quadrants(intensity_cube, LidarCHM, coordinate_array):
    # Crop LidarCHM
    cropped_LidarCHM = crop_to_non_nan(LidarCHM)
    print(f"Original shape: {LidarCHM.shape}")
    print(f"Cropped shape: {cropped_LidarCHM.shape}")

    row_start = np.argmax(~np.all(np.isnan(LidarCHM), axis=1))
    row_end = len(LidarCHM) - np.argmax(~np.all(np.isnan(LidarCHM), axis=1)[::-1])
    col_start = np.argmax(~np.all(np.isnan(LidarCHM), axis=0))
    col_end = len(LidarCHM[0]) - np.argmax(~np.all(np.isnan(LidarCHM), axis=0)[::-1])
    # Crop Intensity
    crop_int = intensity_cube[row_start:row_end, col_start:col_end, ...]
    # Split the array
    coordinate_array = coordinate_array[row_start:row_end, col_start:col_end]

    quadrants, ground_truths, coordinate_arrays = split_into_quadrants(crop_int, cropped_LidarCHM, coordinate_array)
    
    return quadrants, ground_truths, (crop_int, cropped_LidarCHM), coordinate_arrays

def generate_dataloaders(quadrants, ground_truths, coordinate_arrays, label, band_opt='P', test_quadrant=3, sep_pol_file=None, union_pol_file=None, output_path=None):

    quad_options = [0, 1, 2, 3]
    if test_quadrant not in quad_options:
        raise ValueError('Choose index 0-3 for test')
    
    # Remove test_quadrant from options
    remaining_quadrants = [x for x in quad_options if x != test_quadrant]

    lidar_dfs = []
    data_sep_pol = []
    for i in quad_options:
        lidar_dfs.append(lidar_cube_to_table(ground_truths[i], coordinate_array=coordinate_arrays[i]))
        data_sep_pol.append(generate_flattened_intensity_and_lidar(quadrants[i], band_opt, lidar_dfs[i], filename=f"{sep_pol_file.split('.csv')[0]}_cuadrant_{i}.csv", coordinate_array=coordinate_arrays[i]))
        convert_mldata_union_polarimetry(data_sep_pol[i], label, filename=f"{union_pol_file.split('.csv')[0]}_cuadrant_{i}.csv")

    train_sep_pol_file = os.path.join(output_path , f'quadrants_mldata_separate_pol__train_data__thick_0__testsize_25.csv')
    test_sep_pol_file = os.path.join(output_path , f'quadrants_mldata_separate_pol__test_data__thick_0__testsize_25.csv')
    train_union_pol_file = os.path.join(output_path , f'quadrants_mldata_union_pol__train_data__thick_0__testsize_25.csv')
    test_union_pol_file = os.path.join(output_path , f'quadrants_mldata_union_pol__test_data__thick_0__testsize_25.csv')

    train_data_sep_pol = []
    for i in remaining_quadrants:
        train_data_sep_pol.append(data_sep_pol[i])
    train_data_sep_pol = pd.concat(train_data_sep_pol)
    train_data_sep_pol.to_csv(train_sep_pol_file,  index=False )
    convert_mldata_union_polarimetry(train_data_sep_pol, label, filename=train_union_pol_file)

    data_sep_pol[test_quadrant].to_csv( test_sep_pol_file,  index=False )
    convert_mldata_union_polarimetry(data_sep_pol[test_quadrant], label, filename=test_union_pol_file)

    return train_data_sep_pol, data_sep_pol[test_quadrant] 
