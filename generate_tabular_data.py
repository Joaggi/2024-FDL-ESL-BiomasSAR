from src.fdlbiomass.utils import load_biomass_data, load_classification_tree_data, crop_lidar_edge_zone, merge_headings
from src.fdlbiomass.geosplit_tabular import gen_square_swath
from src.fdlbiomass.preprocessing.generate_quadrants import generate_quadrants, generate_dataloaders
from src.fdlbiomass.flatten_UTM_cube import generate_flattened_intensity_and_lidar, generate_flattened_intensity_and_lidar_PL, generate_flattened_intensity_and_class, generate_flattened_intensity_and_class_PL, lidar_cube_to_table, convert_mldata_union_polarimetry, classification_tree_image_to_table
import pandas as pd 
import os 
import sys
import argparse
import numpy as np 



def generate_ml_data(parent_path, band_opt, output_path, class_parent_path, label, crop_lidar, regenerate_data, type):
    """ Convert the UTM power cube into flat data for tabular ML 
    Inputs:
    parent path: location of input data e.g. contains the P or L folders 
    band_opt: which band : P, L, PL 
    output_path: where should processed data be stored
    """
    print('Starting to generate flattened ML data')
    sep_pol_file = os.path.join(output_path, f'{band_opt}_{label}_seperate_pol_ml_data.csv') #should this be 'separate'
    union_pol_file = os.path.join(output_path , f'{band_opt}_{label}_union_pol_ml_data.csv')
 
   
   
    if (not regenerate_data) and os.path.exists(sep_pol_file) and os.path.exists(union_pol_file):
        print('Data files already exist. Skipping generation.')
        data_sep_pol = pd.read_csv(sep_pol_file)
        data_union_pol = pd.read_csv(union_pol_file)
        return data_sep_pol, data_union_pol
    elif regenerate_data:
        print('Starting to regenerate flattened ML data')
    


    data = load_biomass_data(parent_path, band_opt, crop_lidar)

    

    if crop_lidar == 'yes' and label=="lidar":
        LidarCHM_original = data['LidarCHM']
        LidarCHM = crop_lidar_edge_zone(LidarCHM_original)
    else:
        LidarCHM = data['LidarCHM']

    # Access the returned objects
    if band_opt == 'P':
        LidarCHM = data['LidarCHM']
        PcubeDeRampedUTM = data['PcubeDeRampedUTM']
    elif band_opt == 'L_Mono':
        PcubeDeRampedDeBiasedUTM = data['PcubeDeRampedDeBiasedUTM']
        PcubeDeRampedUTM = PcubeDeRampedDeBiasedUTM[:,:,:,:,0,:]
    elif band_opt == 'L_Bi':
        PcubeDeRampedDeBiasedUTM = data['PcubeDeRampedDeBiasedUTM']
        PcubeDeRampedUTM = PcubeDeRampedDeBiasedUTM[:,:,:,:,1,:]
    elif band_opt == 'PL':
        PcubeDeRampedUTM = data['PcubeDeRampedUTM']
        PcubeDeRampedDeBiasedUTM = data['PcubeDeRampedDeBiasedUTM']
        PcubeDeRampedDeBiasedUTM_Mono = PcubeDeRampedDeBiasedUTM[:,:,:,:,0,:]
        PcubeDeRampedDeBiasedUTM_Bi = PcubeDeRampedDeBiasedUTM[:,:,:,:,1,:]
    else:
        print('select: P, L_Mono, L_Bi, PL')   
        return

    coordinate_array = np.zeros(PcubeDeRampedUTM.shape[:2]).astype('str')
    for x in range(coordinate_array.shape[0]):
        for y in range(coordinate_array.shape[1]):
            coordinate_array[x, y] = f"{x} {y}"

    PcubeDeRampedUTM = merge_headings(PcubeDeRampedUTM)

    if type == 'non_quadrants':
        if label == 'lidar':
            if crop_lidar == 'yes':
                LidarCHM_original = data['LidarCHM']
                LidarCHM = crop_lidar_edge_zone(LidarCHM_original)
            else:
                LidarCHM = data['LidarCHM']
        else:
            print('The label is classification.')

        print('Converting data ..')


        if label == "lidar":
            lidar_df = lidar_cube_to_table(LidarCHM, coordinate_array=coordinate_array)
            if band_opt in ['P', 'L_Mono', 'L_Bi']:
                data_sep_pol = generate_flattened_intensity_and_lidar(PcubeDeRampedUTM, band_opt, lidar_df, filename=sep_pol_file, coordinate_array=coordinate_array)
            elif band_opt == 'PL':
                data_sep_pol = generate_flattened_intensity_and_lidar_PL(PcubeDeRampedUTM, PcubeDeRampedDeBiasedUTM_Mono, PcubeDeRampedDeBiasedUTM_Bi, lidar_df, filename=sep_pol_file)
            
            print('Flattened data generated')
        elif label == 'class':
            tree_classification = load_classification_tree_data(class_parent_path)
            class_df = classification_tree_image_to_table(tree_classification, coordinate_array=coordinate_array)
            if band_opt in ['P', 'L_Mono', 'L_Bi']:
                data_sep_pol = generate_flattened_intensity_and_class(PcubeDeRampedUTM, band_opt, class_df, filename=sep_pol_file, coordinate_array=coordinate_array)
            elif band_opt == 'PL':
                data_sep_pol = generate_flattened_intensity_and_class_PL(PcubeDeRampedUTM, PcubeDeRampedDeBiasedUTM_Mono, PcubeDeRampedDeBiasedUTM_Bi, class_df, filename=sep_pol_file)
            print('Flattened data generated')
        else:
            print('label error')
        
            
        print('Starting to generate union polarimetry data')
        print('cl_ml_data shape is:', data_sep_pol.shape)
        data_union_pol = convert_mldata_union_polarimetry(data_sep_pol, label, filename=union_pol_file)
        print('Union complete')
        assert (sum(data_union_pol.duplicated(['x','y'])) == 0)
        print('Not duplicates found for x and y coordinates')
        return data_sep_pol, data_union_pol

    else:

        if label == "lidar":
            if band_opt in ['P', 'L_Mono', 'L_Bi']:
                quadrants, ground_truths, _, coordinate_array = generate_quadrants(PcubeDeRampedUTM, LidarCHM_original, coordinate_array)
                return generate_dataloaders(quadrants, ground_truths, coordinate_array, label, band_opt, test_quadrant=3, sep_pol_file=sep_pol_file, union_pol_file=union_pol_file, output_path=output_path)

 

def generate_training_geo_split(data_sep_pol, data_union_pol, outpath, label):
    print('Starting geo split')
    for data, name in zip([data_sep_pol, data_union_pol], ['separate_pol', 'union_pol']):
        print(f'Starting split for {name} {label}')
        gen_square_swath(data, name, outpath, label)
        print(f'Complete for {name} {label}')
    print('Geo split complete')

def main(args):
    # Set folder paths and band options 
    os.makedirs(args.output_path, exist_ok=True)
    data_sep_pol, data_union_pol = generate_ml_data(args.parent_path, args.band_opt, args.output_path, args.classification_parent_path, args.label, args.crop_lidar, args.regenerate_data, args.type)
    if(args.type == 'non_quadrants'):
        generate_training_geo_split(data_sep_pol, data_union_pol, args.output_path, args.label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Tabular data generator')
    parser.add_argument('--output_path', default="/home/joag/local/mldata", help='the path to store the generated data')
    parser.add_argument('--parent_path', default="/data/2024-esl-biomasssar-development-landing/tomosar-matlab/", help='the parent path to look for the generated data')
    parser.add_argument('--classification_parent_path', default="/home/joag/data/tree_species/", help='the classification parent path to look for the generated data')
    parser.add_argument('--label', default='lidar', help='lidar for regression task or class for classification task')
    parser.add_argument('--band_opt', default='P',  help='band selection for the data-generation (P, L_Mono, L_Bi, PL)')
    parser.add_argument('--crop_lidar', default='yes',  help='Crop the edge zero value zone in LiDAR data')
    parser.add_argument('--regenerate_data', default='yes',  help='Regenerate data or not')
    parser.add_argument('--type', default='quadrants',  help='Regenerate data or not')

    args = parser.parse_args()
    #print(args)
    main(args)

