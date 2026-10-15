import numpy  as np 
import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, ConcatDataset
import os
import h5py
from src.fdlbiomass.tomograph_reconstruction.tomo_dataset_fn import SLC_QuadrantDataset 
import random 

import os
import h5py
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler

class MinMaxScaler:
    def __init__(self, feature_range=(0, 1)):
        self.feature_range = feature_range
        self.min = None
        self.max = None
        self.scale = None
        self.min_scale = None

    def fit(self, data):
        self.min = torch.min(data)
        self.max = torch.max(data)
        self.scale = (self.feature_range[1] - self.feature_range[0]) / (self.max - self.min)
        self.min_scale = self.feature_range[0] - self.min * self.scale

    def transform(self, data):
        return data * self.scale + self.min_scale

    def inverse_transform(self, data):
        return (data - self.min_scale) / self.scale

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = torch.mean(data)
        self.std = torch.std(data)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return data * self.std + self.mean

def get_scaler(scaler_type):
    if scaler_type == 'minmax':
        return MinMaxScaler()
    elif scaler_type == 'standard':
        return StandardScaler()
    else:
        raise ValueError("Invalid scaler type. Choose 'minmax' or 'standard'.")


def crop_full_data_to_valid_ground_truth(full_data, ground_truth):
    # Check if the x and y dimensions match
    if full_data.shape[:2] != ground_truth.shape:
        raise ValueError("Shape mismatch in x and y dimensions")
    
    # Create a mask where ground truth is not NaN
    valid_mask = ~np.isnan(ground_truth)
    print('valid mask shape:', valid_mask.shape)
    
    # Expand the mask to match the shape of full_data
    expanded_mask = np.repeat(valid_mask[:, :, np.newaxis], full_data.shape[2], axis=2)
    print('expanded mask shape ', expanded_mask.shape)
    # Apply the mask to full_data
    cropped_data = np.where(expanded_mask, full_data, np.nan)
    print('crop data shape' , cropped_data.shape)
    return cropped_data



def split_data_into_quadrants(full_data, ground_truth, reconstruction=False):
    # First, use the previous function to crop the data
    if reconstruction is True :
        cropped_data=full_data
    
    else:
        cropped_data = crop_full_data_to_valid_ground_truth(full_data, ground_truth)
    
    # Define the splitting levels
    x = cropped_data.shape[1] // 2
    y = 600
    y2 = y // 2

    # Ensure y doesn't exceed the height of cropped_data
    y = min(y, cropped_data.shape[0] - 1)
    y2 = min(y2, y - 1)

    # Calculate the new dimensions for each quadrant
    top_height = y
    bottom_height = cropped_data.shape[0] - y
    left_width = x
    right_width = cropped_data.shape[1] - x

    # Create quadrants
    q1_data = cropped_data[:top_height, x:, :]
    q2a_data = cropped_data[:y2, :left_width, :]
    q2b_data = cropped_data[y2:top_height, :left_width, :]
    q3_data = cropped_data[y:, :left_width, :]
    q4_data = cropped_data[y:, x:, :]

    q1_truth = ground_truth[:top_height, x:]
    q2a_truth = ground_truth[:y2, :left_width]
    q2b_truth = ground_truth[y2:top_height, :left_width]
    q3_truth = ground_truth[y:, :left_width]
    q4_truth = ground_truth[y:, x:]

    return {
        'q1': (q1_data, q1_truth),
        'q2a': (q2a_data, q2a_truth),
        'q2b': (q2b_data, q2b_truth),
        'q3': (q3_data, q3_truth),
        'q4': (q4_data, q4_truth)
    }



def create_datasets(quadrants, role_mapping, patch_size, stride, gt_zero_thresholddataset):
    datasets = {}
    
    for role, quadrant_indices in role_mapping.items():
        print(f'Creating dataset for {role}')
        role_datasets = []
        
        # Ensure quadrant_indices is always a list
        if not isinstance(quadrant_indices, list):
            quadrant_indices = [quadrant_indices]
        
        for quadrant_index in quadrant_indices:
            quadrant_key = f'q{quadrant_index}'
            if quadrant_key in quadrants:
                q_data, q_truth = quadrants[quadrant_key]
                dataset = SLC_QuadrantDataset(q_data, q_truth, patch_size, stride, gt_zero_thres =gt_zero_thresholddataset)
                role_datasets.append(dataset)
            else:
                print(f"Warning: No valid dataset found for role '{role}' (quadrant {quadrant_key})")
        
        if role_datasets:
            if len(role_datasets) == 1:
                datasets[role] = role_datasets[0]
            else:
                datasets[role] = ConcatDataset(role_datasets)
        else:
            print(f"Error: No valid datasets found for role '{role}'")
    
    return datasets




def load_process_h5_data(h5_file_path, config, num_images):
    """"
    Load the h5 files for tomo recon
    crop to valid patches, and select the layers based on indices selected
    """
    # Load data
    with h5py.File(h5_file_path, 'r') as f:
        full_data = f['full_data'][:]
        ground_truth = f['ground_truth'][:]

    # Process data
    cropped_data = crop_full_data_to_valid_ground_truth(full_data, ground_truth)
    if config.exclude_height_thresh.lower() =='yes':
            print('starting to exclude lower than 5m')
            mask = ground_truth < config.height_exclusion
            ground_truth[mask] = np.nan
            cropped_data[mask] = np.nan
    X,  Y,_ = cropped_data.shape 
    
    if num_images is not None and num_images < 28:
        selected_indices = [0]  # Always include the first image
        if config.first_set_SLC ==True:
            print('taking first set no random')
            selected_indices.extend(range(1, min(28, num_images)))
        else:
            print('random selection')
            random_seed = config.random_seed
            random.seed(random_seed)
            np.random.seed(random_seed)
            selected_indices.extend(np.random.choice(range(1, 28), size=num_images-1, replace=False))
        print('Selected indices ', selected_indices )
        selected_indices.sort()
        
        # Reshape the data to separate the 28 images
        reshaped_data = cropped_data.reshape(X, Y, 28, 3)
        # Select the subset of images
        selected_data = reshaped_data[:, :, selected_indices, :]
        # Reshape back to original format
        cropped_data = selected_data.reshape(X, Y, -1)
        return cropped_data, ground_truth


def load_process_data(config, num_images, base_path):
    """
    Loads data, checking if config pol is single or multi channel 
    """
    valid_pols = ['HH', 'HV', 'VV']

    if isinstance(config.polarimetric_channel, str):
        if config.polarimetric_channel not in valid_pols:
            raise ValueError(f"Invalid polarization channel: {config.polarimetric_channel}. Must be one of {valid_pols}")
        
        h5_file_path = os.path.join(base_path, f'{config.band_opt}__heading_{config.heading}__pol_{config.polarimetric_channel}_final_data_run1.h5')
        return load_process_h5_data(h5_file_path, config, num_images)

    elif isinstance(config.polarimetric_channel, list) and all(isinstance(item, str) for item in config.polarimetric_channel):
        if not all(pol in valid_pols for pol in config.polarimetric_channel):
            raise ValueError(f"Invalid polarization channel(s) in {config.polarimetric_channel}. Must be from {valid_pols}")
        
        if len(set(config.polarimetric_channel)) == 1:
            # If all polarizations are the same, process as single pol
            return load_process_data_multipol(config._replace(pol=config.pol[0]), num_images)
        
        data_list = []
        gt = None
        for pol in config.polarimetric_channel:
            h5_file_path = os.path.join(base_path, f'{config.band_opt}__heading_{config.heading}__pol_{pol}_final_data_run1.h5')
            data, current_gt = load_process_h5_data(h5_file_path, config, num_images)
            data_list.append(data)
            if gt is None:
                gt = current_gt  # Store ground truth from the first polarization
        
        combined_data = np.concatenate(data_list, axis=-1)  # Concatenate along the last axis
        if combined_data.shape[2] != num_images * 3 * len(config.polarimetric_channel):
            raise ValueError('Stacking shape not right for multi channel polarimatey; investigate ')
        return combined_data, gt

    else:
        raise ValueError("config.polarimetric_channel must be a string or a list of strings")


def setup_tomo_dataloaders(config, patch_size, stride, base_path, num_images):
    """
    Set up tomo loaders for single polarimatery for tomo recon 

    """
    cropped_data, ground_truth= load_process_data(config, num_images, base_path)

    quadrants = split_data_into_quadrants(cropped_data, ground_truth)
    datasets = create_datasets(quadrants, config.role_mapping, patch_size, stride, config.gt_zero_thresholddataset)
    train_dataset = datasets['train']
    val_dataset = datasets['val']
    test_dataset = datasets['test']
    
    # Create scalers for input and ground truth
    input_scaler = get_scaler(config['scaler_type'])
    gt_scaler = get_scaler(config['scaler_type'])

    # Fit the scalers on the training data
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    input_data = []
    gt_data = []
    
    for batch in train_loader:
        input_data.append(batch[0])
        gt_data.append(batch[1])

    input_data = torch.cat(input_data, dim=0)
    print('inp shape ', input_data.shape )
    gt_data = torch.cat(gt_data, dim=0)
    print('gt shape ', gt_data.shape)
    
    input_scaler.fit(input_data)
    gt_scaler.fit(gt_data ) 
    
    # Function to apply transformation to a batch
    def transform_batch(batch):
        inputs, targets, masks = zip(*batch)
        transformed_inputs = input_scaler.transform(torch.stack(inputs))
        transformed_targets = gt_scaler.transform(torch.stack(targets))
        return transformed_inputs, transformed_targets, torch.stack(masks)
        
    
    # Create DataLoaders with transformed data
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=transform_batch)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=transform_batch)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=transform_batch)

    return train_loader, val_loader, test_loader, (input_scaler, gt_scaler), (cropped_data, ground_truth)

