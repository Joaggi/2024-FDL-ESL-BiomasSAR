import torch
from torch.utils.data import DataLoader

import numpy as np
from src.fdlbiomass.utils import merge_headings , load_biomass_data, split_into_quadrants, crop_to_non_nan
from src.fdlbiomass.datasets.quadrant_dataset_pband import QuadrantPatchDataset
from torch.utils.data import Dataset, DataLoader



def check_pol_channels(polarimetric_channels):
    if polarimetric_channels is None:
        return None
    
    if isinstance(polarimetric_channels, str):
        return polarimetric_channels

    if isinstance(polarimetric_channels, list):
        # Convert to set to remove duplicates, then back to list
        unique_channels = list(set(polarimetric_channels))
        
        # Check if there were any duplicates
        if len(unique_channels) != len(polarimetric_channels):
            print("Warning: Duplicate polarimetric channels were found and removed.")
            print(f"Original channels: {polarimetric_channels}")
            print(f"Unique channels: {unique_channels}")
        
        # Ensure all channels are valid
        valid_channels = ['HH', 'HV', 'VV']
        for channel in unique_channels:
            if channel not in valid_channels:
                raise ValueError(f"Invalid polarimetric channel: {channel}. Valid channels are {valid_channels}")
        
        return unique_channels
    
    raise ValueError("polarimetric_channels must be None, a string, or a list of strings")


def generate_quadrants(parent_path, polarimetric_channels=None, band_opt = None, exclude_height_thresh='no'):
    polarimetric_channels = check_pol_channels(polarimetric_channels)
    num_channels = len(polarimetric_channels) if isinstance(polarimetric_channels, list) else 1
    data = load_biomass_data(parent_path, band_opt)
      
    LidarCHM = data['LidarCHM']
    if band_opt=='P':
        pcube = data['PcubeDeRampedUTM']
        mono_bi_ind=False 
        print('PCube shape ', pcube.shape )
        
        
    elif band_opt =='L_Mono':
        pcube = data['PcubeDeRampedDeBiasedUTM']
        mono_bi_ind = 0 
        pcube=pcube[:,:,:,:,mono_bi_ind, :]
    elif band_opt =='L_Bi':
        pcube = data['PcubeDeRampedDeBiasedUTM']
        mono_bi_ind = 1 
        
        pcube=pcube[:,:,:,:,mono_bi_ind, :]
    # elif band_opt=='PL':
    else:
        print('Band opt must be "P", "L_Bi", "L_Mono" ' )
        
    merged_power_cube = merge_headings(pcube)
    print('merged power cube shape ',merged_power_cube.shape )
    intensity_cube=select_polarimetry_monoband(merged_power_cube, polarimetric_channels, band_opt)
    
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

    if exclude_height_thresh =='Yes':
        print('starting to exclude lower than 5m')
        mask = cropped_LidarCHM < 5
        cropped_LidarCHM[mask] = np.nan
        crop_int[mask] = np.nan

    # Split the array
    quadrants, ground_truths = split_into_quadrants(crop_int, cropped_LidarCHM)
    
    
    return quadrants, ground_truths, num_channels, (crop_int, cropped_LidarCHM)


def select_polarimetry_monoband(merged_power_cube,polarimetric_channels, band  ):
    if band =='P':
        if polarimetric_channels is None:
            intensity_cube = merged_power_cube
        elif isinstance(polarimetric_channels, str):
            pol_ind = {'HH': 0, 'HV': 1, 'VV': 2}[polarimetric_channels]
            intensity_cube = merged_power_cube[:,:,:,pol_ind:pol_ind+1]
        elif isinstance(polarimetric_channels, list):
            print('Starting union polarimetry')
            pol_indices = [{'HH': 0, 'HV': 1, 'VV': 2}[ch] for ch in polarimetric_channels]
            intensity_cube = merged_power_cube[:,:,:,pol_indices]
        else:
            raise ValueError("Invalid polarimetric_channels input")
    elif band in ['L_Bi', 'L_Mono']:
        if polarimetric_channels is None:
            intensity_cube = merged_power_cube
        elif isinstance(polarimetric_channels, str):
            pol_ind = {'HH': 0, 'HV': 1, 'VV': 2}[polarimetric_channels]
            intensity_cube = merged_power_cube[:,:,:,pol_ind:pol_ind+1 ]
        elif isinstance(polarimetric_channels, list):
            print('Starting union polarimetry')
            pol_indices = [{'HH': 0, 'HV': 1, 'VV': 2}[ch] for ch in polarimetric_channels]
            intensity_cube = merged_power_cube[:,:,:,pol_indices]
        else:
            raise ValueError("Invalid polarimetric_channels input")
    return intensity_cube


import torch
from torch.utils.data import DataLoader

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

def generate_dataloaders(quadrants, ground_truths, patch_size, stride, test_quadrant=3, batch_size=4, scaler_type='minmax'):
    """
    Generate dataloaders for training, testing, and validation from quadrant data.

    This function creates a QuadrantPatchDataset, sets up scalers for input and ground truth data,
    and returns DataLoader objects for training, testing, and validation. All dataloaders now include masks.

    Parameters:
    quadrants (list): List of quadrant data arrays.
    ground_truths (list): List of ground truth data arrays corresponding to the quadrants.
    patch_size (tuple): Size of patches to be extracted from the quadrants.
    stride (tuple): Stride for patch extraction.
    test_quadrant (int, optional): Index of the quadrant to be used for testing. Defaults to 3.
    batch_size (int, optional): Batch size for the dataloaders. Defaults to 4.
    scaler_type (str, optional): Type of scaler to use ('minmax' or 'standard'). Defaults to 'minmax'.

    Returns:
    tuple: Contains:
        - train_loader (DataLoader): DataLoader for training data.
        - test_loader (DataLoader): DataLoader for test data.
        - val_loader (DataLoader): DataLoader for validation data.
        - dataset (QuadrantPatchDataset): The dataset object.
        - (input_scaler, gt_scaler) (tuple): Scalers for input and ground truth data.

    Raises:
    ValueError: If an invalid test_quadrant index is provided.

    Note:
    This function assumes that there are exactly 4 quadrants and uses a specific split strategy:
    One quadrant for testing, one for validation, and two for training.
    """
    quad_options = [0, 1, 2, 3]
    if test_quadrant not in quad_options:
        raise ValueError('Choose index 0-3 for test')
    
    remaining_quadrants = [x for x in quad_options if x != test_quadrant]
    train_quadrants = remaining_quadrants[:2]
    if len(train_quadrants)!=2:
        raise ValueError('Not 2 train quadrants')
    val_quadrant = remaining_quadrants[2]
    
    train_dataset = QuadrantPatchDataset(
        quadrants, ground_truths, 
        train_quadrants=train_quadrants, 
        test_quadrant=test_quadrant, 
        val_quadrant=val_quadrant, 
        patch_size=patch_size, 
        stride=stride
    )
    train_dataset.set_mode('train')

    test_dataset = QuadrantPatchDataset(
        quadrants, ground_truths, 
        train_quadrants=train_quadrants, 
        test_quadrant=test_quadrant, 
        val_quadrant=val_quadrant, 
        patch_size=patch_size, 
        stride=stride
    )
    test_dataset.set_mode('test')

    val_dataset = QuadrantPatchDataset(
        quadrants, ground_truths, 
        train_quadrants=train_quadrants, 
        test_quadrant=test_quadrant, 
        val_quadrant=val_quadrant, 
        patch_size=patch_size, 
        stride=stride
    )
    val_dataset.set_mode('val')
    validation_passed = train_dataset.validate_masks()

    if not validation_passed:
        print("Validation failed. Please check your dataset and masking logic.")
    else:
        print("Validation passed. Proceeding with training.")

    # Create scalers for input and ground truth
    input_scaler = get_scaler(scaler_type)
    gt_scaler = get_scaler(scaler_type)

    # # Fit the scalers on the training data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    
    input_data = []
    gt_data = []
    for batch in train_loader:
        input_data.append(batch[0])
        gt_data.append(batch[1])
    
    input_data = torch.cat(input_data, dim=0)
    gt_data = torch.cat(gt_data, dim=0)
    print('inp datas shape pre scaler ', input_data.shape)
    print('gt  datas shape pre scaler ', gt_data.shape)
    input_scaler.fit(input_data)
    gt_scaler.fit(gt_data)
    
    # Function to apply transformation to a batch
    def transform_batch(batch):
        inputs, targets, masks = zip(*batch)
        transformed_inputs = input_scaler.transform(torch.stack(inputs))
        transformed_targets = gt_scaler.transform(torch.stack(targets))
        return transformed_inputs, transformed_targets, torch.stack(masks)

    # Create DataLoaders with transformed data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=transform_batch)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=transform_batch)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=transform_batch)

    return train_loader, test_loader, val_loader, train_dataset, (input_scaler, gt_scaler), val_quadrant
    



import ast 

def convert_pol_ch(polarimetric_channels):
    # Try to convert to a list if it's a string representation of a list
    try:
        polarimetric_channels = ast.literal_eval(polarimetric_channels)
    except (ValueError, SyntaxError):
        # If it's not a valid list representation, keep it as is (likely a single channel string)
        pass

    # If it's a list with a single string, extract the string
    if isinstance(polarimetric_channels, list) and len(polarimetric_channels) == 1:
        polarimetric_channels = polarimetric_channels[0]

    print(f"Polarimetric channels: {polarimetric_channels}")
    print(f"Type: {type(polarimetric_channels)}")
    return polarimetric_channels