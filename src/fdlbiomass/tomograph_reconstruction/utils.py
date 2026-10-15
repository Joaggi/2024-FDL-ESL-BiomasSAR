import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from scipy.io import loadmat
from netCDF4 import Dataset
import sys 
import os 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.fdlbiomass.utils import read_biomass_data
import os 
from functools import wraps
import time 
import h5py
import dask.array as da 
from src.fdlbiomass.tomograph_reconstruction.covariance_matrix import calculate_covariance_matrix_tutorial, compare_matrices_detailed
import cupy as cp 
from src.fdlbiomass.dataloader_fns import merge_headings

def read_SLC(parent_path, band_opt, heading, polarimetric_channel):
    """
    Load TomoSAR data and related parameters.
    
    Parameters:
    parent_path (str): Path to the parent directory
    heading (str): Either 'NW' (290°) or 'SE' (110°)
    polarimetric_channel (str): 'HH', 'HV' or 'VV'
    
    Returns:
    dict: A dictionary containing all loaded data and parameters
    """
    
    # Determine the polarimetric index based on the channel
    pol_ind = {'HH': 1, 'HV': 2, 'VV': 3}[polarimetric_channel]
    if band_opt =='L_Bi':
        band = 'L'
        monostatic_tomography  = False 
    elif band_opt =='L_Mono':
        band ='L'
        monostatic_tomography=True
    elif band_opt =='P':
        band='P'
        monostatic_tomography=None  


    # Loading scene parameters
    parent_path=os.path.join(parent_path, band) 
    curr_fullpath = os.path.join(parent_path, heading, 'Util', 'SLC', 'SceneParameters.mat')
    scene_parameters = loadmat(curr_fullpath)
    
    ground_range_axis = scene_parameters['ground_range_axis'].squeeze()
    azimuth_axis = scene_parameters['azimuth_axis'].squeeze()
    Ngr = len(ground_range_axis)
    Naz = len(azimuth_axis)
    


    # Loading phase-to-height factors
    kz = np.zeros((Ngr, Naz, scene_parameters['N'][0][0]), dtype=np.single)
    for n in range(scene_parameters['N'][0][0]):
        filename_kz = scene_parameters['filenames_kz'][n][0]
        if isinstance(filename_kz, np.ndarray):
            filename_kz = str(filename_kz[0])
        curr_fullpath = os.path.join(parent_path, heading, 'Util', 'SLC', 'kz', filename_kz)
        # print(f'Loading {curr_fullpath}...')
        kz[:, :, n] = read_biomass_data(curr_fullpath, [], 2)
        # print('done.')
    
    # Loading sensor trajectories
    curr_fullpath = os.path.join(parent_path, heading, 'Util', 'SLC', 'FocusingCenters.mat')
    focusing_centers = loadmat(curr_fullpath)
    
    # Loading DTMs
    curr_fullpath = os.path.join(parent_path, heading, 'Util', 'SLC', 'MetaSensingDTM.mat')
    meta_sensing_dtm = loadmat(curr_fullpath)
    curr_fullpath = os.path.join(parent_path, heading, 'Util', 'SLC', 'TomoDTMsr.mat')
    tomo_dtm_sr = loadmat(curr_fullpath)
    
    # Loading SLC images
    I = np.zeros((Ngr, Naz, scene_parameters['N'][0][0]), dtype=np.complex64)


    for n in range(scene_parameters['N'][0][0]):
        curr_fullpath = os.path.join(parent_path, heading, 'SLC', 'PoliMi', scene_parameters['filenames_slc'][n, pol_ind-1][0])
        # print(f'Loading {curr_fullpath}...')
        I[:, :, n] = read_biomass_data(curr_fullpath, [], 1)

        # print('done.')

    # Print shapes before returning
    print(f"\nShapes for {heading} heading:")
    print(f"I_SLC shape: {I.shape}")
    print(f"kz shape: {kz.shape}")
    # print(f"MetaSensingDTM shape: {meta_sensing_dtm['meta_sensing_dtm'].shape}")
    # print(f"TomoDTMsr shape: {tomo_dtm_sr['tomo_dtm_sr'].shape}")
    # Prepare the return dictionary
    return_data = {
        'scene_parameters': scene_parameters,
        'ground_range_axis': ground_range_axis,
        'azimuth_axis': azimuth_axis,
        'kz': kz,
        'focusing_centers': focusing_centers,
        'meta_sensing_dtm': meta_sensing_dtm,
        'tomo_dtm_sr': tomo_dtm_sr,
        'pol_ind': pol_ind, 
        'I_SLC' : I, 
    }
    
    return return_data



def process_topo_phase_P(I, kz, tomo_dtm_sr, meta_sensing_dtm ):
    """
    Process all ground indexes to create ground-steered SAR images.

    Parameters:
    - I: Complex SAR image data (3D array: ground range, azimuth, tomographic)
    - kz: Vertical wavenumbers (3D array: ground range, azimuth, tomographic)
    - tomo_dtm_sr: Dictionary containing 'TomoDTMsr' (2D array: ground range, azimuth)
    - meta_sensing_dtm: Dictionary containing 'MetaSensingDTM' (2D array: ground range, azimuth)

    Returns:
    - Igs: Ground-steered SAR images for all ground indexes (3D array: ground range, azimuth, tomographic)
    """

    # Get dimensions
    Ngr, Naz, Ntomo = I.shape

    # Initialize the output array
    Igs = np.zeros_like(I, dtype=np.complex128)

    for gr_ind in range(Ngr):
        # Ground steering
        dtm_diff = (tomo_dtm_sr['TomoDTMsr'][gr_ind, :] - meta_sensing_dtm['MetaSensingDTM'][gr_ind, :])
        exp_term = np.exp(-1j * kz[gr_ind, :, :] * dtm_diff[:, np.newaxis])
        Igs[gr_ind, :, :] = I[gr_ind, :, :] * exp_term


    if Igs.shape != I.shape:
            raise ValueError('topo phase procesing has changed shape')
        
    return Igs


def process_topo_phase_L(I, kz, tomo_dtm_sr, meta_sensing_dtm, scene_parameters, band_opt ):
    """
    - this is only for ground index? do we need to ground steer for azimuth axis) 

    process for L band 
    - note compred to P band when we subset by n_tomo_gr (which varies depednding on if mmono or bistatic) it can invert the order if GR index used. 

    """
    if band_opt =='L_Bi':
        monostatic_tomography = False 
    elif band_opt =='L_Mono':
        monostatic_tomography=True 

    mono_aq = scene_parameters['monostatic_acquisition']
    if isinstance(mono_aq[0], list):
        n_tomo_bool = [bool(item[0]) for item in mono_aq]
    else:
        n_tomo_bool = [bool(item) for item in mono_aq]
    if not monostatic_tomography:
        n_tomo_bool = [not x for x in n_tomo_bool]
    
    # Get dimensions
    Ngr, Naz, Ntomo = I.shape

    # Initialize the output array
    Igs = np.zeros_like(I[:,:, n_tomo_bool ], dtype=np.complex128)

    print('Igs shape', Igs.shape)
    for gr_ind in range(Ngr):
        # Ground steering
        dtm_diff = (tomo_dtm_sr['TomoDTMsr'][gr_ind, :] - meta_sensing_dtm['MetaSensingDTM'][gr_ind, :])
        exp_term = np.exp(-1j * kz[gr_ind, :, n_tomo_bool].T * dtm_diff[:, np.newaxis])
        Igs[gr_ind, :, :] = I[gr_ind, :, n_tomo_bool].T * exp_term


    if Igs.shape != I[:,:, n_tomo_bool ].shape:
            raise ValueError('topo phase procesing has changed shape')
        
    return Igs





def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} executed in {end_time - start_time:.2f} seconds")
        return result
    return wrapper

@timer
def load_dummy_data(size):
    print("Generating sample data...")
    rows, cols, nTrack = size, size, 28
    # rows, cols, nTrack = 200, 3500, 2
    cols=cols
    img_stack = np.random.random((rows, cols, nTrack)) + 1j * np.random.random((rows, cols, nTrack))
    return img_stack 




def load_igs_data(parent_path, heading, polarimetric_channel, output_path, band_opt):
    img_stack_path = os.path.join(output_path, f'{band_opt}_{polarimetric_channel}_{heading}_img_stack.npy')
    if not os.path.exists(img_stack_path):
        print('Loading SLC and doing topo phase correction')
        sar_data_dict = read_SLC(parent_path, band_opt, heading, polarimetric_channel)
        if band_opt=='P':
            img_stack = process_topo_phase_P(sar_data_dict['I_SLC'], sar_data_dict['kz'], sar_data_dict['tomo_dtm_sr'], sar_data_dict['meta_sensing_dtm'])
        elif band_opt in ['L_Bi', 'L_Mono']:
            img_stack = process_topo_phase_L(sar_data_dict['I_SLC'], sar_data_dict['kz'], sar_data_dict['tomo_dtm_sr'], sar_data_dict['meta_sensing_dtm'], sar_data_dict['scene_parameters'], band_opt)
        np.save(img_stack_path, img_stack)
        print('Phase correction complete')
    else:
        print('Loading data')
        img_stack = np.load(img_stack_path)

    print('Img stack shape', img_stack.shape)
    return img_stack 



def save_result(result, output_file):
    with h5py.File(output_file, 'w') as f:
        dset = f.create_dataset('data', shape=result.shape, dtype=result.dtype)
        da.store(result, dset)


@timer 
def test_cov_matches_tutorial_cov_fn(cov_matrix_to_compare, input_data_stack):
    """
    Compare test covariance matrix totutroial cov matrix 
    """
    if isinstance(cov_matrix_to_compare, cp.ndarray):
        cov_matrix_to_compare = cp.asnumpy(cov_matrix_to_compare)
    if isinstance(cov_matrix_to_compare, da.Array):
        
        cov_matrix_to_compare = cov_matrix_to_compare.compute()

    x, y, z = input_data_stack.shape 
    
    if x > 700 or y > 700 or z > 700:
        print('Large dataset - trying to compute will be v slow. Consider using subset ')
        return None
    
    print('Starting comparison to tutorial')
    C = calculate_covariance_matrix_tutorial(input_data_stack, outname='test', overwrite=True)
    
    C = cp.asarray(C)
    
    max_diff, mean_diff, percentage_exceed = compare_matrices_detailed(C, cov_matrix_to_compare)
    
    if percentage_exceed > 0.01:  # If more than 0.01% of elements exceed tolerance
        print("Matrices are significantly different")
        raise ValueError('Matrices not essentially identical')
    else:
        print("Matrices are essentially identical")



import glob 

def reconstruct_data(filepath, chunk_directory, original_shape, chunk_sizes):
    """
    Re creates covariance matrix from chunks that dask produces 
    
    """
    print('Chunk dir ', chunk_directory)
    print('output file pat ' , filepath )
    op , gt =reconstruct_cubes_from_chunks(chunk_directory, original_shape, chunk_sizes)
        # Save the chunks to an HDF5 file
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('full_data', data=op)
        f.create_dataset('ground_truth', data=gt)
    print('full data saved')

def load_ground_truth(base_path_to_lidar_CHM='/home/grace.bcolverd/lidar-chm', size=True, heading=None ):
    # Generate dummy ground truth data
    if size==True :
        path = os.path.join(base_path_to_lidar_CHM, f'CHM_lidar_{heading}_radar_coord.mat' ) 
        ground_truth =loadmat (path )['CHM_lidar']
        print('gt shape ', ground_truth.shape )
    else:
        ground_truth = np.random.rand(*(size, size)).astype(np.float32)
        
    return ground_truth


# def reconstruct_cube_from_chunks(chunk_directory, original_shape, chunk_sizes):
#     """
#     Reconstruct a cube from chunks of various sizes.
    
#     :param chunk_directory: Path to the directory containing chunk files
#     :param original_shape: Original shape of the full cube (x, y, z)
#     :param chunk_sizes: Tuple of (x_chunk_size, y_chunk_size, z_chunk_size)
#     :return: Reconstructed numpy array
#     """
#     chunk_files = sorted(glob.glob(chunk_directory + '*_*_*.h5'))
#     if len(chunk_files) == 0:
#         raise ValueError('Chunks list empty')
    
#     # Initialize an empty array to hold the reconstructed data
#     reconstructed_cube = np.zeros(original_shape, dtype=np.float32)
    
#     max_x_idx, max_y_idx, max_z_idx = -1, -1, -1
    
#     for chunk_file in chunk_files:
#         print('Processing chunk file:', chunk_file)
#         # Extract chunk indices from filename
#         filename = os.path.basename(chunk_file)
#         x_idx, y_idx, z_idx = map(int, filename.split('.')[0].split('_')[-3:])
        
#         max_x_idx = max(max_x_idx, x_idx)
#         max_y_idx = max(max_y_idx, y_idx)
#         max_z_idx = max(max_z_idx, z_idx)
#         print(max_x_idx, max_y_idx, max_z_idx)
        
#         # Load the chunk data
#         with h5py.File(chunk_file, 'r') as f:
#             if 'data' not in f:
#                 raise KeyError(f"'data' key not found in file {chunk_file}")
#             chunk_data = f['data'][:]
        
#         print('Chunk shape:', chunk_data.shape)
        
#         # Calculate the position of this chunk in the final array
#         x_start = x_idx * chunk_sizes[0]
#         y_start = y_idx * chunk_sizes[1]
#         z_start = z_idx * chunk_sizes[2]
        
#         # Calculate end positions
#         x_end = min(x_start + chunk_sizes[0], original_shape[0])
#         y_end = min(y_start + chunk_sizes[1], original_shape[1])
#         z_end = min(z_start + chunk_sizes[2], original_shape[2])
        
#         # Check if the chunk fits within the original shape
#         if x_end > original_shape[0] or y_end > original_shape[1] or z_end > original_shape[2]:
#             raise ValueError(f"Chunk at position ({x_idx}, {y_idx}, {z_idx}) exceeds the original shape. "
#                              f"Chunk ends at ({x_end}, {y_end}, {z_end}), but original shape is {original_shape}")
        
#         # Place the chunk in the correct position
#         reconstructed_cube[x_start:x_end, y_start:y_end, z_start:z_end] = chunk_data[:x_end-x_start, :y_end-y_start, :z_end-z_start]
    
#     # Check if we've filled the entire cube
#     expected_x_chunks = (original_shape[0] + chunk_sizes[0] - 1) // chunk_sizes[0]
#     expected_y_chunks = (original_shape[1] + chunk_sizes[1] - 1) // chunk_sizes[1]
#     expected_z_chunks = (original_shape[2] + chunk_sizes[2] - 1) // chunk_sizes[2]
#     if max_x_idx + 1 < expected_x_chunks or max_y_idx + 1 < expected_y_chunks or max_z_idx + 1 < expected_z_chunks:
#         raise ValueError(f"Mismatch in number of chunks. Expected {expected_x_chunks}x{expected_y_chunks}x{expected_z_chunks}, "
#                          f"but got {max_x_idx + 1}x{max_y_idx + 1}x{max_z_idx + 1}")
    
#     # # Check if there are any unfilled areas
#     # if np.any(reconstructed_cube == 0):
#     #     raise ValueError("The reconstructed cube contains unfilled areas (zeros)")
    
#     return reconstructed_cube


import numpy as np
import glob
import os
import h5py

def reconstruct_cubes_from_chunks(chunk_directory, original_shape, chunk_sizes):
    """
    Reconstruct cubes for data and ground truth from chunks of various sizes.
    
    :param chunk_directory: Path to the directory containing chunk files
    :param original_shape: Original shape of the full cube (x, y, z)
    :param chunk_sizes: Tuple of (x_chunk_size, y_chunk_size, z_chunk_size)
    :return: Tuple of (reconstructed_data_cube, reconstructed_ground_truth)
    """
    chunk_files = sorted(glob.glob(os.path.join(chunk_directory,  '*_*_*.h5') ))
    if len(chunk_files) == 0:
        raise ValueError('Chunks list empty')
    
    # Initialize empty arrays to hold the reconstructed data
    reconstructed_data_cube = np.zeros(original_shape, dtype=np.float32)
    reconstructed_ground_truth = np.zeros((original_shape[0], original_shape[1]), dtype=np.float32)
    
    max_x_idx, max_y_idx, max_z_idx = -1, -1, -1
    
    for chunk_file in chunk_files:
        print('Processing chunk file:', chunk_file)
        # Extract chunk indices from filename
        filename = os.path.basename(chunk_file)
        x_idx, y_idx, z_idx = map(int, filename.split('.')[0].split('_')[-3:])
        
        max_x_idx = max(max_x_idx, x_idx)
        max_y_idx = max(max_y_idx, y_idx)
        max_z_idx = max(max_z_idx, z_idx)
        print(max_x_idx, max_y_idx, max_z_idx)
        
        # Load the chunk data
        with h5py.File(chunk_file, 'r') as f:
            if 'data' not in f or 'ground_truth' not in f:
                raise KeyError(f"'data' or 'ground_truth' key not found in file {chunk_file}")
            chunk_data = f['data'][:]
            chunk_ground_truth = f['ground_truth'][:]
        
        print('Data chunk shape:', chunk_data.shape)
        print('Ground truth chunk shape:', chunk_ground_truth.shape)
        
        # Calculate the position of this chunk in the final arrays
        x_start = x_idx * chunk_sizes[0]
        y_start = y_idx * chunk_sizes[1]
        z_start = z_idx * chunk_sizes[2]
        
        # Calculate end positions
        x_end = min(x_start + chunk_sizes[0], original_shape[0])
        y_end = min(y_start + chunk_sizes[1], original_shape[1])
        z_end = min(z_start + chunk_sizes[2], original_shape[2])
        
        # Check if the chunk fits within the original shape
        if x_end > original_shape[0] or y_end > original_shape[1] or z_end > original_shape[2]:
            raise ValueError(f"Chunk at position ({x_idx}, {y_idx}, {z_idx}) exceeds the original shape. "
                             f"Chunk ends at ({x_end}, {y_end}, {z_end}), but original shape is {original_shape}")
        
        # Place the chunks in the correct positions
        reconstructed_data_cube[x_start:x_end, y_start:y_end, z_start:z_end] = chunk_data[:x_end-x_start, :y_end-y_start, :z_end-z_start]
        reconstructed_ground_truth[x_start:x_end, y_start:y_end] = chunk_ground_truth[:x_end-x_start, :y_end-y_start]
    
    # Check if we've filled the entire data cube
    expected_x_chunks = (original_shape[0] + chunk_sizes[0] - 1) // chunk_sizes[0]
    expected_y_chunks = (original_shape[1] + chunk_sizes[1] - 1) // chunk_sizes[1]
    expected_z_chunks = (original_shape[2] + chunk_sizes[2] - 1) // chunk_sizes[2]
    if max_x_idx + 1 < expected_x_chunks or max_y_idx + 1 < expected_y_chunks or max_z_idx + 1 < expected_z_chunks:
        raise ValueError(f"Mismatch in number of chunks. Expected {expected_x_chunks}x{expected_y_chunks}x{expected_z_chunks}, "
                         f"but got {max_x_idx + 1}x{max_y_idx + 1}x{max_z_idx + 1}")
    
    # # Check if there are any unfilled areas
    # if np.any(reconstructed_data_cube == 0):
    #     raise ValueError("The reconstructed data cube contains unfilled areas (zeros)")
    # if np.any(reconstructed_ground_truth == 0):
    #     raise ValueError("The reconstructed ground truth contains unfilled areas (zeros)")
    
    return reconstructed_data_cube, reconstructed_ground_truth