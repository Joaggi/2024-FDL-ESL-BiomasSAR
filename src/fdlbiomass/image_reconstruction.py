import torch 
import numpy as np 
import matplotlib.pyplot as plt
import numpy as np
import math
import os 
import torch
import tempfile
import io
import wandb 

from mpl_toolkits.mplot3d import Axes3D
from src.fdlbiomass.utils import crop_lidar_edge_zone

def create_log_prediction_image(raw_data_tuple, MODEL,  scalers, patch_size,  polarimetric_channels, log_to_wandb=True ):

    """ Generates predictions for model for input data and compares to ground truth 
    Inputs:
    Create and log the prediction of the model
    raw_data_tuple: (cropint, cropped lidar) cropped intensity image 3d, and cropped lidar gt 2d
    model: trained model to use for predictions
    scales: input-scale and gt-scaler: genreated from dataloader_fms, used to invert/ normalise inputs  / outputs 
    patch size: used for reconstructing grid
    
    Outputs: logs image to wndb 
    """
    if isinstance(polarimetric_channels, list):
        print("TODO fix plotter for multi channel - polarimetric_channels is a list")
        return None 
    crop_int, cropped_LidarCHM = raw_data_tuple
    cropped_LidarCHM = crop_lidar_edge_zone(cropped_LidarCHM)
    input_scaler, gt_scaler = scalers
    cube_shape = (321, 665, 36)
    if len(crop_int.shape)!=4:
        raise ValueError('Intensity cube not of expected shape, expecting 4 dims')
    patches, patch_info, padding_info = create_patches(crop_int[:,:,:,0], patch_size)
    predictions = process_patches(patches, MODEL, input_scaler, gt_scaler)


    full_prediction = reconstruct_prediction(predictions, cube_shape, patch_info, padding_info)
    
    error = visualize_with_quadrants(full_prediction, cropped_LidarCHM, cube_shape, log= log_to_wandb)
    _ = plot_3d_lidar_prediction(full_prediction, cropped_LidarCHM, cube_shape,log_to_wandb=log_to_wandb ,elev=15, azim=270)



def create_patches(cube, patch_size):
    """
    Create non-overlapping patches from the input cube.
    
    :param cube: Input 3D array of shape (H, W, D)
    :param patch_size: Tuple of (patch_height, patch_width, patch_depth)
    :return: List of patches and padding information
    """
    print('Generating patches')
    H, W, D = cube.shape
    if isinstance(patch_size, tuple):
        None  
    elif isinstance(patch_size, str):
        patch_size = tuple(map(int, patch_size.strip('()').split(',')))
    
    
    ph, pw, _ = patch_size
    
    # Calculate padding
    pad_h = (ph - H % ph) % ph
    pad_w = (pw - W % pw) % pw
    
    # Pad the cube
    padded_cube = np.pad(cube, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    
    # Calculate number of patches in each dimension
    n_h = (H + pad_h) // ph
    n_w = (W + pad_w) // pw

    
    patches = []
    for i in range(n_h):
        for j in range(n_w):
            patch = padded_cube[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :]
            patches.append(patch)
    
    
    return patches, (n_h, n_w), (pad_h, pad_w)

def process_patches(patches, model, input_scaler, gt_scaler):
    print('Making predictions')
    predictions = []
    model.eval()
    # Move model to CPU
    device = torch.device("cpu")
    model.to(device)
    with torch.no_grad():
        for i, patch in enumerate(patches):
            try:
                # Print the shape of the patch
                # print(f"Patch {i} shape: {patch.shape}")
                
                # Transform and prepare the input tensor
                input_tensor = input_scaler.transform(torch.from_numpy(patch).float())
                
                # Add batch and channel dimensions if needed
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
                elif input_tensor.dim() == 4:
                    input_tensor = input_tensor.unsqueeze(0)
                
                # Print the shape of the input tensor
                # print(f"Input tensor {i} shape: {input_tensor.shape}")
                
                # Process the patch
                output, _ ,_  = model(input_tensor)
                output = gt_scaler.inverse_transform(output)
                
                predictions.append(output.squeeze().cpu().numpy())
            except Exception as e:
                print(f"Error processing patch {i}: {str(e)}")
                raise
    return predictions


def reconstruct_prediction(predictions, original_shape, patch_info, padding_info):
    """
    Reconstruct the full prediction from patches.
    
    :param predictions: List of predictions (2D)
    :param original_shape: Original shape of the input cube (H, W, D)
    :param patch_info: Tuple of (n_h, n_w) - number of patches in each dimension
    :param padding_info: Tuple of (pad_h, pad_w) - padding in each dimension
    :return: Reconstructed prediction (2D)
    """
    H, W, _ = original_shape

    n_h, n_w  = patch_info
    pad_h, pad_w  = padding_info
    ph, pw =( H + pad_h) // n_h, (W + pad_w) // n_w
    print( H , W )
    print(pad_h, pad_w)
    print(ph, pw)
    
    full_prediction = np.zeros((H + pad_h, W + pad_w))
    
    idx = 0
    for i in range(n_h):
        for j in range(n_w):
            full_prediction[i*ph:(i+1)*ph, j*pw:(j+1)*pw] = predictions[idx]
            idx += 1
    
    # Remove padding
    full_prediction = full_prediction[:H, :W]
    
    return full_prediction



def visualize_with_quadrants(full_prediction, ground_truth, cube_shape, log=True):
    # Ensure ground truth and prediction have the same shape
    min_height = min(full_prediction.shape[0], ground_truth.shape[0])
    min_width = min(full_prediction.shape[1], ground_truth.shape[1])

    full_prediction = full_prediction[:min_height, :min_width]
    ground_truth = ground_truth[:min_height, :min_width]

    # Calculate error
    error = np.abs(full_prediction - ground_truth)

    # Calculate quadrant boundaries
    height, width = full_prediction.shape
    mid_h, mid_w = height // 2, width // 2

    # Visualize the full prediction, ground truth, and error
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(44, 8))
    fig.suptitle(f"Prediction, Ground Truth, and Absolute Error", fontsize=16)
    fs = 10 

    # Plot prediction
    im1 = ax1.imshow(full_prediction, cmap='viridis', aspect='auto')
    ax1.set_title("Full Prediction", fontsize=14)
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    # Plot ground truth
    im2 = ax2.imshow(ground_truth, cmap='viridis', aspect='auto')
    ax2.set_title("Ground Truth", fontsize=14)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Plot error
    im3 = ax3.imshow(error, cmap='hot', aspect='auto')
    ax3.set_title("Absolute Error", fontsize=14)
    plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    # Add quadrant lines and labels to all plots
    for ax in [ax1, ax2, ax3]:
        ax.axhline(y=mid_h, color='r', linestyle='-', linewidth=0.5)
        ax.axvline(x=mid_w, color='r', linestyle='-', linewidth=0.5)

        ax.text(mid_w/2, mid_h/2, 'Train', color='white', ha='center', va='center', fontsize=fs, fontweight='bold')
        ax.text(mid_w*1.5, mid_h/2, 'Train', color='white', ha='center', va='center', fontsize=fs, fontweight='bold')
        ax.text(mid_w/2, mid_h*1.5, 'Val', color='white', ha='center', va='center', fontsize=fs, fontweight='bold')
        ax.text(mid_w*1.5, mid_h*1.5, 'Test', color='white', ha='center', va='center', fontsize=fs, fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
        plt.savefig(tmp_file.name, format='png', dpi=300, bbox_inches='tight')
    

    print('Start log')
    if log:
        plt.close(fig)
        # Log the image using the temporary file path
        wandb.log({"Prediction_GroundTruth_Error": wandb.Image(tmp_file.name)})

    # Remove the temporary file
    os.unlink(tmp_file.name)

    # Calculate and print Mean Absolute Error
    mae = np.mean(error)
    print(f"Overall Mean Absolute Error: {mae:.4f}")

    # Calculate and print Mean Absolute Error for each quadrant
    quadrant_names = ['Train (top-left)', 'Train (top-right)', 'Val (bottom-left)', 'Test (bottom-right)']
    quadrant_maes = {}
    for i in range(2):
        for j in range(2):
            quad_error = error[i*mid_h:(i+1)*mid_h, j*mid_w:(j+1)*mid_w]
            quad_mae = np.mean(quad_error)
            quadrant_name = quadrant_names[i*2+j]
            print(f"Quadrant MAE - {quadrant_name}: {quad_mae:.4f}")
            quadrant_maes[f"MAE_{quadrant_name}"] = quad_mae

    if log:
        # Log MAE values to wandb
        wandb.log({
            f"Overall_MAE": mae,
            **{f"{k}_": v for k, v in quadrant_maes.items()}
        })

    return error




def plot_3d_lidar_prediction(prediction, ground_truth, cube_shape, log_to_wandb=True, elev=30, azim=45):
    """
    Create two 3D surface plots of the LiDAR prediction and ground truth with correct aspect ratio,
    handling NaN values, and optional logging to wandb.
    
    :param prediction: 2D numpy array of the predicted LiDAR data
    :param ground_truth: 2D numpy array of the ground truth LiDAR data
    :param cube_shape: Original shape of the input cube (H, W, D)
    :param log_to_wandb: Boolean, whether to log the plot to wandb (default: True)
    :param elev: Elevation angle for 3D plot view (default: 30)
    :param azim: Azimuth angle for 3D plot view (default: 45)
    """
    y, x = np.mgrid[0:prediction.shape[0], 0:prediction.shape[1]]

    # Calculate the aspect ratio
    aspect_ratio = prediction.shape[1] / prediction.shape[0]

    # Create the figure
    fig = plt.figure(figsize=(10, 20))
    
    cmap = 'RdYlGn'

    # Set the z-limits for both plots
    max_height = max(np.nanmax(prediction), np.nanmax(ground_truth))
    max_height = 35
    min_height = min(np.nanmin(prediction), np.nanmin(ground_truth))

    # Plot prediction
    ax1 = fig.add_subplot(211, projection='3d')
    surf1 = ax1.plot_surface(x, y, np.ma.masked_invalid(prediction), cmap=cmap, edgecolor='none')
    fig.colorbar(surf1, ax=ax1, shrink=0.2, aspect=20, pad=0.1)
    ax1.set_zlim(min_height, max_height)
    ax1.set_title("Predicted LiDAR", y=1.0, pad=10)

    # Plot ground truth
    ax2 = fig.add_subplot(212, projection='3d')
    surf2 = ax2.plot_surface(x, y, np.ma.masked_invalid(ground_truth), cmap=cmap, edgecolor='none')
    fig.colorbar(surf2, ax=ax2, shrink=0.2, aspect=20, pad=0.1)
    ax2.set_zlim(min_height, 35)
    ax2.set_title("Ground Truth LiDAR", y=1.0, pad=10)

    # Set correct aspect ratio and view for both plots
    for ax in [ax1, ax2]:
        ax.set_box_aspect((aspect_ratio, 1, 1))  # Adjust the 0.3 to change the height of the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
    #     ax.view_init(elev=elev, azim=azim)  # Set the viewing angle

    # Adjust subplot positions
    plt.subplots_adjust(left=0.1, right=0.9, bottom=0.05, top=0.95, hspace=0.1)

    if log_to_wandb:
        # Save the plot to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
            plt.savefig(tmp_file.name, format='png', dpi=300, bbox_inches='tight')
        
        plt.close(fig)  # Close the figure to free up memory

        # Log the image using the temporary file path
        wandb.log({"3D_LiDAR_Prediction_vs_GroundTruth": wandb.Image(tmp_file.name)})

        # Remove the temporary file
        os.unlink(tmp_file.name)

        print("3D LiDAR prediction plot logged to wandb")
    else:
        plt.show()
        plt.close(fig)  # Close the figure to free up memory

    return fig  # Return the figure object in case further modifications are needed