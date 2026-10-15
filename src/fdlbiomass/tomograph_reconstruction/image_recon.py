import wandb 
import tempfile
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os 

import torch
import numpy as np
import matplotlib.pyplot as plt

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import numpy as np

def create_patches(cube, patch_size):
    """
    Create patches from the input cube with stride half the patch size.
    
    :param cube: Input 3D array of shape (H, W, D)
    :param patch_size: Tuple of (patch_height, patch_width)
    :return: List of patches, patch info, and padding info
    """
    print('Generating patches')
    H, W, D = cube.shape
    ph, pw = patch_size if isinstance(patch_size, tuple) else map(int, patch_size.strip('()').split(','))
    
    # Calculate stride (half the patch size)
    stride_h, stride_w = ph // 2, pw // 2
    
    # Calculate padding
    pad_h = (stride_h - H % stride_h) % stride_h
    pad_w = (stride_w - W % stride_w) % stride_w
    
    # Pad the cube
    padded_cube = np.pad(cube, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    
    # Calculate number of patches in each dimension
    n_h = (H + pad_h) // stride_h
    n_w = (W + pad_w) // stride_w
    
    patches = []
    positions = []
    for i in range(n_h):
        for j in range(n_w):
            y_start, x_start = i * stride_h, j * stride_w
            patch = padded_cube[y_start:y_start+ph, x_start:x_start+pw, :]
            patches.append(patch)
            positions.append((y_start, x_start))
    
    return patches, positions, (n_h, n_w), (pad_h, pad_w)


def reconstruct_prediction(predictions, positions, original_shape, patch_info, padding_info, patch_size):
    """
    Reconstruct the full 2D prediction from non-overlapping central parts of patches.
    
    :param predictions: List of 2D predictions (patch_height x patch_width)
    :param positions: List of (y, x) positions for each patch
    :param original_shape: Original shape of the input image (H, W)
    :param patch_info: Tuple of (n_h, n_w) - number of patches in each dimension
    :param padding_info: Tuple of (pad_h, pad_w) - padding in each dimension
    :param patch_size: Tuple of (patch_height, patch_width)
    :return: Reconstructed 2D prediction (H x W)
    """
    H, W = original_shape[:2]  # We only need the first two dimensions
    pad_h, pad_w = padding_info
    ph, pw = patch_size
    
    # Calculate stride (half the patch size)
    stride_h, stride_w = ph // 2, pw // 2
    
    # Initialize the reconstruction array
    full_prediction = np.zeros((H + pad_h, W + pad_w))
    
    for pred, (y, x) in zip(predictions, positions):
        # Extract the central part of the prediction
        central_pred = pred[stride_h:stride_h*2, stride_w:stride_w*2]
        
        # Place the central prediction in the full prediction array
        full_prediction[y+stride_h:y+stride_h*2, x+stride_w:x+stride_w*2] = central_pred
    
    # Remove padding
    full_prediction = full_prediction[:H, :W]
    
    return full_prediction

# Update the plot_recon_tomo function to use the new implementation
def plot_recon_tomo(cropped_data, ground_truth, model, input_scaler, gt_scaler, patch_size, wandb_log=True, local_save=False, save_path=None):
    patches, positions, patch_info, padding_info = create_patches(cropped_data, patch_size)
    predictions = process_patches(patches, model, input_scaler, gt_scaler)
    full_prediction = reconstruct_prediction(predictions, positions, ground_truth.shape, patch_info, padding_info, patch_size)
    visualize_quadrants_multiple_ax(full_prediction, ground_truth, wandb_log=wandb_log, local_save=local_save, save_path=save_path)



# def plot_recon_tomo(cropped_data, ground_truth, model,  input_scaler,gt_scaler , patch_size):
#     patches, patch_info, padding_info = create_patches(cropped_data, patch_size)
#     predictions = process_patches(patches, model, input_scaler, gt_scaler)
#     full_prediction = reconstruct_prediction(predictions, ground_truth.shape, patch_info, padding_info)
#     visualize_quadrants_multiple_ax(full_prediction, ground_truth, wandb_log= True )

# def create_patches(cube, patch_size):
#     """
#     Create non-overlapping patches from the input cube.
    
#     :param cube: Input 3D array of shape (H, W, D)
#     :param patch_size: Tuple of (patch_height, patch_width, patch_depth)
#     :return: List of patches and padding information
#     """
#     print('Generating patches')
#     H, W, D = cube.shape
#     if isinstance(patch_size, tuple):
#         None  
#     elif isinstance(patch_size, str):
#         patch_size = tuple(map(int, patch_size.strip('()').split(',')))
    
    
#     ph, pw = patch_size
    
#     # Calculate padding
#     pad_h = (ph - H % ph) % ph
#     pad_w = (pw - W % pw) % pw
    
#     # Pad the cube
#     padded_cube = np.pad(cube, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    
#     # Calculate number of patches in each dimension
#     n_h = (H + pad_h) // ph
#     n_w = (W + pad_w) // pw

    
#     patches = []
#     for i in range(n_h):
#         for j in range(n_w):
#             patch = padded_cube[i*ph:(i+1)*ph, j*pw:(j+1)*pw, :]
#             patches.append(patch)
    
    
#     return patches, (n_h, n_w), (pad_h, pad_w)

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
                input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
                
                # Print the shape of the input tensor
                # print(f"Input tensor {i} shape: {input_tensor.shape}")
                
                # Process the patch
                output  = model(input_tensor)
                output = gt_scaler.inverse_transform(output)
                
                predictions.append(output.squeeze().cpu().numpy())
            except Exception as e:
                print(f"Error processing patch {i}: {str(e)}")
                raise
    return predictions



# def reconstruct_prediction(predictions, original_shape, patch_info, padding_info):
#     """
#     Reconstruct the full prediction from patches.
    
#     :param predictions: List of predictions (2D)
#     :param original_shape: Original shape of the input cube (H, W, D)
#     :param patch_info: Tuple of (n_h, n_w) - number of patches in each dimension
#     :param padding_info: Tuple of (pad_h, pad_w) - padding in each dimension
#     :return: Reconstructed prediction (2D)
#     """
#     H, W = original_shape

#     n_h, n_w  = patch_info
#     pad_h, pad_w  = padding_info
#     ph, pw =( H + pad_h) // n_h, (W + pad_w) // n_w
#     print( H , W )
#     print(pad_h, pad_w)
#     print(ph, pw)
    
#     full_prediction = np.zeros((H + pad_h, W + pad_w))
    
#     idx = 0
#     for i in range(n_h):
#         for j in range(n_w):
#             full_prediction[i*ph:(i+1)*ph, j*pw:(j+1)*pw] = predictions[idx]
#             idx += 1
    
#     # Remove padding
#     full_prediction = full_prediction[:H, :W]
    
#     return full_prediction

def split_data_into_quadrants_imge(full_data, ground_truth, reconstruction=False):
    # First, use the previous function to crop the data

    cropped_data=full_data
    

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
    q1_data = cropped_data[:top_height, x:]
    q2a_data = cropped_data[:y2, :left_width]
    q2b_data = cropped_data[y2:top_height, :left_width]
    q3_data = cropped_data[y:, :left_width]
    q4_data = cropped_data[y:, x:]

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


import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import wandb
import os

def visualize_quadrants_multiple_ax(full_prediction, ground_truth, wandb_log=False, local_save=False, save_path=None):
    # Split the data into quadrants
    quadrants = split_data_into_quadrants_imge(full_prediction, ground_truth, reconstruction=True)
    
    # Create a new figure with three subplots
    fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    
    # Calculate the full error map
    full_error = np.abs(full_prediction - ground_truth)
    
    # Define data for each subplot
    plot_data = [
        (full_prediction, "Prediction", 'viridis'),
        (ground_truth, "Ground Truth", 'viridis'),
        (full_error, "Absolute Error", 'hot')
    ]
    # Define colors for quadrant borders
    colors = {'q1': 'red', 'q2a': 'blue', 'q2b': 'green', 'q3': 'yellow', 'q4': 'magenta'}
    
    for idx, (data, title, cmap) in enumerate(plot_data):
        ax = axs[idx]
        
        # Plot the data
        im = ax.imshow(data, cmap=cmap, aspect='auto')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Add rectangles and labels for each quadrant
        for quad, (pred, truth) in quadrants.items():
            y, x = pred.shape[:2]
            if quad == 'q1':
                rect = Rectangle((data.shape[1]//2, 0), x, y, fill=False, edgecolor=colors[quad], linewidth=2)
            elif quad == 'q2a':
                rect = Rectangle((0, 0), x, y, fill=False, edgecolor=colors[quad], linewidth=2)
            elif quad == 'q2b':
                rect = Rectangle((0, truth.shape[0]), x, y, fill=False, edgecolor=colors[quad], linewidth=2)
            elif quad == 'q3':
                rect = Rectangle((0, data.shape[0] - y), x, y, fill=False, edgecolor=colors[quad], linewidth=2)
            else:  # q4
                rect = Rectangle((data.shape[1]//2, data.shape[0] - y), x, y, fill=False, edgecolor=colors[quad], linewidth=2)
            
            ax.add_patch(rect)
            
            # Add label
            cx = rect.get_x() + rect.get_width()/2
            cy = rect.get_y() + rect.get_height()/2
            ax.text(cx, cy, quad.upper(), ha='center', va='center', fontsize=12, color=colors[quad], 
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
        # Set title
        ax.set_title(f"{title} with Quadrants", fontsize=16)
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Adjust layout and display
    plt.tight_layout()
    
    # Log to wandb if requested
    if wandb_log:
        wandb.log({"quadrant_visualization": wandb.Image(fig)})
        print("Figure logged to wandb")
    
    # Save locally if requested
    if local_save:
        if save_path is None:
            save_path = 'quadrant_visualization.png'
        else:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved locally to {save_path}")
    
    # Close the figure to free up memory
    plt.close(fig)

# def visualize_quadrants_multiple_ax(full_prediction, ground_truth, wandb_log, ):
#     # Split the data into quadrants
#     quadrants = split_data_into_quadrants_imge(full_prediction, ground_truth, reconstruction=True)
    
#     # Create a new figure with three subplots
#     fig, axs = plt.subplots(1, 3, figsize=(30, 10))
    
#     # Calculate the full error map
#     full_error = np.abs(full_prediction - ground_truth)
    
#     # Define data for each subplot
#     plot_data = [
#         (full_prediction, "Prediction", 'viridis'),
#         (ground_truth, "Ground Truth", 'viridis'),
#         (full_error, "Absolute Error", 'hot')
#     ]
#     # Define colors for quadrant borders
#     colors = {'q1': 'red', 'q2a': 'blue', 'q2b': 'green', 'q3': 'yellow', 'q4': 'magenta'}
    
#     for idx, (data, title, cmap) in enumerate(plot_data):
#         ax = axs[idx]
        
#         # Plot the data
#         im = ax.imshow(data, cmap=cmap, aspect='auto')
        
#         # Add colorbar
#         plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
#         # Add rectangles and labels for each quadrant
#         for quad, (pred, truth) in quadrants.items():
#             y, x = pred.shape[:2]
#             if quad == 'q1':
#                 rect = Rectangle((data.shape[1]//2, 0), x, y, fill=False, edgecolor=colors[quad], linewidth=2)
#             elif quad == 'q2a':
#                 rect = Rectangle((0, 0), x, y, fill=False, edgecolor=colors[quad], linewidth=2)
#             elif quad == 'q2b':
#                 rect = Rectangle((0, truth.shape[0]), x, y, fill=False, edgecolor=colors[quad], linewidth=2)
#             elif quad == 'q3':
#                 rect = Rectangle((0, data.shape[0] - y), x, y, fill=False, edgecolor=colors[quad], linewidth=2)
#             else:  # q4
#                 rect = Rectangle((data.shape[1]//2, data.shape[0] - y), x, y, fill=False, edgecolor=colors[quad], linewidth=2)
            
#             ax.add_patch(rect)
            
#             # Add label
#             cx = rect.get_x() + rect.get_width()/2
#             cy = rect.get_y() + rect.get_height()/2
#             ax.text(cx, cy, quad.upper(), ha='center', va='center', fontsize=12, color=colors[quad], 
#                     bbox=dict(facecolor='white', edgecolor='none', alpha=0.7))
        
#         # Set title
#         ax.set_title(f"{title} with Quadrants", fontsize=16)
        
#         # Remove axis ticks
#         ax.set_xticks([])
#         ax.set_yticks([])
    
#     # Adjust layout and display
#     plt.tight_layout()
#         # Log to wandb if requested
#     if wandb_log:
#         wandb.log({"quadrant_visualization": wandb.Image(fig)})
#         print("Figure logged to wandb")




