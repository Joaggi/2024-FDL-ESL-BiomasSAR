import numpy as np
import torch
import torch.nn as nn
import wandb 
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy 

import copy

class EarlyStopping:
    """
    Early stopping to prevent overfitting in training neural networks.

    This class implements the early stopping mechanism, which monitors a given metric
    (either Mean Absolute Error or Root Mean Squared Error) and stops the training process 
    if no improvement is seen for a specified number of consecutive epochs.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        metric (str): The metric to use for early stopping ('mae' or 'rmse').
        counter (int): Counter for epochs with no improvement.
        best_score (float): Best score observed during training.
        early_stop (bool): Flag indicating whether to stop training.
        best_model (dict): State dictionary of the best model.
        best_epoch (int): Epoch number where the best score was achieved.
        best_mae (float): Best MAE score.
        best_rmse (float): Best RMSE score.

    Methods:
        __call__(val_mae, val_rmse, model, epoch): Check for improvement and update early stopping state.
    """
    def __init__(self, patience=10, min_delta=0, metric='mae'):
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric.lower()
        if self.metric not in ['mae', 'rmse']:
            raise ValueError("Metric must be either 'mae' or 'rmse'")
        self.counter = 0
        self.best_score = float('inf')  # Initialize with infinity for minimization
        self.early_stop = False
        self.best_model = None
        self.best_epoch = None
        self.best_mae = float('inf')
        self.best_rmse = float('inf')

    def __call__(self, val_mae, val_rmse, model, epoch):
        """
        Check if the model has improved and update the early stopping state.

        This method is called after each epoch to determine if the model has improved.
        If there's no improvement for 'patience' number of epochs, it sets the
        early_stop flag to True.

        Args:
            val_mae (float): Validation Mean Absolute Error for the current epoch.
            val_rmse (float): Validation Root Mean Squared Error for the current epoch.
            model (torch.nn.Module): The current model.
            epoch (int): The current epoch number.

        Returns:
            tuple: A tuple containing the best model state_dict, the best epoch number,
                   the best MAE, and the best RMSE.
        """
        current_score = val_mae if self.metric == 'mae' else val_rmse
        print(f"Current {self.metric.upper()}: {current_score}, Best {self.metric.upper()}: {self.best_score}")

        if current_score < self.best_score - self.min_delta:  # Improvement
            print(f"Improvement found in {self.metric.upper()}")
            self.best_score = current_score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.best_mae = val_mae
            self.best_rmse = val_rmse
            self.counter = 0
        else:  # No improvement
            print(f"No improvement in {self.metric.upper()}")
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        print(f"After check - Best {self.metric.upper()}: {self.best_score}, Counter: {self.counter}")
        return self.best_model, self.best_epoch, self.best_mae, self.best_rmse

# class EarlyStopping:
#     """
#     Early stopping to prevent overfitting in training neural networks.

#     This class implements the early stopping mechanism, which monitors a given metric
#     (in this case, Mean Absolute Error) and stops the training process if no improvement
#     is seen for a specified number of consecutive epochs.

#     Attributes:
#         patience (int): Number of epochs to wait for improvement before stopping.
#         min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
#         counter (int): Counter for epochs with no improvement.
#         best_score (float): Best score observed during training.
#         early_stop (bool): Flag indicating whether to stop training.
#         best_model (dict): State dictionary of the best model.
#         best_epoch (int): Epoch number where the best score was achieved.

#     Methods:
#         __call__(val_mae, model, epoch): Check for improvement and update early stopping state.
#     """
#     def __init__(self, patience=10, min_delta=0):
#         self.patience = patience
#         self.min_delta = min_delta
#         self.counter = 0
#         self.best_score = None
#         self.early_stop = False
#         self.best_model = None
#         self.best_epoch = None

#     def __call__(self, val_mae, model, epoch):
#         """
#         Check if the model has improved and update the early stopping state.

#         This method is called after each epoch to determine if the model has improved.
#         If there's no improvement for 'patience' number of epochs, it sets the
#         early_stop flag to True.
#         We invert mae to move to 'higher is better'

#         Args:
#             val_mae (float): Validation Mean Absolute Error for the current epoch.
#             model (torch.nn.Module): The current model.
#             epoch (int): The current epoch number.

#         Returns:
#             tuple: A tuple containing the best model state_dict and the best epoch number.
#         """
#         score = -val_mae
#         print(f"Current score: {score}, Best score: {self.best_score}, Counter: {self.counter}")
#         if self.best_score is None:
#             self.best_score = score
#             self.best_model = copy.deepcopy(model.state_dict())
#             self.best_epoch = epoch
#             print("Initial best score set")
#         elif score > self.best_score + self.min_delta:  # Improvement
#             print("Improvement found")
#             self.best_score = score
#             self.best_model = copy.deepcopy(model.state_dict())
#             self.best_epoch = epoch
#             self.counter = 0
#         else:  # No improvement
#             print("No improvement")
#             self.counter += 1
#             if self.counter >= self.patience:
#                 self.early_stop = True
#         print(f"After check - Best score: {self.best_score}, Counter: {self.counter}")
#         return self.best_model, self.best_epoch
    
    
def log_images_to_wandb(image_pairs, label, gt_scaler, sample_idx=0):
    """
    Log input images, their reconstructed versions, and ground truth to Weights & Biases (wandb) for visualization.
    
    This function processes up to 8 batches of image pairs, preparing them for logging to wandb.
    It creates a single figure with three subplots for each batch, showing the input slice,
    reconstruction slice, and reconstruction mean, all with ground truth overlaid.
    
    Args:
        image_pairs (list of tuples): A list where each tuple contains an input image,
                                      its reconstructed version, and the ground truth for a batch.
                                      Each image should be a torch.Tensor.
        label (str): A label to identify the dataset, typically 'train' or 'val'.
        gt_scaler: The scaler used to normalize the ground truth data.
        sample_idx (int, optional): The index of the sample within each batch to plot.
                                    Defaults to 0.
    
    Returns:
        None
    
    Raises:
        ValueError: If either the input_image or reconstructed_image is not a torch.Tensor.
    """
    
    figures = []
    print('Len of image pairs is ', len(image_pairs))
    
    def normalize_image(img):
        """Normalize image to 0-255 range, handling edge cases."""
        img_min, img_max = np.nanmin(img), np.nanmax(img)
        if img_min == img_max or np.isnan(img_min) or np.isnan(img_max):
            return np.zeros_like(img, dtype=np.uint8)
        else:
            normalized = (img - img_min) / (img_max - img_min)
            return (normalized * 255).clip(0, 255).astype(np.uint8)
    
    for i, (input_image, reconstructed_image, batch_target) in enumerate(image_pairs):
        if i >= 8:
            break
        if not isinstance(input_image, torch.Tensor):
            raise ValueError("input_image must be a torch.Tensor")
        if not isinstance(reconstructed_image, torch.Tensor):
            raise ValueError("reconstructed_image must be a torch.Tensor")
        
        bt = gt_scaler.inverse_transform(batch_target)
        bt = bt[sample_idx, :, bt.shape[2]//2].detach().cpu().numpy() 
        
        input_slice = input_image[sample_idx, 0].detach().cpu().numpy()
        recon_slice = reconstructed_image[sample_idx, 0].detach().cpu().numpy()
        recon_slice_mean = reconstructed_image[sample_idx].mean(dim=0).detach().cpu().numpy()
        
        # Normalize images
        input_slice = normalize_image(input_slice)
        recon_slice = normalize_image(recon_slice)
        recon_slice_mean = normalize_image(recon_slice_mean)
        
        # Create figure with three subplots
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Sample {i+1}", fontsize=20)
        lw = 2
        font_size = 14

        # Input slice (Ch 0)
        axs[0].imshow(np.transpose(input_slice), origin='lower', cmap='gray')
        axs[0].plot(range(len(bt)), bt, color='red', linewidth=lw)
        axs[0].set_title("Input (ch=0)", fontsize=font_size)
        axs[0].tick_params(axis='both', which='major', labelsize=font_size)

        # Reconstruction slice (Ch 0)
        axs[1].imshow(np.transpose(recon_slice), origin='lower', cmap='gray')
        axs[1].plot(range(len(bt)), bt, color='red', linewidth=lw)
        axs[1].set_title("Reconstruction (ch=0)", fontsize=font_size)
        axs[1].tick_params(axis='both', which='major', labelsize=font_size)

        # Reconstruction mean
        axs[2].imshow(np.transpose(recon_slice_mean), origin='lower', cmap='gray')
        axs[2].plot(range(len(bt)), bt, color='red', linewidth=lw)
        axs[2].set_title("Reconstruction (mean over channels)", fontsize=font_size)
        axs[2].tick_params(axis='both', which='major', labelsize=font_size)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        figures.append(wandb.Image(fig))
        plt.close(fig)
    
    # Log the figures to WandB
    wandb.log({f"{label}_Combined_Visualizations": figures})
    
