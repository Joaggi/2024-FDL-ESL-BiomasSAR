

from src.fdlbiomass.quadrant_dataset_pband import QuadrantPatchDataset
from src.fdlbiomass.utils import merge_headings , load_biomass_data, split_into_quadrants, crop_to_non_nan
from src.fdlbiomass.unet_model import UNet3D
from src.fdlbiomass.simple_conv import SimpleConv3D

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import wandb
from sklearn.metrics import r2_score, mean_absolute_error
import os 
from dotenv import load_dotenv


def generate_quadrants(parent_path, polarimetric_channel ):

    acquisition_mode = 'monostatic'
    band_opt = 'P'
    heading = 'NW'
    data = load_biomass_data(parent_path, band_opt, heading, acquisition_mode, polarimetric_channel)
      
    # load vals, option here for L band
    # PcubeNormUTM = data['PcubeNormUTM']
    LidarCHM = data['LidarCHM']
    PcubeDeRampedUTM = data['PcubeDeRampedUTM']
    pol_ind = data['pol_ind']

    merged_power_cube = merge_headings(PcubeDeRampedUTM)
    intensity_cube = merged_power_cube[:,:,:,pol_ind]
    
    # Crop LidarCHM
    cropped_LidarCHM = crop_to_non_nan(LidarCHM)

    print(f"Original shape: {LidarCHM.shape}")
    print(f"Cropped shape: {cropped_LidarCHM.shape}")

    
    row_start = np.argmax(~np.all(np.isnan(LidarCHM), axis=1))
    row_end = len(LidarCHM) - np.argmax(~np.all(np.isnan(LidarCHM), axis=1)[::-1])
    col_start = np.argmax(~np.all(np.isnan(LidarCHM), axis=0))
    col_end = len(LidarCHM[0]) - np.argmax(~np.all(np.isnan(LidarCHM), axis=0)[::-1])

    # Crop Intensity
    crop_int = intensity_cube[row_start:row_end, col_start:col_end, : ]

    # Split the array
    quadrants, ground_truths = split_into_quadrants(crop_int , cropped_LidarCHM)
    return quadrants, ground_truths


def generate_dataloaders(quadrants, ground_truths, patch_size, stride, test_quadrant=3, batch_size=4):
    quad_options = [0, 1, 2, 3]
    if test_quadrant not in quad_options:
        raise ValueError('Choose index 0-3 for test')
    
    # Remove test_quadrant from options
    remaining_quadrants = [x for x in quad_options if x != test_quadrant]
    
    # Assign train and val quadrants
    train_quadrants = remaining_quadrants[:2]  # Use first two remaining quadrants for training
    val_quadrant = remaining_quadrants[2]  # Use the last remaining quadrant for validation
    
    dataset = QuadrantPatchDataset(
        quadrants, ground_truths, 
        train_quadrants=train_quadrants, 
        test_quadrant=test_quadrant, 
        val_quadrant=val_quadrant, 
        patch_size=patch_size, 
        stride=stride
    )

    # Create DataLoaders
    dataset.set_mode('train')
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    dataset.set_mode('test')
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    dataset.set_mode('val')
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, val_loader, dataset




# Set variables 
parent_path = '/home/grace.bcolverd/'
polarimetric_channel = 'HH'
test_quadrant = 3

num_epochs = 50
lr =  0.001
batch_size = 4 
patch_size = (64, 64, 36)
stride = (32,32,36)
model_name = 'simple_conv' 
# model_name = 'unet3d'

def main():
    # Initalise local variables
    load_dotenv()
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if not wandb_api_key:
            raise ValueError("WANDB_API_KEY not found in environment variables")
    wandb.login(key=wandb_api_key)
    
    # Load data 
    quadrants, ground_truths = generate_quadrants(parent_path, polarimetric_channel)
    train_loader, test_loader, val_loader, dataset= generate_dataloaders(quadrants, ground_truths, patch_size, stride, test_quadrant=test_quadrant, batch_size=batch_size)

    # Initialize wandb
    wandb.init(entity='fdl-biomass', project="3d-unet-chm", config={
        "learning_rate":lr ,
        "epochs": num_epochs,
        "batch_size": batch_size,
        "test_quadrant": test_quadrant, 
        "patch_size": str(patch_size) , 
        "stride": str(stride),
        "modelType": model_name , 
    })

    # Initialize the model, loss function, and optimizer
    if model_name == 'simple_conv':
        model  = SimpleConv3D(in_channels=1) 
    elif model_name =='unet3d':
        model = UNet3D(in_channels=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr )

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        dataset.set_mode('train')
        for batch_input, batch_target in train_loader:
            batch_input = batch_input.unsqueeze(1).to(device)  # Add channel dimension
            batch_target = batch_target.unsqueeze(1).to(device)  # Add channel dimension
            
            optimizer.zero_grad()
            outputs = model(batch_input)
            loss = criterion(outputs, batch_target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            print(f"Batch input range: {batch_input.min().item()} to {batch_input.max().item()}")
            print(f"Batch target range: {batch_target.min().item()} to {batch_target.max().item()}")
            print(f"Model output range: {outputs.min().item()} to {outputs.max().item()}")
            print(f"Loss: {loss.item()}")
        # Validation
        model.eval()
        val_loss = 0
        dataset.set_mode('val')
        with torch.no_grad():
            for batch_input, batch_target in val_loader:
                batch_input = batch_input.unsqueeze(1).to(device)  # Add channel dimension
                batch_target = batch_target.unsqueeze(1).to(device)  # Add channel dimension
                outputs = model(batch_input)
                loss = criterion(outputs, batch_target)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

    print("Training completed!")

    # Evaluation on test set, calc metrics batch wise for mem optimisation 
    model.eval()
    test_loss = 0
    total_samples = 0
    sum_squared_error = 0
    sum_absolute_error = 0
    sum_true = 0
    sum_pred = 0
    sum_true_squared = 0
    sum_pred_squared = 0

    dataset.set_mode('test')
    with torch.no_grad():
        for batch_input, batch_target in test_loader:
            batch_input = batch_input.unsqueeze(1).to(device)
            batch_target = batch_target.unsqueeze(1).to(device)
            outputs = model(batch_input)
            loss = criterion(outputs, batch_target)
            test_loss += loss.item() * batch_input.size(0)
            
            # Move tensors to CPU and convert to numpy for metric calculation
            true = batch_target.cpu().numpy().flatten()
            pred = outputs.cpu().numpy().flatten()
            
            # Update sums for metric calculation
            total_samples += len(true)
            sum_squared_error += np.sum((true - pred) ** 2)
            sum_absolute_error += np.sum(np.abs(true - pred))
            sum_true += np.sum(true)
            sum_pred += np.sum(pred)
            sum_true_squared += np.sum(true ** 2)
            sum_pred_squared += np.sum(pred ** 2)

    # Calculate final metrics
    test_loss /= total_samples
    mse = sum_squared_error / total_samples
    mae = sum_absolute_error / total_samples
    mean_true = sum_true / total_samples
    mean_pred = sum_pred / total_samples
    r2 = 1 - (sum_squared_error / (sum_true_squared - (sum_true ** 2 / total_samples)))

    print(f"Test Loss: {test_loss:.4f}, R2: {r2:.4f}, MAE: {mae:.4f}")

    # Log final test metrics
    wandb.log({
        "test_loss": test_loss,
        "test_r2": r2,
        "test_mae": mae
    })

    # Save the model
    torch.save(model.state_dict(), "3d_unet_chm_model.pth")

    # Finish wandb run
    wandb.finish()


if __name__ == "__main__": 
    main()