from sklearn.metrics import r2_score, mean_absolute_error
import torch
import torch.optim as optim
import wandb
import numpy as np
import h5py 
import os 

# from src.fdlbiomass.tomograph_reconstruction.tomo_dataset_fn import  collate_fn
from src.fdlbiomass.tomograph_reconstruction.visualisatons import * 
from src.fdlbiomass.tomograph_reconstruction.tomo_loader_fns import setup_tomo_dataloaders
# from src.fdlbiomass.tomograph_reconstruction.mae_recon import generate_weights
from src.fdlbiomass.models.td_unet import UNet_2d , ShallowUNet
from src.fdlbiomass.loss_fns import masked_mse_loss_sum

from src.fdlbiomass.tomograph_reconstruction.wandb_fns import EarlyStopping
from src.fdlbiomass.tomograph_reconstruction.image_recon import plot_recon_tomo 
 


import os
import json
from datetime import datetime
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn 


def select_td_model(config, num_channels):
    if config.model_name == 'unet_2d':
        model = UNet_2d(num_channels, out_channels=1)
    elif config.model_name == 'shallow_unet':
        model = ShallowUNet(num_channels, out_channels=1)
    return model 

def save_trained_model(model, base_path):
    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print('saving model at ', timestamp)
    # Create a directory for this run
    run_dir = os.path.join(base_path, 'models', f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save the model state dict
    model_path = os.path.join(run_dir,   "model_state_dict.pth")
    torch.save(model.state_dict(), model_path)

    print(f"Model state dict saved in {model_path}")
    wandb.save(model_path)

    # Optionally, you can log the save path to wandb
    wandb.log({"model_save_path": model_path})

    return model_path


def full_loop_train_model():
    torch.cuda.empty_cache()
    config_defaults = {
        'epochs': 5,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'polarimetric_channel': 'HH',
        'heading':'NW', 
        'batchnorm': 'no',
        'dropout_rate': 0.0,
        "patch_stride_pair": {"patch_size": "(64,64,36)", "stride": "(32,32,36)"},
        "scaler_type": "minmax",
        "model_name": "unet2d",
        'role_mapping': {'train': [1,4],'val': '2a','test': '2b'}, 
        'early_stopping_patience': 10,
        'early_stopping_min_delta': 0.001,
        'lr_patience': 5,
        'lr_factor': 0.1,
        'min_lr': 1e-6,
        'gt_zero_thresholddataset':0.15,
        'exclude_height_thresh': 'Yes',
        'height_exclusion':5, 
        'n_images': 28,
        'first_set_SLC': False,
        'random_seed' : 42, 
    }

    base_path = '/home/grace.bcolverd/esl_dev/tomo_recon'
    save_model=False 
    
    wandb.init(config=config_defaults)
    config = wandb.config
    
    patch_size = tuple(map(int, config['patch_stride_pair']['patch_size'].strip('()').split(',')))
    stride = tuple(map(int, config['patch_stride_pair']['stride'].strip('()').split(',')))
    train_loader , val_loader, test_loader, (input_scaler, gt_scaler), (cropped_data, ground_truth) = setup_tomo_dataloaders(config , patch_size, stride, base_path, config.n_images)

    wandb.config.update({
        "optimizer": "Adam",
        "loss_function": "masked_mse_loss_sum",
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    })

    channel_multiplier=1 
    vector_extract_multiplier = 3 
    if isinstance(config.polarimetric_channel, list) and all(isinstance(item, str) for item in config.polarimetric_channel):
        if len(config.polarimetric_channel) >1:
            channel_multiplier = len(config.polarimetric_channel)
        
    num_channels = config.n_images * vector_extract_multiplier * channel_multiplier
    model = select_td_model(config, num_channels)

    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.functional.mse_loss
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_factor, 
                                  patience=config.lr_patience, min_lr=config.min_lr)



    best_model_state_dict,  best_score = train_model(model, train_loader, val_loader, optimizer, criterion, device, config.epochs, gt_scaler, scheduler, config.early_stopping_patience, config.early_stopping_min_delta)
    model.load_state_dict(best_model_state_dict)

    # test_mae, test_loss = test(model, test_loader, criterion, device, weights, cropped_data.shape)
    test_avg_loss, test_avg_mae = test_model(model, test_loader , criterion, device, gt_scaler)
    # create_log_prediction_image(cropped_data_tuple, model, (input_scaler, gt_scaler), patch_size, polarimetric_channels  )
    if save_model is True:
        save_trained_model(model, base_path)
    
    print('starting tomo recon ')
    
    plot_recon_tomo(cropped_data, ground_truth, model,  input_scaler,gt_scaler, patch_size )
    wandb.finish()


def calculate_central_mae(output, target, mask):
    # Assuming output, target, and mask are of shape [batch_size,x,x]
    
    if len(output.shape)==4:
        output = output.squeeze(1)
    
    batch_size, height, width = output.shape
    
    # Calculate the start and end indices for the central 32x32 region
    start_h, start_w = height // 4, width // 4
    end_h, end_w = start_h + height // 2, start_w + width // 2
    
    # Extract the central region
    output_central = output[:,  start_h:end_h, start_w:end_w]
    target_central = target[:,  start_h:end_h, start_w:end_w]
    mask_central = mask[:,  start_h:end_h, start_w:end_w]
    
    # Calculate MAE only for the central region
    mae = torch.abs(output_central - target_central) * mask_central
    mae = mae.sum() / (mask_central.sum() + 1e-8)  # Adding small epsilon to avoid division by zero
    
    return mae


# Training function (same as before)
def train_model(model, train_loader, val_loader,  optimizer, criterion, device, num_epochs=10, gt_scaler=None, scheduler=None, early_stopping_patience=10, early_stopping_min_delta=0.001 ):
    """
    Implement mixed precision training as shown in https://pytorch.org/blog/what-every-user-should-know-about-mixed-precision-training-in-pytorch/
    """
    early_stopping = EarlyStopping(patience=early_stopping_patience, min_delta=early_stopping_min_delta)
    scaler = torch.cuda.amp.GradScaler()
    # best_val_mae = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, (data, target, mask) in enumerate(train_loader):
            data, target, mask = data.to(device), target.to(device), mask.to(device)
            
            optimizer.zero_grad()
            # loss = criterion(output[mask], target[mask])
            # with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
            output  = model(data)
            output = output.squeeze(1)
            
            loss = criterion(output, target)
            # loss.backward()
            scaler.scale(loss).backward()
            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        avg_loss = train_loss / len(train_loader)
        wandb.log({"epoch": epoch + 1, "train_loss": avg_loss,})

        # Validation
        model.eval()
        val_loss = 0.0
        val_mae=0.0

        with torch.no_grad():
            for data, target, mask in val_loader:
                data, target, mask = data.to(device), target.to(device), mask.to(device)
                # with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                output = model(data)
                output = output.squeeze(1)
                         # Inverse transform outputs and targets for metric calculation
                vloss = criterion(output, target) 
                
                output = gt_scaler.inverse_transform(output)
                target = gt_scaler.inverse_transform(target)

                mae = calculate_central_mae(output, target, mask)
                val_loss += vloss
                val_mae += mae.item()
        
        # Calculate average validation metrics
        val_loss /= len(val_loader)
        val_mae /= len(val_loader)
        wandb.log({"val_loss": val_loss, "val_mae": val_mae})
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training loss {avg_loss:.4f}')
        print(f'val Loss: {val_loss:.4f}')
        print(f'Validation MAE: {val_mae:.4f}')

        # Learning rate adjustment
        if scheduler is not None:
            print('Learning rate adjustment? ')
            scheduler.step(val_mae)
            current_lr = optimizer.param_groups[0]['lr']
            wandb.log({"learning_rate": current_lr})
            print(f'Current Learning Rate: {current_lr:.6f}')

       # Early stopping
        best_model, best_epoch = early_stopping(val_mae, model, epoch)
        if early_stopping.early_stop:
            print(f"Early stopping triggered. Best model was from epoch {best_epoch + 1}")
            break

        # # Save the best model
        # if val_mae < best_val_mae:
        #     best_val_mae = val_mae
        #     torch.save(model.state_dict(), 'best_model.pth')
        #     print("Saved best model so far ")


        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Training Loss: {train_loss/len(train_loader):.4f}')
        
    print("Training completed!")
    # After training
    wandb.log({
        "best_val_mae": -early_stopping.best_score,
        "best_val_epoch": early_stopping.best_epoch
    })
    return best_model, early_stopping.best_score
    
    

from sklearn.metrics import mean_squared_error, r2_score

def test_model(model, test_loader, criterion, device, gt_scaler):
    model.eval()
    total_mae = 0.0
    total_loss = 0.0 
    total_samples = 0
    # all_predictions = []
    # all_targets = []
    all_outputs = []
    all_targets = []
    

    with torch.no_grad():
        for data, target, mask in test_loader:
            data, target, mask = data.to(device), target.to(device), mask.to(device)
            output = model(data)
            output = output.squeeze(1)
            loss = criterion(output, target)
            output = gt_scaler.inverse_transform(output)
            target = gt_scaler.inverse_transform(target)
            
            mae = calculate_central_mae(output, target, mask)
            total_mae += mae 
            total_loss+=loss
            total_samples += data.size(0)

            output = output.squeeze(dim=1)
            # Store predictions and targets for further analysis
            # all_predictions.append(output[mask].cpu().numpy())
            # all_targets.append(target[mask].cpu().numpy())

            all_outputs.extend(output[mask].cpu().numpy())
            all_targets.extend(target[mask].cpu().numpy())
    
    print('total samples' , total_samples)
    avg_mae = total_mae /  len(test_loader)
    avg_loss = total_loss / len(test_loader)
    # all_predictions = np.concatenate(all_predictions)

    # all_targets = np.concatenate(all_targets)

    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    test_rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))
    r2 = r2_score(all_targets, all_outputs)


    print(f"Test Loss: {avg_loss:.4f}")
    print(f"Test avg_mae: {avg_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")

    # Log metrics to wandb
    wandb.log({
        "test_mae": avg_mae,
        "test_loss": avg_loss,
        "test_rmse": test_rmse,
        "test_r2": r2
    })

    

    return avg_loss, avg_mae


