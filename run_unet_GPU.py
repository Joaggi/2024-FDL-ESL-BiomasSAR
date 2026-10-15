import argparse


from src.fdlbiomass.models.unet_model import UNet3D, UNet3D_MP
from src.fdlbiomass.models.simple_unet import simple_UNet3D2 
from src.fdlbiomass.models.simple_resnet import simple_UNet3D2_Residual 
from src.fdlbiomass.dataloader_fns import generate_quadrants, generate_dataloaders, convert_pol_ch
from src.fdlbiomass.wandb_fns import EarlyStopping, log_images_to_wandb
from src.fdlbiomass.loss_fns import calculate_mae, select_loss
from src.fdlbiomass.image_reconstruction import create_log_prediction_image 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from sklearn.metrics import r2_score, mean_absolute_error
import os 
from dotenv import load_dotenv
import torch.optim as optim
from torch.cuda.amp import autocast
from torch.profiler import profile, record_function, ProfilerActivity
# import ast 


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)





def initialize_model(args, num_channels):
    """
    Initialize the model based on the provided arguments.

    Args:
    - args: An argument parser object containing model_name, batchnorm, and dropout_rate.
    - num_channels: Number of input channels for the model.

    Returns:
    - model: The initialized model.
    - batchnorm_bool: Boolean indicating if batch normalization is used or 'N/A' if not applicable.
    - dropout_rate: The dropout rate used in the model or 'N/A' if not applicable.
    """

    if args.model_name in ['unet3d', 'unet_resnet', 'simple_unet2', 'unet3d_mp']:
        if args.batchnorm == 'no':
            batchnorm_bool = False
        elif args.batchnorm == 'yes':
            batchnorm_bool = True
        else:
            raise ValueError('Wrong type of batch norm')
        
        print('Batch norm:', batchnorm_bool)
        dropout_rate = args.dropout_rate
        
        if args.model_name == 'unet3d':
            model = UNet3D(in_channels=num_channels, use_batchnorm=batchnorm_bool, dropout_rate=dropout_rate, z_collapse_method= args.z_collapse_method)
        elif args.model_name == 'simple_unet2':
            model = simple_UNet3D2(in_channels=num_channels, use_batchnorm=batchnorm_bool, dropout_rate=dropout_rate, z_collapse_method= args.z_collapse_method)
        elif args.model_name =='unet3d_mp':
            model = UNet3D_MP(in_channels=num_channels, use_batchnorm=batchnorm_bool, dropout_rate=dropout_rate, z_collapse_method= args.z_collapse_method)
        elif args.model_name =='unet_resnet':
            model = simple_UNet3D2_Residual(in_channels=num_channels, use_batchnorm=batchnorm_bool, dropout_rate=dropout_rate)
    else:
        raise ValueError('Unsupported model name')
    
    return model, batchnorm_bool, dropout_rate



def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs, scalers, early_stopping_metric):
    set_seed(42) 
    input_scaler, gt_scaler = scalers 
    early_stopping = EarlyStopping(patience=10, min_delta=0.001,  metric=early_stopping_metric)
    best_model = None
    best_val_mae = float('inf')
    
    for epoch in range(num_epochs):
        image_pairs=[] 
        model.train()
        epoch_loss = 0
        total_valid_elements = 0

        
        for batch_idx, (batch_input, batch_target, mask) in enumerate(train_loader):
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            mask = mask.to(device)
            

            optimizer.zero_grad()
            
            with autocast():
                outputs, input_slice, reconstructed_slice = model(batch_input)
                _, loss = criterion(outputs, batch_target, mask)

            loss.backward()
            optimizer.step()
            
            # Check for NaN values
            if torch.isnan(loss):
                print(f"NaN loss detected at epoch {epoch}, batch {batch_idx}")
                print(f"Input range: {batch_input.min().item()} to {batch_input.max().item()}")
                print(f"Target range: {batch_target.min().item()} to {batch_target.max().item()}")
                print(f"Output range: {outputs.min().item()} to {outputs.max().item()}")
                raise ValueError("NaN loss detected")
            
            total_valid_elements += mask.sum().item()
            epoch_loss += loss.item()
            image_pairs.append((input_slice, reconstructed_slice, batch_target))
            
        if epoch % 5 ==0:
            log_images_to_wandb(image_pairs, 'train', gt_scaler)
         
        # Log average losses for the epoch
        avg_loss = epoch_loss / len(train_loader)
        
        
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_loss,
        })
        
        # Validation
        val_loss, val_mae, val_rmse = validate(model, val_loader, criterion, device, gt_scaler)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {avg_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val MAE: {val_mae:.4f}",
        f"Val RMSE: {val_rmse:.4f}")

        best_model, best_epoch, best_mae, best_rmse = early_stopping(val_mae, val_rmse, model, epoch)

        if early_stopping.early_stop:
            print(f"Early stopping triggered. Best model was from epoch {best_epoch + 1}")
            print(f"for best epoch:   MAE: {best_mae},   RMSE: {best_rmse}")
            break
        

    print("Training completed!")
    # After training
    
    wandb.log({
        "early_stopping_score": -early_stopping.best_score,
        "best_val_rmse": best_rmse, 
        "best_val_mae": best_mae, 
        "best_val_epoch": early_stopping.best_epoch
    })
   

    return best_model, early_stopping.best_score

import torch
import numpy as np
from sklearn.metrics import mean_squared_error
import wandb

def validate(model, val_loader, criterion, device, gt_scaler):
    model.eval()
    val_loss = 0
    total_mae = 0
    total_valid_elements = 0
    all_outputs = []
    all_targets = []

    with torch.no_grad():
        for batch_input, batch_target, mask in val_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            mask = mask.to(device)

            outputs, _, _ = model(batch_input)
            _, loss = criterion(outputs, batch_target, mask)
            
            # Inverse transform outputs and targets for metric calculation
            outputs = gt_scaler.inverse_transform(outputs)
            batch_target = gt_scaler.inverse_transform(batch_target)
            
            val_loss += loss.item()
            
            # Calculate MAE
            batch_mae, batch_total_mae, batch_valid_elements = calculate_mae(outputs, batch_target, mask)
            
            total_mae += batch_total_mae
            total_valid_elements += batch_valid_elements
            
            # Collect outputs and targets for RMSE calculation
            all_outputs.extend(outputs[mask].cpu().numpy())
            all_targets.extend(batch_target[mask].cpu().numpy())
        
    avg_val_loss = val_loss / len(val_loader)
    avg_val_mae = total_mae / total_valid_elements if total_valid_elements > 0 else 0

    # Calculate RMSE
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)
    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))

    wandb.log({
        "val_loss": avg_val_loss,
        "val_mae": avg_val_mae,
        "val_rmse": rmse,
        "val_valid_elements": total_valid_elements
    })

    return avg_val_loss, avg_val_mae, rmse


import torch
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
import wandb

def test(model, test_loader, criterion, device, gt_scaler):
    model.eval()
    model = model.to(device)
    test_loss = 0
    test_mse = 0
    all_outputs = []
    all_targets = []
    all_masks = []
    total_valid_elements = 0
    total_mae = 0
    
    with torch.no_grad():
        for batch_input, batch_target, mask in test_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            mask = mask.to(device)
            
            outputs, _, _ = model(batch_input)

            _, loss = criterion(outputs, batch_target, mask)
            
            test_loss += loss.item()


            # Inverse transform outputs and targets for metric calculation
            outputs = gt_scaler.inverse_transform(outputs)
            batch_target = gt_scaler.inverse_transform(batch_target)
            
            # Calculate MAE
            batch_mae, batch_total_mae, batch_valid_elements = calculate_mae(outputs, batch_target, mask)
            total_mae += batch_total_mae
            total_valid_elements += batch_valid_elements
            
            all_outputs.extend(outputs[mask].cpu().numpy())
            all_targets.extend(batch_target[mask].cpu().numpy())
            # all_masks.extend(mask.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    avg_test_mse = test_mse / len(test_loader)
    avg_test_mae = total_mae / total_valid_elements if total_valid_elements > 0 else 0

    # Calculate metrics using only valid (unmasked) data
    all_outputs = np.array(all_outputs)
    all_targets = np.array(all_targets)


    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))

    
    r2 = r2_score(all_targets, all_outputs)
    rmse = np.sqrt(mean_squared_error(all_targets, all_outputs))

    print(f"Test - Loss: {avg_test_loss:.4f}, MSE: {avg_test_mse:.4f}")
    print(f"R2: {r2:.4f}, MAE: {avg_test_mae:.4f}, RMSE: {rmse:.4f}")

    # Log final test metrics
    wandb.log({
        "test_loss": avg_test_loss,
        "test_mse": avg_test_mse,
        "test_r2": r2,
        "test_mae": avg_test_mae,
        "test_rmse": rmse,
        "test_valid_elements": total_valid_elements
    })

    return avg_test_loss, avg_test_mse, r2, avg_test_mae, rmse

def main(args):
    torch.cuda.empty_cache()
    geo_split = 'quadrant'
    patch_size = tuple(map(int, args.patch_size.strip('()').split(',')))
    stride = tuple(map(int, args.stride.strip('()').split(',')))
    polarimetric_channels = convert_pol_ch(args.pol_ch)
    
    # Create a custom run name
    run_name = f"{args.model_name}_bs{args.batch_size}_lr{args.lr}_polch{polarimetric_channels}_testquad{args.test_quadrant}_geosplit{geo_split}_ps{patch_size}_str{stride}_scaler:{args.scaler_type}_batchnorm{args.batchnorm}_dropout{args.dropout_rate}_loss{args.loss_fn}_zcol{args.z_collapse_method}_epochs{args.num_epochs}"
    model_path =  os.path.join(args.output_path, f"{run_name}_{args.model_name}_chm_model.pth") 
    print('Model path ', model_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    # check if model saved 
    if os.path.exists(model_path):
        print('Run exists, aborting')
        return None 
    
    # Initialize local variables
    load_dotenv()
    wandb_api_key = os.getenv('WANDB_API_KEY')
    if not wandb_api_key:
        raise ValueError("WANDB_API_KEY not found in environment variables")
    wandb.login(key=wandb_api_key)
    
    # Load data 
    quadrants, ground_truths, num_channels ,raw_data_tuple = generate_quadrants(args.parent_path, polarimetric_channels, args.exclude_height_thresh)
    train_loader, test_loader, val_loader, dataset, scalers , val_quad = generate_dataloaders(quadrants, ground_truths, patch_size, stride, test_quadrant=args.test_quadrant, batch_size=args.batch_size, scaler_type=args.scaler_type)

    model, batchnorm_bool, dropout_rate = initialize_model(args, num_channels)

    # Initialize wandb
    wandb.init( entity='fdl-biomass', project=args.project_name, name=run_name, 
               config={
        "learning_rate": args.lr,
        "epochs": args.num_epochs,
        "batch_size": args.batch_size,
        "test_quadrant": args.test_quadrant, 
        "val_quadrante": val_quad, 
        "patch_size": str(patch_size), 
        "stride": str(stride),
        "modelType": args.model_name, 
        "polarimetric_channel": polarimetric_channels, 
        "scaler_type": args.scaler_type,
        "batchnorm_bool": args.batchnorm,
        "dropout_rate": args.dropout_rate, 
        "loss_fn" : args.loss_fn,
        "z_collapse_fn" : args.z_collapse_method ,
        "exclude_height_thresh": 'Yes', 
    })

    total_params = sum(p.numel() for p in model.parameters())
    print('Model param count: ', total_params)
    print('Model summary: ')
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # criterion = nn.MSELoss(reduction='sum')
    
    criterion=select_loss(args)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train the model
    best_model_state_dict, _ = train(model, train_loader, val_loader, optimizer, criterion, device, args.num_epochs, scalers)
    model.load_state_dict(best_model_state_dict)
    
    # Test the model
    _ = test(model, test_loader, criterion, device, scalers[1] )
    create_log_prediction_image(raw_data_tuple, model, scalers, patch_size,polarimetric_channels  )
   

    # Save the model
    torch.save(model.state_dict(), model_path ) 

    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run U-Net model training with given settings.')
    parser.add_argument('--parent_path', type=str, required=True, help='Parent path')
    parser.add_argument('--output_path', type=str, required=True, help='Output path')
    parser.add_argument('--pol_ch', type=str, required=True, help='Polarimetric channels')
    parser.add_argument('--test_quadrant', type=int, required=True, help='Test quadrant')
    parser.add_argument('--num_epochs', type=int, required=True, help='Number of epochs')
    parser.add_argument('--lr', type=float, required=True, help='Learning rate')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size')
    parser.add_argument('--patch_size', type=str, required=True, help='Patch size')
    parser.add_argument('--stride', type=str, required=True, help='Stride')
    parser.add_argument('--model_name', type=str, required=True, help='Model name')
    parser.add_argument('--project_name', type=str, required=True, help='Name of wanb project')
    parser.add_argument('--scaler_type', type=str, required=False, default='standard', help='Type of scaling: minmax or  standard')
    parser.add_argument('--batchnorm', type=str, required=False, default='yes',  help='Should batchnorm be run? Used in UNET3d.')
    parser.add_argument('--dropout_rate', type=float, required=False, default=0, help='Dropout rate ofr 3dUNET')
    parser.add_argument('--loss_fn', type=str , required=True , help='Loss fn to sue: dice, mse, combined (all custom using mask)')
    parser.add_argument('--z_collapse_method', type=str, required=False, default='conv', help='Type of collapse for 36 z dims, conv, global_avg_pool or progressive_z, for the simple and 3d unet models')
    
    # Parse the arguments
    args = parser.parse_args()

    main(args)