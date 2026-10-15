import argparse
from src.fdlbiomass.dataloader_fns import generate_quadrants, generate_dataloaders, convert_pol_ch
from src.fdlbiomass.loss_fns import  masked_mse_loss_sum
from src.fdlbiomass.image_reconstruction import create_log_prediction_image 

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
import os 
from dotenv import load_dotenv 
from datetime import datetime

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import os
torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

from run_unet_GPU import train, test, initialize_model
from src.fdlbiomass.tomograph_reconstruction.train_model import save_trained_model 

parent_path ='/home/grace.bcolverd/esl_dev/tomosar-matlab'


def train_model():
    torch.cuda.empty_cache()
    config_defaults = {
        'epochs': 5,
        'batch_size': 10,
        'learning_rate': 1e-3,
        'polarimetric_channel': 'HH',
        'batchnorm':  'no',
        'dropout_rate': 0.0, 
         "patch_stride_pair": {"patch_size": "(64,64,36)", "stride": "(32,32,36)"},
        "test_quadrant":3, 
        "scaler_type": "standard", 
        "model_name": "simple_unet",
        'true_bn_method': 'Yess:)',
         'z_collapse_method' : 'conv' }
    
    # Initialize a new wandb run
    wandb.init(config=config_defaults)
    
    # Config is a variable that holds and saves hyperparameters and inputs
    config = wandb.config
    print('starting train model ')
    
    
    geo_split = 'quadrant'
    patch_size = tuple(map(int, config['patch_stride_pair']['patch_size'].strip('()').split(',')))
    stride = tuple(map(int, config['patch_stride_pair']['stride'].strip('()').split(',')))
    polarimetric_channels = convert_pol_ch(config.polarimetric_channel)
    band_opt = config['band_opt']
    print('Starting band ', band_opt)
    
    quadrants, ground_truths, num_channels, raw_data_tuple = generate_quadrants(parent_path=parent_path, polarimetric_channels=polarimetric_channels, band_opt=band_opt, exclude_height_thresh=config.exclude_height_thresh)
  
    train_loader, test_loader, val_loader, dataset, scalers, val_quad = generate_dataloaders(quadrants, ground_truths, patch_size, stride, test_quadrant=config.test_quadrant, batch_size=config.batch_size, scaler_type=config.scaler_type)

    wandb.config.update({"val_quad":val_quad })

    model, batchnorm_bool, dropout_rate = initialize_model(config, num_channels)


    run_name = f"{config.model_name}_bs{config.batch_size}_lr{config.learning_rate}_polch{polarimetric_channels}_testquad{config.test_quadrant}_geosplit{geo_split}_ps{patch_size}_str{stride}_scaler:{config.scaler_type}_batchnorm{config.batchnorm}_dropout{config.dropout_rate}"
    total_params = sum(p.numel() for p in model.parameters())
    print('Model param count: ', total_params)
    print('Model summary: ')
    print(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = masked_mse_loss_sum
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    # Train the model
    best_model_state_dict, best_mae = train(model, train_loader, val_loader, optimizer, criterion, device, config.epochs, scalers, config.early_stopping_metric)
    print('Best MAE/RMSE ', best_mae ) 
    model.load_state_dict(best_model_state_dict)

    


    # test the model 
    _, gt_scaler = scalers
    _ = test(model, test_loader, criterion, device, gt_scaler)

    # save the model if performance good
    if config.early_stopping_metric =='mae':
        save_thresh = 2.9
    elif config.early_stopping_metric =='rmse':
        save_thresh = 3.5

    if abs(best_mae) < save_thresh:
        create_log_prediction_image(raw_data_tuple, model, scalers, patch_size, polarimetric_channels )
        if config.save == 'yes':
                save_trained_model(model, config.base_path)

    wandb.finish()

