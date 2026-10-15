from src.fdlbiomass.models.unet_model import UNet3D, UNet3D_MP
from src.fdlbiomass.models.simple_conv import SimpleConv3D
from src.fdlbiomass.models.simple_unet import simple_UNet3D2 
from src.fdlbiomass.models.simple_resnet import simple_UNet3D2_Residual 


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
    if args.model_name == 'simple_conv':
        batchnorm_bool = 'N/A'
        dropout_rate = 'N/A'
        model = SimpleConv3D(in_channels=num_channels)
    elif args.model_name in ['unet3d', 'unet_resnet', 'simple_unet2', 'unet3d_mp']:
        if args.batchnorm == 'no':
            batchnorm_bool = False
        elif args.batchnorm == 'yes':
            batchnorm_bool = True
        else:
            raise ValueError('Wrong type of batch norm')
        
        print('Batch norm:', batchnorm_bool)
        dropout_rate = args.dropout_rate
        
        if args.model_name == 'unet3d':
            model = UNet3D(in_channels=num_channels, use_batchnorm=batchnorm_bool, dropout_rate=dropout_rate, z_collapse_method= args.z_collapse_method, )
        elif args.model_name == 'simple_unet2':
            model = simple_UNet3D2(in_channels=num_channels, use_batchnorm=batchnorm_bool, dropout_rate=dropout_rate, z_collapse_method= args.z_collapse_method, final_z_dim=args.final_z_dim)
        elif args.model_name =='unet3d_mp':
            model = UNet3D_MP(in_channels=num_channels, use_batchnorm=batchnorm_bool, dropout_rate=dropout_rate, z_collapse_method= args.z_collapse_method)
        elif args.model_name =='unet_resnet':
            model = simple_UNet3D2_Residual(in_channels=num_channels, use_batchnorm=batchnorm_bool, dropout_rate=dropout_rate, final_z_dim=args.final_z_dim)
    else:
        raise ValueError('Unsupported model name')
    
    return model, batchnorm_bool, dropout_rate