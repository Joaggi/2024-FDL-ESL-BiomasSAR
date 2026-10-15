import torch 
import torch.nn.functional as F

def select_loss(args):
    """
    Dont use dice loss or combined 
    """
    if args.loss_fn == 'mse':
        criterion = masked_mse_loss_sum
    # elif args.loss_fn == 'dice':
    #     criterion = masked_dice_loss
    # elif args.loss_fn == 'combined':
    #     criterion = masked_combined_loss
    else:
        raise ValueError("Invalid loss function. Choose 'mse' ")
    return criterion 

def calculate_all_losses(pred, target, mask):
    mse_total, mse_norm = masked_mse_loss_sum(pred, target, mask)
    dice = masked_dice_loss(pred, target, mask)
    combined, _, _ = masked_combined_loss(pred, target, mask)
    return mse_norm, dice, combined

def calculate_mae(outputs, targets, mask):
    """
    Calculate Mean Absolute Error (MAE) for masked data.
    """
    mae = torch.abs(outputs - targets) * mask
    total_mae = mae.sum().item()
    total_valid_elements = mask.sum().float().item()
    avg_mae = total_mae / total_valid_elements if total_valid_elements > 0 else 0
    return avg_mae, total_mae, total_valid_elements

    
def masked_mse_loss_sum(pred, target, mask):
    """
    Calculate the masked Mean Squared Error (MSE) loss.

    This function computes both the total and normalized MSE loss, applying a mask to consider
    only valid elements. The mask is used to exclude certain elements (e.g., padded areas) from
    the loss calculation.

    Parameters:
    pred (torch.Tensor): The model's predictions. Shape: (batch_size, height, width)
    target (torch.Tensor): The ground truth values. Shape: (batch_size, height, width)
    mask (torch.Tensor): A boolean mask indicating valid elements. Shape: (batch_size, height, width)

    Returns:
    tuple: A tuple containing:
        - total_loss (torch.Tensor): The sum of squared errors for valid elements.
        - normalized_loss (torch.Tensor): The mean squared error for valid elements.

    Raises:
    AssertionError: If the shapes of pred, target, and mask do not match.

    Notes:
    - The function assumes all inputs are 3D tensors (batch_size, height, width).
    - If the mask is all False (no valid elements), the normalized loss will be equal to the total loss.
    - This loss is useful for scenarios where certain parts of the input should be ignored,
      such as padded regions in satellite
    """
    # Ensure pred and target have the same shape
    # print('pred shape ', pred.shape)
    # print('target shape ', target.shape)
    assert pred.shape == target.shape, "Predictions and targets must have the same shape"
    assert pred.shape == mask.shape, "Mask must have the same shape as pred and target"
    # Calculate squared difference
    loss = (pred - target)**2
    masked_loss = loss * mask
    
    # Sum the loss
    total_loss = masked_loss.sum()
    
    # Calculate normalized loss
    num_valid_elements = mask.sum()
    normalized_loss = total_loss / num_valid_elements if num_valid_elements > 0 else total_loss
    
    return total_loss, normalized_loss




def masked_dice_loss(pred, target, mask ):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch

    Pulled from https://gist.github.com/weiliu620/52d140b22685cf9552da4899e2160183 and added masking
    """
    if torch.all(~mask):
        print('Mask All false')
        raise ValueError("All elements in the mask are False. Cannot compute Dice loss.")
    
    pred = pred[mask]
    target=target[mask]

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    dice = 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )
    # if dice < 0:
        # print("Warning: Dice loss out of expected range [0, 1]", dice )
    return dice 



def masked_combined_loss(pred, target, mask, alpha=0.5, beta=0.5):
    _, mse_loss = masked_mse_loss_sum(pred, target, mask)
    dice_loss = masked_dice_loss(pred, target, mask)
    combined_loss = alpha * dice_loss + beta * mse_loss
    return combined_loss, dice_loss, mse_loss


def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(tflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )