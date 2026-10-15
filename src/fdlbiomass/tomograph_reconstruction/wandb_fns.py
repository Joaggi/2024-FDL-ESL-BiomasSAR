



import copy 

class EarlyStopping:
    """
    Early stopping to prevent overfitting in training neural networks.

    This class implements the early stopping mechanism, which monitors a given metric
    (in this case, Mean Absolute Error) and stops the training process if no improvement
    is seen for a specified number of consecutive epochs.

    Attributes:
        patience (int): Number of epochs to wait for improvement before stopping.
        min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        counter (int): Counter for epochs with no improvement.
        best_score (float): Best score observed during training.
        early_stop (bool): Flag indicating whether to stop training.
        best_model (dict): State dictionary of the best model.
        best_epoch (int): Epoch number where the best score was achieved.

    Methods:
        __call__(val_mae, model, epoch): Check for improvement and update early stopping state.
    """
    def __init__(self, patience=10, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model = None
        self.best_epoch = None

    def __call__(self, val_mae, model, epoch):
        """
        Check if the model has improved and update the early stopping state.

        This method is called after each epoch to determine if the model has improved.
        If there's no improvement for 'patience' number of epochs, it sets the
        early_stop flag to True.
        We invert mae to move to 'higher is better'

        Args:
            val_mae (float): Validation Mean Absolute Error for the current epoch.
            model (torch.nn.Module): The current model.
            epoch (int): The current epoch number.

        Returns:
            tuple: A tuple containing the best model state_dict and the best epoch number.
        """
        score = -val_mae
        print(f"Current score: {score}, Best score: {self.best_score}, Counter: {self.counter}")
        if self.best_score is None:
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            print("Initial best score set")
        elif score > self.best_score + self.min_delta:  # Improvement
            print("Improvement found")
            self.best_score = score
            self.best_model = copy.deepcopy(model.state_dict())
            self.best_epoch = epoch
            self.counter = 0
        else:  # No improvement
            print("No improvement")
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        print(f"After check - Best score: {self.best_score}, Counter: {self.counter}")
        return self.best_model, self.best_epoch
    