import unittest
import numpy as np


def masked_mse_loss_sum(pred, target, mask):
    """ Copy of the masked loss from UNET
    """
    # Ensure pred and target have the same shape
    assert pred.shape == target.shape, "Predictions and targets must have the same shape"
    assert pred.shape == mask.shape, "Mask must have the same shape as pred and target"
    
    # Calculate squared difference
    loss = (pred - target)**2
    
    # Apply mask
    masked_loss = loss * mask
    
    # Sum the loss
    total_loss = masked_loss.sum()
    
    # Calculate normalized loss
    num_valid_elements = mask.sum()
    normalized_loss = total_loss / num_valid_elements if num_valid_elements > 0 else total_loss
    
    return total_loss, normalized_loss

class TestMaskedMSELossSum(unittest.TestCase):
    
    def test_normal_case(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        target = np.array([1.5, 2.5, 3.5, 4.5])
        mask = np.array([1, 1, 0, 1])
        total_loss, normalized_loss = masked_mse_loss_sum(pred, target, mask)
        expected_total_loss = ((1.0 - 1.5)**2 + (2.0 - 2.5)**2 + (4.0 - 4.5)**2)
        expected_normalized_loss = expected_total_loss / 3
        self.assertAlmostEqual(total_loss, expected_total_loss)
        self.assertAlmostEqual(normalized_loss, expected_normalized_loss)
    
    def test_all_zeros_in_mask(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        target = np.array([1.5, 2.5, 3.5, 4.5])
        mask = np.array([0, 0, 0, 0])
        total_loss, normalized_loss = masked_mse_loss_sum(pred, target, mask)
        self.assertEqual(total_loss, 0.0)
        self.assertEqual(normalized_loss, 0.0)
    
    def test_all_ones_in_mask(self):
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        target = np.array([1.5, 2.5, 3.5, 4.5])
        mask = np.array([1, 1, 1, 1])
        total_loss, normalized_loss = masked_mse_loss_sum(pred, target, mask)
        expected_total_loss = ((1.0 - 1.5)**2 + (2.0 - 2.5)**2 + (3.0 - 3.5)**2 + (4.0 - 4.5)**2)
        expected_normalized_loss = expected_total_loss / 4
        self.assertAlmostEqual(total_loss, expected_total_loss)
        self.assertAlmostEqual(normalized_loss, expected_normalized_loss)
    
    def test_shape_mismatch(self):
        pred = np.array([1.0, 2.0, 3.0])
        target = np.array([1.5, 2.5, 3.5, 4.5])
        mask = np.array([1, 1, 1, 1])
        with self.assertRaises(AssertionError):
            masked_mse_loss_sum(pred, target, mask)
        
        pred = np.array([1.0, 2.0, 3.0, 4.0])
        target = np.array([1.5, 2.5, 3.5, 4.5])
        mask = np.array([1, 1, 1])
        with self.assertRaises(AssertionError):
            masked_mse_loss_sum(pred, target, mask)

if __name__ == "__main__":
    unittest.main()
