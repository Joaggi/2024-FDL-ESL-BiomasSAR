import unittest
import torch
import torch.nn.functional as F
import os 
import sys 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.fdlbiomass.loss_fns import select_loss, calculate_all_losses, calculate_mae, masked_mse_loss_sum
from src.fdlbiomass.loss_fns import masked_dice_loss, masked_combined_loss, dice_loss

class TestLossFunctions(unittest.TestCase):
    
    def setUp(self):
        # Sample tensors for testing
        self.pred = torch.tensor([[[0.1, 0.2], [0.3, 0.4]], [[0.5, 0.6], [0.7, 0.8]]], dtype=torch.float32)
        self.target = torch.tensor([[[0.1, 0.25], [0.35, 0.45]], [[0.55, 0.65], [0.75, 0.85]]], dtype=torch.float32)
        self.mask = torch.tensor([[[True, False ], [True, True]], [[False, True], [True, True]]], dtype=torch.bool)

    def test_select_loss(self):
        args = lambda: None
        args.loss_fn = 'mse'
        self.assertEqual(select_loss(args), masked_mse_loss_sum)

        args.loss_fn = 'dice'
        self.assertEqual(select_loss(args), masked_dice_loss)

        args.loss_fn = 'combined'
        self.assertEqual(select_loss(args), masked_combined_loss)

        args.loss_fn = 'invalid'
        with self.assertRaises(ValueError):
            select_loss(args)

    def test_calculate_all_losses(self):
        mse_norm, dice, combined = calculate_all_losses(self.pred, self.target, self.mask)
        self.assertIsInstance(mse_norm, torch.Tensor)
        self.assertIsInstance(dice, torch.Tensor)
        self.assertIsInstance(combined, torch.Tensor)
        
        # Check that losses are non-negative
        self.assertGreaterEqual(mse_norm.item(), 0)
        self.assertGreaterEqual(dice.item(), 0)
        self.assertGreaterEqual(combined.item(), 0)

    def test_calculate_mae(self):
        avg_mae, total_mae, total_valid_elements = calculate_mae(self.pred, self.target, self.mask)
        self.assertIsInstance(avg_mae, float)
        self.assertIsInstance(total_mae, float)
        self.assertIsInstance(total_valid_elements, float)
        self.assertGreaterEqual(total_valid_elements, 0)
        
        # Check that MAE is calculated correctly
        expected_total_mae = ((0.1 - 0.1) + abs(0.3 - 0.35) + abs(0.4 - 0.45) + 
                              abs(0.6 - 0.65) + abs(0.7 - 0.75) + abs(0.8 - 0.85))
        expected_avg_mae = expected_total_mae / 6  # 6 valid elements
        self.assertAlmostEqual(avg_mae, expected_avg_mae, places=6)

    def test_masked_mse_loss_sum(self):
        total_loss, normalized_loss = masked_mse_loss_sum(self.pred, self.target, self.mask)
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(normalized_loss, torch.Tensor)
        
        # Check that masking is applied correctly
        expected_mse = (((0.1 - 0.1)**2 + (0.3 - 0.35)**2 + (0.4 - 0.45)**2 + 
                         (0.6 - 0.65)**2 + (0.7 - 0.75)**2 + (0.8 - 0.85)**2) / 6)
        self.assertAlmostEqual(normalized_loss.item(), expected_mse, places=6)
    
    def test_masked_dice_loss(self):
        # Test with a mask of all ones
        mask_all_ones = torch.ones_like(self.mask, dtype=torch.bool)
        loss_all_ones = masked_dice_loss(self.pred, self.target, mask_all_ones)
        print(f"Dice loss with mask of all ones: {loss_all_ones.item()}")
        self.assertAlmostEqual(loss_all_ones.item(), dice_loss(self.pred, self.target).item(), places=5)

        # Test with a mask of all False values
        mask_all_false = torch.zeros_like(self.mask, dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "All elements in the mask are False. Cannot compute Dice loss."):
            loss_all_false = masked_dice_loss(self.pred, self.target, mask_all_false)


        # Test the case where the prediction and target are the same
        self.assertAlmostEqual(masked_dice_loss(self.pred, self.pred, self.mask).item(), 0.0, places=5)

    def test_masked_combined_loss(self):
        total_loss, dice_loss, mse_loss = masked_combined_loss(self.pred, self.target, self.mask)
        self.assertIsInstance(total_loss, torch.Tensor)
        self.assertIsInstance(dice_loss, torch.Tensor)
        self.assertIsInstance(mse_loss, torch.Tensor)
        
        alpha, beta = 0.5, 0.5
        expected_combined = alpha * dice_loss + beta * mse_loss
        self.assertAlmostEqual(total_loss.item(), expected_combined.item(), places=6)

        # Test with perfect prediction
        perfect_pred = self.target.clone()
        perfect_total_loss, perfect_dice_loss, perfect_mse_loss = masked_combined_loss(perfect_pred, self.target, self.mask)
        self.assertAlmostEqual(perfect_total_loss.item(), 0, places=6)
        self.assertAlmostEqual(perfect_dice_loss.item(), 0, places=6)
        self.assertAlmostEqual(perfect_mse_loss.item(), 0, places=6)
    
    def test_mask_application(self):
        # Create pred and target with clear differences in masked vs unmasked areas
        pred = torch.tensor([[[0.1, 1.0], [0.3, 0.4]], [[1.0, 0.6], [0.7, 0.8]]], dtype=torch.float32)
        target = torch.tensor([[[0.1, 0.0], [0.3, 0.4]], [[0.0, 0.6], [0.7, 0.8]]], dtype=torch.float32)
        mask = torch.tensor([[[True, False ], [True, True]], [[False, True], [True, True]]], dtype=torch.bool)
        masked_loss, masked_norm_loss = masked_mse_loss_sum(pred, target, mask)
        unmasked_loss = F.mse_loss(pred, target, reduction='mean')

        # Check that masked loss is less than unmasked loss
        self.assertLess(masked_norm_loss.item(), unmasked_loss.item())

        # Check that masked loss is correct
        expected_masked_loss = (0.0 + 0.0 + 0.0 + 0.0 + 0.0 + 0.0) / 6  # Sum of squared differences for non-masked elements
        self.assertAlmostEqual(masked_norm_loss.item(), expected_masked_loss, places=6)

        # Check that unmasked loss is correct
        expected_unmasked_loss = (0.0 + 1.0 + 0.0 + 0.0 + 1.0 + 0.0 + 0.0 + 0.0) / 8  # Sum of all squared differences
        self.assertAlmostEqual(unmasked_loss.item(), expected_unmasked_loss, places=6)

if __name__ == '__main__':
    unittest.main()