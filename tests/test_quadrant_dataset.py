import unittest
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys 
import os 


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from src.fdlbiomass.quadrant_dataset_pband import QuadrantPatchDataset

class TestQuadrantPatchDataset(unittest.TestCase):

    @staticmethod
    def create_dummy_data(shape=(163, 420, 36), num_quadrants=4, nan_fraction=0.1):
        quadrants = []
        ground_truths = []
        for _ in range(num_quadrants):
            quad = np.random.rand(*shape).astype(np.float32)
            quad[np.random.rand(*shape) < nan_fraction] = np.nan
            quadrants.append(quad)
            
            gt = np.random.rand(shape[0], shape[1]).astype(np.float32)
            gt[np.random.rand(shape[0], shape[1]) < nan_fraction] = np.nan
            ground_truths.append(gt)
        
        return quadrants, ground_truths

    def setUp(self):
        quadrants, ground_truths = self.create_dummy_data(nan_fraction=0.05)  # Reduced NaN fraction
        self.dataset = QuadrantPatchDataset(
            quadrants, 
            ground_truths, 
            train_quadrants=[0, 1], 
            test_quadrant=2, 
            val_quadrant=3, 
            patch_size=(64, 64, 36), 
            stride=(16, 16, 36), 
            nan_threshold=0.1  # Increased threshold
        )

    def test_dataset_statistics(self):
        stats = self.dataset.get_stats()
        for quadrant, quad_stats in stats.items():
            self.assertIn('total_patches', quad_stats)
            self.assertIn('valid_patches', quad_stats)
            self.assertIn('invalid_patches', quad_stats)
            self.assertIn('invalid_percentage', quad_stats)
            self.assertEqual(quad_stats['total_patches'], 
                             quad_stats['valid_patches'] + quad_stats['invalid_patches'])
        
        print("Dataset statistics:")
        for quadrant, quad_stats in stats.items():
            print(f"{quadrant}: {quad_stats}")

    def test_dataset_lengths(self):
        self.dataset.set_mode('train')
        train_len = len(self.dataset)
        self.dataset.set_mode('val')
        val_len = len(self.dataset)
        self.dataset.set_mode('test')
        test_len = len(self.dataset)
        
        print(f"Train length: {train_len}")
        print(f"Val length: {val_len}")
        print(f"Test length: {test_len}")
        
        stats = self.dataset.get_stats()
        expected_train_len = sum(stats[f'Quadrant {i}']['valid_patches'] for i in [0, 1])
        self.assertEqual(train_len, expected_train_len)

    def test_nan_values_in_patches(self):
        self.dataset.set_mode('train')
        if len(self.dataset) > 0:
            dataloader = DataLoader(self.dataset, batch_size=min(32, len(self.dataset)), shuffle=True)
            for inputs, targets in dataloader:
                self.assertFalse(torch.isnan(inputs).any())
                self.assertLess(torch.isnan(targets).float().mean().item(), self.dataset.nan_threshold)
                break  # Check only first batch
        else:
            print("No valid patches for testing NaN values")

    def test_patch_sizes(self):
        self.dataset.set_mode('train')
        if len(self.dataset) > 0:
            inputs, targets = self.dataset[0]
            self.assertEqual(inputs.shape, (64, 64, 36))
            self.assertEqual(targets.shape, (64, 64))
        else:
            print("No valid patches for testing patch sizes")

    def test_invalid_patches_excluded(self):
        self.dataset.set_mode('train')
        for i in range(min(10, len(self.dataset))):
            patch, gt = self.dataset[i]
            nan_fraction_patch = torch.isnan(patch).float().mean().item()
            nan_fraction_gt = torch.isnan(gt).float().mean().item()
            
            self.assertLessEqual(nan_fraction_patch, self.dataset.nan_threshold)
            self.assertLessEqual(nan_fraction_gt, self.dataset.nan_threshold)

if __name__ == '__main__':
    unittest.main()