import math
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F 

class QuadrantPatchDataset(Dataset):
    def __init__(self, quadrants, ground_truths, train_quadrants, test_quadrant, val_quadrant, 
                 patch_size=(64, 64, 36), stride=(16, 16, 36), nan_threshold=0.05, gt_zeros_threshold=0.25):
        """Patch Dataset. For test and Val: patches have stride == patch size. Only use overlaps in training data. ensures correct MAE. 

        quadrants: list of 4 intensity cubes (163, 420, 36) for single channel or (163, 420, 36, N) for multi-channel
        ground_truths: list of 4 ground truth lidar data (163, 420)
        train_quadrants: list of quadrant indices for training
        test_quadrant: index of the quadrant for testing
        val_quadrant: index of the quadrant for validation
        patch_size: size of the patches to generate (default: (64, 64, 36))
        stride: stride for patch generation (default: (16, 16, 36))
        nan_threshold: maximum fraction of NaN values allowed in a patch (default: 0.05)
        gt_zeros_threshold: max fraction of 0s in ground truth (needed to exclude edges of lidar) - thresh value??
        """

        self.gt_zeros_threshold = gt_zeros_threshold
        self.quadrants = [torch.tensor(q, dtype=torch.float32) for q in quadrants]
        self.ground_truths = [torch.tensor(gt, dtype=torch.float32) for gt in ground_truths]
        
        self.train_quadrants = train_quadrants
        self.test_quadrant = test_quadrant
        self.val_quadrant = val_quadrant
        
        self.patch_size = patch_size if isinstance(patch_size, tuple) else (patch_size, patch_size, self.quadrants[0].shape[2])
        self.stride = stride if isinstance(stride, tuple) else (stride, stride, self.quadrants[0].shape[2])
        
        self.nan_threshold = nan_threshold
        self.mode = 'train'  # Default mode
        

        self.multi_channel = len(self.quadrants[0].shape) == 4
        self.num_channels = self.quadrants[0].shape[-1] if self.multi_channel else 1
        self.original_shapes = [q.shape[:2] for q in self.quadrants]
        self.padded_quadrants, self.padded_ground_truths = self._pad_data()
        self.patches_per_quadrant = self._calculate_patches_per_quadrant()
        self.valid_indices = self._get_valid_indices()
  
        
        # Print some statistics about valid patches
        total_patches = sum(len(indices) for indices in self.valid_indices.values())
        print(f"Total valid patches: {total_patches}")
        for quadrant, indices in self.valid_indices.items():
            print(f"Quadrant {quadrant}: {len(indices)} valid patches")
    

        print("Dataset Initialization Information:")
        print(f"Patch size: {self.patch_size}")
        print(f"Stride: {self.stride}")

        for i, quadrant in enumerate(self.quadrants):
            print(f"\nQuadrant {i}:")
            print(f"  Original shape: {quadrant.shape}")
            print(f"  Padded shape: {self.padded_quadrants[i].shape}")
            print(f"  Ground truth shape: {self.ground_truths[i].shape}")
            print(f"  Padded ground truth shape: {self.padded_ground_truths[i].shape}")


    def _pad_data(self):
        padded_quadrants = []
        padded_ground_truths = []
        
        for quad, gt in zip(self.quadrants, self.ground_truths):
            h, w = quad.shape[:2]
            pad_h = (self.patch_size[0] - h % self.stride[0]) % self.stride[0]
            pad_w = (self.patch_size[1] - w % self.stride[1]) % self.stride[1]
            
            # Pad quadrant
            if self.multi_channel:
                padded_quad = F.pad(quad, (0, 0, 0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
            else:
                padded_quad = F.pad(quad, (0, 0, 0, pad_w, 0, pad_h), mode='constant', value=0)
            padded_quadrants.append(padded_quad)
            
            # Pad ground truth
            padded_gt = F.pad(gt, (0, pad_w, 0, pad_h), mode='constant', value=0)
            padded_ground_truths.append(padded_gt)
            
            print(f"Quadrant shape: {quad.shape} -> {padded_quad.shape}")
            print(f"Ground truth shape: {gt.shape} -> {padded_gt.shape}")
        
        return padded_quadrants, padded_ground_truths

    def _calculate_patches_per_quadrant(self):
        h, w = self.padded_quadrants[0].shape[:2]
        rows = (h - self.patch_size[0]) // self.stride[0] + 1
        cols = (w - self.patch_size[1]) // self.stride[1] + 1
        return rows * cols

    def _get_valid_indices(self):
        valid_indices = {i: [] for i in range(len(self.padded_quadrants))}
        for quadrant in range(len(self.padded_quadrants)):
            h, w = self.padded_quadrants[quadrant].shape[:2]
            cols = (w - self.patch_size[1]) // self.stride[1] + 1
            for patch_idx in range(self.patches_per_quadrant):
                row = (patch_idx // cols) * self.stride[0]
                col = (patch_idx % cols) * self.stride[1]
                intensity_patch, gt_patch = self._get_patch(quadrant, patch_idx)
                if self._is_valid_patch(intensity_patch, gt_patch, quadrant, row, col):
                    valid_indices[quadrant].append(patch_idx)
        return valid_indices

    def _is_valid_patch(self, intensity_patch, gt_patch, quadrant, row, col):
        original_h, original_w = self.original_shapes[quadrant]
        valid_h = min(self.patch_size[0], original_h - row)
        valid_w = min(self.patch_size[1], original_w - col)
        
        valid_intensity_patch = intensity_patch[:valid_h, :valid_w]
        valid_gt_patch = gt_patch[:valid_h, :valid_w]
        
        intensity_nan_fraction = torch.isnan(valid_intensity_patch).float().mean().item()
        gt_zero_fraction = (valid_gt_patch == 0).float().mean().item()
        
        return (intensity_nan_fraction <= self.nan_threshold and 
                gt_zero_fraction <= self.gt_zeros_threshold)


    def calculate_zero_percentages(self):
        zero_percentages = []
        all_zero_count = 0
        non_zero_count = 0
        for quadrant in range(4):
            for patch_idx in self.valid_indices[quadrant]:
                _, ground_truth_patch = self._get_patch(quadrant, patch_idx)
                zero_percentage = (ground_truth_patch == 0).float().mean().item() * 100
                zero_percentages.append(zero_percentage)
                if zero_percentage == 100:
                    all_zero_count += 1
                else:
                    non_zero_count += 1
        return zero_percentages, all_zero_count, non_zero_count

    
    def _get_patch(self, quadrant_idx, patch_idx):
        h, w = self.padded_quadrants[quadrant_idx].shape[:2]
        cols = (w - self.patch_size[1]) // self.stride[1] + 1
        row = (patch_idx // cols) * self.stride[0]
        col = (patch_idx % cols) * self.stride[1]

        if self.multi_channel:
            intensity_patch = self.padded_quadrants[quadrant_idx][
                row:row+self.patch_size[0], 
                col:col+self.patch_size[1], 
                :self.patch_size[2],
                :
            ]
        else:
            intensity_patch = self.padded_quadrants[quadrant_idx][
                row:row+self.patch_size[0], 
                col:col+self.patch_size[1], 
                :self.patch_size[2]
            ]
        
        ground_truth_patch = self.padded_ground_truths[quadrant_idx][
            row:row+self.patch_size[0], 
            col:col+self.patch_size[1]
        ]
        
        return intensity_patch, ground_truth_patch

    
    def __len__(self):
        if self.mode is None:
            raise ValueError('Dataset mode not set. Call set_mode() before using the dataset')
        if self.mode == 'train':
            return sum(len(self.valid_indices[q]) for q in self.train_quadrants)
        elif self.mode == 'test':
            return len(self.valid_indices[self.test_quadrant])
        else:  # val
            return len(self.valid_indices[self.val_quadrant])
    
    def _get_mask(self, quadrant_idx, row, col):
        original_h, original_w = self.original_shapes[quadrant_idx]
        mask = torch.ones(self.patch_size[:2], dtype=torch.bool)
        
        if row + self.patch_size[0] > original_h:
            mask[-(row + self.patch_size[0] - original_h):] = False
        if col + self.patch_size[1] > original_w:
            mask[:, -(col + self.patch_size[1] - original_w):] = False
        
        return mask

    def __getitem__(self, idx):
        if self.mode is None:
            raise ValueError('Dataset mode not set. Call set_mode() before using the dataset')
        if self.mode == 'train':
            quadrant_idx = self.train_quadrants[0]
            for q in self.train_quadrants:
                if idx >= len(self.valid_indices[q]):
                    idx -= len(self.valid_indices[q])
                else:
                    quadrant_idx = q
                    break
        elif self.mode == 'test':
            quadrant_idx = self.test_quadrant
        else:  # val
            quadrant_idx = self.val_quadrant

        patch_idx = self.valid_indices[quadrant_idx][idx]
        intensity_patch, ground_truth_patch = self._get_patch(quadrant_idx, patch_idx)
        
        h, w = self.padded_quadrants[quadrant_idx].shape[:2]
        cols = (w - self.patch_size[1]) // self.stride[1] + 1
        row = (patch_idx // cols) * self.stride[0]
        col = (patch_idx % cols) * self.stride[1]
        
        mask = self._get_mask(quadrant_idx, row, col)
        
        # Replace NaN values with 0 in the intensity patch
        intensity_patch = torch.nan_to_num(intensity_patch, nan=0.0)
        
        # Ensure the intensity patch has the correct shape and channel order
        if not self.multi_channel:
            intensity_patch = intensity_patch.unsqueeze(-1)
        intensity_patch = intensity_patch.permute(3, 0, 1, 2)
        
        return intensity_patch, ground_truth_patch, mask

    def validate_masks(self):
        print("Validating masks...")
        
        error_found = False

        for mode in ['train', 'test', 'val']:
            self.set_mode(mode)
            all_true_count = 0
            total_count = 0
            masks_with_false = 0
            
            for idx in range(len(self)):
                _, _, mask = self[idx]
                total_count += 1
                
                if torch.all(mask):
                    all_true_count += 1
                else:
                    masks_with_false += 1

            print(f"{mode.capitalize()} mode: {all_true_count}/{total_count} all-True masks")
            
            if mode in ['test', 'val'] and all_true_count == total_count:
                print(f"Error: All masks in {mode} data are all-True")
                error_found = True
            
            if mode == 'train':
                print(f"Train mode: {masks_with_false}/{total_count} masks have at least one False element")

        if error_found:
            print("Validation failed: All masks are all-True in test or validation data.")
        else:
            print("Mask validation complete. No errors found in test or validation data.")

        self.set_mode('train')
        return not error_found
            
    def set_mode(self, mode):
        assert mode in ['train', 'test', 'val'], "Mode must be 'train', 'test', or 'val'"
        self.mode = mode
    


    def get_stats(self):
        stats = {}
        for quadrant in range(4):
            total_patches = self.patches_per_quadrant
            valid_patches = len(self.valid_indices[quadrant])
            invalid_patches = total_patches - valid_patches
            stats[f'Quadrant {quadrant}'] = {
                'total_patches': total_patches,
                'valid_patches': valid_patches,
                'invalid_patches': invalid_patches,
                'invalid_percentage': (invalid_patches / total_patches) * 100
            }
        return stats
    
    def get_exclusion_stats(self):
        stats = {
            'total_patches': 0,
            'excluded_patches': 0,
            'excluded_due_to_zeros': 0,
            'excluded_due_to_nans': 0
        }
        
        for quadrant in range(4):
            h, w = self.padded_quadrants[quadrant].shape[:2]
            cols = (w - self.patch_size[1]) // self.stride[1] + 1
            for patch_idx in range(self.patches_per_quadrant):
                stats['total_patches'] += 1
                row = (patch_idx // cols) * self.stride[0]
                col = (patch_idx % cols) * self.stride[1]
                intensity_patch, gt_patch = self._get_patch(quadrant, patch_idx)
                
                if not self._is_valid_patch(intensity_patch, gt_patch, quadrant, row, col):
                    stats['excluded_patches'] += 1
                    original_h, original_w = self.original_shapes[quadrant]
                    valid_h = min(self.patch_size[0], original_h - row)
                    valid_w = min(self.patch_size[1], original_w - col)
                    valid_gt_patch = gt_patch[:valid_h, :valid_w]
                    valid_intensity_patch = intensity_patch[:valid_h, :valid_w]
                    
                    gt_zero_fraction = (valid_gt_patch == 0).float().mean().item()
                    intensity_nan_fraction = torch.isnan(valid_intensity_patch).float().mean().item()
                    
                    if gt_zero_fraction > self.gt_zeros_threshold:
                        stats['excluded_due_to_zeros'] += 1
                    elif intensity_nan_fraction > self.nan_threshold:
                        stats['excluded_due_to_nans'] += 1
        
        return stats


    def visualize_patches(self, quadrant_idx):
        quadrant = self.padded_quadrants[quadrant_idx]
        h, w = quadrant.shape[:2]
        patch_map = torch.zeros((h, w), dtype=torch.int)
        
        for patch_idx in self.valid_indices[quadrant_idx]:
            cols = (w - self.patch_size[1]) // self.stride[1] + 1
            row = (patch_idx // cols) * self.stride[0]
            col = (patch_idx % cols) * self.stride[1]
            patch_map[row:row+self.patch_size[0], col:col+self.patch_size[1]] += 1
        
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.imshow(patch_map.cpu().numpy(), cmap='viridis')
        plt.colorbar(label='Patch overlap count')
        plt.title(f'Valid Patches for Quadrant {quadrant_idx}')
        plt.show()