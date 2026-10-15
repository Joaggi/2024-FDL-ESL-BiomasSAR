import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np



import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

# def collate_fn(batch):
#     patches, truths, masks = zip(*batch)
#     return torch.stack(patches), torch.stack(truths), torch.stack(masks)

class SLC_QuadrantDataset(Dataset):
    def __init__(self, quadrant_data, quadrant_truth, patch_size, stride, null_threshold=0.00, gt_zero_thres=0.15):
        if not isinstance(patch_size, tuple) or len(patch_size) != 2:
            raise ValueError("patch_size must be a tuple (x, x) where x is the same for both dimensions")
        
        if not isinstance(stride, (int, tuple)):
            raise ValueError("stride must be an integer or a tuple of two integers")
        
        self.null_threshold = null_threshold
        self.gt_zero_thres = gt_zero_thres

        self.patch_size = tuple(int(x) for x in patch_size)
        self.stride = (int(stride), int(stride)) if isinstance(stride, int) else tuple(int(x) for x in stride)
        
        self.data = quadrant_data
        self.ground_truth = quadrant_truth
        
        # Create mask with True for original data
        self.mask = np.ones_like(self.ground_truth, dtype=bool)
        
        # Calculate padding
        self.pad_h = (self.patch_size[0] - self.data.shape[0] % self.patch_size[0]) % self.patch_size[0]
        self.pad_w = (self.patch_size[1] - self.data.shape[1] % self.patch_size[1]) % self.patch_size[1]
        
        # Apply padding
        self.data = np.pad(self.data, ((0, self.pad_h), (0, self.pad_w), (0, 0)), mode='constant', constant_values=np.nan)
        self.ground_truth = np.pad(self.ground_truth, ((0, self.pad_h), (0, self.pad_w)), mode='constant', constant_values=np.nan)
        
        # Pad mask with False for padded parts
        self.mask = np.pad(self.mask, ((0, self.pad_h), (0, self.pad_w)), mode='constant', constant_values=False)
        
        # Calculate number of patches
        self.num_patches_h = (self.data.shape[0] - self.patch_size[0]) // self.stride[0] + 1
        self.num_patches_w = (self.data.shape[1] - self.patch_size[1]) // self.stride[1] + 1

        # Create a list of valid patch indices
        self.valid_patches = self._get_valid_patches()

        print(f"Data shape: {self.data.shape}")
        print(f"Patch size: {self.patch_size}")
        print(f"Stride: {self.stride}")
        print(f"Number of valid patches: {len(self.valid_patches)}")

    def _get_valid_patches(self):
        valid_patches = []
        for idx in range(self.num_patches_h * self.num_patches_w):
            h_idx = idx // self.num_patches_w
            w_idx = idx % self.num_patches_w
            
            h_start = int(h_idx * self.stride[0])
            w_start = int(w_idx * self.stride[1])
            
            h_end = int(h_start + self.patch_size[0])
            w_end = int(w_start + self.patch_size[1])
            
            patch = self.data[h_start:h_end, w_start:w_end, :]
            truth = self.ground_truth[h_start:h_end, w_start:w_end]
            
            # Check for excessive nulls in both intensity and ground truth
            intensity_null_ratio = np.isnan(patch).sum() / patch.size
            truth_null_ratio = np.isnan(truth).sum() / truth.size
            
            # Check for excessive zeros in ground truth
            truth_zero_ratio = (truth == 0).sum() / truth.size
            
            if (intensity_null_ratio <= self.null_threshold and 
                truth_null_ratio <= self.null_threshold and 
                truth_zero_ratio <= self.gt_zero_thres):
                valid_patches.append(idx)
        
        return valid_patches

    def __len__(self):
        return len(self.valid_patches)

    def __getitem__(self, idx):
        patch_idx = self.valid_patches[idx]
        h_idx = patch_idx // self.num_patches_w
        w_idx = patch_idx % self.num_patches_w
        
        h_start = int(h_idx * self.stride[0])
        w_start = int(w_idx * self.stride[1])
        
        h_end = int(h_start + self.patch_size[0])
        w_end = int(w_start + self.patch_size[1])
        
        patch = self.data[h_start:h_end, w_start:w_end, :]
        truth = self.ground_truth[h_start:h_end, w_start:w_end]
        mask = self.mask[h_start:h_end, w_start:w_end]
        
        # Rearrange dimensions to make channels first
        patch = np.transpose(patch, (2, 0, 1))
        
        return torch.from_numpy(patch).float(), torch.from_numpy(truth).float(), torch.from_numpy(mask).bool()
    
