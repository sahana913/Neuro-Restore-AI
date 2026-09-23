"""
Module: src/dataset/dataloader.py
Purpose: Standard PyTorch Dataset and DataLoader for 2D Brain MRI Slices.
Works seamlessly with both synthetic dummy data and real preprocessed ISLES data.
"""

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class StrokeBrainDataset(Dataset):
    """
    Loads 2D brain MRI slices saved as .npz files.
    Each file contains:
      - 'dwi': (256, 256) float32
      - 'adc': (256, 256) float32
      - 'mask': (256, 256) uint8 {0, 1}
      - 'voxel_spacing': (3,) float32
    """
    def __init__(self, data_dir: str, modality: str = "dwi"):
        """
        modality options:
          - "dwi": returns (1, 256, 256) tensor
          - "adc": returns (1, 256, 256) tensor
          - "both": returns (2, 256, 256) tensor (stacked DWI + ADC)
        """
        self.data_dir = data_dir
        self.modality = modality.lower()
        self.file_paths = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        if len(self.file_paths) == 0:
            raise FileNotFoundError(f"No .npz files found in {data_dir}!")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        data = np.load(file_path)

        dwi = data['dwi'].astype(np.float32)
        adc = data['adc'].astype(np.float32)
        mask = data['mask'].astype(np.float32)

        if self.modality == "dwi":
            img_tensor = torch.from_numpy(dwi).unsqueeze(0) # (1, 256, 256)
        elif self.modality == "adc":
            img_tensor = torch.from_numpy(adc).unsqueeze(0) # (1, 256, 256)
        elif self.modality == "both":
            img_tensor = torch.from_numpy(np.stack([dwi, adc], axis=0)) # (2, 256, 256)
        else:
            raise ValueError(f"Unknown modality: {self.modality}")

        mask_tensor = torch.from_numpy(mask).unsqueeze(0) # (1, 256, 256)

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "file_name": os.path.basename(file_path)
        }

def get_dataloader(data_dir: str = "data/dummy", batch_size: int = 4, shuffle: bool = True, modality: str = "dwi"):
    dataset = StrokeBrainDataset(data_dir=data_dir, modality=modality)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

if __name__ == "__main__":
    loader = get_dataloader(data_dir="data/dummy", batch_size=2)
    sample_batch = next(iter(loader))
    print("Dataloader working successfully!")
    print("Image batch shape:", sample_batch["image"].shape)
    print("Mask batch shape: ", sample_batch["mask"].shape)
