"""
Script: create_dummy_data.py
Purpose: Generates realistic synthetic 2D brain MRI slices (DWI, ADC, Mask)
         so Members 2, 3, and 4 can immediately build & test their models
         without waiting for the full ISLES 2022 dataset to be downloaded.

Output format for each slice (.npz):
  - 'dwi': 2D float32 array (256, 256), range [0.0, 1.0] (stroke is bright)
  - 'adc': 2D float32 array (256, 256), range [0.0, 1.0] (stroke is dark)
  - 'mask': 2D uint8 array (256, 256), values {0, 1} (stroke lesion = 1)
  - 'voxel_spacing': float32 array [dx, dy, dz] in mm
"""

import os
import numpy as np
import matplotlib.pyplot as plt

def generate_synthetic_brain_slice(slice_idx: int, size: int = 256):
    """
    Creates a synthetic brain MRI slice with realistic shape and stroke lesion.
    """
    y, x = np.meshgrid(np.arange(size), np.arange(size), indexing='ij')
    center_y, center_x = size // 2, size // 2

    # 1. Brain Skull / Parenchyma Ellipse
    a, b = size * 0.38, size * 0.32
    brain_mask = ((x - center_x) ** 2 / (b ** 2) + (y - center_y) ** 2 / (a ** 2)) <= 1.0

    # 2. Simulated Brain Tissue Textures (Ventricles & Parenchyma)
    # Ventricles (dark butterfly-shaped region in the center)
    v_left = ((x - (center_x - 15)) ** 2 / (8 ** 2) + (y - center_y) ** 2 / (30 ** 2)) <= 1.0
    v_right = ((x - (center_x + 15)) ** 2 / (8 ** 2) + (y - center_y) ** 2 / (30 ** 2)) <= 1.0
    ventricles = v_left | v_right

    # Base brain tissue intensities
    base_tissue = np.zeros((size, size), dtype=np.float32)
    # Smooth gradient + baseline intensity
    tissue_vals = 0.45 + 0.1 * np.sin(x / 20.0) + 0.05 * np.cos(y / 25.0)
    base_tissue[brain_mask] = tissue_vals[brain_mask]
    base_tissue[ventricles] = 0.15 # Dark cerebrospinal fluid in ventricles

    # 3. Create Stroke Lesion (in 80% of slices; some slices are healthy/negative)
    has_lesion = (slice_idx % 5 != 0) # e.g. slice 0, 5 have no lesion (healthy)
    lesion_mask = np.zeros((size, size), dtype=np.uint8)

    dwi = base_tissue.copy()
    adc = 1.0 - base_tissue.copy() # ADC is naturally inverse contrast to DWI
    adc[~brain_mask] = 0.0

    if has_lesion:
        # Place lesion randomly in left or right hemisphere
        side = -1 if (slice_idx % 2 == 0) else 1
        lesion_cx = center_x + side * int(size * 0.15 + (slice_idx * 3) % 20)
        lesion_cy = center_y + int(((slice_idx * 7) % 40) - 20)
        radius = int(size * 0.05 + (slice_idx % 4) * 3) # varying sizes

        dist_from_lesion = np.sqrt((x - lesion_cx) ** 2 + (y - lesion_cy) ** 2)
        lesion_pixels = (dist_from_lesion <= radius) & brain_mask
        lesion_mask[lesion_pixels] = 1

        # DWI: Acute stroke glows bright (hyper-intense)
        dwi[lesion_pixels] = np.clip(dwi[lesion_pixels] + 0.45, 0.0, 1.0)
        # ADC: Acute stroke is dark (restricted diffusion / hypo-intense)
        adc[lesion_pixels] = np.clip(adc[lesion_pixels] - 0.45, 0.0, 1.0)

    # 4. Add Realistic MRI Scanner Noise (Rician/Gaussian approximation)
    noise_dwi = np.random.normal(0, 0.03, (size, size))
    noise_adc = np.random.normal(0, 0.03, (size, size))
    dwi = np.clip(dwi + noise_dwi, 0.0, 1.0).astype(np.float32)
    adc = np.clip(adc + noise_adc, 0.0, 1.0).astype(np.float32)

    # Zero-out air outside brain
    dwi[~brain_mask] = 0.0
    adc[~brain_mask] = 0.0

    voxel_spacing = np.array([1.0, 1.0, 2.0], dtype=np.float32) # mm: x, y, z

    return dwi, adc, lesion_mask, voxel_spacing


def create_dummy_dataset(output_dir: str = "data/dummy", num_slices: int = 12):
    """
    Generates and saves the synthetic dataset into output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Generating {num_slices} synthetic brain MRI slices in '{output_dir}'...")

    for i in range(num_slices):
        dwi, adc, mask, spacing = generate_synthetic_brain_slice(slice_idx=i)
        file_path = os.path.join(output_dir, f"sample_{i+1:03d}.npz")
        np.savez_compressed(
            file_path,
            dwi=dwi,
            adc=adc,
            mask=mask,
            voxel_spacing=spacing
        )

    print(f"Successfully created {num_slices} .npz files!")

    # Generate a visual preview image so the team can inspect the data
    preview_path = os.path.join(output_dir, "preview.png")
    sample_file = os.path.join(output_dir, "sample_001.npz")
    data = np.load(sample_file)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(data['dwi'], cmap='gray')
    axes[0].set_title("DWI (Stroke is Bright)")
    axes[0].axis('off')

    axes[1].imshow(data['adc'], cmap='gray')
    axes[1].set_title("ADC (Stroke is Dark)")
    axes[1].axis('off')

    axes[2].imshow(data['mask'], cmap='Reds', interpolation='none')
    axes[2].set_title("Stroke Mask (1 = Lesion)")
    axes[2].axis('off')

    # Overlay
    axes[3].imshow(data['dwi'], cmap='gray')
    axes[3].imshow(data['mask'], cmap='autumn', alpha=0.45)
    axes[3].set_title("DWI + Lesion Overlay")
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(preview_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved visual preview to '{preview_path}'")


if __name__ == "__main__":
    create_dummy_dataset(output_dir="data/dummy", num_slices=12)
