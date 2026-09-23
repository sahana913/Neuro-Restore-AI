from pathlib import Path
import nibabel as nib
import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROJECT_ROOT = Path(r"D:\NeuroRestore-AI")
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "ISLES-2022"

CASE_ID = "sub-strokecase0001"


# --------------------------------------------------
# Locate files
# --------------------------------------------------

case_dir = DATASET_ROOT / CASE_ID
derivatives_dir = DATASET_ROOT / "derivatives" / CASE_ID

flair_path = next(
    (case_dir / "ses-0001" / "anat").glob("*_FLAIR.nii.gz")
)

adc_path = next(
    (case_dir / "ses-0001" / "dwi").glob("*_adc.nii.gz")
)

dwi_path = next(
    (case_dir / "ses-0001" / "dwi").glob("*_dwi.nii.gz")
)

mask_candidates = list(
    derivatives_dir.rglob("*_msk.nii.gz")
)

if not mask_candidates:
    raise FileNotFoundError(
        f"No mask found for {CASE_ID}"
    )

mask_path = mask_candidates[0]


# --------------------------------------------------
# Load NIfTI
# --------------------------------------------------

images = {
    "FLAIR": flair_path,
    "ADC": adc_path,
    "DWI": dwi_path,
    "MASK": mask_path,
}


def inspect_nifti(name, path):
    img = nib.load(path)

    data = img.get_fdata()

    print("=" * 70)
    print(f"{name}")
    print("=" * 70)

    print(f"File: {path}")
    print(f"Shape: {img.shape}")
    print(f"Data type: {data.dtype}")

    print(
        "Voxel spacing:",
        img.header.get_zooms()
    )

    print(
        "Orientation:",
        nib.aff2axcodes(img.affine)
    )

    print(
        "Intensity min:",
        np.min(data)
    )

    print(
        "Intensity max:",
        np.max(data)
    )

    print(
        "Intensity mean:",
        np.mean(data)
    )

    print(
        "Intensity std:",
        np.std(data)
    )

    print()


# --------------------------------------------------
# Inspect all modalities
# --------------------------------------------------

print("\nNEURORESTORE AI")
print("ISLES 2022 DATASET INSPECTION")
print(f"Case: {CASE_ID}\n")

for name, path in images.items():
    inspect_nifti(name, path)


# --------------------------------------------------
# Compare spatial information
# --------------------------------------------------

print("=" * 70)
print("SPATIAL COMPATIBILITY")
print("=" * 70)

loaded = {
    name: nib.load(path)
    for name, path in images.items()
}

for name, img in loaded.items():
    print(
        f"{name}: "
        f"shape={img.shape}, "
        f"spacing={img.header.get_zooms()}, "
        f"orientation={nib.aff2axcodes(img.affine)}"
    )


# --------------------------------------------------
# Mask information
# --------------------------------------------------

mask_data = loaded["MASK"].get_fdata()

print("\n" + "=" * 70)
print("MASK INFORMATION")
print("=" * 70)

unique_values = np.unique(mask_data)

print("Unique mask values:", unique_values)
print("Number of unique values:", len(unique_values))
print(
    "Lesion voxels:",
    np.sum(mask_data > 0)
)