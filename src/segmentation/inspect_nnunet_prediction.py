from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "nnunet_case0001"
)


def main():
    files = list(PREDICTION_DIR.glob("*.nii.gz"))

    if not files:
        raise FileNotFoundError(
            f"No NIfTI prediction found in: {PREDICTION_DIR}"
        )

    print("PREDICTION FILES")
    print("================")

    for file in files:
        nii = nib.load(file)
        data = nii.get_fdata()

        print(f"File: {file.name}")
        print(f"Shape: {data.shape}")
        print(f"Spacing: {nii.header.get_zooms()[:3]}")
        print(f"Orientation: {nib.aff2axcodes(nii.affine)}")
        print(f"Unique labels: {np.unique(data).tolist()}")
        print(f"Lesion voxels: {int(np.sum(data == 1))}")
        print()


if __name__ == "__main__":
    main()