from pathlib import Path

import nibabel as nib
import numpy as np


# ============================================================
# NEURORESTORE AI
# ISLES 2022
# MRI INTENSITY NORMALIZATION
# ============================================================

PROJECT_ROOT = Path(r"D:\NeuroRestore-AI")

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ISLES-2022"
)

CASE_ID = "sub-strokecase0001"

CASE_DIR = (
    PROCESSED_ROOT
    / CASE_ID
    / "ses-0001"
)


# ============================================================
# INPUT FILES
# ============================================================

DWI_PATH = (
    CASE_DIR
    / f"{CASE_ID}_dwi_aligned.nii.gz"
)

ADC_PATH = (
    CASE_DIR
    / f"{CASE_ID}_adc_aligned.nii.gz"
)

FLAIR_PATH = (
    CASE_DIR
    / f"{CASE_ID}_FLAIR_aligned.nii.gz"
)

MASK_PATH = (
    CASE_DIR
    / f"{CASE_ID}_mask_aligned.nii.gz"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

NORMALIZED_DIR = (
    CASE_DIR
    / "normalized"
)


# ============================================================
# NORMALIZATION PARAMETERS
# ============================================================

LOW_PERCENTILE = 1.0
HIGH_PERCENTILE = 99.0

EPSILON = 1e-8


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def load_nifti(path):
    """
    Load a NIfTI image and return the image object
    and floating-point image data.
    """

    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist:\n{path}"
        )

    image = nib.load(str(path))

    data = image.get_fdata(
        dtype=np.float32
    )

    return image, data


def replace_invalid_values(data):
    """
    Replace NaN and infinite values with zero.
    """

    data = np.nan_to_num(
        data,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return data.astype(
        np.float32,
        copy=False,
    )


def create_foreground_mask(data):
    """
    Create a simple foreground mask.

    ISLES images contain background values around zero.
    We use non-zero voxels as the initial foreground mask.
    """

    foreground = (
        np.abs(data) > EPSILON
    )

    return foreground


def normalize_mri(
    data,
    modality_name,
):
    """
    Robust MRI intensity normalization.

    Steps:
        1. Replace NaN/Inf.
        2. Identify non-zero foreground.
        3. Calculate robust 1st and 99th percentiles.
        4. Clip intensities to that range.
        5. Calculate mean/std on the foreground.
        6. Apply z-score normalization.
        7. Set background to zero.

    This function is intended for MRI images only.
    It must NOT be used on the segmentation mask.
    """

    print()
    print(
        f"Normalizing {modality_name}..."
    )

    data = replace_invalid_values(
        data
    )

    foreground = create_foreground_mask(
        data
    )

    foreground_values = data[
        foreground
    ]

    if foreground_values.size == 0:
        raise RuntimeError(
            f"{modality_name}: "
            "No non-zero foreground voxels found."
        )

    # --------------------------------------------------------
    # Original statistics
    # --------------------------------------------------------

    original_min = float(
        np.min(foreground_values)
    )

    original_max = float(
        np.max(foreground_values)
    )

    original_mean = float(
        np.mean(foreground_values)
    )

    original_std = float(
        np.std(foreground_values)
    )

    print(
        f"Original foreground min: "
        f"{original_min:.6f}"
    )

    print(
        f"Original foreground max: "
        f"{original_max:.6f}"
    )

    print(
        f"Original foreground mean: "
        f"{original_mean:.6f}"
    )

    print(
        f"Original foreground std: "
        f"{original_std:.6f}"
    )

    # --------------------------------------------------------
    # Robust percentile clipping
    # --------------------------------------------------------

    lower_bound = float(
        np.percentile(
            foreground_values,
            LOW_PERCENTILE,
        )
    )

    upper_bound = float(
        np.percentile(
            foreground_values,
            HIGH_PERCENTILE,
        )
    )

    print(
        f"{LOW_PERCENTILE:g}th percentile: "
        f"{lower_bound:.6f}"
    )

    print(
        f"{HIGH_PERCENTILE:g}th percentile: "
        f"{upper_bound:.6f}"
    )

    if upper_bound <= lower_bound:
        raise RuntimeError(
            f"{modality_name}: "
            "Invalid percentile range."
        )

    clipped = np.clip(
        data,
        lower_bound,
        upper_bound,
    )

    # --------------------------------------------------------
    # Calculate statistics after clipping
    # --------------------------------------------------------

    clipped_foreground_values = clipped[
        foreground
    ]

    mean_value = float(
        np.mean(
            clipped_foreground_values
        )
    )

    std_value = float(
        np.std(
            clipped_foreground_values
        )
    )

    print(
        f"Clipped foreground mean: "
        f"{mean_value:.6f}"
    )

    print(
        f"Clipped foreground std: "
        f"{std_value:.6f}"
    )

    if std_value < EPSILON:
        raise RuntimeError(
            f"{modality_name}: "
            "Standard deviation is too small."
        )

    # --------------------------------------------------------
    # Z-score normalization
    # --------------------------------------------------------

    normalized = (
        clipped - mean_value
    ) / (
        std_value + EPSILON
    )

    # --------------------------------------------------------
    # Restore background to zero
    # --------------------------------------------------------

    normalized[
        ~foreground
    ] = 0.0

    normalized = normalized.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    normalized_foreground = normalized[
        foreground
    ]

    final_min = float(
        np.min(
            normalized_foreground
        )
    )

    final_max = float(
        np.max(
            normalized_foreground
        )
    )

    final_mean = float(
        np.mean(
            normalized_foreground
        )
    )

    final_std = float(
        np.std(
            normalized_foreground
        )
    )

    print(
        f"Normalized foreground min: "
        f"{final_min:.6f}"
    )

    print(
        f"Normalized foreground max: "
        f"{final_max:.6f}"
    )

    print(
        f"Normalized foreground mean: "
        f"{final_mean:.6f}"
    )

    print(
        f"Normalized foreground std: "
        f"{final_std:.6f}"
    )

    return normalized


def save_nifti(
    data,
    reference_image,
    output_path,
):
    """
    Save normalized data using the reference
    NIfTI affine and header.
    """

    output_image = nib.Nifti1Image(
        data.astype(np.float32),
        reference_image.affine,
        reference_image.header.copy(),
    )

    output_image.set_data_dtype(
        np.float32
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        output_image,
        str(output_path),
    )


def validate_output(
    output_path,
    expected_shape,
):
    """
    Load an output NIfTI file and verify that it
    exists and has the expected shape.
    """

    if not output_path.exists():
        raise RuntimeError(
            f"Output file was not created:\n"
            f"{output_path}"
        )

    image = nib.load(
        str(output_path)
    )

    data = image.get_fdata()

    if data.shape != expected_shape:
        raise RuntimeError(
            f"Shape mismatch for {output_path.name}: "
            f"{data.shape} != {expected_shape}"
        )

    if not np.all(
        np.isfinite(data)
    ):
        raise RuntimeError(
            f"Non-finite values detected in "
            f"{output_path.name}"
        )

    print(
        f"✓ {output_path.name}"
    )
    print(
        f"  Shape: {data.shape}"
    )
    print(
        f"  Dtype: {data.dtype}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "NEURORESTORE AI"
    )

    print(
        "ISLES 2022 MRI INTENSITY NORMALIZATION"
    )

    print(
        f"Case: {CASE_ID}"
    )

    # --------------------------------------------------------
    # Load aligned images
    # --------------------------------------------------------

    print_header(
        "STEP 1 - LOADING ALIGNED DATA"
    )

    dwi_image, dwi = load_nifti(
        DWI_PATH
    )

    adc_image, adc = load_nifti(
        ADC_PATH
    )

    flair_image, flair = load_nifti(
        FLAIR_PATH
    )

    mask_image, mask = load_nifti(
        MASK_PATH
    )

    print(
        "✓ DWI loaded."
    )

    print(
        "✓ ADC loaded."
    )

    print(
        "✓ FLAIR loaded."
    )

    print(
        "✓ MASK loaded."
    )

    # --------------------------------------------------------
    # Validate common shape
    # --------------------------------------------------------

    print_header(
        "STEP 2 - VALIDATING INPUT SHAPES"
    )

    expected_shape = dwi.shape

    print(
        f"DWI:   {dwi.shape}"
    )

    print(
        f"ADC:   {adc.shape}"
    )

    print(
        f"FLAIR: {flair.shape}"
    )

    print(
        f"MASK:  {mask.shape}"
    )

    if not (
        dwi.shape
        == adc.shape
        == flair.shape
        == mask.shape
    ):
        raise RuntimeError(
            "ERROR: Input shapes do not match."
        )

    print(
        "\n✓ All inputs have matching shapes."
    )

    # --------------------------------------------------------
    # Validate affines
    # --------------------------------------------------------

    print_header(
        "STEP 3 - VALIDATING AFFINES"
    )

    reference_affine = dwi_image.affine

    for name, image in [
        ("DWI", dwi_image),
        ("ADC", adc_image),
        ("FLAIR", flair_image),
        ("MASK", mask_image),
    ]:

        difference = np.max(
            np.abs(
                reference_affine
                - image.affine
            )
        )

        print(
            f"{name}: "
            f"maximum affine difference = "
            f"{difference:.10f}"
        )

        if not np.allclose(
            reference_affine,
            image.affine,
        ):
            raise RuntimeError(
                f"ERROR: {name} affine does not match DWI."
            )

    print(
        "\n✓ All affines match."
    )

    # --------------------------------------------------------
    # Validate mask
    # --------------------------------------------------------

    print_header(
        "STEP 4 - VALIDATING MASK"
    )

    mask = np.rint(
        mask
    ).astype(
        np.uint8
    )

    mask_unique = np.unique(
        mask
    )

    print(
        f"Mask unique values: "
        f"{mask_unique}"
    )

    if not np.all(
        np.isin(
            mask_unique,
            [0, 1],
        )
    ):
        raise RuntimeError(
            "ERROR: Mask is not binary."
        )

    lesion_voxels = int(
        np.sum(mask == 1)
    )

    print(
        f"Lesion voxels: "
        f"{lesion_voxels}"
    )

    print(
        "✓ Mask is binary."
    )

    # --------------------------------------------------------
    # Normalize modalities
    # --------------------------------------------------------

    print_header(
        "STEP 5 - MODALITY NORMALIZATION"
    )

    dwi_normalized = normalize_mri(
        dwi,
        "DWI",
    )

    adc_normalized = normalize_mri(
        adc,
        "ADC",
    )

    flair_normalized = normalize_mri(
        flair,
        "FLAIR",
    )

    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    print_header(
        "STEP 6 - CREATING OUTPUT DIRECTORY"
    )

    NORMALIZED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Output directory:\n"
        f"{NORMALIZED_DIR}"
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    dwi_output = (
        NORMALIZED_DIR
        / f"{CASE_ID}_dwi_normalized.nii.gz"
    )

    adc_output = (
        NORMALIZED_DIR
        / f"{CASE_ID}_adc_normalized.nii.gz"
    )

    flair_output = (
        NORMALIZED_DIR
        / f"{CASE_ID}_FLAIR_normalized.nii.gz"
    )

    mask_output = (
        NORMALIZED_DIR
        / f"{CASE_ID}_mask.nii.gz"
    )

    # --------------------------------------------------------
    # Save normalized modalities
    # --------------------------------------------------------

    print_header(
        "STEP 7 - SAVING NORMALIZED DATA"
    )

    save_nifti(
        dwi_normalized,
        dwi_image,
        dwi_output,
    )

    print(
        f"Saved:\n{dwi_output}"
    )

    save_nifti(
        adc_normalized,
        adc_image,
        adc_output,
    )

    print(
        f"Saved:\n{adc_output}"
    )

    save_nifti(
        flair_normalized,
        flair_image,
        flair_output,
    )

    print(
        f"Saved:\n{flair_output}"
    )

    # --------------------------------------------------------
    # Save mask without normalization
    # --------------------------------------------------------

    mask_nifti = nib.Nifti1Image(
        mask,
        mask_image.affine,
        mask_image.header.copy(),
    )

    mask_nifti.set_data_dtype(
        np.uint8
    )

    nib.save(
        mask_nifti,
        str(mask_output),
    )

    print(
        f"Saved unchanged binary mask:\n"
        f"{mask_output}"
    )

    # --------------------------------------------------------
    # Validate outputs
    # --------------------------------------------------------

    print_header(
        "STEP 8 - OUTPUT VALIDATION"
    )

    validate_output(
        dwi_output,
        expected_shape,
    )

    validate_output(
        adc_output,
        expected_shape,
    )

    validate_output(
        flair_output,
        expected_shape,
    )

    # Validate mask separately
    mask_check_image = nib.load(
        str(mask_output)
    )

    mask_check = mask_check_image.get_fdata()

    print(
        f"\nMask output shape: "
        f"{mask_check.shape}"
    )

    print(
        f"Mask output unique values: "
        f"{np.unique(mask_check)}"
    )

    if mask_check.shape != expected_shape:
        raise RuntimeError(
            "ERROR: Saved mask shape is incorrect."
        )

    if not np.all(
        np.isin(
            np.unique(mask_check),
            [0, 1],
        )
    ):
        raise RuntimeError(
            "ERROR: Saved mask is no longer binary."
        )

    print(
        "✓ Mask output validated."
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print_header(
        "STEP 9 - FINAL VERIFICATION"
    )

    normalized_files = {
        "DWI": dwi_output,
        "ADC": adc_output,
        "FLAIR": flair_output,
        "MASK": mask_output,
    }

    for name, path in normalized_files.items():

        image = nib.load(
            str(path)
        )

        data = image.get_fdata()

        print(
            f"{name}: "
            f"shape={data.shape}, "
            f"spacing={image.header.get_zooms()[:3]}, "
            f"orientation={nib.aff2axcodes(image.affine)}"
        )

    print_header(
        "NORMALIZATION SUCCESS"
    )

    print(
        "✓ DWI normalized independently."
    )

    print(
        "✓ ADC normalized independently."
    )

    print(
        "✓ FLAIR normalized independently."
    )

    print(
        "✓ Mask preserved as binary 0/1."
    )

    print(
        "✓ NaN/Inf values checked."
    )

    print(
        "✓ All outputs retain the common spatial grid."
    )

    print()
    print(
        "Normalized data saved to:"
    )

    print(
        NORMALIZED_DIR
    )


if __name__ == "__main__":
    main()