from pathlib import Path

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to


# ============================================================
# NEURORESTORE AI
# ISLES 2022 - CASE SPATIAL ALIGNMENT
# ============================================================

PROJECT_ROOT = Path(r"D:\NeuroRestore-AI")

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ISLES-2022"
)

PROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ISLES-2022"
)

CASE_ID = "sub-strokecase0001"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def find_case_files(case_id):
    """
    Locate the four required NIfTI files for one ISLES 2022 case.
    """

    case_dir = DATASET_ROOT / case_id

    if not case_dir.exists():
        raise FileNotFoundError(
            f"Case directory not found:\n{case_dir}"
        )

    anat_dir = case_dir / "ses-0001" / "anat"
    dwi_dir = case_dir / "ses-0001" / "dwi"
    mask_dir = (
        DATASET_ROOT
        / "derivatives"
        / case_id
        / "ses-0001"
    )

    flair_files = list(anat_dir.glob("*_FLAIR.nii.gz"))
    adc_files = list(dwi_dir.glob("*_adc.nii.gz"))
    dwi_files = list(dwi_dir.glob("*_dwi.nii.gz"))
    mask_files = list(mask_dir.glob("*_msk.nii.gz"))

    if len(flair_files) != 1:
        raise RuntimeError(
            f"Expected exactly one FLAIR file, found {len(flair_files)}"
        )

    if len(adc_files) != 1:
        raise RuntimeError(
            f"Expected exactly one ADC file, found {len(adc_files)}"
        )

    if len(dwi_files) != 1:
        raise RuntimeError(
            f"Expected exactly one DWI file, found {len(dwi_files)}"
        )

    if len(mask_files) != 1:
        raise RuntimeError(
            f"Expected exactly one mask file, found {len(mask_files)}"
        )

    return {
        "FLAIR": flair_files[0],
        "ADC": adc_files[0],
        "DWI": dwi_files[0],
        "MASK": mask_files[0],
    }


def load_nifti(path):
    """
    Load a NIfTI file.
    """

    image = nib.load(str(path))

    return image


def print_image_information(name, image):
    """
    Print shape, spacing, orientation, dtype and affine.
    """

    data = image.get_fdata()

    spacing = image.header.get_zooms()[:3]
    orientation = nib.aff2axcodes(image.affine)

    print(f"\n{name}")
    print("-" * 75)

    print(f"Shape:       {image.shape}")
    print(f"Spacing:     {spacing}")
    print(f"Orientation: {orientation}")
    print(f"Data type:   {data.dtype}")

    print("Affine:")
    print(image.affine)

    print(
        f"Intensity range: "
        f"{np.nanmin(data):.6f} → {np.nanmax(data):.6f}"
    )


def compare_affines(image_a, image_b, name_a, name_b):
    """
    Compare two affine matrices.
    """

    difference = np.max(
        np.abs(image_a.affine - image_b.affine)
    )

    print(
        f"\nMaximum affine difference "
        f"({name_a} vs {name_b}): {difference:.8f}"
    )

    if np.allclose(image_a.affine, image_b.affine):
        print("Affine status: IDENTICAL")
    else:
        print("Affine status: DIFFERENT")


def save_nifti(data, reference_image, output_path, dtype):
    """
    Save a NIfTI image using the reference image affine/header.
    """

    data = np.asarray(data, dtype=dtype)

    output_image = nib.Nifti1Image(
        data,
        reference_image.affine,
        reference_image.header.copy(),
    )

    output_image.set_data_dtype(dtype)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    nib.save(
        output_image,
        str(output_path),
    )


def resample_image_to_reference(
    image,
    reference_image,
    interpolation_order,
):
    """
    Resample an image onto the reference image grid.

    interpolation_order:
        0 = nearest neighbor
        1 = linear
    """

    target = (
        reference_image.shape[:3],
        reference_image.affine,
    )

    resampled = resample_from_to(
        image,
        target,
        order=interpolation_order,
    )

    return resampled


# ============================================================
# MAIN ALIGNMENT PIPELINE
# ============================================================

def main():

    print_header("NEURORESTORE AI")
    print("ISLES 2022 SPATIAL ALIGNMENT")
    print(f"Case: {CASE_ID}")

    # --------------------------------------------------------
    # Locate files
    # --------------------------------------------------------

    print_header("STEP 1 - LOCATING FILES")

    files = find_case_files(CASE_ID)

    for name, path in files.items():
        print(f"{name}:")
        print(path)

    # --------------------------------------------------------
    # Load images
    # --------------------------------------------------------

    print_header("STEP 2 - LOADING NIFTI FILES")

    flair = load_nifti(files["FLAIR"])
    adc = load_nifti(files["ADC"])
    dwi = load_nifti(files["DWI"])
    mask = load_nifti(files["MASK"])

    print("All NIfTI files loaded successfully.")

    # --------------------------------------------------------
    # Original image information
    # --------------------------------------------------------

    print_header("STEP 3 - ORIGINAL IMAGE INFORMATION")

    print_image_information("FLAIR", flair)
    print_image_information("ADC", adc)
    print_image_information("DWI", dwi)
    print_image_information("MASK", mask)

    # --------------------------------------------------------
    # DWI is the reference grid
    # --------------------------------------------------------

    print_header("STEP 4 - DWI REFERENCE GRID")

    print("DWI will be used as the common spatial reference.")

    print(f"Reference shape: {dwi.shape}")
    print(
        f"Reference spacing: "
        f"{dwi.header.get_zooms()[:3]}"
    )
    print(
        f"Reference orientation: "
        f"{nib.aff2axcodes(dwi.affine)}"
    )

    # --------------------------------------------------------
    # Compare original affines
    # --------------------------------------------------------

    print_header("STEP 5 - AFFINE COMPARISON")

    compare_affines(
        dwi,
        adc,
        "DWI",
        "ADC",
    )

    compare_affines(
        dwi,
        flair,
        "DWI",
        "FLAIR",
    )

    compare_affines(
        dwi,
        mask,
        "DWI",
        "MASK",
    )

    # --------------------------------------------------------
    # Resample all images onto DWI grid
    # --------------------------------------------------------

    print_header("STEP 6 - RESAMPLING")

    print("Resampling FLAIR to DWI grid...")
    print("Interpolation: linear")

    flair_aligned = resample_image_to_reference(
        flair,
        dwi,
        interpolation_order=1,
    )

    print("FLAIR resampling complete.")

    print("\nResampling ADC to DWI grid...")
    print("Interpolation: linear")

    adc_aligned = resample_image_to_reference(
        adc,
        dwi,
        interpolation_order=1,
    )

    print("ADC resampling complete.")

    print("\nResampling MASK to DWI grid...")
    print("Interpolation: nearest neighbor")

    mask_aligned = resample_image_to_reference(
        mask,
        dwi,
        interpolation_order=0,
    )

    print("MASK resampling complete.")

    # --------------------------------------------------------
    # DWI does not need spatial resampling
    # --------------------------------------------------------

    dwi_aligned = dwi

    # --------------------------------------------------------
    # Convert data
    # --------------------------------------------------------

    dwi_data = dwi_aligned.get_fdata()
    adc_data = adc_aligned.get_fdata()
    flair_data = flair_aligned.get_fdata()
    mask_data = mask_aligned.get_fdata()

    # --------------------------------------------------------
    # Validate shapes
    # --------------------------------------------------------

    print_header("STEP 7 - ALIGNMENT VALIDATION")

    print(f"DWI aligned shape:   {dwi_data.shape}")
    print(f"ADC aligned shape:   {adc_data.shape}")
    print(f"FLAIR aligned shape: {flair_data.shape}")
    print(f"MASK aligned shape:  {mask_data.shape}")

    expected_shape = dwi.shape[:3]

    all_shapes_match = (
        dwi_data.shape == expected_shape
        and adc_data.shape == expected_shape
        and flair_data.shape == expected_shape
        and mask_data.shape == expected_shape
    )

    if all_shapes_match:
        print("\n✓ All image shapes match the DWI reference grid.")
    else:
        raise RuntimeError(
            "\nERROR: Image shapes do not match after resampling."
        )

    # --------------------------------------------------------
    # Validate mask
    # --------------------------------------------------------

    print_header("STEP 8 - MASK VALIDATION")

    mask_binary = np.rint(mask_data).astype(np.uint8)

    unique_values = np.unique(mask_binary)

    print(f"Mask unique values: {unique_values}")

    if not np.all(np.isin(unique_values, [0, 1])):
        raise RuntimeError(
            "ERROR: Mask contains values other than 0 and 1."
        )

    lesion_voxels = int(np.sum(mask_binary == 1))

    print(f"Lesion voxels: {lesion_voxels}")

    print("✓ Mask is binary.")

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_dir = (
        PROCESSED_ROOT
        / CASE_ID
        / "ses-0001"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    dwi_output = output_dir / f"{CASE_ID}_dwi_aligned.nii.gz"
    adc_output = output_dir / f"{CASE_ID}_adc_aligned.nii.gz"
    flair_output = output_dir / f"{CASE_ID}_FLAIR_aligned.nii.gz"
    mask_output = output_dir / f"{CASE_ID}_mask_aligned.nii.gz"

    # --------------------------------------------------------
    # Save aligned images
    # --------------------------------------------------------

    print_header("STEP 9 - SAVING ALIGNED DATA")

    save_nifti(
        dwi_data,
        dwi,
        dwi_output,
        np.float32,
    )

    print(f"Saved DWI:   {dwi_output}")

    save_nifti(
        adc_data,
        dwi,
        adc_output,
        np.float32,
    )

    print(f"Saved ADC:   {adc_output}")

    save_nifti(
        flair_data,
        dwi,
        flair_output,
        np.float32,
    )

    print(f"Saved FLAIR: {flair_output}")

    save_nifti(
        mask_binary,
        dwi,
        mask_output,
        np.uint8,
    )

    print(f"Saved MASK:  {mask_output}")

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    print_header("STEP 10 - FINAL VERIFICATION")

    output_files = {
        "DWI": dwi_output,
        "ADC": adc_output,
        "FLAIR": flair_output,
        "MASK": mask_output,
    }

    for name, path in output_files.items():

        if not path.exists():
            raise RuntimeError(
                f"ERROR: Output file was not created: {path}"
            )

        image = nib.load(str(path))

        print(
            f"{name}: "
            f"shape={image.shape}, "
            f"spacing={image.header.get_zooms()[:3]}, "
            f"orientation={nib.aff2axcodes(image.affine)}"
        )

    print_header("ALIGNMENT SUCCESS")

    print("✓ DWI, ADC, FLAIR and MASK are now on one common grid.")
    print("✓ Reference grid: DWI")
    print("✓ Target shape: (112, 112, 73)")
    print("✓ Target spacing: approximately (2, 2, 2) mm")
    print("✓ Mask interpolation: nearest neighbor")
    print("✓ Image interpolation: linear")
    print()
    print("Processed case saved to:")
    print(output_dir)


if __name__ == "__main__":
    main()