from pathlib import Path
import shutil
import sys

import nibabel as nib
import numpy as np
from scipy.ndimage import zoom


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "ISLES-2022"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "ISLES-2022"
SPLIT_FILE = PROJECT_ROOT / "data" / "splits" / "test.txt"

OUTPUT_ROOT = PROJECT_ROOT / "data" / "nnunet_inference" / "test"

# DWI is the reference modality, matching our preprocessing.
REFERENCE_MODALITY = "DWI"

EXPECTED_CHANNELS = {
    "0000": "DWI",
    "0001": "ADC",
    "0002": "FLAIR",
}


# ============================================================
# HELPERS
# ============================================================

def load_case_ids():
    if not SPLIT_FILE.exists():
        raise FileNotFoundError(f"Test split not found: {SPLIT_FILE}")

    case_ids = [
        line.strip()
        for line in SPLIT_FILE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    return case_ids


def find_raw_files(case_id):
    """
    Locate original ISLES-2022 DWI, ADC and FLAIR files.
    """

    case_root = RAW_ROOT / case_id

    if not case_root.exists():
        raise FileNotFoundError(f"Raw case not found: {case_root}")

    nii_files = list(case_root.rglob("*.nii.gz"))

    dwi_files = [
        p for p in nii_files
        if "_dwi.nii.gz" in p.name.lower()
        and "_adc.nii.gz" not in p.name.lower()
    ]

    adc_files = [
        p for p in nii_files
        if "_adc.nii.gz" in p.name.lower()
    ]

    flair_files = [
        p for p in nii_files
        if "_flair.nii.gz" in p.name.lower()
    ]

    if len(dwi_files) != 1:
        raise RuntimeError(
            f"{case_id}: expected exactly 1 DWI file, found {len(dwi_files)}"
        )

    if len(adc_files) != 1:
        raise RuntimeError(
            f"{case_id}: expected exactly 1 ADC file, found {len(adc_files)}"
        )

    if len(flair_files) != 1:
        raise RuntimeError(
            f"{case_id}: expected exactly 1 FLAIR file, found {len(flair_files)}"
        )

    return {
        "DWI": dwi_files[0],
        "ADC": adc_files[0],
        "FLAIR": flair_files[0],
    }


def same_geometry(img1, img2, atol=1e-5):
    return (
        img1.shape == img2.shape
        and np.allclose(img1.header.get_zooms()[:3],
                        img2.header.get_zooms()[:3],
                        atol=atol)
        and np.allclose(img1.affine, img2.affine, atol=atol)
    )


def resample_to_reference(source_img, reference_img, order):
    """
    Resample source image to the exact reference grid.

    order=1 -> linear interpolation
    order=0 -> nearest-neighbor
    """

    source_data = source_img.get_fdata(dtype=np.float32)
    reference_shape = reference_img.shape[:3]

    # We use scipy zoom only when needed.
    zoom_factors = np.array(reference_shape) / np.array(source_data.shape[:3])

    resampled = zoom(
        source_data,
        zoom_factors,
        order=order,
        mode="nearest",
        prefilter=(order > 1),
    )

    # Guard against tiny shape differences caused by interpolation.
    if resampled.shape != reference_shape:
        fixed = np.zeros(reference_shape, dtype=np.float32)

        min_shape = tuple(
            min(resampled.shape[i], reference_shape[i])
            for i in range(3)
        )

        fixed[
            :min_shape[0],
            :min_shape[1],
            :min_shape[2],
        ] = resampled[
            :min_shape[0],
            :min_shape[1],
            :min_shape[2],
        ]

        resampled = fixed

    return nib.Nifti1Image(
        resampled.astype(np.float32),
        reference_img.affine,
        reference_img.header.copy(),
    )


def prepare_case(case_id):
    print()
    print("=" * 70)
    print(f"Preparing {case_id}")
    print("=" * 70)

    raw_files = find_raw_files(case_id)

    print(f"DWI   : {raw_files['DWI']}")
    print(f"ADC   : {raw_files['ADC']}")
    print(f"FLAIR : {raw_files['FLAIR']}")

    # --------------------------------------------------------
    # Load original MRI volumes
    # --------------------------------------------------------

    dwi_img = nib.load(str(raw_files["DWI"]))
    adc_img = nib.load(str(raw_files["ADC"]))
    flair_img = nib.load(str(raw_files["FLAIR"]))

    print()
    print("Original geometry:")
    print(f"  DWI   : shape={dwi_img.shape[:3]}, spacing={dwi_img.header.get_zooms()[:3]}")
    print(f"  ADC   : shape={adc_img.shape[:3]}, spacing={adc_img.header.get_zooms()[:3]}")
    print(f"  FLAIR : shape={flair_img.shape[:3]}, spacing={flair_img.header.get_zooms()[:3]}")

    # --------------------------------------------------------
    # DWI = reference
    # --------------------------------------------------------

    reference_img = dwi_img

    # --------------------------------------------------------
    # Align ADC and FLAIR to DWI grid
    #
    # ADC/FLAIR: linear interpolation
    # DWI: unchanged
    # --------------------------------------------------------

    if same_geometry(adc_img, reference_img):
        adc_aligned = adc_img
        print("ADC already matches DWI geometry.")
    else:
        adc_aligned = resample_to_reference(
            adc_img,
            reference_img,
            order=1,
        )
        print("ADC resampled to DWI grid.")

    if same_geometry(flair_img, reference_img):
        flair_aligned = flair_img
        print("FLAIR already matches DWI geometry.")
    else:
        flair_aligned = resample_to_reference(
            flair_img,
            reference_img,
            order=1,
        )
        print("FLAIR resampled to DWI grid.")

    # --------------------------------------------------------
    # Output folder
    # --------------------------------------------------------

    case_output = OUTPUT_ROOT / case_id
    case_output.mkdir(parents=True, exist_ok=True)

    dwi_output = case_output / f"{case_id}_0000.nii.gz"
    adc_output = case_output / f"{case_id}_0001.nii.gz"
    flair_output = case_output / f"{case_id}_0002.nii.gz"

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    nib.save(
        nib.Nifti1Image(
            dwi_img.get_fdata(dtype=np.float32),
            dwi_img.affine,
            dwi_img.header.copy(),
        ),
        str(dwi_output),
    )

    nib.save(
        adc_aligned,
        str(adc_output),
    )

    nib.save(
        flair_aligned,
        str(flair_output),
    )

    # --------------------------------------------------------
    # Final verification
    # --------------------------------------------------------

    check_dwi = nib.load(str(dwi_output))
    check_adc = nib.load(str(adc_output))
    check_flair = nib.load(str(flair_output))

    images = {
        "DWI": check_dwi,
        "ADC": check_adc,
        "FLAIR": check_flair,
    }

    reference_shape = check_dwi.shape[:3]
    reference_spacing = check_dwi.header.get_zooms()[:3]
    reference_affine = check_dwi.affine

    for modality, img in images.items():

        if img.shape[:3] != reference_shape:
            raise RuntimeError(
                f"{case_id}: {modality} shape mismatch: "
                f"{img.shape[:3]} vs {reference_shape}"
            )

        if not np.allclose(
            img.header.get_zooms()[:3],
            reference_spacing,
            atol=1e-5,
        ):
            raise RuntimeError(
                f"{case_id}: {modality} spacing mismatch."
            )

        if not np.allclose(
            img.affine,
            reference_affine,
            atol=1e-5,
        ):
            raise RuntimeError(
                f"{case_id}: {modality} affine mismatch."
            )

        data = img.get_fdata(dtype=np.float32)

        if not np.isfinite(data).all():
            raise RuntimeError(
                f"{case_id}: {modality} contains NaN or Inf."
            )

    print()
    print("Final geometry:")
    print(f"  Shape      : {reference_shape}")
    print(f"  Spacing    : {reference_spacing}")
    print(f"  Orientation: {nib.orientations.aff2axcodes(reference_affine)}")

    print()
    print("Created:")
    print(f"  {dwi_output.name}")
    print(f"  {adc_output.name}")
    print(f"  {flair_output.name}")

    print(f"\n{case_id}: PASS")

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("NEURORESTORE AI")
    print("nnU-Net Test Set Preparation")
    print("=" * 70)

    case_ids = load_case_ids()

    print(f"\nTest cases requested: {len(case_ids)}")
    print(f"Output directory: {OUTPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    successful = []
    failed = []

    for index, case_id in enumerate(case_ids, start=1):

        print()
        print(f"[{index}/{len(case_ids)}] {case_id}")

        try:
            prepare_case(case_id)
            successful.append(case_id)

        except Exception as exc:
            print(f"\n{case_id}: FAILED")
            print(f"Reason: {exc}")
            failed.append((case_id, str(exc)))

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"Requested : {len(case_ids)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed    : {len(failed)}")

    if failed:
        print("\nFailed cases:")
        for case_id, reason in failed:
            print(f"  {case_id}: {reason}")

        print("\nSTATUS: FAIL")
        sys.exit(1)

    # Count expected files.
    output_files = list(OUTPUT_ROOT.rglob("*.nii.gz"))

    expected_files = len(case_ids) * 3

    print(f"\nExpected NIfTI files: {expected_files}")
    print(f"Actual NIfTI files  : {len(output_files)}")

    if len(output_files) != expected_files:
        print("STATUS: FAIL")
        sys.exit(1)

    print("\nSTATUS: PASS")
    print("ALL TEST CASES PREPARED FOR NNUNET INFERENCE.")


if __name__ == "__main__":
    main()