from pathlib import Path
import json

import nibabel as nib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREDICTION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "nnunet_test38"
)

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ISLES-2022"
)

SPLITS_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "metrics"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


PER_CASE_OUTPUT = OUTPUT_DIR / "nnunet_test38_per_case.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "nnunet_test38_summary.json"


# ============================================================
# METRIC FUNCTIONS
# ============================================================

def calculate_metrics(prediction, ground_truth):
    """
    Calculate binary segmentation metrics.
    """

    prediction = prediction.astype(bool)
    ground_truth = ground_truth.astype(bool)

    tp = np.logical_and(prediction, ground_truth).sum()
    fp = np.logical_and(prediction, ~ground_truth).sum()
    fn = np.logical_and(~prediction, ground_truth).sum()
    tn = np.logical_and(~prediction, ~ground_truth).sum()

    # Dice
    denominator = 2 * tp + fp + fn

    if denominator == 0:
        dice = 1.0
    else:
        dice = (2 * tp) / denominator

    # IoU
    union = tp + fp + fn

    if union == 0:
        iou = 1.0
    else:
        iou = tp / union

    # Precision
    precision_denominator = tp + fp

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = tp / precision_denominator

    # Recall
    recall_denominator = tp + fn

    if recall_denominator == 0:
        recall = 1.0
    else:
        recall = tp / recall_denominator

    return {
        "Dice": float(dice),
        "IoU": float(iou),
        "Precision": float(precision),
        "Recall": float(recall),
        "TP": int(tp),
        "FP": int(fp),
        "FN": int(fn),
        "TN": int(tn),
    }


# ============================================================
# GROUND-TRUTH SEARCH
# ============================================================

def find_ground_truth(case_id):
    """
    Find the processed ground-truth mask for a case.

    Preferred:
        *_mask.nii.gz

    Fallback:
        *_mask_aligned.nii.gz
    """

    case_dir = PROCESSED_DIR / case_id

    if not case_dir.exists():
        return None

    # Search recursively for normal mask
    normal_masks = list(case_dir.rglob("*_mask.nii.gz"))

    if normal_masks:
        return normal_masks[0]

    # Fallback to aligned mask
    aligned_masks = list(case_dir.rglob("*_mask_aligned.nii.gz"))

    if aligned_masks:
        return aligned_masks[0]

    return None


# ============================================================
# GEOMETRY CHECK
# ============================================================

def check_geometry(pred_img, gt_img):
    """
    Check shape and voxel spacing.
    """

    pred_shape = pred_img.shape
    gt_shape = gt_img.shape

    if pred_shape != gt_shape:
        raise ValueError(
            f"Shape mismatch: prediction={pred_shape}, "
            f"ground_truth={gt_shape}"
        )

    pred_spacing = pred_img.header.get_zooms()[:3]
    gt_spacing = gt_img.header.get_zooms()[:3]

    if not np.allclose(pred_spacing, gt_spacing, atol=1e-4):
        raise ValueError(
            f"Spacing mismatch: prediction={pred_spacing}, "
            f"ground_truth={gt_spacing}"
        )


# ============================================================
# VOLUME
# ============================================================

def calculate_volume_ml(mask, image):
    """
    Calculate lesion volume in milliliters.
    """

    voxel_volume_mm3 = np.prod(
        image.header.get_zooms()[:3]
    )

    voxel_count = np.sum(mask)

    volume_mm3 = voxel_count * voxel_volume_mm3

    volume_ml = volume_mm3 / 1000.0

    return float(volume_ml)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    print("=" * 70)
    print("ISLES 2022 PRETRAINED NNUNET - 38 CASE EVALUATION")
    print("=" * 70)

    if not PREDICTION_DIR.exists():
        raise FileNotFoundError(
            f"Prediction directory not found:\n{PREDICTION_DIR}"
        )

    prediction_files = sorted(
        PREDICTION_DIR.glob("*.nii.gz")
    )

    print(f"\nPrediction directory:")
    print(PREDICTION_DIR)

    print(f"\nPrediction files found: {len(prediction_files)}")

    if len(prediction_files) == 0:
        raise RuntimeError(
            "No prediction files found."
        )

    results = []

    failed_cases = []

    # ========================================================
    # CASE LOOP
    # ========================================================

    for index, prediction_path in enumerate(prediction_files, start=1):

        case_id = prediction_path.name.replace(
            ".nii.gz",
            ""
        )

        print(
            f"\n[{index}/{len(prediction_files)}] "
            f"{case_id}"
        )

        try:

            # ------------------------------------------------
            # Load prediction
            # ------------------------------------------------

            pred_img = nib.load(
                str(prediction_path)
            )

            pred_data = pred_img.get_fdata()

            prediction = pred_data > 0.5

            # ------------------------------------------------
            # Find ground truth
            # ------------------------------------------------

            gt_path = find_ground_truth(case_id)

            if gt_path is None:
                raise FileNotFoundError(
                    f"Ground-truth mask not found for {case_id}"
                )

            gt_img = nib.load(
                str(gt_path)
            )

            gt_data = gt_img.get_fdata()

            ground_truth = gt_data > 0.5

            # ------------------------------------------------
            # Geometry validation
            # ------------------------------------------------

            check_geometry(
                pred_img,
                gt_img
            )

            # ------------------------------------------------
            # Metrics
            # ------------------------------------------------

            metrics = calculate_metrics(
                prediction,
                ground_truth
            )

            # ------------------------------------------------
            # Volumes
            # ------------------------------------------------

            gt_volume_ml = calculate_volume_ml(
                ground_truth,
                gt_img
            )

            pred_volume_ml = calculate_volume_ml(
                prediction,
                pred_img
            )

            volume_error_ml = (
                pred_volume_ml - gt_volume_ml
            )

            absolute_volume_error_ml = abs(
                volume_error_ml
            )

            # ------------------------------------------------
            # Result row
            # ------------------------------------------------

            result = {
                "case_id": case_id,

                "Dice": metrics["Dice"],
                "IoU": metrics["IoU"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],

                "TP": metrics["TP"],
                "FP": metrics["FP"],
                "FN": metrics["FN"],
                "TN": metrics["TN"],

                "gt_voxels": int(
                    np.sum(ground_truth)
                ),

                "pred_voxels": int(
                    np.sum(prediction)
                ),

                "gt_volume_ml": gt_volume_ml,
                "pred_volume_ml": pred_volume_ml,

                "volume_error_ml": volume_error_ml,
                "absolute_volume_error_ml":
                    absolute_volume_error_ml,
            }

            results.append(result)

            print(
                f"Dice      : {metrics['Dice']:.6f}"
            )

            print(
                f"IoU       : {metrics['IoU']:.6f}"
            )

            print(
                f"Precision : {metrics['Precision']:.6f}"
            )

            print(
                f"Recall    : {metrics['Recall']:.6f}"
            )

            print(
                f"GT volume : {gt_volume_ml:.3f} ml"
            )

            print(
                f"Pred vol  : {pred_volume_ml:.3f} ml"
            )

        except Exception as error:

            print(
                f"FAILED: {error}"
            )

            failed_cases.append(
                {
                    "case_id": case_id,
                    "error": str(error)
                }
            )

    # ========================================================
    # DATAFRAME
    # ========================================================

    if len(results) == 0:
        raise RuntimeError(
            "No cases were successfully evaluated."
        )

    df = pd.DataFrame(results)

    df = df.sort_values(
        "case_id"
    ).reset_index(drop=True)

    # ========================================================
    # AGGREGATE METRICS
    # ========================================================

    metric_names = [
        "Dice",
        "IoU",
        "Precision",
        "Recall",
        "gt_volume_ml",
        "pred_volume_ml",
        "absolute_volume_error_ml",
    ]

    aggregate_metrics = {}

    for metric in metric_names:

        values = df[metric].astype(float)

        aggregate_metrics[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    # ========================================================
    # LESION / NO-LESION CASES
    # ========================================================

    lesion_cases = int(
        (df["gt_voxels"] > 0).sum()
    )

    no_lesion_cases = int(
        (df["gt_voxels"] == 0).sum()
    )

    # ========================================================
    # EXPERIMENT INFORMATION
    # ========================================================

    summary = {

        "experiment": {
            "name":
                "Pretrained ISLES22 nnU-Net baseline",

            "dataset":
                "ISLES 2022",

            "split":
                "held-out test",

            "test_cases":
                int(len(prediction_files)),

            "model":
                "nnU-Net v2 3D full-resolution",

            "checkpoint":
                "checkpoint_best.pth",

            "fold":
                0,

            "device":
                "CPU",

            "tta":
                False,
        },

        "evaluation": {

            "cases_evaluated":
                int(len(df)),

            "cases_failed":
                int(len(failed_cases)),

            "lesion_cases":
                lesion_cases,

            "no_lesion_cases":
                no_lesion_cases,
        },

        "aggregate_metrics":
            aggregate_metrics,

        "failed_cases":
            failed_cases,
    }

    # ========================================================
    # SAVE PER-CASE RESULTS
    # ========================================================

    df.to_csv(
        PER_CASE_OUTPUT,
        index=False
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=2
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print("\n")
    print("=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)

    print(
        f"Cases evaluated : {len(df)}"
    )

    print(
        f"Cases failed    : {len(failed_cases)}"
    )

    print(
        f"Lesion cases    : {lesion_cases}"
    )

    print(
        f"No-lesion cases : {no_lesion_cases}"
    )

    print("\nAggregate metrics:")

    print(
        f"Mean Dice      : "
        f"{aggregate_metrics['Dice']['mean']:.6f}"
    )

    print(
        f"Median Dice    : "
        f"{aggregate_metrics['Dice']['median']:.6f}"
    )

    print(
        f"Mean IoU       : "
        f"{aggregate_metrics['IoU']['mean']:.6f}"
    )

    print(
        f"Mean Precision : "
        f"{aggregate_metrics['Precision']['mean']:.6f}"
    )

    print(
        f"Mean Recall    : "
        f"{aggregate_metrics['Recall']['mean']:.6f}"
    )

    print("\nOutput files:")

    print(
        f"Per-case CSV:\n{PER_CASE_OUTPUT}"
    )

    print(
        f"\nSummary JSON:\n{SUMMARY_OUTPUT}"
    )

    # ========================================================
    # STATUS
    # ========================================================

    if len(failed_cases) == 0:
        print("\nSTATUS: PASS")
        print(
            "38-CASE NNUNET BASELINE EVALUATION COMPLETE."
        )
    else:
        print("\nSTATUS: WARNING")
        print(
            f"{len(failed_cases)} case(s) failed evaluation."
        )


if __name__ == "__main__":
    main()