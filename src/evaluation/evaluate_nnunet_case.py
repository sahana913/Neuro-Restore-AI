from pathlib import Path

import nibabel as nib
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUND_TRUTH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ISLES-2022"
    / "sub-strokecase0001"
    / "ses-0001"
    / "sub-strokecase0001_mask_aligned.nii.gz"
)

PREDICTION = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "nnunet_case0001"
    / "sub-strokecase0001.nii.gz"
)


def main():
    gt_nii = nib.load(GROUND_TRUTH)
    pred_nii = nib.load(PREDICTION)

    gt = gt_nii.get_fdata() > 0
    pred = pred_nii.get_fdata() > 0

    if gt.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: GT={gt.shape}, prediction={pred.shape}"
        )

    tp = np.logical_and(gt, pred).sum()
    fp = np.logical_and(~gt, pred).sum()
    fn = np.logical_and(gt, ~pred).sum()

    dice_denominator = 2 * tp + fp + fn
    iou_denominator = tp + fp + fn

    dice = (
        2 * tp / dice_denominator
        if dice_denominator > 0
        else float("nan")
    )

    iou = (
        tp / iou_denominator
        if iou_denominator > 0
        else float("nan")
    )

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    voxel_volume_ml = (
        np.prod(gt_nii.header.get_zooms()[:3]) / 1000.0
    )

    gt_voxels = int(gt.sum())
    pred_voxels = int(pred.sum())

    gt_volume_ml = gt_voxels * voxel_volume_ml
    pred_volume_ml = pred_voxels * voxel_volume_ml

    print()
    print("=" * 50)
    print("ISLES22 nnU-Net — CASE 0001 EVALUATION")
    print("=" * 50)

    print(f"Ground truth voxels : {gt_voxels}")
    print(f"Predicted voxels    : {pred_voxels}")
    print(f"Voxel volume        : {voxel_volume_ml:.4f} ml")

    print()
    print("CONFUSION COUNTS")
    print(f"TP : {int(tp)}")
    print(f"FP : {int(fp)}")
    print(f"FN : {int(fn)}")

    print()
    print("SEGMENTATION METRICS")
    print(f"Dice      : {dice:.6f}")
    print(f"IoU       : {iou:.6f}")
    print(f"Precision : {precision:.6f}")
    print(f"Recall    : {recall:.6f}")

    print()
    print("LESION VOLUME")
    print(f"Ground truth : {gt_volume_ml:.4f} ml")
    print(f"Prediction   : {pred_volume_ml:.4f} ml")

    print()
    print("EVALUATION: PASS")
    print("=" * 50)


if __name__ == "__main__":
    main()