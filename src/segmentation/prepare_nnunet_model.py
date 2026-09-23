from pathlib import Path
import json
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models" / "isles22_nnunet"
CHECKPOINT = MODEL_DIR / "fold_0" / "checkpoint_best.pth"


def main():
    print("Loading checkpoint...")
    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    dataset_json = checkpoint["init_args"]["dataset_json"]
    plans = checkpoint["init_args"]["plans"]

    dataset_path = MODEL_DIR / "dataset.json"
    plans_path = MODEL_DIR / "plans.json"

    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(dataset_json, f, indent=2)

    with open(plans_path, "w", encoding="utf-8") as f:
        json.dump(plans, f, indent=2)

    print()
    print("MODEL METADATA CREATED")
    print("----------------------")
    print(f"Dataset JSON : {dataset_path}")
    print(f"Plans JSON   : {plans_path}")
    print()
    print("Dataset:")
    print(json.dumps(dataset_json, indent=2))
    print()
    print("Configuration:", plans["configurations"].keys())
    print("Plans name:", plans["plans_name"])
    print("Dataset name:", plans["dataset_name"])


if __name__ == "__main__":
    main()