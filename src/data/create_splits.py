from pathlib import Path
import json
import random


# ============================================================
# NEURORESTORE AI
# ISLES 2022
# PATIENT-LEVEL TRAIN / VALIDATION / TEST SPLIT
# ============================================================

PROJECT_ROOT = Path(r"D:\NeuroRestore-AI")

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "ISLES-2022"
)

SPLIT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "splits"
)


# ============================================================
# SPLIT CONFIGURATION
# ============================================================

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_SEED = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def print_header(title):
    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


def find_subjects():
    """
    Find all ISLES 2022 subject directories.

    Only directories following the expected
    'sub-strokecaseXXXX' naming convention are included.
    """

    if not DATASET_ROOT.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist:\n"
            f"{DATASET_ROOT}"
        )

    subjects = []

    for item in DATASET_ROOT.iterdir():

        if not item.is_dir():
            continue

        if not item.name.startswith(
            "sub-strokecase"
        ):
            continue

        subjects.append(
            item.name
        )

    subjects.sort()

    return subjects


def validate_ratios():
    """
    Verify that train/validation/test ratios
    add up to 1.
    """

    total = (
        TRAIN_RATIO
        + VAL_RATIO
        + TEST_RATIO
    )

    if abs(total - 1.0) > 1e-8:
        raise ValueError(
            "Train, validation and test ratios "
            f"must sum to 1.0. Current value: {total}"
        )


def split_subjects(subjects):
    """
    Create a reproducible patient-level split.
    """

    subjects = list(subjects)

    rng = random.Random(
        RANDOM_SEED
    )

    rng.shuffle(
        subjects
    )

    total = len(subjects)

    train_count = int(
        total * TRAIN_RATIO
    )

    val_count = int(
        total * VAL_RATIO
    )

    test_count = (
        total
        - train_count
        - val_count
    )

    train_subjects = sorted(
        subjects[
            :train_count
        ]
    )

    val_subjects = sorted(
        subjects[
            train_count:
            train_count + val_count
        ]
    )

    test_subjects = sorted(
        subjects[
            train_count + val_count:
        ]
    )

    return (
        train_subjects,
        val_subjects,
        test_subjects,
    )


def validate_split(
    all_subjects,
    train_subjects,
    val_subjects,
    test_subjects,
):
    """
    Verify:
    - all subjects are included
    - no duplicates exist
    - no subject occurs in more than one split
    """

    all_set = set(
        all_subjects
    )

    train_set = set(
        train_subjects
    )

    val_set = set(
        val_subjects
    )

    test_set = set(
        test_subjects
    )

    # --------------------------------------------------------
    # Check duplicates inside each split
    # --------------------------------------------------------

    if len(train_set) != len(
        train_subjects
    ):
        raise RuntimeError(
            "Duplicate patient found in TRAIN split."
        )

    if len(val_set) != len(
        val_subjects
    ):
        raise RuntimeError(
            "Duplicate patient found in VALIDATION split."
        )

    if len(test_set) != len(
        test_subjects
    ):
        raise RuntimeError(
            "Duplicate patient found in TEST split."
        )

    # --------------------------------------------------------
    # Check overlap
    # --------------------------------------------------------

    train_val_overlap = (
        train_set & val_set
    )

    train_test_overlap = (
        train_set & test_set
    )

    val_test_overlap = (
        val_set & test_set
    )

    if train_val_overlap:
        raise RuntimeError(
            "TRAIN and VALIDATION splits overlap:\n"
            f"{sorted(train_val_overlap)}"
        )

    if train_test_overlap:
        raise RuntimeError(
            "TRAIN and TEST splits overlap:\n"
            f"{sorted(train_test_overlap)}"
        )

    if val_test_overlap:
        raise RuntimeError(
            "VALIDATION and TEST splits overlap:\n"
            f"{sorted(val_test_overlap)}"
        )

    # --------------------------------------------------------
    # Check complete coverage
    # --------------------------------------------------------

    combined_set = (
        train_set
        | val_set
        | test_set
    )

    if combined_set != all_set:

        missing = all_set - combined_set
        extra = combined_set - all_set

        raise RuntimeError(
            "Dataset split does not contain "
            "exactly the original subjects.\n"
            f"Missing: {sorted(missing)}\n"
            f"Extra: {sorted(extra)}"
        )

    print(
        "✓ No patient overlap between splits."
    )

    print(
        "✓ Every patient is included exactly once."
    )


def save_split_file(
    path,
    subjects,
):
    """
    Save a list of patient IDs to a text file.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        for subject in subjects:
            file.write(
                subject + "\n"
            )


def save_json(
    path,
    data,
):
    """
    Save split information as JSON.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print_header(
        "NEURORESTORE AI"
    )

    print(
        "ISLES 2022 PATIENT-LEVEL DATA SPLIT"
    )

    # --------------------------------------------------------
    # Validate configuration
    # --------------------------------------------------------

    print_header(
        "STEP 1 - VALIDATING SPLIT CONFIGURATION"
    )

    validate_ratios()

    print(
        f"Training ratio:   {TRAIN_RATIO:.0%}"
    )

    print(
        f"Validation ratio: {VAL_RATIO:.0%}"
    )

    print(
        f"Test ratio:       {TEST_RATIO:.0%}"
    )

    print(
        f"Random seed:      {RANDOM_SEED}"
    )

    # --------------------------------------------------------
    # Find patients
    # --------------------------------------------------------

    print_header(
        "STEP 2 - FINDING SUBJECTS"
    )

    subjects = find_subjects()

    print(
        f"Total subjects found: "
        f"{len(subjects)}"
    )

    if len(subjects) == 0:
        raise RuntimeError(
            "No ISLES 2022 subject directories found."
        )

    print(
        "\nFirst 5 subjects:"
    )

    for subject in subjects[:5]:
        print(
            f"  {subject}"
        )

    print(
        "\nLast 5 subjects:"
    )

    for subject in subjects[-5:]:
        print(
            f"  {subject}"
        )

    # --------------------------------------------------------
    # Create split
    # --------------------------------------------------------

    print_header(
        "STEP 3 - CREATING PATIENT-LEVEL SPLIT"
    )

    (
        train_subjects,
        val_subjects,
        test_subjects,
    ) = split_subjects(
        subjects
    )

    print(
        f"TRAIN subjects: "
        f"{len(train_subjects)}"
    )

    print(
        f"VALIDATION subjects: "
        f"{len(val_subjects)}"
    )

    print(
        f"TEST subjects: "
        f"{len(test_subjects)}"
    )

    # --------------------------------------------------------
    # Validate split
    # --------------------------------------------------------

    print_header(
        "STEP 4 - VALIDATING SPLIT"
    )

    validate_split(
        subjects,
        train_subjects,
        val_subjects,
        test_subjects,
    )

    # --------------------------------------------------------
    # Print examples
    # --------------------------------------------------------

    print_header(
        "STEP 5 - SPLIT EXAMPLES"
    )

    print(
        "\nTRAIN examples:"
    )

    for subject in train_subjects[:10]:
        print(
            f"  {subject}"
        )

    print(
        "\nVALIDATION examples:"
    )

    for subject in val_subjects[:10]:
        print(
            f"  {subject}"
        )

    print(
        "\nTEST examples:"
    )

    for subject in test_subjects[:10]:
        print(
            f"  {subject}"
        )

    # --------------------------------------------------------
    # Save text files
    # --------------------------------------------------------

    print_header(
        "STEP 6 - SAVING SPLIT FILES"
    )

    train_path = (
        SPLIT_OUTPUT_DIR
        / "train.txt"
    )

    val_path = (
        SPLIT_OUTPUT_DIR
        / "val.txt"
    )

    test_path = (
        SPLIT_OUTPUT_DIR
        / "test.txt"
    )

    save_split_file(
        train_path,
        train_subjects,
    )

    save_split_file(
        val_path,
        val_subjects,
    )

    save_split_file(
        test_path,
        test_subjects,
    )

    print(
        f"Saved TRAIN:\n{train_path}"
    )

    print(
        f"Saved VALIDATION:\n{val_path}"
    )

    print(
        f"Saved TEST:\n{test_path}"
    )

    # --------------------------------------------------------
    # Save JSON metadata
    # --------------------------------------------------------

    print_header(
        "STEP 7 - SAVING SPLIT METADATA"
    )

    split_metadata = {
        "dataset": "ISLES 2022",
        "random_seed": RANDOM_SEED,
        "ratios": {
            "train": TRAIN_RATIO,
            "validation": VAL_RATIO,
            "test": TEST_RATIO,
        },
        "counts": {
            "total": len(subjects),
            "train": len(train_subjects),
            "validation": len(val_subjects),
            "test": len(test_subjects),
        },
        "subjects": {
            "train": train_subjects,
            "validation": val_subjects,
            "test": test_subjects,
        },
    }

    json_path = (
        SPLIT_OUTPUT_DIR
        / "split_metadata.json"
    )

    save_json(
        json_path,
        split_metadata,
    )

    print(
        f"Saved metadata:\n{json_path}"
    )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print_header(
        "DATASET SPLIT SUCCESS"
    )

    print(
        "✓ Patient-level split created."
    )

    print(
        "✓ No patient appears in multiple splits."
    )

    print(
        "✓ All patients are accounted for."
    )

    print()
    print(
        f"TRAIN:      {len(train_subjects)} patients"
    )

    print(
        f"VALIDATION: {len(val_subjects)} patients"
    )

    print(
        f"TEST:       {len(test_subjects)} patients"
    )

    print()
    print(
        "Split files saved to:"
    )

    print(
        SPLIT_OUTPUT_DIR
    )


if __name__ == "__main__":
    main()