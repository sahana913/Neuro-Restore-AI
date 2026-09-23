from pathlib import Path
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CASE_ID = "sub-strokecase0001"
SESSION = "ses-0001"

SOURCE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ISLES-2022"
    / CASE_ID
    / SESSION
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "nnunet_inference"
    / "case0001"
)

MODALITIES = {
    "0000": SOURCE / f"{CASE_ID}_dwi_aligned.nii.gz",
    "0001": SOURCE / f"{CASE_ID}_adc_aligned.nii.gz",
    "0002": SOURCE / f"{CASE_ID}_FLAIR_aligned.nii.gz",
}


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    print(f"Preparing {CASE_ID} for nnU-Net...")
    print()

    for channel, source in MODALITIES.items():

        if not source.exists():
            raise FileNotFoundError(
                f"Missing file: {source}"
            )

        destination = OUTPUT / f"{CASE_ID}_{channel}.nii.gz"

        shutil.copy2(source, destination)

        print(f"Channel {channel}")
        print(f"  Meaning : {['DWI', 'ADC', 'FLAIR'][int(channel)]}")
        print(f"  Source  : {source.name}")
        print(f"  Output  : {destination.name}")
        print()

    print("NNUNET CASE PREPARATION: PASS")
    print()
    print("Channel mapping:")
    print("  0000 = DWI")
    print("  0001 = ADC")
    print("  0002 = FLAIR")


if __name__ == "__main__":
    main()