import sys
import platform

import torch
import numpy as np
import nibabel as nib
import monai
import streamlit
import plotly


def main():
    print("=" * 60)
    print("NEURORESTORE AI - ENVIRONMENT CHECK")
    print("=" * 60)

    # Python information
    print(f"Python        : {sys.version}")
    print(f"Platform      : {platform.platform()}")

    # PyTorch information
    print()
    print(f"PyTorch       : {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Package versions
    print()
    print(f"NumPy         : {np.__version__}")
    print(f"NiBabel       : {nib.__version__}")
    print(f"MONAI         : {monai.__version__}")
    print(f"Streamlit     : {streamlit.__version__}")
    print(f"Plotly        : {plotly.__version__}")

    # CPU test
    print()
    print("CPU TEST:")

    x = torch.randn(2, 3)
    y = torch.randn(2, 3)
    z = x + y

    print(f"Tensor shape  : {z.shape}")
    print(f"Tensor device : {z.device}")

    print()
    print("Environment verification completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()