
# 🧠 NEURORESTORE AI

### AI-Powered Brain MRI Restoration and Ischemic Stroke Lesion Segmentation

> **Research / Educational Prototype — Not a Clinical Diagnostic System**

NEURORESTORE AI is a research-oriented medical imaging project designed to investigate whether **MRI restoration/enhancement using a VAE-GAN can improve downstream ischemic stroke lesion segmentation**.

The project uses multimodal brain MRI data from the **ISLES 2022** dataset and combines MRI preprocessing, generative restoration, automated lesion segmentation, quantitative evaluation, and visualization into a reproducible pipeline.

---

## 📌 Research Question

The central research question of this project is:

> **Does VAE-GAN-based MRI restoration improve downstream ischemic stroke lesion segmentation compared with direct MRI-to-U-Net segmentation?**

Two pipelines are investigated:

### Baseline

```text
DWI + ADC + FLAIR
        ↓
     nnU-Net
        ↓
Stroke Lesion Mask

Proposed

DWI + ADC + FLAIR
        ↓
   VAE-GAN Restoration
        ↓
   Restored MRI
        ↓
     U-Net / nnU-Net
        ↓
Stroke Lesion Mask

The final comparison will determine whether restoration provides measurable benefits for lesion segmentation.


---

🎯 Project Objectives

The main objectives are:

Process multimodal brain MRI data consistently.

Align DWI, ADC and FLAIR modalities.

Normalize MRI intensities.

Create reproducible patient-level train/validation/test splits.

Establish a pretrained segmentation baseline.

Develop a VAE-GAN-based MRI restoration component.

Evaluate segmentation before and after restoration.

Measure restoration quality.

Analyze segmentation failure cases.

Provide an interactive visualization interface.

Maintain reproducible experiments and documentation.



---

🧠 Why This Project?

Ischemic stroke can produce lesions that vary considerably in size, location and appearance across MRI modalities.

MRI modalities such as:

Diffusion-Weighted Imaging (DWI)

Apparent Diffusion Coefficient (ADC)

Fluid-Attenuated Inversion Recovery (FLAIR)


provide complementary information.

However, MRI data can contain variations caused by:

scanner characteristics

acquisition conditions

noise

intensity differences

spatial resolution

modality-specific characteristics


This project investigates whether a generative restoration stage can improve the quality of MRI representations before segmentation.

The goal is not simply to build another segmentation model, but to experimentally evaluate:

> Does improving the MRI representation improve the final segmentation task?




---

🏥 Dataset

The project uses the:

ISLES 2022 Dataset

Ischemic Stroke Lesion Segmentation Challenge 2022

The dataset contains multimodal MRI scans and corresponding ischemic stroke lesion masks.

Modalities Used

Channel	Modality

Channel 0	DWI
Channel 1	ADC
Channel 2	FLAIR


The lesion mask contains:

0 → Background
1 → Infarct / Stroke lesion


---

📊 Dataset Statistics

The downloaded ISLES 2022 dataset contains:

250 patient cases

1000 NIfTI files

4 NIfTI files per case:

DWI

ADC

FLAIR

lesion mask



After preprocessing:

Total patients          : 250
Patients with lesion    : 247
Patients without lesion : 3

Total axial slices      : 15,684
Brain/foreground slices : 14,577
Excluded non-brain      : 1,107

The dataset is split at the patient level to prevent patient leakage.


---

📂 Dataset Format

The MRI data is stored in NIfTI format using compressed .nii.gz files.

These are 3D medical imaging volumes, not ordinary PNG/JPG images.

Example:

data/
└── processed/
    └── ISLES-2022/
        └── sub-strokecase0001/
            └── ses-0001/
                ├── *_dwi_aligned.nii.gz
                ├── *_adc_aligned.nii.gz
                ├── *_FLAIR_aligned.nii.gz
                ├── *_mask_aligned.nii.gz
                └── normalized/
                    ├── *_dwi_normalized.nii.gz
                    ├── *_adc_normalized.nii.gz
                    ├── *_FLAIR_normalized.nii.gz
                    └── *_mask.nii.gz

Why .nii.gz?

NIfTI is commonly used for volumetric medical imaging.

A single file represents a 3D MRI volume, containing many axial slices.

PNG files are generated separately only for visualization and reporting.


---

🔄 Project Pipeline

The overall pipeline is:

ISLES 2022
                  │
                  ▼
        Multimodal MRI Data
        DWI + ADC + FLAIR
                  │
                  ▼
          Data Inspection
                  │
                  ▼
        Spatial Alignment
                  │
                  ▼
        Intensity Normalization
                  │
                  ▼
        Patient-Level Splitting
                  │
          ┌───────┴────────┐
          │                │
          ▼                ▼
      Baseline          Proposed
          │                │
          ▼                ▼
     Segmentation      VAE-GAN
      Model              │
          │              ▼
          │        Restored MRI
          │              │
          │              ▼
          │         Segmentation
          │              │
          └───────┬──────┘
                  ▼
          Quantitative Evaluation
                  │
                  ▼
       Visualization + Analysis
                  │
                  ▼
            Streamlit App


---

🧹 Data Preprocessing

The preprocessing pipeline includes:

1. Dataset Inspection

The raw ISLES 2022 dataset is inspected for:

patient folders

MRI modalities

lesion masks

dimensions

spacing

orientation

missing files



---

2. Spatial Alignment

DWI is used as the common reference grid.

ADC and FLAIR are spatially aligned to the DWI reference.

The lesion mask uses nearest-neighbor interpolation to preserve its binary nature.

Example processed geometry:

DWI     → reference grid
ADC     → aligned to DWI
FLAIR   → aligned to DWI
Mask    → aligned to DWI


---

3. Intensity Normalization

MRI modalities are normalized independently.

The preprocessing performs:

1. foreground identification


2. percentile clipping


3. z-score normalization



The lesion mask remains binary.


---

4. Patient-Level Dataset Splitting

The dataset is split into:

Split	Patients

Train	175
Validation	37
Test	38
Total	250


A fixed random seed of 42 is used.

No patient appears in multiple splits.


---

📐 Important Data Property

The number of slices along the Z-axis is not identical for every patient.

Observed Z dimensions range from:

25 → 76 slices

The in-plane dimensions are standardized for the 2D dataset where required.

The project therefore does not assume that every 3D patient volume has exactly the same number of slices.


---

🧪 Baseline Segmentation

A pretrained ISLES 2022 nnU-Net v2 model is used as the segmentation baseline.

The model is a:

3D Full-Resolution nnU-Net

with:

Input channels:
DWI
ADC
FLAIR

The checkpoint used is:

checkpoint_best.pth

The model is used for research evaluation and is not a clinical diagnostic system.


---

⚙️ Baseline Model Configuration

The pretrained checkpoint uses:

Configuration : 3d_fullres
Architecture  : PlainConvUNet
Input channels: 3
Output labels : 2
Classes       : Background + Infarct
Spacing       : 2 × 2 × 2 mm

The model was successfully loaded and executed using CPU inference.


---

🖥️ Hardware Considerations

The development environment is designed around limited hardware.

Current development machine:

RAM       : 8 GB
GPU       : None
CUDA      : False
OS        : Windows
Python    : 3.11.9

The pretrained nnU-Net model is used for inference instead of training a large 3D segmentation model locally.

Heavy training experiments can be performed on:

Google Colab

Kaggle

other GPU environments


while final inference and project integration remain CPU-compatible where possible.


---

📊 Baseline Evaluation

The pretrained nnU-Net model was evaluated on the project's 38-case test split.

Evaluation setup

Cases evaluated : 38
Model           : nnU-Net v2 3D full-resolution
Checkpoint      : checkpoint_best.pth
Fold            : 0
Device          : CPU
TTA             : Disabled

Observed Results

Metric	Mean	Median

Dice	0.7058	0.8241
IoU	0.5999	0.7009
Precision	0.8195	0.9245
Recall	0.6915	0.8278


These values are results from this project's evaluation run and should not be interpreted as official ISLES 2022 benchmark results.


---

⚠️ Important Evaluation Limitation

The pretrained checkpoint was originally trained using a dataset split containing 220 cases.

The overlap between that training set and this project's 38-case test split has not been established.

Therefore:

> The 38-case evaluation should be treated as an external evaluation of the downloaded pretrained checkpoint, not necessarily as a completely independent benchmark.



This limitation will be explicitly considered when reporting final experimental results.


---

🔍 Failure Case Analysis

The project also analyzes cases where segmentation performance is lower.

Examples of observed failure patterns include:

Under-segmentation

The predicted lesion is substantially smaller than the ground-truth lesion.

Typical pattern:

High precision
Low recall


---

Over-segmentation

The predicted lesion covers a larger region than the ground truth.

Typical pattern:

Lower precision
Higher recall


---

Small-lesion miss

Very small lesions may be completely missed.


---

False positives

The model may predict lesion regions in cases where the ground-truth mask contains no lesion.

Failure cases are retained for qualitative analysis rather than being removed from the evaluation.


---

🧬 VAE-GAN Restoration

The next major component of NEURORESTORE AI is the VAE-GAN restoration stage.

The intended architecture is:

DWI + ADC + FLAIR
        ↓
   VAE Encoder
        ↓
   Latent Space
        ↓
   VAE Decoder
        ↓
 Generative Refinement
        ↓
 Restored MRI

The restoration model is intended to improve the representation of the MRI before segmentation.

The research experiment compares:

Baseline:
MRI → Segmentation

Proposed:
MRI → VAE-GAN → Segmentation


---

❗ VAE-GAN Model Status

The VAE-GAN component is currently under development.

A compatible pretrained VAE-GAN specifically designed for the ISLES 2022 DWI + ADC + FLAIR restoration task has not yet been established.

Therefore, the project does not claim that the VAE-GAN stage is already complete.

The final implementation will only use a pretrained model if its architecture, input modalities and preprocessing requirements are compatible with this project.

If a suitable pretrained model cannot be established, a lightweight VAE-GAN may be trained using an appropriate GPU environment.


---

📏 Evaluation Metrics

The project evaluates both restoration and segmentation.

Restoration Metrics

Potential metrics include:

MSE

PSNR

SSIM


These measure how closely the restored image represents the reference image.


---

Segmentation Metrics

The following metrics are used:

Dice Similarity Coefficient

Measures overlap between prediction and ground truth.

Dice = 2TP / (2TP + FP + FN)

Intersection over Union

IoU = TP / (TP + FP + FN)

Precision

Measures the proportion of predicted lesion pixels that are correct.

Precision = TP / (TP + FP)

Recall

Measures the proportion of ground-truth lesion pixels that are detected.

Recall = TP / (TP + FN)


---

🧮 Lesion Measurements

The project also calculates:

lesion voxel count

lesion volume

predicted lesion volume

absolute volume error


This allows comparison between:

Ground Truth Lesion
        vs
Predicted Lesion


---

🖼️ Visualization

The project generates PNG visualizations for human inspection.

For example:

DWI | ADC | FLAIR | Ground Truth Mask

These PNG files are not the original dataset.

They are generated from the .nii.gz volumes for:

visual inspection

debugging

failure analysis

reports

presentations

research documentation


The actual model input remains the NIfTI .nii.gz data.


---

🌐 Streamlit Dashboard

A Streamlit-based interface is planned for the final system.

The dashboard will provide:

MRI visualization

modality selection

original/restored comparison

segmentation visualization

lesion measurements

quantitative metrics

failure-case inspection

experiment comparison


Planned workflow:

Upload / Select MRI
        ↓
Preprocessing
        ↓
Restoration
        ↓
Segmentation
        ↓
Visualization
        ↓
Metrics


---

📁 Project Structure

NeuroRestore-AI/
│
├── app/
│   ├── pages/
│   └── components/
│
├── src/
│   ├── data/
│   │   ├── inspect_dataset.py
│   │   ├── align_case.py
│   │   ├── normalize_case.py
│   │   ├── create_splits.py
│   │   ├── prepare_nnunet_case.py
│   │   └── prepare_nnunet_test.py
│   │
│   ├── vaegan/
│   │
│   ├── segmentation/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── losses.py
│   │   ├── prepare_nnunet_model.py
│   │   └── inspect_nnunet_prediction.py
│   │
│   ├── evaluation/
│   │   ├── evaluate_nnunet_case.py
│   │   └── evaluate_nnunet_test38.py
│   │
│   └── utils/
│
├── configs/
│
├── notebooks/
│
├── models/
│   └── isles22_nnunet/
│       ├── dataset.json
│       ├── plans.json
│       └── fold_0/
│           └── checkpoint_best.pth
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── nnunet_inference/
│   └── splits/
│       ├── train.txt
│       ├── val.txt
│       ├── test.txt
│       └── split_metadata.json
│
├── outputs/
│   ├── figures/
│   ├── predictions/
│   ├── metrics/
│   ├── reports/
│   └── failure_cases/
│
├── tests/
│
├── docs/
│   ├── architecture/
│   ├── experiments/
│   └── screenshots/
│
├── scripts/
│   └── download_assets.ps1
│
├── PROJECT_CONTRACT.md
├── requirements.txt
├── .gitignore
└── README.md


---

🛠️ Technology Stack

Component	Technology

Programming Language	Python 3.11
Deep Learning	PyTorch
Medical Imaging	NiBabel
Medical AI	MONAI
Segmentation	nnU-Net v2
Generative Model	VAE-GAN
Numerical Computing	NumPy
Scientific Computing	SciPy
Data Processing	Pandas
Visualization	Matplotlib / Plotly
Dashboard	Streamlit
Version Control	Git / GitHub



---

📦 Installation

Clone the repository:

git clone -b develop https://github.com/sahana913/Neuro-Restore-AI.git

Enter the project directory:

cd Neuro-Restore-AI

Create a virtual environment:

python -m venv .venv

Activate it on Windows PowerShell:

.\.venv\Scripts\Activate.ps1

Upgrade pip:

python -m pip install --upgrade pip

Install dependencies:

pip install -r requirements.txt


---

📥 Large Assets

Large medical datasets and model checkpoints are intentionally not stored directly in GitHub.

The repository keeps:

GitHub
   ↓
Source code
Configuration
Documentation
Metadata
Evaluation scripts

Large assets are stored separately:

Google Drive
   ↓
Pretrained checkpoint
Test inference data
Processed dataset

This keeps the GitHub repository manageable.


---

📂 Required External Assets

For reproducing the pretrained nnU-Net inference, the following assets are required.

Model checkpoint

models/
└── isles22_nnunet/
    └── fold_0/
        └── checkpoint_best.pth

The checkpoint is intentionally excluded from Git using .gitignore.

Test inference data

data/
└── nnunet_inference/
    └── test_flat/

The prepared test data is also excluded from Git.

Download links for these assets should be provided separately to project collaborators.


---

▶️ Running nnU-Net Inference

After placing the pretrained checkpoint and test data in the required directories:

nnUNetv2_predict_from_modelfolder `
    -i ".\data\nnunet_inference\test_flat" `
    -o ".\outputs\predictions\nnunet_test38" `
    -m ".\models\isles22_nnunet" `
    -f 0 `
    -chk checkpoint_best.pth `
    -device cpu `
    -npp 1 `
    -nps 1 `
    --disable_tta

The predictions will be generated in:

outputs/
└── predictions/
    └── nnunet_test38/


---

📊 Running Evaluation

After inference:

python .\src\evaluation\evaluate_nnunet_test38.py

Evaluation outputs:

outputs/
└── metrics/
    ├── nnunet_test38_per_case.csv
    └── nnunet_test38_summary.json


---

🔬 Reproducibility

The project follows reproducible experiment practices.

Important controls include:

fixed random seed

patient-level splitting

explicit preprocessing

fixed modality order

saved model configuration

saved evaluation metrics

per-case predictions

failure-case analysis

documented hardware

documented software versions



---

🔐 Data and Privacy

The project uses the public ISLES 2022 research dataset.

No private patient dataset is intentionally included in this repository.

Raw and processed medical imaging data should not be committed to GitHub.

The .gitignore configuration excludes large data and model files.


---

⚠️ Medical Safety Disclaimer

NEURORESTORE AI is a:

> Research and educational prototype.



It is not a medical device and must not be used for:

clinical diagnosis

treatment decisions

emergency medical decisions

patient management

replacing a radiologist or physician


Model predictions may contain errors, including false positives and false negatives.

All results should be interpreted in a research context.


---

📌 Current Project Status

Completed

[x] Project architecture

[x] GitHub repository

[x] Project contract

[x] ISLES 2022 dataset acquisition

[x] Dataset inspection

[x] MRI modality identification

[x] Spatial preprocessing

[x] DWI/ADC/FLAIR alignment

[x] Mask alignment

[x] Intensity normalization

[x] Patient-level train/validation/test split

[x] Full 250-case preprocessing

[x] Preprocessing validation

[x] 2D dataset preparation and validation

[x] 2D U-Net architecture

[x] Segmentation loss functions

[x] Training-step validation

[x] Pretrained ISLES 2022 nnU-Net acquisition

[x] nnU-Net model loading

[x] Test-set preparation

[x] 38-case CPU inference

[x] Quantitative baseline evaluation

[x] Failure-case analysis


In Progress

[ ] Compatible pretrained VAE-GAN identification

[ ] VAE-GAN restoration implementation

[ ] Restoration evaluation

[ ] Restored MRI → segmentation experiment

[ ] Baseline vs proposed comparison

[ ] Final Streamlit dashboard

[ ] Final experiment report



---

🗺️ Future Work

The planned development stages are:

Phase 1
Project Setup
        ↓
Phase 2
Dataset Acquisition
        ↓
Phase 3
Preprocessing
        ↓
Phase 4
Baseline Segmentation
        ↓
Phase 5
VAE-GAN Restoration
        ↓
Phase 6
Restoration + Segmentation
        ↓
Phase 7
Quantitative Evaluation
        ↓
Phase 8
Failure Analysis
        ↓
Phase 9
Streamlit Dashboard
        ↓
Phase 10
Final Documentation


---

👥 Team Structure

The project is designed for parallel team development.

Member 1 — Dataset & Preprocessing

Responsible for:

ISLES dataset

preprocessing

alignment

normalization

data splits

dataset validation


Main directory:

src/data/


---

Member 2 — VAE-GAN

Responsible for:

VAE-GAN architecture

pretrained model investigation

restoration pipeline

restoration evaluation


Main directory:

src/vaegan/


---

Member 3 — Segmentation

Responsible for:

U-Net / nnU-Net

segmentation pipeline

model inference

segmentation metrics


Main directory:

src/segmentation/


---

Member 4 — Evaluation & Application

Responsible for:

evaluation

visualizations

failure-case analysis

Streamlit dashboard


Main directories:

src/evaluation/
app/


---

🌿 Git Branching Strategy

The project uses:

main
  ↓
Stable version

develop
  ↓
Team development

feature/data-preprocessing
feature/vaegan
feature/unet
feature/streamlit-evaluation



