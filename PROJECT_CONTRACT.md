# NEURORESTORE AI — PROJECT CONTRACT

## AI-Powered Brain MRI Restoration and Ischemic Stroke Lesion Segmentation

**Project Type:** Research / Educational Decision-Support Prototype  
**Dataset:** ISLES 2022  
**Primary Framework:** PyTorch + MONAI  
**Frontend:** Streamlit  
**Development Platform:** Windows + CPU  
**Repository:** NeuroRestore-AI

---

# 1. PROJECT PURPOSE

NeuroRestore AI is an AI-assisted research and educational decision-support system for brain MRI.

The system will:

1. Accept multimodal brain MRI scans.
2. Load DWI, ADC and FLAIR modalities.
3. Preprocess the MRI data.
4. Restore/enhance MRI using a VAE-GAN-based model.
5. Segment ischemic stroke lesions using a 2D U-Net.
6. Calculate lesion measurements.
7. Compare baseline U-Net against VAE-GAN + U-Net.
8. Present results through a Streamlit dashboard.

The primary research question is:

> Does VAE-GAN-based MRI restoration improve downstream ischemic stroke lesion segmentation compared with direct MRI-to-U-Net segmentation?

---

# 2. MEDICAL SAFETY

NeuroRestore AI is strictly a:

> AI-assisted research and educational decision-support prototype.

It is NOT:

- A clinical diagnostic device.
- A replacement for a radiologist or physician.
- A system for making clinical treatment decisions.
- A clinically validated medical device.

The project documentation, GitHub README, Streamlit application and presentation must not claim:

> "AI diagnoses stroke."

Preferred wording:

> "AI-assisted research and decision-support system for brain MRI restoration and ischemic stroke lesion segmentation."

---

# 3. HARDWARE CONSTRAINTS

Primary development hardware:

- Operating System: Windows
- RAM: 8 GB
- GPU: None
- CUDA: Not available
- Primary device: CPU

Therefore:

- Models must initially be CPU-compatible.
- Prefer 2D MRI processing.
- Avoid large 3D architectures during initial development.
- Avoid unnecessarily large batch sizes.
- Avoid loading the entire dataset into RAM.
- Use memory-efficient data loading.
- Training may be performed on Google Colab, Kaggle or another GPU environment if required.
- Final inference must remain CPU-compatible where practical.

---

# 4. PYTHON ENVIRONMENT

Required Python version:

```text
Python 3.11.9