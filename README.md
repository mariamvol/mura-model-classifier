# MURA Model Classifier

Automated fracture detection in musculoskeletal X-ray images from the **MURA dataset**
using deep learning and computer vision techniques.

This repository provides:
- a ready-to-use **PyTorch inference package**,
- pretrained **per-anatomical-area models**,
- and a full **analytical report** describing the methodology and experiments.

The project was developed as an academic research work and is intended for
**educational and research purposes only**.

---

## Project Overview

Musculoskeletal Radiographs (MURA) present several challenges for automated analysis:
- high intra-class variability,
- subtle fracture patterns,
- strong anatomical differences between body parts.

To address this, the project follows a **per-area modeling strategy**:
each anatomical region is trained with its own dedicated classifier.

### Supported anatomical areas
- `XR_ELBOW`
- `XR_FINGER`
- `XR_FOREARM`
- `XR_HAND`
- `XR_HUMERUS`
- `XR_SHOULDER`
- `XR_WRIST`

Each area has an independently trained model and checkpoint.

---

## Methodology Summary

The full methodology is described in detail in the analytical note, but the main steps are:

1. **Dataset preparation**
   - MURA dataset with image-level and study-level labels
   - Explicit separation by anatomical area
   - Study-aware aggregation for evaluation

2. **Baseline architecture**
   - DenseNet-121 pretrained on ImageNet
   - Binary classification head (fracture / no fracture)

3. **Training strategy**
   - Area-specific training
   - Class imbalance handling
   - Two-phase fine-tuning strategy
   - Best model selection based on validation metrics

4. **Hyperparameter optimization**
   - Automated tuning using Optuna
   - Separate studies per anatomical area

5. **Evaluation**
   - Image-level metrics
   - Study-level aggregation and AUC evaluation

6. **Interpretability**
   - Grad-CAM based visualization
   - Qualitative localization analysis of fracture regions

7. **Advanced experiments (research stage)**
   - CAM-based regularization
   - Hybrid architectures (DenseNet + Transformer backbones)

---

## Analytical Note

A detailed analytical report describing the dataset analysis, experimental setup,
training strategies, evaluation protocols, and results is available here:

**[Analytical Note (PDF)](docs/Analytical_Note_MURA_Vol_2025.pdf)**

The document includes:
- MURA dataset analysis,
- per-area classification rationale,
- DenseNet baseline and improvements,
- Optuna hyperparameter optimization,
- CAM / Grad-CAM interpretability analysis,
- two-phase training strategy,
- hybrid DenseNet + Transformer experiments,
- quantitative and qualitative results.

---

## Installation

Install directly from GitHub:

```bash
pip install git+https://github.com/mariamvol/mura-model-classifier.git
```

---

## Requirements:

- Python ≥ 3.9
- PyTorch
- torchvision
- NumPy, pandas, scikit-learn

---

## Inference Usage
### Single image inference
```
mura-infer \
  --image path/to/image.png \
  --area XR_WRIST
```
On first run, the corresponding pretrained weights will be automatically downloaded
from the GitHub release and cached locally.

---

## Using custom checkpoint
```
mura-infer \
  --image path/to/image.png \
  --area XR_WRIST \
  --ckpt /path/to/XR_WRIST_FINAL_best.pt
```

---

## Output

The command prints:
- predicted fracture probability (sigmoid output),
- binary prediction using a configurable threshold (default: 0.5).
  
Example output:
```
area=XR_WRIST
prob_fracture=0.8731
pred=1
```

---

## Model Details

- Architecture: DenseNet-121
- Input resolution: 224 × 224
- Normalization: ImageNet statistics
- Output: single logit → sigmoid probability

Each anatomical area uses its own independently trained model.

---

## Disclaimer

This project is provided for research and educational purposes only.

It is **not a medical device** and must not be used for clinical decision-making
or diagnostic purposes.

---

## License

This project is released under the MIT License.
