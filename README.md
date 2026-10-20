# Predicting Alzheimer's Disease Progression from Brain MRI

Bachelor's thesis project (FESB, University of Split, July 2026). Transfer learning with a pretrained 3D ResNet-10 (MedicalNet) on preprocessed structural MRI scans from the longitudinal **OASIS-2** dataset, predicting whether a patient's dementia rating will worsen at their next visit.

> **Research code, not a clinical tool.** The model is not intended for diagnosis or use on real patients (see [Limitations](#limitations)).

## Key findings

| Model (target: `worsen`) | Balanced accuracy | AUC | PR-AUC |
| --- | --- | --- | --- |
| Metadata only | 0.549 ± 0.086 | 0.466 | 0.219 |
| **MRI only** | **0.667 ± 0.078** | **0.632** | **0.307** |
| MRI + metadata (fusion) | 0.659 ± 0.069 | 0.624 | 0.284 |

- The MRI-only model beats the metadata-only baseline by +0.118 balanced accuracy (p = 0.0001) and +0.166 AUC (p = 0.0010), using paired tests over the same 25 folds.
- Metadata alone performs at roughly chance level (AUC 0.466), while the preprocessed scan carries real predictive signal.
- Fusing in metadata did not help (p = 0.07 and p = 0.30 vs. MRI-only).
- On the harder `convert` task (healthy → impaired, 15 positive subjects), the MRI model reaches 0.739 balanced accuracy vs. 0.599 for metadata, with a very wide confidence interval (± 0.164).

## Dataset

[OASIS-2](https://sites.wustl.edu/oasisbrains/) (Open Access Series of Imaging Studies): 150 older adults (60–96 years), 373 imaging sessions, all scanned on one 1.5 T scanner with an identical MP-RAGE sequence. Participants fall into three groups: 72 stable non-demented, 64 demented, and 14 who converted from non-demented to demented.

The data is **not included** in this repository. Download it from the OASIS website and follow its data use terms.

### Task definition

Consecutive visits per subject are paired as *current → next*. The model only sees the current visit's scan and metadata; the label comes from the next visit.

| Target | Classes | Description |
| --- | --- | --- |
| `cdr3` | 3 | CDR at next visit (0 / 0.5 / ≥ 1). Dropped as the main target: copying the current CDR is already a strong baseline |
| **`worsen`** | 2 | Next-visit CDR higher than current (main target, 31 positives out of 223 pairs) |
| `convert` | 2 | Healthy at current visit and impaired at next (15 positives out of 133 pairs) |

The clinical group label is excluded from the inputs to prevent data leakage.

## Preprocessing pipeline

Seven steps, each reading from one directory tree and writing to a mirrored one, so any step can be resumed, inspected visually, and quality-checked.

1. **Extraction and organization**: merge downloaded archives into one folder per session
2. **NIfTI conversion and reorientation**: Analyze 7.5 → NIfTI, RAS+ orientation via `nibabel`
3. **Bias field correction**: N4 (SimpleITK) with anisotropic diffusion denoising and Li-threshold mask
4. **Skull stripping**: SynthStrip (official Docker container)
5. **Spatial registration**: ANTs SyN to the MNI152NLin2009cAsym template (via TemplateFlow)
6. **Intensity normalization and cropping**: percentile clipping (1st–99th), z-score inside the brain mask, crop to 154 × 181 × 144
7. **Averaging**: average repeated acquisitions per session to improve SNR (373 final volumes)

Automated QC runs after the steps (file counts and shapes, CV before/after N4, registration MI/NCC/Jacobian, NaN checks, intensity statistics), plus visual overlays for skull stripping.

## Model

- **Image branch**: MedicalNet ResNet-10 encoder (72 pretrained weight tensors loaded, segmentation head removed), global average pooling → 512-d → 128-d. Input is an axial slab of 96 × 96 × 32 voxels.
- **Tabular branch** (deliberately weak, 8-d with high dropout): `cur_CDR`, `MMSE`, `nWBV`, `gap_days`, `EDUC`, `is_female`.
- **Fusion**: concatenated 128 + 8 features → classification head. Branches can run separately or together to measure each source's contribution.
- **Fine-tuning**: last residual block (`layer4`) plus heads, or fully frozen encoder.

## Training and evaluation

- Repeated stratified group k-fold cross-validation (`StratifiedGroupKFold`, 5 folds × 5 repeats = 25 evaluations), grouped by subject so no patient appears in both train and validation.
- Cross-entropy loss, AdamW, cosine learning-rate decay, mixed precision, trained on Kaggle GPUs.
- Light augmentation: flipping and small noise.
- Metrics suited to class imbalance: balanced accuracy, macro-F1, ROC-AUC, PR-AUC, with bootstrap 95% CIs.
- Model variants compared with paired Wilcoxon and t-tests over identical folds.

## Ablations

| Configuration | Balanced accuracy |
| --- | --- |
| 96×96×32, fine-tune `layer4` (baseline) | **0.667** |
| 128×128×64, fine-tune `layer4` | 0.645 |
| 96×96×32, frozen encoder | 0.618 |
| Medial temporal lobe crop, fine-tune `layer4` | 0.591 |

Higher resolution, full freezing, and hippocampal cropping all failed to improve results. Cropping significantly hurt (−0.076, p = 0.0007), suggesting the model uses global cues such as ventricle enlargement and overall atrophy. Overall, performance appears limited by the number of positive examples rather than model capacity.

## Limitations

- Small, imbalanced dataset: 31 worsening cases among 223 pairs.
- Single source dataset, no external validation.
- At the operating point used, recall for `worsen` is about 48% and precision about 35% (74 of 155 true worsenings caught, 135 false alarms among 960 stable cases). This is at best a rough screening signal.
- Wide confidence intervals, especially on `convert`, so numbers should not be treated as definitive.

## Repository structure

<!-- TODO: replace with your actual layout -->
```
.
├── preprocessing/     # 7-step pipeline + QC scripts
├── model/             # dataset, ResNet-10 encoder, fusion model
├── experiments/       # training / cross-validation / ablations
├── notebooks/         # analysis and figures
└── README.md
```

## Getting started

<!-- TODO: fill in with your real entry points and versions -->
```bash
git clone <repo-url>
cd <repo>
pip install -r requirements.txt

# 1. Download OASIS-2 and place it in data/raw/
# 2. Run the preprocessing pipeline
# 3. Download MedicalNet resnet_10 weights
# 4. Train and evaluate
```

SynthStrip is run through Docker, so Docker needs to be installed for the skull-stripping step.

## Tech stack

Python, PyTorch, nibabel, SimpleITK, ANTs, SynthStrip (Docker), TemplateFlow, scikit-learn, NumPy, Pandas.

## References

- Marcus et al. *Open Access Series of Imaging Studies: Longitudinal MRI Data in Nondemented and Demented Older Adults.* J Cogn Neurosci, 2010 (OASIS-2)
- Chen, Ma, Zheng. *Med3D: Transfer Learning for 3D Medical Image Analysis.* 2019 (MedicalNet)
- Hoopes et al. *SynthStrip: Skull-Stripping for Any Brain Image.* NeuroImage, 2022
- Tustison et al. *N4ITK: Improved N3 Bias Correction.* IEEE TMI, 2010
- Avants et al. *Symmetric Diffeomorphic Image Registration with Cross-Correlation.* Med Image Anal, 2008

## Author

**Josip Bilić**: B.Sc. thesis, Faculty of Electrical Engineering, Mechanical Engineering and Naval Architecture (FESB), University of Split. Mentor: prof. dr. sc. Tamara Grujić.
