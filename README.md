# Redundancy-Aware Radiomic Feature Selection for IDH and 1p/19q Prediction in Glioma

Code for the paper:

> **Redundancy-Aware Radiomic Feature Selection Enables Compact, Interpretable Prediction of IDH Mutation and 1p/19q Codeletion Status in Glioma from MRI**
> Claudio Espinoza, Roberto León, Steren Chabert, Rodrigo Salas.
> Universidad Técnica Federico Santa María · Universidad de Valparaíso · Millennium Science Initiative iHealth · MEDING.

The repository contains only the procedure: how the radiomic feature vectors are generated from the MRI, and how the models are trained and evaluated on them. No data and no results are stored. Every table and figure in the paper is reproduced by running the steps below.

## Method in brief

1. **ROIs.** Three regions per patient, all derived from the whole-tumour mask supplied with EGD:
   - **tumour**: the supplied mask;
   - **mirrored**: the mask reflected about the mid-sagittal plane (contralateral reference);
   - **boundary shell**: dilation minus erosion of the mask with a spherical element of radius 3 voxels.
2. **Radiomic vectors.** PyRadiomics on T1, T2 and FLAIR for the original image plus 17 filtered images (8 wavelet, exponential, logarithm, square, square root, gradient, 2D LBP and three 3D LBP components). That gives 1,688 features per ROI and contrast. Each contrast contributes tumour (1,688), mirrored (1,688), tumour-minus-mirrored for non-shape features (1,674) and border (1,688) features. Across T1, T2 and FLAIR that makes 20,214 variables, and age and sex bring the total to 20,216.
3. **Primary selection**, fitted inside each training fold:
   - ANOVA baseline;
   - **GCTL**: GroupedCoeThrLasso;
   - **CCTK**: ContrastCollapseTopK;
   - **CCGL**: ContrastCollapseGlobalLasso.
4. **SHAP reduction.** TreeSHAP from an XGBoost reference model keeps the top 10 variables.
5. **Interpretable models.** Bayesian networks (HC+BIC, TAN) and a zero-order Takagi–Sugeno ANFIS with 10 rules.
6. **Evaluation.** A stratified 85/15 development/held-out split with a fixed seed, then 10-fold stratified cross-validation on the development set.

## Requirements

### Data (not included)

The **Erasmus Glioma Database (EGD)** is distributed by its curators under a data use agreement and cannot be redistributed:

> van der Voort, S. R. et al. *The Erasmus Glioma Database (EGD): Structural MRI scans, WHO 2016 subtypes, and segmentations of 774 patients with glioma.* Data in Brief 37, 107191 (2021).

After obtaining access, place the files so that the tree looks like this:

```
<repo>/
├── original_datasets/
│   └── dataset_EGD/
│       └── MR_EGD-XXXX/
│           ├── 1_T1/NIFTI/T1.nii.gz
│           ├── 3_T2/NIFTI/T2.nii.gz
│           ├── 4_FLAIR/NIFTI/FLAIR.nii.gz
│           └── 5_MASK/NIFTI/MASK.nii.gz
├── tabular_data/
│   ├── Genetic_and_Histological_labels.xlsx
│   └── Clinical_data.xlsx
└── radiomics/
    └── radiomics_embeddings/        # created by step 3; holds the per-patient vectors
```

This is the layout of the EGD download (curator-registered to MNI152, 1 mm isotropic). No further image preprocessing is applied.

### Software: two separate environments

Feature extraction and modelling use **different Python environments** because the libraries are incompatible:

| Environment | File | Python | Used for |
|---|---|---|---|
| `radiomics_env` | `environment-radiomics.yml` | 3.7.1 | PyRadiomics 3.1.0a2 + SimpleITK 2.1. PyRadiomics has no builds for recent Python versions. |
| `glioma_env` | `environment-modeling.yml` | 3.12.7 | XGBoost 2.1, SHAP 0.46, pgmpy 0.1.26, PyTorch 2.5, scikit-learn 1.5. These require modern Python and NumPy. |

```bash
conda env create -f environment-radiomics.yml
conda env create -f environment-modeling.yml
```

Each environment needs its own Jupyter kernel. Select the matching kernel when you open each notebook.

**MATLAB** (R2024b was used, with the Image Processing Toolbox) is also needed for one step: building the boundary-shell masks.

## Procedure

### Step 1: boundary-shell masks (MATLAB)

```matlab
cd matlab
periphery_masks_EGD
```

This writes `PERIPHERY_MASK.nii.gz` next to each `MASK.nii.gz`.

### Step 2: radiomic extraction (`radiomics_env`)

Notebook: `radiomics/get_radiomics/pyradiomics_features_EGD_normal_tissue_contrasting.ipynb`, run with `radiomics/get_radiomics/` as the working directory.

The batch-extraction cell is run **twice** (instructions are in the notebook):

| Run | Mask | `mirror` | Output |
|---|---|---|---|
| a | `MASK.nii.gz` | `True` | `radiomics_features.csv` (tumour + mirrored ROI) |
| b | `PERIPHERY_MASK.nii.gz` | `False` | `border_radiomics_features_EGD.csv` (boundary shell) |

Both files are written to the repository root, with one row per patient and contrast. Extraction uses PyRadiomics defaults with `enableAllFeatures()` and `enableAllImageTypes()`. It covers patients with a known IDH or 1p/19q label (483 patients).

### Step 3: per-patient vectors (`radiomics_env` or `glioma_env`)

Notebook: `radiomics/get_radiomics/processEGDcsv.ipynb`.

It collapses the three rows per patient into one and writes:

- `radiomics/radiomics_embeddings/t1t2flair_mirror_radiomics_features_EGD.csv` (483 × 10,129)
- `radiomics/radiomics_embeddings/border_features_EGD.csv` (483 × 5,064)

### Step 4: final embedding (`glioma_env`)

`radiomics/get_data.py` (imported by the notebooks, with `radiomics/` as the working directory):

- reads both vectors, the clinical data and the labels;
- computes tumour-minus-mirrored differences for the non-shape features;
- concatenates tumour, mirrored, difference and border features, plus age and sex;
- imputes missing sex (mode) and age (mean);
- returns the stratified development/held-out split.

| Experiment | Task | Cohort |
|---|---|---|
| `get_data('EGD_IDH', test_size, random_state)` | IDH mutation | 467 patients |
| `get_data('EGD_COD', test_size, random_state)` | 1p/19q codeletion | 149 IDH-mutant grade II–III patients |

`test_size` is 0.15. The seed is set in each notebook: `PIPELINE_FINAL.ipynb` uses 42 and `radiomics_clean.ipynb` uses 24.

### Step 5: models and evaluation (`glioma_env`)

Run with `radiomics/` as the working directory.

| Notebook | Paper section | Content |
|---|---|---|
| `PIPELINE_FINAL.ipynb` | 2.4, 2.7, 3.2–3.4 | Modular pipeline: GCTL / CCTK / CCGL selectors, XGBoost, SHAP top-10, Bayesian networks (HC+BIC, TAN) and ANFIS, with 10-fold CV and the held-out test. |
| `radiomics_clean.ipynb` | 2.7, 3.2–3.3 | Compact end-to-end version: selection, XGBoost, ANFIS, BN (HC+BIC, TAN), result tables and network plots. |
| `XGBoost_py_radiomics_clean.ipynb` | 2.5, 3.1 | t-SNE projections of ANOVA-ranked variables; SHAP-to-ANFIS cross-validation with per-fold reports and rule books. |
| `ML_techniques_testing.ipynb` | 2.6, Table 3 | Preliminary classifier benchmark (XGBoost, SVM, RF, kNN, NB, Elastic Net, LR, two MLPs, BN) on ANOVA-reduced features. |
| `clustering.ipynb` | 2.5, Figs. 5–6 | Exploratory K-means, Gaussian mixture and DBSCAN on ANOVA- and SHAP-ranked subsets. |

The notebooks write their reports (fold CSVs, ANFIS rule books, BN graphs, SHAP rankings) to sub-folders of `radiomics/`. Those folders are git-ignored.

## Notes

- As stated in the paper (Section 2.8), the preliminary benchmark (`ML_techniques_testing.ipynb`) and the exploratory clustering (`clustering.ipynb`) were produced with earlier scripts. They call an older `get_data(experiment, contrast, test_size, random_state, wavelet=...)` signature, so before running them, change the call to the current `get_data(experiment, test_size, random_state)`.
- Standardisation is fitted once on the complete development set before the folds are generated (see the limitations in the paper).

## Citation

If you use this code, please cite the paper above.
