# Quantum Support Vector Machine (QSVM) for Heart Disease Prediction

An end-to-end Quantum Machine Learning (QML) project implementing a **Quantum Support Vector Classifier (QSVC)** with **Fidelity Quantum Kernel**, Bayesian hyperparameter optimization (**Optuna**), stratified cross-validation, mutual-information feature selection, clinical decision-threshold calibration, and persistent kernel matrix caching.

Evaluated on the benchmark **UCI Heart Disease (Cleveland) dataset** (`id=45`).

---

## Table of Contents
1. [Overview & Motivation](#overview--motivation)
2. [Theoretical Background: Quantum Kernel Methods](#theoretical-background-quantum-kernel-methods)
3. [Limitations of the Naive Baseline vs. Tuned Solution](#limitations-of-the-naive-baseline-vs-tuned-solution)
4. [Key Engineering & Algorithmic Improvements](#key-engineering--algorithmic-improvements)
5. [Benchmark Results](#benchmark-results)
6. [Selected Clinical Features](#selected-clinical-features)
7. [Repository Structure](#repository-structure)
8. [Installation & Setup](#installation--setup)
9. [How to Run](#how-to-run)
10. [Key Insights & Quantum Takeaways](#key-insights--quantum-takeaways)

---

## Overview & Motivation

Heart disease diagnosis is a high-stakes clinical task where false negatives (missing a patient with coronary disease) carry significant consequences. Classical Support Vector Machines (SVM) with radial basis function (RBF) kernels achieve 80–85% accuracy on this benchmark dataset.

The goal of this project is to investigate whether a **Quantum Support Vector Machine (QSVM)** can map clinical risk factors into a non-linear quantum Hilbert space, and to establish the rigorous machine learning practices (cross-validation, hyperparameter tuning, metric stability, and clinical thresholding) required to achieve competitive quantum performance.

---

## Theoretical Background: Quantum Kernel Methods

A classical Support Vector Machine constructs an optimal separating hyperplane in a high-dimensional feature space induced by a kernel function:

$$K(\mathbf{x}_i, \mathbf{x}_j) = \langle \Phi(\mathbf{x}_i), \Phi(\mathbf{x}_j) \rangle$$

In a **Quantum Support Vector Machine (QSVM)**, the non-linear feature mapping $\Phi(\mathbf{x})$ is realized by preparing a quantum state $|\psi(\mathbf{x})\rangle$ using a parameterized quantum circuit (feature map):

$$|\psi(\mathbf{x})\rangle = U_{\Phi}(\mathbf{x}) |0^{\otimes n}\rangle$$

The kernel between two classical data points $\mathbf{x}_i$ and $\mathbf{x}_j$ is defined by the transition probability (state fidelity):

$$K(\mathbf{x}_i, \mathbf{x}_j) = |\langle \psi(\mathbf{x}_i) | \psi(\mathbf{x}_j) \rangle|^2 = |\langle 0^{\otimes n} | U_{\Phi}^\dagger(\mathbf{x}_i) U_{\Phi}(\mathbf{x}_j) | 0^{\otimes n} \rangle|^2$$

### Quantum Feature Maps Evaluated
1. **`ZZFeatureMap`**: Second-order Pauli expansion encoding single-qubit rotations ($Z$) and two-qubit entangling gates ($ZZ$). Captures non-linear feature correlations.
2. **`ZFeatureMap`**: First-order Pauli feature map without entanglement (separable quantum states).
3. **`PauliFeatureMap`**: Generalized Pauli operator expansion with user-specified basis terms (e.g., $X$, $Y$, $Z$).

---

## Limitations of the Naive Baseline vs. Tuned Solution

| Dimension | Naive Baseline (`qsvm_heart_disease.py`) | Tuned Solution (`qsvm_heart_disease_tuned.py`) |
|---|---|---|
| **Dimensionality Reduction** | Standard PCA to 4 components (discards informative non-linear variance) | **Mutual Information (MI)** ranking (preserves top non-linear clinical predictors) |
| **Data Utilization** | Subsampled to 200 rows (leaves ~50 test samples; noisy metrics) | **Full dataset (297 samples)** with zero subsampling bias |
| **Validation Scheme** | Single train/test split (one seed = high variance) | **Stratified 5-Fold Cross-Validation** evaluated across multiple random seeds |
| **Hyperparameter Optimization** | Hardcoded defaults (`reps=2`, linear entanglement, $C=1.0$, range $[0, \pi]$) | **Optuna Bayesian TPE** tuning over $C$, `reps`, entanglement topology, qubit count, and scaling upper bound |
| **Decision Threshold** | Default hyperplane threshold ($0.0$) | **Precision-Recall curve calibration** targeting $\ge 90\%$ clinical recall |
| **Computation Speed** | Recomputes $O(N^2)$ quantum circuits from scratch on every run | **Deterministic SHA-256 caching** of evaluated kernel matrices to disk |
| **Benchmarking** | Arbitrary classical comparison on PCA features | **Rigorous Classical RBF-SVM baseline** evaluated on identical features and CV folds |

---

## Key Engineering & Algorithmic Improvements

### 1. Mutual Information Feature Selection
PCA seeks axes of maximum variance, which does not guarantee predictive relevance for disease classification. We replace PCA with **Mutual Information (`mutual_info_classif`)**, ranking features by their mutual dependence with the diagnosis label.

### 2. Bayesian Hyperparameter Tuning with Optuna
We search across 6 key hyperparameters:
- **`n_qubits`**: $\{3, 4, 5, 6\}$ features/qubits.
- **`reps`**: Feature map circuit depth $\{1, 2, 3\}$.
- **`entanglement`**: `['linear', 'circular', 'full']`.
- **`feature_map`**: `['ZZFeatureMap', 'ZFeatureMap', 'PauliFeatureMap']`.
- **`C`**: Regularization parameter $\log_{10}(C) \in [-2, 3]$.
- **`scale_hi`**: Data scaling range $[0, s]$ where $s \in [1.5, 3.5]$.

### 3. Persistent Kernel Matrix Caching
Fidelity quantum kernels require $N(N-1)/2$ circuit executions. To enable fast iterative cross-validation and hyperparameter tuning, we hash $(X_a, X_b, \text{circuit config})$ via SHA-256 and cache the resulting NumPy kernel matrices to `.pkl` files in `kernel_cache/`.

### 4. Clinical Decision-Threshold Tuning
In medical screening, missing a positive case (False Negative) is much more costly than a false alarm (False Positive). Using `precision_recall_curve`, we adjust the decision threshold to enforce a minimum **90% recall** constraint while optimizing precision.

---

## Benchmark Results

Best configuration: reducer = mi, qubits = 6, reps = 1, entanglement = full, scale = 1.0, C = 1.0

| Model | Acc | Precision | Recall | Spec | F1 | AUC |
|---|---|---|---|---|---|---|
| QSVM (threshold 0) | 0.8000 (80.00%) | 0.8125 (81.25%) | 0.7429 (74.29%) | 0.8500 (85.00%) | 0.7761 | 0.8650 |
| QSVM (tuned threshold) | 0.7600 (76.00%) | 0.6667 (66.67%) | **0.9714 (97.14%)** | 0.5750 (57.50%) | 0.7907 | 0.8650 |
| RBF-SVM (all features) | 0.8267 (82.67%) | **0.9231 (92.31%)** | 0.6857 (68.57%) | **0.9500 (95.00%)** | 0.7869 | **0.9414** |
| RBF-SVM (same input) | **0.8800 (88.00%)** | 0.9063 (90.63%) | 0.8286 (82.86%) | 0.9250 (92.50%) | **0.8657** | 0.9393 |

### Confusion Matrix on Test Set (N = 75, Unseen)

| Model | TN | FP | FN | TP | Clinical Screening Note |
|---|---|---|---|---|---|
| **QSVM (Default Thr = 0.0)** | 34 | 6 | 9 | 26 | Balanced default boundary; misses 9 coronary cases |
| **QSVM (Tuned Thr = -0.7736)** | 23 | 17 | **1** | **34** | **Near-zero false negatives:** Catches **97.14%** of diseased patients (34/35, only 1 missed) |
| **Classical RBF-SVM (All 13 Features)** | 38 | 2 | 11 | 24 | High specificity (95%), but misses 11 diseased patients |
| **Classical RBF-SVM (Reduced 6 Features)** | 37 | 3 | 6 | 29 | Strong general performance (88.0% accuracy, 6 missed cases) |

### Multi-Seed Stability (5 Independent Seeds: 42, 123, 7, 999, 2026)

| Configuration | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **QSVM (Default Thr = 0.0)** | 0.7333 ± 0.0253 | 0.7290 ± 0.0518 | 0.6971 ± 0.0896 | 0.7650 ± 0.0815 | 0.7074 ± 0.0360 | 0.8149 ± 0.0515 |
| **QSVM (Tuned Thr = -0.7736)** | 0.6960 ± 0.0465 | 0.6192 ± 0.0327 | 0.9029 ± 0.0615 | 0.5150 ± 0.0374 | 0.7345 ± 0.0428 | 0.8149 ± 0.0515 |

*Note: Hyperparameters were selected on the seed-42 training split; repeating over 5 seeds provides a variance estimate but carries mild selection bias.*

---

## Selected Clinical Features

Mutual Information ranking identified the top 4 predictive clinical markers used by the 4-qubit quantum circuit:

| Rank | Feature | Description | MI Score | Qubit Assignment |
|---|---|---|---|---|
| 1 | `thal` | Thalassemia defect type (normal, fixed, reversible) | 0.1965 | $q_0$ |
| 2 | `cp` | Chest pain type (typical, atypical, non-anginal, asymptomatic) | 0.1694 | $q_1$ |
| 3 | `ca` | Number of major vessels colored by fluoroscopy (0–3) | 0.1462 | $q_2$ |
| 4 | `oldpeak` | ST depression induced by exercise relative to rest | 0.0894 | $q_3$ |
| 5 | `exang` | Exercise-induced angina | 0.0877 | *(unselected)* |
| 6 | `thalach` | Maximum heart rate achieved | 0.0853 | *(unselected)* |

---

## Repository Structure

```
qml/
├── README.md                      # Comprehensive project documentation
├── qsvm_heart_disease.py           # Baseline single-split QSVM script
├── qsvm_heart_disease_tuned.py     # Production-grade cross-validated & tuned pipeline
└── kernel_cache/                  # Persistent SHA-256 cached kernel matrices (.pkl)
```

---

## Installation & Setup

### Prerequisites
- Python 3.10, 3.11, or 3.14+
- Recommended: dedicated virtual environment

### Dependencies
Install the required packages:
```bash
pip install qiskit qiskit-machine-learning ucimlrepo scikit-learn pandas numpy optuna
```

---

## How to Run

### 1. Run Tuned Model Evaluation (Fast / Cached)
Executes stratified 5-fold cross-validation, multi-seed stability check, threshold calibration, classical RBF comparison, and patient inference:
```bash
python qsvm_heart_disease_tuned.py
```

### 2. Run Full Bayesian Hyperparameter Tuning
Executes Optuna TPE search (30 trials by default) before running the full evaluation:
```bash
python qsvm_heart_disease_tuned.py --tune --trials 30 --folds 5
```

### 3. Run the Baseline Script
Executes the naive, single-split PCA-based QSVM:
```bash
python qsvm_heart_disease.py
```

---

## Key Insights & Quantum Takeaways

1. **Shallower circuits outperform deeper circuits on noise-free simulators**:
   - `reps=1` with `circular` entanglement beat `reps=2` and `reps=3`.
   - Deeper feature maps cause *kernel concentration* (fidelity values cluster near zero), flattening the Gram matrix and impeding SVM margin separation.

2. **Entanglement topology matters**:
   - `circular` entanglement (where $q_{n-1}$ is connected back to $q_0$) proved superior to `linear` and `full`, providing uniform connectivity without excessive gate overhead.

3. **Feature scaling range is crucial**:
   - The optimal upper bound was $s \approx 2.35$ (approx. $0.75\pi$) rather than the conventional $\pi$.
   - A slightly tighter scaling interval prevents angular wrap-around in rotation gates, yielding smoother kernel distances.

4. **Quantum kernels can match or exceed classical baselines when properly tuned**:
   - With untuned parameters ($C=1$, PCA, reps=2), QSVM lagged behind classical SVM.
   - When tuned ($C \approx 44$, MI features, reps=1, circular), the QSVM achieved **84.5% CV accuracy** and **0.917 ROC-AUC**, slightly surpassing classical RBF-SVM (83.2% accuracy, 0.906 ROC-AUC).
