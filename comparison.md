# QSVM Optimization Comparison

## Dataset and evaluation protocol

- Dataset: bundled Cleveland heart-disease CSV; 303 source records, 297 complete rows used, 6 rows removed.
- Target: original 0 maps to no disease; values above 0 map to disease.
- Evaluation: 5 outer × 5 inner stratified folds, seed 42. Feature selection, circuit/kernel selection, C, and decision threshold are selected within outer training data.
- Primary comparison metric: ROC-AUC; threshold metrics use inner out-of-fold F2 calibration.
- The three stages use the same outer-fold seed and classical all-feature baseline. The reduced classical baseline follows the selected quantum feature reduction, so it can differ between stages.

## How many iterations?

There were **3 distinct completed full-search stages** to reach the current result. The first 128-setting fidelity-only stage was executed twice (120.1 s and 271.1 s; the latter run is archived), so there were **4 completed full training executions** across those stages. There were also two quick 3×3 smoke runs and one expanded-search attempt that stopped during fold 1 because the Windows console could not print a Greek gamma character; that interrupted attempt produced no result set.

| Stage | Quantum candidates per outer fold | Search scope | Runtime of archived full run |
|---|---:|---|---:|
| 1 — Fidelity baseline | 128 | Fidelity kernel; 4/6 qubits | 271.1 s |
| 2 — One-qubit projected kernel | 512 | Fidelity + order-1 projected kernel; 4/6 qubits | 166.0 s |
| 3 — Expanded projected search (current) | 1344 | Fidelity + order-1/order-2 projected kernels; 4/6/8 qubits | 526.0 s |

### Smoke and interrupted runs

| Run | Search settings | Outer-fold QSVM AUCs | Runtime/status |
|---|---|---|---|
| Quick fidelity smoke | 3×3 CV; 2 settings | 0.861, 0.740, 0.802 | 13.4 s; pipeline smoke only |
| Quick projected smoke | 3×3 CV; 4 settings | 0.835, 0.708, 0.809 | 12.2 s; pipeline smoke only |
| Expanded-search failed attempt | 8 qubits + order-1/order-2 projections | Not reported | Stopped during first-fold logging due to Windows cp1252 output encoding; fixed by using ASCII console text and rerun successfully |

## PCA status

**Yes. PCA is implemented and was included in all three full searches.** It competed against mutual-information feature selection as the reducer, fitted only on each training fold. Each full search included the same number of MI and PCA candidates. PCA was selected in **0 of 5 outer folds at every stage**, and the final full-data model selected MI. The current model therefore uses MI; PCA remains available as a searched alternative.

## Aggregate nested-CV results

Values are mean ± standard deviation across the five outer folds. All six reported metrics are included.

### 1 — Fidelity baseline

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Classical RBF (all features) | 0.747 ± 0.059 | 0.654 ± 0.051 | 0.971 ± 0.031 | 0.556 ± 0.084 | 0.781 ± 0.045 | 0.907 ± 0.052 |
| Classical RBF (same reduced input) | 0.693 ± 0.038 | 0.608 ± 0.034 | 0.949 ± 0.033 | 0.475 ± 0.068 | 0.741 ± 0.028 | 0.893 ± 0.070 |
| QSVM | 0.653 ± 0.109 | 0.589 ± 0.075 | 0.934 ± 0.109 | 0.412 ± 0.245 | 0.716 ± 0.056 | 0.875 ± 0.074 |

### 2 — One-qubit projected kernel

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Classical RBF (all features) | 0.747 ± 0.059 | 0.654 ± 0.051 | 0.971 ± 0.031 | 0.556 ± 0.084 | 0.781 ± 0.045 | 0.907 ± 0.052 |
| Classical RBF (same reduced input) | 0.710 ± 0.053 | 0.625 ± 0.052 | 0.949 ± 0.033 | 0.506 ± 0.107 | 0.752 ± 0.035 | 0.887 ± 0.066 |
| QSVM | 0.626 ± 0.095 | 0.569 ± 0.066 | 0.919 ± 0.129 | 0.375 ± 0.240 | 0.695 ± 0.047 | 0.884 ± 0.068 |

### 3 — Expanded projected search (current)

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Classical RBF (all features) | 0.747 ± 0.059 | 0.654 ± 0.051 | 0.971 ± 0.031 | 0.556 ± 0.084 | 0.781 ± 0.045 | 0.907 ± 0.052 |
| Classical RBF (same reduced input) | 0.751 ± 0.095 | 0.676 ± 0.096 | 0.912 ± 0.062 | 0.613 ± 0.148 | 0.775 ± 0.077 | 0.910 ± 0.057 |
| QSVM | 0.778 ± 0.091 | 0.707 ± 0.092 | 0.897 ± 0.072 | 0.675 ± 0.116 | 0.790 ± 0.082 | 0.916 ± 0.053 |

## Per-fold results

This section contains the complete outer-fold metrics and confusion counts for every stage/model.

### 1 — Fidelity baseline

| Fold | Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | TN | FP | FN | TP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Classical RBF (all features) | 0.783 | 0.692 | 0.964 | 0.625 | 0.806 | 0.955 | 20 | 12 | 1 | 27 |
| 1 | Classical RBF (same reduced input) | 0.750 | 0.659 | 0.964 | 0.562 | 0.783 | 0.948 | 18 | 14 | 1 | 27 |
| 1 | QSVM | 0.717 | 0.628 | 0.964 | 0.500 | 0.761 | 0.921 | 16 | 16 | 1 | 27 |
| 2 | Classical RBF (all features) | 0.800 | 0.700 | 1.000 | 0.625 | 0.824 | 0.916 | 20 | 12 | 0 | 28 |
| 2 | Classical RBF (same reduced input) | 0.700 | 0.619 | 0.929 | 0.500 | 0.743 | 0.916 | 16 | 16 | 2 | 26 |
| 2 | QSVM | 0.700 | 0.614 | 0.964 | 0.469 | 0.750 | 0.908 | 15 | 17 | 1 | 27 |
| 3 | Classical RBF (all features) | 0.661 | 0.581 | 0.926 | 0.438 | 0.714 | 0.823 | 14 | 18 | 2 | 25 |
| 3 | Classical RBF (same reduced input) | 0.644 | 0.568 | 0.926 | 0.406 | 0.704 | 0.781 | 13 | 19 | 2 | 25 |
| 3 | QSVM | 0.695 | 0.645 | 0.741 | 0.656 | 0.690 | 0.751 | 21 | 11 | 7 | 20 |
| 4 | Classical RBF (all features) | 0.712 | 0.619 | 0.963 | 0.500 | 0.754 | 0.896 | 16 | 16 | 1 | 26 |
| 4 | Classical RBF (same reduced input) | 0.695 | 0.610 | 0.926 | 0.500 | 0.735 | 0.873 | 16 | 16 | 2 | 25 |
| 4 | QSVM | 0.458 | 0.458 | 1.000 | 0.000 | 0.628 | 0.864 | 0 | 32 | 0 | 27 |
| 5 | Classical RBF (all features) | 0.780 | 0.675 | 1.000 | 0.594 | 0.806 | 0.943 | 19 | 13 | 0 | 27 |
| 5 | Classical RBF (same reduced input) | 0.678 | 0.587 | 1.000 | 0.406 | 0.740 | 0.949 | 13 | 19 | 0 | 27 |
| 5 | QSVM | 0.695 | 0.600 | 1.000 | 0.438 | 0.750 | 0.932 | 14 | 18 | 0 | 27 |
### 2 — One-qubit projected kernel

| Fold | Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | TN | FP | FN | TP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Classical RBF (all features) | 0.783 | 0.692 | 0.964 | 0.625 | 0.806 | 0.955 | 20 | 12 | 1 | 27 |
| 1 | Classical RBF (same reduced input) | 0.717 | 0.628 | 0.964 | 0.500 | 0.761 | 0.918 | 16 | 16 | 1 | 27 |
| 1 | QSVM | 0.650 | 0.571 | 1.000 | 0.344 | 0.727 | 0.919 | 11 | 21 | 0 | 28 |
| 2 | Classical RBF (all features) | 0.800 | 0.700 | 1.000 | 0.625 | 0.824 | 0.916 | 20 | 12 | 0 | 28 |
| 2 | Classical RBF (same reduced input) | 0.783 | 0.703 | 0.929 | 0.656 | 0.800 | 0.914 | 21 | 11 | 2 | 26 |
| 2 | QSVM | 0.667 | 0.595 | 0.893 | 0.469 | 0.714 | 0.890 | 15 | 17 | 3 | 25 |
| 3 | Classical RBF (all features) | 0.661 | 0.581 | 0.926 | 0.438 | 0.714 | 0.823 | 14 | 18 | 2 | 25 |
| 3 | Classical RBF (same reduced input) | 0.644 | 0.568 | 0.926 | 0.406 | 0.704 | 0.781 | 13 | 19 | 2 | 25 |
| 3 | QSVM | 0.678 | 0.633 | 0.704 | 0.656 | 0.667 | 0.773 | 21 | 11 | 8 | 19 |
| 4 | Classical RBF (all features) | 0.712 | 0.619 | 0.963 | 0.500 | 0.754 | 0.896 | 16 | 16 | 1 | 26 |
| 4 | Classical RBF (same reduced input) | 0.729 | 0.641 | 0.926 | 0.562 | 0.758 | 0.870 | 18 | 14 | 2 | 25 |
| 4 | QSVM | 0.458 | 0.458 | 1.000 | 0.000 | 0.628 | 0.884 | 0 | 32 | 0 | 27 |
| 5 | Classical RBF (all features) | 0.780 | 0.675 | 1.000 | 0.594 | 0.806 | 0.943 | 19 | 13 | 0 | 27 |
| 5 | Classical RBF (same reduced input) | 0.678 | 0.587 | 1.000 | 0.406 | 0.740 | 0.950 | 13 | 19 | 0 | 27 |
| 5 | QSVM | 0.678 | 0.587 | 1.000 | 0.406 | 0.740 | 0.953 | 13 | 19 | 0 | 27 |
### 3 — Expanded projected search (current)

| Fold | Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | TN | FP | FN | TP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | Classical RBF (all features) | 0.783 | 0.692 | 0.964 | 0.625 | 0.806 | 0.955 | 20 | 12 | 1 | 27 |
| 1 | Classical RBF (same reduced input) | 0.717 | 0.628 | 0.964 | 0.500 | 0.761 | 0.927 | 16 | 16 | 1 | 27 |
| 1 | QSVM | 0.767 | 0.684 | 0.929 | 0.625 | 0.788 | 0.946 | 20 | 12 | 2 | 26 |
| 2 | Classical RBF (all features) | 0.800 | 0.700 | 1.000 | 0.625 | 0.824 | 0.916 | 20 | 12 | 0 | 28 |
| 2 | Classical RBF (same reduced input) | 0.733 | 0.650 | 0.929 | 0.562 | 0.765 | 0.924 | 18 | 14 | 2 | 26 |
| 2 | QSVM | 0.783 | 0.703 | 0.929 | 0.656 | 0.800 | 0.901 | 21 | 11 | 2 | 26 |
| 3 | Classical RBF (all features) | 0.661 | 0.581 | 0.926 | 0.438 | 0.714 | 0.823 | 14 | 18 | 2 | 25 |
| 3 | Classical RBF (same reduced input) | 0.627 | 0.564 | 0.815 | 0.469 | 0.667 | 0.813 | 15 | 17 | 5 | 22 |
| 3 | QSVM | 0.644 | 0.583 | 0.778 | 0.531 | 0.667 | 0.835 | 17 | 15 | 6 | 21 |
| 4 | Classical RBF (all features) | 0.712 | 0.619 | 0.963 | 0.500 | 0.754 | 0.896 | 16 | 16 | 1 | 26 |
| 4 | Classical RBF (same reduced input) | 0.797 | 0.727 | 0.889 | 0.719 | 0.800 | 0.919 | 23 | 9 | 3 | 24 |
| 4 | QSVM | 0.797 | 0.727 | 0.889 | 0.719 | 0.800 | 0.924 | 23 | 9 | 3 | 24 |
| 5 | Classical RBF (all features) | 0.780 | 0.675 | 1.000 | 0.594 | 0.806 | 0.943 | 19 | 13 | 0 | 27 |
| 5 | Classical RBF (same reduced input) | 0.881 | 0.812 | 0.963 | 0.812 | 0.881 | 0.964 | 26 | 6 | 1 | 26 |
| 5 | QSVM | 0.898 | 0.839 | 0.963 | 0.844 | 0.897 | 0.976 | 27 | 5 | 1 | 26 |

## Selected settings by outer fold

### 1 — Fidelity baseline

| Fold | Reducer | Qubits | Reps | Entanglement | Kernel | Projection order | Bandwidth multiplier | Scale | C | Inner-CV AUC | OOF threshold | OOF F2 |
|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mi | 6 | 2 | linear | fidelity | — | — | 1 | 0.1 | 0.8723 | -0.5939 | 0.8516 |
| 2 | mi | 6 | 1 | linear | fidelity | — | — | 1 | 0.1 | 0.8841 | -0.8966 | 0.8553 |
| 3 | mi | 6 | 2 | linear | fidelity | — | — | 1 | 0.1 | 0.9117 | -0.0553 | 0.8781 |
| 4 | mi | 6 | 1 | full | fidelity | — | — | 1 | 0.1 | 0.8685 | -1.0070 | 0.8413 |
| 5 | mi | 6 | 2 | linear | fidelity | — | — | 1 | 0.1 | 0.8799 | -0.8806 | 0.8706 |
### 2 — One-qubit projected kernel

| Fold | Reducer | Qubits | Reps | Entanglement | Kernel | Projection order | Bandwidth multiplier | Scale | C | Inner-CV AUC | OOF threshold | OOF F2 |
|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mi | 6 | 2 | full | projected | — | 1.0 | 3.142 | 0.1 | 0.8870 | -0.9534 | 0.8511 |
| 2 | mi | 6 | 2 | full | projected | — | 1.0 | 3.142 | 0.1 | 0.8876 | -0.7842 | 0.8609 |
| 3 | mi | 6 | 1 | linear | projected | — | 10.0 | 1 | 0.1 | 0.9187 | 0.0213 | 0.8719 |
| 4 | mi | 4 | 2 | full | projected | — | 1.0 | 1 | 10.0 | 0.8761 | -0.8773 | 0.8206 |
| 5 | mi | 6 | 1 | full | projected | — | 0.1 | 3.142 | 10.0 | 0.8874 | -0.7760 | 0.8685 |
### 3 — Expanded projected search (current)

| Fold | Reducer | Qubits | Reps | Entanglement | Kernel | Projection order | Bandwidth multiplier | Scale | C | Inner-CV AUC | OOF threshold | OOF F2 |
|---:|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | mi | 8 | 1 | full | projected | 1 | 0.1 | 3.142 | 10.0 | 0.8950 | -0.3693 | 0.8639 |
| 2 | mi | 8 | 2 | linear | projected | 1 | 0.1 | 1 | 10.0 | 0.9017 | -0.5936 | 0.8688 |
| 3 | mi | 8 | 1 | full | projected | 1 | 0.1 | 3.142 | 10.0 | 0.9339 | -0.3869 | 0.8932 |
| 4 | mi | 8 | 1 | full | projected | 1 | 0.1 | 3.142 | 10.0 | 0.9113 | -0.3741 | 0.8767 |
| 5 | mi | 8 | 2 | full | projected | 1 | 0.1 | 3.142 | 10.0 | 0.9041 | -0.2507 | 0.8711 |

## Final full-data review models

These are refits for inference after nested evaluation. Their inner-CV scores are not independent estimates; the outer-fold tables above are the performance comparison.

### 1 — Fidelity baseline

- Full-data selected config: `{"C": 0.1, "entanglement": "linear", "mean_cv_roc_auc": 0.9013227513227513, "qubits": 6, "reducer": "mi", "reps": 1, "scale": 1.0, "std_cv_roc_auc": 0.05624198496346784}`
- OOF threshold: `-0.303916`; OOF F2: `0.8717`
- Final kernel effective rank: `27.600`; off-diagonal mean: `0.113`; minimum eigenvalue: `-5.850e-15`
- Dataset SHA-256: `868cd646a53ffbc55611192bae7496911a163937c185710f6b8aa11549b88df1`; Python 3.14.3, NumPy 2.4.2, scikit-learn 1.8.0, Qiskit 2.4.1.

### 2 — One-qubit projected kernel

- Full-data selected config: `{"C": 10.0, "entanglement": "full", "gamma_multiplier": 0.1, "kernel_type": "projected", "mean_cv_roc_auc": 0.9059482473544973, "qubits": 6, "reducer": "mi", "reps": 1, "scale": 3.141592653589793, "std_cv_roc_auc": 0.048202444110484695}`
- OOF threshold: `-0.565691`; OOF F2: `0.8573`
- Final kernel effective rank: `1.697`; off-diagonal mean: `0.898`; minimum eigenvalue: `-7.754e-16`
- Dataset SHA-256: `868cd646a53ffbc55611192bae7496911a163937c185710f6b8aa11549b88df1`; Python 3.14.3, NumPy 2.4.2, scikit-learn 1.8.0, Qiskit 2.4.1.

### 3 — Expanded projected search (current)

- Full-data selected config: `{"C": 1.0, "entanglement": "full", "gamma_multiplier": 0.1, "kernel_type": "projected", "mean_cv_roc_auc": 0.914277447089947, "projection_order": 1, "qubits": 8, "reducer": "mi", "reps": 2, "scale": 3.141592653589793, "std_cv_roc_auc": 0.046101922129959824}`
- OOF threshold: `-0.256786`; OOF F2: `0.8662`
- Final kernel effective rank: `1.731`; off-diagonal mean: `0.902`; minimum eigenvalue: `-4.622e-16`
- Dataset SHA-256: `868cd646a53ffbc55611192bae7496911a163937c185710f6b8aa11549b88df1`; Python 3.14.3, NumPy 2.4.2, scikit-learn 1.8.0, Qiskit 2.4.1.

## What changed across iterations

1. **Fidelity-only search (128 candidates):** selected 6-qubit MI configurations, but the QSVM trailed the all-feature RBF baseline in mean AUC and accuracy.
2. **Add local one-qubit projected kernels (512 candidates):** the projected kernel was selected in all outer folds, but mean QSVM AUC remained below the classical baseline.
3. **Add 8 qubits and one-/two-qubit Pauli projections (1,344 candidates):** the selected outer-fold models consistently used 8-qubit, order-1 projected kernels. This stage produced a small positive point-estimate edge over the all-feature classical baseline.

## Interpretation and limits

In the final run, QSVM mean outer-fold AUC was 0.916 versus 0.907 for all-feature RBF, and accuracy was 0.778 versus 0.747. The QSVM specificity was higher (0.675 vs 0.556) while recall was lower (0.897 vs 0.971). The five-fold variability is substantial and overlapping; this comparison does not establish statistical significance. The final full-data projected kernel is also highly concentrated (effective rank ≈1.73; off-diagonal mean ≈0.90).

All experiments used exact statevector simulation. They establish neither a quantum hardware speed advantage nor a formal computational quantum advantage. The predictive difference should be described as a favorable but preliminary nested-CV result.

Full per-stage artifacts remain in `outputs/fidelity_only_baseline/`, `outputs/projected_1body_baseline/`, and the current `outputs/` directory. The earlier pre-revision single-split results are in `outputs/historical_single_split/`.
