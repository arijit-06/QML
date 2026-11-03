# Nested CV model review

Dataset: `heart_disease_cleveland.csv` (297 rows used; 6 dropped)
Outer/inner CV: 5 × 5; seed 42; mode: full

## Outer-fold performance

Scores below are the procedure-level estimate. Model settings and the threshold were selected using only each outer training partition.

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Classical RBF (all features) | 0.747 ± 0.059 | 0.654 ± 0.051 | 0.971 ± 0.031 | 0.556 ± 0.084 | 0.781 ± 0.045 | 0.907 ± 0.052 |
| Classical RBF (same reduced input) | 0.710 ± 0.053 | 0.625 ± 0.052 | 0.949 ± 0.033 | 0.506 ± 0.107 | 0.752 ± 0.035 | 0.887 ± 0.066 |
| QSVM | 0.626 ± 0.095 | 0.569 ± 0.066 | 0.919 ± 0.129 | 0.375 ± 0.240 | 0.695 ± 0.047 | 0.884 ± 0.068 |

## Interpretation

The all-feature classical RBF-SVM had the highest mean outer-fold ROC-AUC (0.907); the QSVM mean was 0.884 (difference +0.023, classical minus quantum). This run does not show a quantum advantage.
The QSVM threshold averaged 0.919 recall and 0.375 specificity; its fold-specific tradeoff varies, so these values should not be read as clinical validation.
Selected quantum settings by outer fold: fold 1: mi/6q, reps=2, full; fold 2: mi/6q, reps=2, full; fold 3: mi/6q, reps=1, linear; fold 4: mi/4q, reps=2, full; fold 5: mi/6q, reps=1, full.

## Per-fold results

| Fold | Model | Accuracy | Recall | Specificity | ROC-AUC | TN | FP | FN | TP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | QSVM | 0.650 | 1.000 | 0.344 | 0.919 | 11 | 21 | 0 | 28 |
| 1 | Classical RBF (all features) | 0.783 | 0.964 | 0.625 | 0.955 | 20 | 12 | 1 | 27 |
| 1 | Classical RBF (same reduced input) | 0.717 | 0.964 | 0.500 | 0.918 | 16 | 16 | 1 | 27 |
| 2 | QSVM | 0.667 | 0.893 | 0.469 | 0.890 | 15 | 17 | 3 | 25 |
| 2 | Classical RBF (all features) | 0.800 | 1.000 | 0.625 | 0.916 | 20 | 12 | 0 | 28 |
| 2 | Classical RBF (same reduced input) | 0.783 | 0.929 | 0.656 | 0.914 | 21 | 11 | 2 | 26 |
| 3 | QSVM | 0.678 | 0.704 | 0.656 | 0.773 | 21 | 11 | 8 | 19 |
| 3 | Classical RBF (all features) | 0.661 | 0.926 | 0.438 | 0.823 | 14 | 18 | 2 | 25 |
| 3 | Classical RBF (same reduced input) | 0.644 | 0.926 | 0.406 | 0.781 | 13 | 19 | 2 | 25 |
| 4 | QSVM | 0.458 | 1.000 | 0.000 | 0.884 | 0 | 32 | 0 | 27 |
| 4 | Classical RBF (all features) | 0.712 | 0.963 | 0.500 | 0.896 | 16 | 16 | 1 | 26 |
| 4 | Classical RBF (same reduced input) | 0.729 | 0.926 | 0.562 | 0.870 | 18 | 14 | 2 | 25 |
| 5 | QSVM | 0.678 | 1.000 | 0.406 | 0.953 | 13 | 19 | 0 | 27 |
| 5 | Classical RBF (all features) | 0.780 | 1.000 | 0.594 | 0.943 | 19 | 13 | 0 | 27 |
| 5 | Classical RBF (same reduced input) | 0.678 | 1.000 | 0.406 | 0.950 | 13 | 19 | 0 | 27 |

## Final full-data review artifact

The separately refit artifact uses settings chosen by CV over all available data. Do not treat its training/CV scores as an independent performance estimate.

- Selected settings: `{"C": 10.0, "entanglement": "full", "gamma_multiplier": 0.1, "kernel_type": "projected", "mean_cv_roc_auc": 0.9059482473544973, "qubits": 6, "reducer": "mi", "reps": 1, "scale": 3.141592653589793, "std_cv_roc_auc": 0.048202444110484695}`
- OOF F2 threshold on standardized margin: `-0.565691` (OOF F2 `0.857`)
- Training-kernel diagnostics: `{"decision_score_center": -0.28044334476567534, "decision_score_scale": 1.6832422415856771, "projected_kernel_gamma": 0.03447257253750038, "train_kernel_condition_number_above_1e-12_relative_floor": 939626157704.6846, "train_kernel_diagonal_max_error": 0.0, "train_kernel_effective_rank": 1.6974732766438574, "train_kernel_min_eigenvalue": -7.754295563924473e-16, "train_kernel_offdiagonal_mean": 0.8977782069047565, "train_kernel_symmetry_max_error": 1.1102230246251565e-16}`

## Files

- `nested_cv_fold_results.csv`: outer-fold metrics.
- `nested_cv_search_results.csv`: inner-search scores for each outer fold.
- `nested_cv_details.json`: selected settings, thresholds, and kernel diagnostics per outer fold.
- `nested_cv_summary.csv`: mean and standard deviation by model.
- `run_metadata.json`: dataset hash, configuration, environment, and runtime.
- `qsvm_review_model.joblib`: fitted full-data inference artifact; for research review only, not clinical use.
