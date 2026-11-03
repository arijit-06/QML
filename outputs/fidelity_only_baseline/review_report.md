# Nested CV model review

Dataset: `heart_disease_cleveland.csv` (297 rows used; 6 dropped)
Outer/inner CV: 5 × 5; seed 42; mode: full

## Outer-fold performance

Scores below are the procedure-level estimate. Model settings and the threshold were selected using only each outer training partition.

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Classical RBF (all features) | 0.747 ± 0.059 | 0.654 ± 0.051 | 0.971 ± 0.031 | 0.556 ± 0.084 | 0.781 ± 0.045 | 0.907 ± 0.052 |
| Classical RBF (same reduced input) | 0.693 ± 0.038 | 0.608 ± 0.034 | 0.949 ± 0.033 | 0.475 ± 0.068 | 0.741 ± 0.028 | 0.893 ± 0.070 |
| QSVM | 0.653 ± 0.109 | 0.589 ± 0.075 | 0.934 ± 0.109 | 0.412 ± 0.245 | 0.716 ± 0.056 | 0.875 ± 0.074 |

## Interpretation

The all-feature classical RBF-SVM had the highest mean outer-fold ROC-AUC (0.907); the QSVM mean was 0.875 (difference +0.031, classical minus quantum). This run does not show a quantum advantage.
The QSVM threshold averaged 0.934 recall and 0.412 specificity; its fold-specific tradeoff varies, so these values should not be read as clinical validation.
Selected quantum settings by outer fold: fold 1: mi/6q, reps=2, linear; fold 2: mi/6q, reps=1, linear; fold 3: mi/6q, reps=2, linear; fold 4: mi/6q, reps=1, full; fold 5: mi/6q, reps=2, linear.

## Per-fold results

| Fold | Model | Accuracy | Recall | Specificity | ROC-AUC | TN | FP | FN | TP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | QSVM | 0.717 | 0.964 | 0.500 | 0.921 | 16 | 16 | 1 | 27 |
| 1 | Classical RBF (all features) | 0.783 | 0.964 | 0.625 | 0.955 | 20 | 12 | 1 | 27 |
| 1 | Classical RBF (same reduced input) | 0.750 | 0.964 | 0.562 | 0.948 | 18 | 14 | 1 | 27 |
| 2 | QSVM | 0.700 | 0.964 | 0.469 | 0.908 | 15 | 17 | 1 | 27 |
| 2 | Classical RBF (all features) | 0.800 | 1.000 | 0.625 | 0.916 | 20 | 12 | 0 | 28 |
| 2 | Classical RBF (same reduced input) | 0.700 | 0.929 | 0.500 | 0.916 | 16 | 16 | 2 | 26 |
| 3 | QSVM | 0.695 | 0.741 | 0.656 | 0.751 | 21 | 11 | 7 | 20 |
| 3 | Classical RBF (all features) | 0.661 | 0.926 | 0.438 | 0.823 | 14 | 18 | 2 | 25 |
| 3 | Classical RBF (same reduced input) | 0.644 | 0.926 | 0.406 | 0.781 | 13 | 19 | 2 | 25 |
| 4 | QSVM | 0.458 | 1.000 | 0.000 | 0.864 | 0 | 32 | 0 | 27 |
| 4 | Classical RBF (all features) | 0.712 | 0.963 | 0.500 | 0.896 | 16 | 16 | 1 | 26 |
| 4 | Classical RBF (same reduced input) | 0.695 | 0.926 | 0.500 | 0.873 | 16 | 16 | 2 | 25 |
| 5 | QSVM | 0.695 | 1.000 | 0.438 | 0.932 | 14 | 18 | 0 | 27 |
| 5 | Classical RBF (all features) | 0.780 | 1.000 | 0.594 | 0.943 | 19 | 13 | 0 | 27 |
| 5 | Classical RBF (same reduced input) | 0.678 | 1.000 | 0.406 | 0.949 | 13 | 19 | 0 | 27 |

## Final full-data review artifact

The separately refit artifact uses settings chosen by CV over all available data. Do not treat its training/CV scores as an independent performance estimate.

- Selected settings: `{"C": 0.1, "entanglement": "linear", "mean_cv_roc_auc": 0.9013227513227513, "qubits": 6, "reducer": "mi", "reps": 1, "scale": 1.0, "std_cv_roc_auc": 0.05624198496346784}`
- OOF F2 threshold on standardized margin: `-0.303916` (OOF F2 `0.872`)
- Training-kernel diagnostics: `{"decision_score_center": -0.13017987391052385, "decision_score_scale": 0.8275132380694803, "train_kernel_condition_number_above_1e-12_relative_floor": 867727756287.4052, "train_kernel_diagonal_max_error": 3.774758283725532e-15, "train_kernel_effective_rank": 27.599539938569727, "train_kernel_min_eigenvalue": -5.850410871163e-15, "train_kernel_offdiagonal_mean": 0.11262064817064558, "train_kernel_symmetry_max_error": 2.3314683517128287e-15}`

## Files

- `nested_cv_fold_results.csv`: outer-fold metrics.
- `nested_cv_search_results.csv`: inner-search scores for each outer fold.
- `nested_cv_details.json`: selected settings, thresholds, and kernel diagnostics per outer fold.
- `nested_cv_summary.csv`: mean and standard deviation by model.
- `run_metadata.json`: dataset hash, configuration, environment, and runtime.
- `qsvm_review_model.joblib`: fitted full-data inference artifact; for research review only, not clinical use.
