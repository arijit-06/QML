# Nested CV model review

Dataset: `heart_disease_cleveland.csv` (297 rows used; 6 dropped)
Outer/inner CV: 5 × 5; seed 42; mode: full

## Outer-fold performance

Scores below are the procedure-level estimate. Model settings and the threshold were selected using only each outer training partition.

| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Classical RBF (all features) | 0.747 ± 0.059 | 0.654 ± 0.051 | 0.971 ± 0.031 | 0.556 ± 0.084 | 0.781 ± 0.045 | 0.907 ± 0.052 |
| Classical RBF (same reduced input) | 0.751 ± 0.095 | 0.676 ± 0.096 | 0.912 ± 0.062 | 0.613 ± 0.148 | 0.775 ± 0.077 | 0.910 ± 0.057 |
| QSVM | 0.778 ± 0.091 | 0.707 ± 0.092 | 0.897 ± 0.072 | 0.675 ± 0.116 | 0.790 ± 0.082 | 0.916 ± 0.053 |

## Interpretation

The expanded 8-qubit projected QSVM had mean outer-fold ROC-AUC 0.916 and accuracy 0.778; the all-feature classical RBF-SVM scored 0.907 AUC and 0.747 accuracy (QSVM minus classical: +0.010 AUC, +0.031 accuracy). This is a modest predictive edge in this five-fold run, not evidence of a statistically robust or computational quantum advantage; the fold-to-fold uncertainty is substantial and the final projected kernel is highly concentrated.
The QSVM threshold averaged 0.897 recall and 0.675 specificity; its fold-specific tradeoff varies, so these values should not be read as clinical validation.
Selected quantum settings by outer fold: fold 1: mi/8q, reps=1, full, projected order 1 γ×0.1; fold 2: mi/8q, reps=2, linear, projected order 1 γ×0.1; fold 3: mi/8q, reps=1, full, projected order 1 γ×0.1; fold 4: mi/8q, reps=1, full, projected order 1 γ×0.1; fold 5: mi/8q, reps=2, full, projected order 1 γ×0.1.

## Per-fold results

| Fold | Model | Accuracy | Recall | Specificity | ROC-AUC | TN | FP | FN | TP |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | QSVM | 0.767 | 0.929 | 0.625 | 0.946 | 20 | 12 | 2 | 26 |
| 1 | Classical RBF (all features) | 0.783 | 0.964 | 0.625 | 0.955 | 20 | 12 | 1 | 27 |
| 1 | Classical RBF (same reduced input) | 0.717 | 0.964 | 0.500 | 0.927 | 16 | 16 | 1 | 27 |
| 2 | QSVM | 0.783 | 0.929 | 0.656 | 0.901 | 21 | 11 | 2 | 26 |
| 2 | Classical RBF (all features) | 0.800 | 1.000 | 0.625 | 0.916 | 20 | 12 | 0 | 28 |
| 2 | Classical RBF (same reduced input) | 0.733 | 0.929 | 0.562 | 0.924 | 18 | 14 | 2 | 26 |
| 3 | QSVM | 0.644 | 0.778 | 0.531 | 0.835 | 17 | 15 | 6 | 21 |
| 3 | Classical RBF (all features) | 0.661 | 0.926 | 0.438 | 0.823 | 14 | 18 | 2 | 25 |
| 3 | Classical RBF (same reduced input) | 0.627 | 0.815 | 0.469 | 0.813 | 15 | 17 | 5 | 22 |
| 4 | QSVM | 0.797 | 0.889 | 0.719 | 0.924 | 23 | 9 | 3 | 24 |
| 4 | Classical RBF (all features) | 0.712 | 0.963 | 0.500 | 0.896 | 16 | 16 | 1 | 26 |
| 4 | Classical RBF (same reduced input) | 0.797 | 0.889 | 0.719 | 0.919 | 23 | 9 | 3 | 24 |
| 5 | QSVM | 0.898 | 0.963 | 0.844 | 0.976 | 27 | 5 | 1 | 26 |
| 5 | Classical RBF (all features) | 0.780 | 1.000 | 0.594 | 0.943 | 19 | 13 | 0 | 27 |
| 5 | Classical RBF (same reduced input) | 0.881 | 0.963 | 0.812 | 0.964 | 26 | 6 | 1 | 26 |

## Final full-data review artifact

The separately refit artifact uses settings chosen by CV over all available data. Do not treat its training/CV scores as an independent performance estimate.

- Selected settings: `{"C": 1.0, "entanglement": "full", "gamma_multiplier": 0.1, "kernel_type": "projected", "mean_cv_roc_auc": 0.914277447089947, "projection_order": 1, "qubits": 8, "reducer": "mi", "reps": 2, "scale": 3.141592653589793, "std_cv_roc_auc": 0.046101922129959824}`
- OOF F2 threshold on standardized margin: `-0.256786` (OOF F2 `0.866`)
- Training-kernel diagnostics: `{"decision_score_center": -0.14403242033571625, "decision_score_scale": 1.1406205838398689, "projected_kernel_gamma": 0.02511468711012667, "projection_order": 1, "train_kernel_condition_number_above_1e-12_relative_floor": 691062908423.2776, "train_kernel_diagonal_max_error": 1.1102230246251565e-16, "train_kernel_effective_rank": 1.7313822734476962, "train_kernel_min_eigenvalue": -4.622289806365019e-16, "train_kernel_offdiagonal_mean": 0.9019667778641388, "train_kernel_symmetry_max_error": 1.1102230246251565e-16}`

## Files

- `nested_cv_fold_results.csv`: outer-fold metrics.
- `nested_cv_search_results.csv`: inner-search scores for each outer fold.
- `nested_cv_details.json`: selected settings, thresholds, and kernel diagnostics per outer fold.
- `nested_cv_summary.csv`: mean and standard deviation by model.
- `run_metadata.json`: dataset hash, configuration, environment, and runtime.
- `qsvm_review_model.joblib`: fitted full-data inference artifact; for research review only, not clinical use.
