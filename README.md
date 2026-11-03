# Quantum Kernel SVM for Cleveland Heart Disease

This repository compares exact noiseless quantum fidelity and projected-kernel SVMs with tuned classical RBF-SVM baselines. The evaluation uses nested stratified cross-validation so feature selection, circuit/kernel selection, regularization, and threshold calibration happen inside each outer training partition.

This is an academic benchmark on a small public dataset. It is not a clinical diagnostic tool and does not establish quantum advantage.

In the current full nested-CV run, an 8-qubit projected QSVM obtained mean outer-fold ROC-AUC 0.916 and accuracy 0.778, versus 0.907 and 0.747 for the all-feature classical RBF-SVM. Treat this as a modest predictive point-estimate edge, not a statistically established result or a computational quantum speedup; see the per-fold values and kernel diagnostics in the review report.

## Pipeline

1. Load the bundled `heart_disease_cleveland.csv` or pass a CSV path. Missing/non-numeric rows are removed; labels `0` and `1..4` map to no disease/disease.
2. In each fold, standardize continuous features and one-hot encode the known categorical Cleveland variables (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`).
3. Search mutual-information selection or PCA; 4, 6, or 8 qubits; 1 or 2 feature-map repetitions; linear or full entanglement; input angle range `[0, 1]` or `[0, π]`; fidelity or projected quantum kernels; one- or two-qubit Pauli-observable projections; projected-kernel bandwidth multipliers; and `C` in `{0.1, 1, 10, 100}`.
4. Compute exact statevectors, fidelity Gram matrices, and kernels over local one- or two-qubit reduced-state observables. Fold encodings, statevectors, and kernel matrices are reused where settings share them; all `C` values reuse each kernel.
5. Select a clinical threshold by inner-fold OOF F2 on standardized SVC margins. A threshold is selected separately within each outer training fold.
6. Compare with RBF-SVMs on all one-hot/continuous inputs and on the same fold-local reduced inputs as the selected quantum model. Classical models use the same outer folds and inner training data.
7. Refit a full-data model for research inference after nested evaluation. Its CV metrics are not an independent estimate; use the nested outer-fold results for model comparison.

## Run

Install the pinned environment from `requirements.txt`, then run from any working directory:

```powershell
python C:\PROGGING\qml\qsvm_heart_disease_tuned.py
```

By default, this runs 5 outer × 5 inner folds and the full search. For a quick pipeline run:

```powershell
python C:\PROGGING\qml\qsvm_heart_disease_tuned.py --quick --outer-folds 3 --inner-folds 3
```

Optional arguments include `--dataset-path`, `--target-column`, `--outer-folds`, `--inner-folds`, `--jobs`, and `--seed`. An invalid dataset path raises an error instead of silently switching datasets. Outputs are rooted at the script directory, independent of the current working directory.

## Review outputs

Current nested-CV files in `outputs/` are overwritten on each run. Results from the previous single-split pipeline are preserved in `outputs/historical_single_split/`:

- `review_report.md`: readable performance summary and per-fold confusion counts.
- `nested_cv_fold_results.csv`: raw outer-fold metrics for each model.
- `nested_cv_summary.csv`: mean and standard deviation across outer folds.
- `nested_cv_search_results.csv`: inner-CV scores for each quantum candidate.
- `nested_cv_details.json`: selected settings, threshold, and kernel diagnostics by fold.
- `run_metadata.json`: dataset SHA-256, feature list, seed, search size, package versions, and elapsed time.
- `qsvm_review_model.joblib`: full-data fitted research inference bundle.

The bundle can be used from Python:

```python
import joblib
import pandas as pd
from qsvm_heart_disease_tuned import predict_frame

model = joblib.load(r"C:\PROGGING\qml\outputs\qsvm_review_model.joblib")
patient = pd.DataFrame([{
    "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260,
    "fbs": 0, "restecg": 2, "thalach": 120, "exang": 1,
    "oldpeak": 2.4, "slope": 2, "ca": 2, "thal": 7,
}])
print(predict_frame(model, patient))
```

The returned decision score is a standardized SVC margin; its threshold was calibrated from OOF scores and is not a probability.

## Reproducibility and interpretation

- Every learned transform is fitted on its training fold only. The dataset file hash and software versions are saved with each run.
- Mutual-information scoring gets an explicit discrete-feature mask and seed. Ties and stochastic jitter are therefore reproducible for a fixed run configuration.
- Kernel reports include symmetry error, diagonal error, minimum eigenvalue, effective rank, condition estimate, and off-diagonal mean.
- The projected kernel uses the local reduced-state representation described by [Huang et al.](https://www.nature.com/articles/s41467-021-22539-9); its bandwidth is selected inside CV and does not by itself imply a quantum computational advantage.
- Highly expressive or concentrated kernels can generalize poorly; see the analysis of [quantum-kernel concentration](https://www.nature.com/articles/s41467-024-49287-w).
- The finite dataset yields uncertain estimates. Review per-fold values and variability; do not report a single split as proof of generalization.
- Statevector simulation is noiseless. Results do not predict hardware performance.

## Files

- `qsvm_heart_disease_tuned.py`: nested evaluation, training, reporting, and `predict_frame` helper.
- `qsvm_heart_disease.py`: original single-split baseline for historical comparison.
- `project.md`: original project write-up and pre-revision benchmark results; those results are historical and are not the current nested-CV evaluation.
- `improvements.md`: review plan and rationale for the changes.
