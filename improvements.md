# QSVM / QML Model Improvement Plan

This review documented the initial implementation in `qsvm_heart_disease_tuned.py` and `qsvm_heart_disease.py`. The recommendations below have now been implemented where applicable; see the completion notes at the end and the generated `outputs/review_report.md` for the trained run.

## Executive summary

The tuned script already has several useful foundations: it keeps preprocessing inside each training fold, builds an exact statevector fidelity kernel, reuses each fold's kernel matrices across the tested values of `C`, and holds a test split out of the tuning loop. For the current 297-row dataset, kernel computation is likely not the biggest risk.

The highest-value improvements are to make the evaluation statistically sound and reproducible, align the data representation with the clinical feature types, and make the documented behavior match the actual scripts. After that, optimize repeated preprocessing/statevector work and make execution/configuration more robust.

## Current model at a glance

- `qsvm_heart_disease_tuned.py` evaluates a fixed 75/25 split (seed 42), then searches a 128-configuration grid using 5-fold stratified CV on the training partition. The grid covers reducer (`mi` or `pca`), qubit count (4 or 6), reps (1 or 2), entanglement (`linear` or `full`), scale (1 or π), and `C` (0.1, 1, 10, 100).
- Each fold fits `StandardScaler`, a reducer, and `MinMaxScaler` on that fold's training data. It then computes statevectors and the train/validation Gram matrices. Each circuit configuration's Gram matrices are reused over its four `C` values.
- The chosen configuration is evaluated on the held-out split. A separate OOF pass selects an F2 threshold; the same numeric threshold is then applied to the final model and reused in the multi-seed summary.
- `qsvm_heart_disease.py` is a simpler PCA + `FidelityQuantumKernel` baseline. It runs at import time, samples at most 200 rows, and uses default SVC settings for its classical comparison.
- Checked-in results show the classical RBF-SVM on the same reduced features ahead of QSVM on the one test split (AUC 0.939 vs 0.865; accuracy 0.88 vs 0.76 for the tuned-threshold QSVM). Treat that as evidence to investigate, not evidence of quantum advantage.

## Prioritized recommendations

### P0 — Fix evaluation design before optimizing the search

#### 1. Use nested validation for model selection and threshold selection

**Current concern:** The 128 candidates are ranked on the same five folds, and the winning configuration's threshold is selected from OOF scores on those same training rows. This makes the CV score and selected threshold optimistic after searching many alternatives. The final held-out test remains useful for this one experiment, but it is a single small split, and its score is also used in the project's conclusions.

**Recommended change:** Use an outer stratified CV loop to estimate generalization. For each outer training partition, perform all feature/reducer, circuit, `C`, and threshold selection using inner CV only, then score once on the outer validation partition. Report outer-fold results (mean, standard deviation, and per-fold values). Keep a final untouched test set only if it is needed for a final demonstration; do not use it to choose settings or make repeated decisions.

For the current small dataset, repeated nested CV may be more informative than a single 75/25 split. A less expensive interim option is to keep the current holdout but describe the CV search score as selection data and report the holdout result as the primary estimate, with uncertainty intervals.

#### 2. Avoid comparing raw SVC scores from different folds to choose one threshold

**Current concern:** `optimize_threshold_f2` concatenates decision scores from independently fitted fold SVCs and searches one threshold across them. SVC decision-function magnitudes can vary with each fold's support vectors and margin geometry. A single absolute threshold across those fold models need not transfer to the final model fit on all training data. The 200-point percentile grid can also miss the exact threshold that maximizes F2.

**Recommended change:** Select the threshold inside each outer training set, then apply it to that outer model's validation scores. Use thresholds derived from sorted unique score values (or a metric helper that evaluates all relevant cut points) rather than an arbitrary 200-point grid. If a single deployed threshold is required, calibrate scores with a nested calibration procedure (for example, sigmoid/isotonic calibration where sample size supports it) and document the procedure. Consider a clinically specified sensitivity target with a corresponding specificity estimate instead of optimizing F2 alone.

#### 3. Make the clinical claim match the evidence

The tuned threshold increases sensitivity on the reported holdout but also lowers specificity substantially (0.575 in the saved table). Describe this as a sensitivity/specificity tradeoff on this dataset, not as demonstrated clinical utility or a diagnostic deployment result. Include confidence intervals for sensitivity, specificity, and AUC; with only 75 test cases, point estimates are noisy. Report that the classical RBF baseline is stronger on the saved comparison.

### P1 — Improve feature handling and reproducibility

#### 4. Treat categorical and ordinal variables deliberately

**Current concern:** All 13 inputs are standardized as numeric. `mutual_info_classif` is called without `discrete_features`, so the dense input is treated as continuous. Several Cleveland variables are categorical or ordinal codes (`cp`, `restecg`, `slope`, `thal`, and binary indicators). Scaling those category codes as if equal numeric gaps had continuous meaning can change MI ranking and the geometry seen by both kernels.

**Recommended change:** Define the feature schema explicitly. Decide whether each field is nominal, ordinal, binary, or continuous; encode nominal categories consistently (for example, one-hot encoding) and mark discrete variables for mutual information. For a compact quantum map, compare principled encodings or domain-preserving feature selection against the current ordinal-value encoding. Keep every encoder fitted inside the relevant CV training fold. Record the feature order and encoding in the exported model metadata.

#### 5. Seed every stochastic operation and record the run configuration

`mutual_info_classif` currently has no explicit random state. Set a configured seed for it (or a fold-specific deterministic seed) so tied/noisy MI scores do not vary between runs. Centralize seeds, folds, grids, and scoring choices in one configuration object/CLI, and include them in `best_config.json` and each result export. Record Python, Qiskit, scikit-learn, NumPy, and dataset version/hash. Avoid blanket warning suppression while tuning; surface relevant warnings and suppress only known, documented noise.

#### 6. Validate dataset schema and numeric values on load

Both scripts assume the target is the last column and convert every target with `> 0`; this is appropriate for the Cleveland 0–4 target but is unsafe for arbitrary custom CSVs. The tuned script silently ignores a non-existent `--dataset-path` and falls back to another data source. Add explicit validation for required feature columns, target column/name, numeric conversion, missing values, binary label distribution, and supported custom-data label conventions. Fail with a direct error instead of silently using a different dataset. Keep dataset source and dropped-row counts with the output artifacts.

### P1 — Reduce wasted computation without changing the kernel

#### 7. Reuse preprocessing and statevectors across related grid configurations

The script already avoids recomputing kernels for the four `C` values of one circuit configuration. However, for a given fold, the same transformed data is independently recalculated for combinations that differ only in `reps` or `entanglement`. Preprocessing depends on reducer, qubits, scale, and fold; statevectors then depend on those transformed features plus reps and entanglement.

**Recommended change:** Refactor a fold-level evaluator that prepares transformed train/validation arrays once per `(fold, reducer, qubits, scale)`, then memoizes statevectors and kernels for each circuit key. Keep cache lifetime bounded to one worker or one fold to avoid unbounded memory. If parallelizing, dispatch coarse groups that can reuse their prepared arrays; benchmark process serialization and memory use against the current `Parallel(n_jobs=-1)` implementation.

#### 8. Measure the real bottleneck before adding parallelism or persistent caches

Capture timings for preprocessing, circuit construction/binding, statevector simulation, Gram multiplication, and SVC fitting separately. The current 297-row problem has small kernels, so process start-up and copying arrays may take a meaningful share of runtime. Compare serial, a bounded worker count, and all-core execution. Avoid nested parallelism if outer configuration parallelism and any lower-level BLAS threading are both active.

Persistent kernel caching is not implemented in the tuned script despite README claims and the presence of a `kernel_cache/` directory. If repeated runs justify it, cache deterministic, versioned intermediate results using a key that includes transformed data bytes, fold/split identity, feature-map definition, Qiskit version, and precision. Write atomically, validate shape/dtype on read, and offer a cache-disable/clear option. Otherwise remove the stale cache claim and artifacts rather than maintaining complexity for a sub-minute workload.

#### 9. Bound memory and avoid unnecessary dense matrices for larger datasets

Exact statevectors require memory proportional to samples × `2**qubits`; dense Gram matrices require quadratic memory in sample count. This is fine for six qubits and 297 rows, but can become the limit for larger external datasets or expanded CV. Add estimates/logging before computation and process validation kernels in row batches when appropriate. For larger datasets, compare approximate quantum-kernel methods (such as Nyström/landmark approximations) against the exact kernel and clearly label them as approximations. Preserve an exact mode as the reference.

#### 10. Make statevector parameter binding explicit and defensive

`compute_statevectors` sorts circuit parameters by parsing their names and zips them to each row. This relies on Qiskit's current parameter naming and silently truncates if dimensions differ. Validate feature width against `n_qubits` and map parameters using an explicit, tested feature-index convention. Reuse the parameterized circuit within a batch as it currently does, and profile binding/simulation before considering lower-level optimizations.

### P2 — Improve model search quality and reporting

#### 11. Tune against the intended objective and use a comparable budget

The quantum search chooses by mean fold ROC-AUC, then tunes F2 after selection. This may not match a screening objective. Predeclare the primary metric and constraints (for example, maximize specificity subject to a minimum sensitivity) and apply the same selection rule to quantum and classical models. Expand the classical baseline's search and validation design consistently; report compute budget and search spaces for both. Include simple baselines (majority, logistic regression, linear SVM, and RBF-SVM) where relevant.

#### 12. Check kernel quality and numerical properties

Export diagnostics per selected configuration: diagonal deviation from 1, symmetry error, minimum eigenvalue (allowing for floating-point tolerance), off-diagonal distribution, effective rank/condition number, and class-conditional kernel similarity. These help identify uninformative/concentrated kernels and catch matrix construction issues. Verify that train kernels are square and validation-kernel columns match training rows before SVC fitting. Keep any correction for small negative eigenvalues explicit; do not silently alter a kernel.

#### 13. Expand robustness analysis beyond fixed hyperparameters on new splits

The current multi-seed block applies the seed-42 configuration and seed-42 threshold to five different train/test splits. The source comment acknowledges fixed hyperparameters, but it also reuses a threshold from another training fit. Label this as a fixed-configuration sensitivity analysis, not a full repeated evaluation of the modeling procedure. For an unbiased procedure-level estimate, repeat model selection and threshold calibration within each split's training data. Report per-seed raw metrics and confidence intervals, not only formatted mean ± standard deviation strings.

## Code and project organization

1. Split the tuned script into import-safe modules: data loading/schema validation, preprocessing, feature map/kernel operations, search/evaluation, reporting, and CLI. Keep a single orchestration entry point.
2. Add a `main()` guard to `qsvm_heart_disease.py`; currently importing it launches data loading, training, reporting, and custom inference.
3. Use paths rooted at the script/project directory or require an explicit output directory. Current `Path("outputs")` and dataset paths depend on the shell's current working directory.
4. Add a dependency lock or `requirements.txt`/`pyproject.toml` with tested Python and library versions. The README currently gives inconsistent Python versions and dependency lists.
5. Reconcile docs with implementation. The README advertises Optuna, `--tune`, `--trials`, circular entanglement, a continuous `scale_hi`, and persistent SHA-256 cache, while the tuned script implements a fixed grid, accepts only `--quick` and `--dataset-path`, uses linear/full entanglement and scales 1/π, and has no kernel cache code. The README's stated best settings also conflict with the checked-in `best_config.json` and tuning table. Make one source of truth and regenerate result documentation from saved run metadata.
6. Keep generated outputs separate from source and label them with run IDs/configuration. Avoid treating checked-in CSV/PNG outputs as current until the configuration and dataset hash match.

## Suggested implementation sequence

1. **Reproducibility/data integrity:** explicit schema, seed control, run metadata, path handling, and correct documentation.
2. **Evaluation correctness:** nested/repeated evaluation and threshold selection confined to training folds; confidence intervals and balanced quantum/classical protocol.
3. **Feature representation:** categorical-aware preprocessing/MI and comparison of encodings.
4. **Profiling and reuse:** timings, fold preprocessing reuse, bounded statevector/kernel caches, controlled parallelism.
5. **Scale-up options:** batched kernel construction and approximate kernels only when dataset size makes exact simulation costly.

## Review checklist for a future implementation

- A held-out outer fold/test observation is never used to choose features, circuit settings, `C`, threshold, or encoding.
- Quantum and classical models use the same outer splits and predeclared primary metric.
- Categorical handling and feature order are recorded and fitted within each training fold.
- Repeated runs with the same configuration and seed reproduce the selected parameters and results within expected numerical tolerance.
- Kernel shape, symmetry, diagonal, and PSD diagnostics are recorded for the selected model.
- Runtime and peak memory are measured before/after compute optimizations on the same machine and dataset.
- Result files identify the data, software versions, split strategy, seed, search space, and model-selection procedure that produced them.

## Implementation status

Implemented in the current revision:

- Nested 5-fold outer/inner model selection and threshold calibration; per-fold metrics are exported.
- Fold-local continuous scaling, one-hot encoding for the known Cleveland categorical fields, explicit discrete-feature MI scoring, and deterministic MI seeds.
- Same-outer-fold classical RBF comparisons on all encoded inputs and on the selected reduced representation.
- Local projected quantum kernels using both one- and two-qubit Pauli-observable features, with CV-selected bandwidth multipliers, searched alongside the fidelity kernel; the circuit search now includes 8 qubits.
- Fold-level reuse of transformations/statevectors, bounded four-worker parallel evaluation, normalized margin thresholds, and kernel numerical diagnostics.
- Dataset/schema/value checks, script-relative paths, import-safe entry points, pinned package versions, dataset hash and run metadata, a serialized full-data model, and a reusable inference helper.
- Updated README and historical labeling for the previous single-split write-up/results.

The current implementation uses one 5-fold outer repetition (5×5 nested CV); repeated nested repetitions remain a useful next step if a tighter estimate is needed. Persistent disk kernel caching and approximate kernels were not added: the exact full search completed in 271 seconds on the final run, and no larger dataset currently justifies the added cache/approximation complexity. The fitted bundle is for research review; its scores are standardized margins rather than probabilities.

Follow-up full search with 8-qubit circuits and one-/two-qubit projected kernels completed in 526 seconds. The selected outer-fold procedure achieved mean ROC-AUC 0.916 and accuracy 0.778 versus 0.907 and 0.747 for the all-feature classical RBF reference. This is a favorable point estimate, but not a robust advantage claim: there are five outer folds, results vary materially by fold, and the full-data projected kernel has effective rank about 1.73 with off-diagonal mean about 0.90.
