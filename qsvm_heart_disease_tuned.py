"""
Quantum Support Vector Machine (QSVM) on UCI Heart Disease (Cleveland, ID=45)
=============================================================================
An academic-grade, leak-free, cross-validated and grid-tuned implementation
comparing a Fidelity Quantum Kernel against tuned classical RBF-SVM baselines.

Addresses research gaps from:
  - Munir et al., AI (MDPI) 2026, 7(5), 180 (no feature map tuning, no threshold tuning)
  - Kumar et al., Sci Rep 2025; Rawat et al., NMAHIB 2025; Ghisoni et al., Sci Rep 2026

Features:
  1. No subsampling: evaluates full 297 cleaned patient records.
  2. Strict train/test isolation: 75/25 stratified split (seed 42).
  3. Preprocessing fit strictly inside folds (StandardScaler -> Reducer -> MinMaxScaler).
  4. Reducers: SelectKBest(mutual_info_classif) vs PCA.
  5. Fast exact statevector quantum kernel: K = |A @ B.conj().T|**2 using zz_feature_map.
  6. Comprehensive grid search over (reducer, qubits, reps, entanglement, scale, C).
  7. Out-Of-Fold (OOF) decision threshold tuning maximizing clinical F2 score (recall-weighted).
  8. Two GridSearchCV-tuned classical RBF-SVM baselines (all 13 features vs same reduced input).
  9. Multi-seed robustness evaluation across 5 independent seeds (mean +/- std).
  10. Automated generation of all academic figures, tables, and summary CSVs in outputs/.
"""
from __future__ import annotations

import os
import sys
import json
import time
import argparse
import warnings
from pathlib import Path

# Safe terminal encoding on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Non-interactive matplotlib backend for headless figure generation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
# pyrefly: ignore [missing-import]
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, fbeta_score
)
from joblib import Parallel, delayed

from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ===========================================================================
# 1. Statevector & Quantum Kernel Utilities
# ===========================================================================
def compute_statevectors(X: np.ndarray, n_qubits: int, reps: int, entanglement: str) -> np.ndarray:
    """
    Compute exact noiseless statevectors for input array X.
    Deterministic parameter assignment guarantees feature-to-qubit alignment.
    """
    qc = zz_feature_map(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)
    sorted_params = sorted(qc.parameters, key=lambda p: int(p.name.split('[')[1].rstrip(']')))
    
    states = []
    for row in X:
        param_dict = dict(zip(sorted_params, row))
        bound_qc = qc.assign_parameters(param_dict)
        sv = Statevector(bound_qc)
        states.append(sv.data)
    return np.array(states)


def compute_kernel_matrix(states_a: np.ndarray, states_b: np.ndarray | None = None) -> np.ndarray:
    """
    Compute Fidelity Quantum Kernel matrix from statevectors:
      K_ij = |<psi(x_i)|psi(x_j)>|^2 = |A @ B.conj().T|**2
    Mathematically identical to FidelityQuantumKernel on noiseless statevector,
    but orders of magnitude faster.
    """
    if states_b is None:
        return np.abs(states_a @ states_a.conj().T) ** 2
    return np.abs(states_a @ states_b.conj().T) ** 2


# ===========================================================================
# 2. Preprocessing Pipeline Builder
# ===========================================================================
def build_preprocessed_fold(X_tr: np.ndarray, y_tr: np.ndarray, X_val: np.ndarray,
                           reducer_type: str, n_qubits: int, scale: float):
    """
    Fit StandardScaler, Reducer, and MinMaxScaler strictly on training fold data.
    Transform validation/test fold without any data leakage.
    """
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr)
    X_val_s = scaler.transform(X_val)

    if reducer_type == "mi":
        reducer = SelectKBest(score_func=mutual_info_classif, k=n_qubits)
        reducer.fit(X_tr_s, y_tr)
    elif reducer_type == "pca":
        reducer = PCA(n_components=n_qubits, random_state=42)
        reducer.fit(X_tr_s)
    else:
        raise ValueError(f"Unknown reducer: {reducer_type}")

    X_tr_r = reducer.transform(X_tr_s)
    X_val_r = reducer.transform(X_val_s)

    minmax = MinMaxScaler(feature_range=(0, scale)).fit(X_tr_r)
    X_tr_q = minmax.transform(X_tr_r)
    X_val_q = minmax.transform(X_val_r)

    return X_tr_q, X_val_q, scaler, reducer, minmax


# ===========================================================================
# 3. Grid Evaluation for a Single Circuit Config (Across All C Values)
# ===========================================================================
def evaluate_circuit_config_on_cv(circuit_cfg: dict, X_train: np.ndarray, y_train: np.ndarray,
                                  c_values: list[float], n_folds: int = 5) -> list[dict]:
    """
    For a given (reducer, qubits, reps, entanglement, scale) circuit configuration:
    Compute 5-fold CV states once, then instantly evaluate all C values.
    Returns list of result dictionaries (one per C value).
    """
    reducer_type = circuit_cfg["reducer"]
    n_qubits = circuit_cfg["qubits"]
    reps = circuit_cfg["reps"]
    entanglement = circuit_cfg["entanglement"]
    scale = circuit_cfg["scale"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Precompute fold kernels once
    fold_kernels = []
    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        X_tr_q, X_val_q, _, _, _ = build_preprocessed_fold(
            X_tr, y_tr, X_val, reducer_type, n_qubits, scale
        )
        states_tr = compute_statevectors(X_tr_q, n_qubits, reps, entanglement)
        states_val = compute_statevectors(X_val_q, n_qubits, reps, entanglement)

        K_tr = compute_kernel_matrix(states_tr)
        K_val = compute_kernel_matrix(states_val, states_tr)
        fold_kernels.append((K_tr, y_tr, K_val, y_val, val_idx))

    # Evaluate each C value against precomputed kernels
    results = []
    for C in c_values:
        fold_aucs = []
        oof_scores = np.zeros(len(X_train))

        for K_tr, y_tr, K_val, y_val, val_idx in fold_kernels:
            clf = SVC(kernel="precomputed", C=C)
            clf.fit(K_tr, y_tr)
            val_scores = clf.decision_function(K_val)
            oof_scores[val_idx] = val_scores
            fold_aucs.append(roc_auc_score(y_val, val_scores))

        mean_auc = float(np.mean(fold_aucs))
        std_auc = float(np.std(fold_aucs))
        oof_auc = float(roc_auc_score(y_train, oof_scores))

        entry = {
            "reducer": reducer_type,
            "qubits": n_qubits,
            "reps": reps,
            "entanglement": entanglement,
            "scale": scale,
            "C": C,
            "mean_cv_roc_auc": mean_auc,
            "std_cv_roc_auc": std_auc,
            "oof_roc_auc": oof_auc,
        }
        results.append(entry)

    return results


# ===========================================================================
# 4. Out-Of-Fold F2 Decision Threshold Optimization
# ===========================================================================
def optimize_threshold_f2(best_cfg: dict, X_train: np.ndarray, y_train: np.ndarray, n_folds: int = 5):
    """
    Perform 5-fold CV using best hyperparameters to gather out-of-fold continuous
    decision scores, then find threshold tau maximizing clinical F2 score (recall-weighted).
    Returns (best_threshold, best_f2, oof_scores).
    """
    reducer_type = best_cfg["reducer"]
    n_qubits = best_cfg["qubits"]
    reps = best_cfg["reps"]
    entanglement = best_cfg["entanglement"]
    scale = best_cfg["scale"]
    C = best_cfg["C"]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_scores = np.zeros(len(X_train))

    for tr_idx, val_idx in skf.split(X_train, y_train):
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        X_tr_q, X_val_q, _, _, _ = build_preprocessed_fold(
            X_tr, y_tr, X_val, reducer_type, n_qubits, scale
        )
        states_tr = compute_statevectors(X_tr_q, n_qubits, reps, entanglement)
        states_val = compute_statevectors(X_val_q, n_qubits, reps, entanglement)

        K_tr = compute_kernel_matrix(states_tr)
        K_val = compute_kernel_matrix(states_val, states_tr)

        clf = SVC(kernel="precomputed", C=C)
        clf.fit(K_tr, y_tr)
        oof_scores[val_idx] = clf.decision_function(K_val)

    # Search thresholds across distribution percentiles (including 0.0)
    percentiles = np.linspace(np.percentile(oof_scores, 2), np.percentile(oof_scores, 98), 200)
    candidate_thresholds = np.unique(np.concatenate([percentiles, [0.0]]))

    best_tau = 0.0
    best_f2 = -1.0
    for tau in candidate_thresholds:
        preds = (oof_scores >= tau).astype(int)
        f2 = fbeta_score(y_train, preds, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_tau = float(tau)

    return best_tau, best_f2, oof_scores


# ===========================================================================
# 5. Metric Calculation Helper
# ===========================================================================
def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    try:
        auc = roc_auc_score(y_true, y_score)
    except Exception:
        auc = float("nan")

    return {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "Specificity": float(spec),
        "F1": float(f1),
        "ROC-AUC": float(auc),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# ===========================================================================
# 6. Main Execution Pipeline
# ===========================================================================
def run_pipeline(quick: bool = False, dataset_path: str | None = None):
    t_start = time.time()
    outputs_dir = Path("outputs")
    outputs_dir.mkdir(exist_ok=True)

    print("=" * 78)
    print("  QUANTUM SUPPORT VECTOR MACHINE (QSVM) - HEART DISEASE PREDICTION")
    print("  UCI Cleveland Dataset (id=45) | Academic Benchmark & Tuning Pipeline")
    print("=" * 78)
    print(f"Python      : {sys.version.split()[0]}")
    import qiskit, sklearn
    print(f"Qiskit      : {qiskit.__version__}")
    print(f"Scikit-Learn: {sklearn.__version__}\n")

    # -----------------------------------------------------------------------
    # Step A: Data Loading & Preprocessing
    # -----------------------------------------------------------------------
    local_csv_path = Path("heart_disease_cleveland.csv")

    if dataset_path and Path(dataset_path).exists():
        print(f"Loading custom dataset from: {dataset_path}")
        df_raw = pd.read_csv(dataset_path)
        target_col = df_raw.columns[-1]
        df = df_raw.dropna().copy()
        df[target_col] = (df[target_col] > 0).astype(int)
        feature_names = [c for c in df.columns if c != target_col]
        X = df[feature_names].values
        y = df[target_col].values
    elif local_csv_path.exists():
        print(f"Loading cached Cleveland dataset from local file: {local_csv_path} ...")
        df_raw = pd.read_csv(local_csv_path)
        target_col = df_raw.columns[-1]
        df = df_raw.dropna().copy()
        df[target_col] = (df[target_col] > 0).astype(int)
        feature_names = [c for c in df.columns if c != target_col]
        X = df[feature_names].values
        y = df[target_col].values
    else:
        print("Fetching UCI Heart Disease dataset (id=45) via ucimlrepo ...")
        try:
            heart = fetch_ucirepo(id=45)
            X_df = heart.data.features.copy()
            y_df = heart.data.targets.copy()
            target_col = y_df.columns[0]
            df_merged = pd.concat([X_df, y_df], axis=1).dropna()
            df_merged.to_csv(local_csv_path, index=False)
            df_merged[target_col] = (df_merged[target_col] > 0).astype(int)
            feature_names = [c for c in df_merged.columns if c != target_col]
            X = df_merged[feature_names].values
            y = df_merged[target_col].values
        except Exception as e:
            print(f"[!] Warning: ucimlrepo network request failed ({e}).")
            print("    Attempting direct download fallback ...")
            try:
                import urllib.request
                url = "https://archive.ics.uci.edu/static/public/45/data.csv"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    df_raw = pd.read_csv(resp)
                    df_raw.to_csv(local_csv_path, index=False)
                target_col = df_raw.columns[-1]
                df = df_raw.dropna().copy()
                df[target_col] = (df[target_col] > 0).astype(int)
                feature_names = [c for c in df.columns if c != target_col]
                X = df[feature_names].values
                y = df[target_col].values
            except Exception as e2:
                raise RuntimeError(
                    f"Unable to fetch UCI dataset due to network connection reset ({e2}). "
                    f"Please verify internet connection or place 'heart_disease_cleveland.csv' in current directory."
                )

    n_samples, n_features = X.shape
    class_counts = dict(zip(*np.unique(y, return_counts=True)))
    print(f"Samples (Cleaned) : {n_samples} (Dropped rows with missing ca/thal)")
    print(f"Features          : {n_features}")
    print(f"Class Balance     : {class_counts} [0: No Disease, 1: Disease]\n")

    # Stratified 75/25 Train/Test Split (Fixed seed 42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=42
    )
    print(f"Train samples     : {len(X_train)} (Class 0: {sum(y_train==0)}, Class 1: {sum(y_train==1)})")
    print(f"Test samples      : {len(X_test)}  (Class 0: {sum(y_test==0)}, Class 1: {sum(y_test==1)})\n")

    # -----------------------------------------------------------------------
    # Step B: Hyperparameter Tuning Grid (5-Fold Stratified CV on X_train)
    # -----------------------------------------------------------------------
    if quick:
        print("[QUICK MODE ACTIVATED: Evaluating minimal parameter grid]")
        reducers = ["mi"]
        qubits_list = [4]
        reps_list = [1]
        entanglements = ["linear"]
        scales = [np.pi]
        c_values = [1.0, 10.0]
    else:
        print("[FULL GRID MODE: 32 Circuit Configurations x 4 C Values = 128 Combinations]")
        reducers = ["mi", "pca"]
        qubits_list = [4, 6]
        reps_list = [1, 2]
        entanglements = ["linear", "full"]
        scales = [1.0, np.pi]
        c_values = [0.1, 1.0, 10.0, 100.0]

    circuit_configs = []
    for red in reducers:
        for q in qubits_list:
            for r in reps_list:
                for ent in entanglements:
                    for s in scales:
                        circuit_configs.append({
                            "reducer": red,
                            "qubits": q,
                            "reps": r,
                            "entanglement": ent,
                            "scale": s,
                        })

    print(f"Dispatching {len(circuit_configs)} circuit configurations across CPU cores using joblib ...")
    t_grid_start = time.time()
    parallel_results = Parallel(n_jobs=-1, verbose=1)(
        delayed(evaluate_circuit_config_on_cv)(cfg, X_train, y_train, c_values, n_folds=5)
        for cfg in circuit_configs
    )
    t_grid_end = time.time()
    print(f"Grid search completed in {t_grid_end - t_grid_start:.2f} seconds.\n")

    # Flatten all results into a DataFrame
    all_grid_rows = [row for batch in parallel_results for row in batch]
    df_tuning = pd.DataFrame(all_grid_rows)
    df_tuning.sort_values(by="mean_cv_roc_auc", ascending=False, inplace=True)
    df_tuning.reset_index(drop=True, inplace=True)

    # Save full tuning results
    df_tuning.to_csv(outputs_dir / "qsvm_tuning_results.csv", index=False)
    df_tuning.to_csv("qsvm_tuning_results.csv", index=False)  # also at root as requested

    top5_df = df_tuning.head(5)
    top5_df.to_csv(outputs_dir / "top5_tuning_configs.csv", index=False)
    print("=" * 78)
    print("  TOP 5 TUNING CONFIGURATIONS (Sorted by 5-Fold CV ROC-AUC)")
    print("=" * 78)
    print(top5_df.to_string(index=False))
    print()

    best_cfg = df_tuning.iloc[0].to_dict()
    print(f"[*] BEST QSVM CONFIGURATION:")
    for k, v in best_cfg.items():
        print(f"    {k:<16s}: {v}")
    print()

    # -----------------------------------------------------------------------
    # Step C: Out-Of-Fold (OOF) Decision Threshold Optimization (Recall-Weighted F2)
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("  OUT-OF-FOLD (OOF) DECISION THRESHOLD TUNING (F2 SCORE)")
    print("=" * 78)
    tuned_threshold, oof_f2, oof_scores = optimize_threshold_f2(best_cfg, X_train, y_train, n_folds=5)
    default_f2 = fbeta_score(y_train, (oof_scores >= 0.0).astype(int), beta=2, zero_division=0)
    print(f"OOF Decision Score Range : [{oof_scores.min():.4f}, {oof_scores.max():.4f}]")
    print(f"Default Threshold (0.00) : OOF F2 = {default_f2:.4f}")
    print(f"Tuned Threshold   ({tuned_threshold:+.4f}) : OOF F2 = {oof_f2:.4f}\n")

    # Save best config JSON
    best_config_export = {
        "reducer": str(best_cfg["reducer"]),
        "qubits": int(best_cfg["qubits"]),
        "reps": int(best_cfg["reps"]),
        "entanglement": str(best_cfg["entanglement"]),
        "scale": float(best_cfg["scale"]),
        "C": float(best_cfg["C"]),
        "mean_cv_roc_auc": float(best_cfg["mean_cv_roc_auc"]),
        "tuned_threshold": float(tuned_threshold),
        "oof_f2_score": float(oof_f2),
    }
    with open(outputs_dir / "best_config.json", "w") as f:
        json.dump(best_config_export, f, indent=4)

    # -----------------------------------------------------------------------
    # Step D: Final Training & Test Evaluation of QSVM (Threshold 0.0 vs Tuned)
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("  FINAL QSVM EVALUATION ON TEST SET (25% Unseen Samples)")
    print("=" * 78)
    # Fit preprocessing pipeline on entire X_train, transform X_test
    X_tr_q, X_te_q, scaler_final, reducer_final, minmax_final = build_preprocessed_fold(
        X_train, y_train, X_test,
        best_cfg["reducer"], int(best_cfg["qubits"]), float(best_cfg["scale"])
    )
    states_train = compute_statevectors(
        X_tr_q, int(best_cfg["qubits"]), int(best_cfg["reps"]), str(best_cfg["entanglement"])
    )
    states_test = compute_statevectors(
        X_te_q, int(best_cfg["qubits"]), int(best_cfg["reps"]), str(best_cfg["entanglement"])
    )
    K_train_final = compute_kernel_matrix(states_train)
    K_test_final = compute_kernel_matrix(states_test, states_train)

    qsvm_final = SVC(kernel="precomputed", C=float(best_cfg["C"]))
    qsvm_final.fit(K_train_final, y_train)

    qsvm_test_scores = qsvm_final.decision_function(K_test_final)
    qsvm_test_pred_default = (qsvm_test_scores >= 0.0).astype(int)
    qsvm_test_pred_tuned = (qsvm_test_scores >= tuned_threshold).astype(int)

    metrics_qsvm_default = compute_all_metrics(y_test, qsvm_test_pred_default, qsvm_test_scores)
    metrics_qsvm_tuned = compute_all_metrics(y_test, qsvm_test_pred_tuned, qsvm_test_scores)

    # -----------------------------------------------------------------------
    # Step E: Tuned Classical Baselines (GridSearchCV on X_train)
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("  TUNED CLASSICAL RBF-SVM BASELINES")
    print("=" * 78)
    param_grid_rbf = {
        "svm__C": [0.1, 1.0, 10.0, 100.0],
        "svm__gamma": ["scale", "auto", 0.01, 0.1, 1.0],
    }
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Baseline A: All 13 features
    pipe_baseline_all = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf"))
    ])
    grid_all = GridSearchCV(pipe_baseline_all, param_grid_rbf, cv=cv_strategy, scoring="roc_auc", n_jobs=-1)
    grid_all.fit(X_train, y_train)
    baseline_all_scores = grid_all.best_estimator_.decision_function(X_test)
    baseline_all_preds = grid_all.best_estimator_.predict(X_test)
    metrics_baseline_all = compute_all_metrics(y_test, baseline_all_preds, baseline_all_scores)
    print(f"Baseline A (All 13 features)     : Best params = {grid_all.best_params_} | CV AUC = {grid_all.best_score_:.4f}")

    # Baseline B: Reduced input matching QSVM
    if best_cfg["reducer"] == "mi":
        red_b = SelectKBest(score_func=mutual_info_classif, k=int(best_cfg["qubits"]))
    else:
        red_b = PCA(n_components=int(best_cfg["qubits"]), random_state=42)

    pipe_baseline_reduced = Pipeline([
        ("scaler", StandardScaler()),
        ("reducer", red_b),
        ("minmax", MinMaxScaler(feature_range=(0, float(best_cfg["scale"])))),
        ("svm", SVC(kernel="rbf"))
    ])
    grid_reduced = GridSearchCV(pipe_baseline_reduced, param_grid_rbf, cv=cv_strategy, scoring="roc_auc", n_jobs=-1)
    grid_reduced.fit(X_train, y_train)
    baseline_reduced_scores = grid_reduced.best_estimator_.decision_function(X_test)
    baseline_reduced_preds = grid_reduced.best_estimator_.predict(X_test)
    metrics_baseline_reduced = compute_all_metrics(y_test, baseline_reduced_preds, baseline_reduced_scores)
    print(f"Baseline B (Reduced {best_cfg['qubits']} features): Best params = {grid_reduced.best_params_} | CV AUC = {grid_reduced.best_score_:.4f}\n")

    # -----------------------------------------------------------------------
    # Step F: Compile Results Table
    # -----------------------------------------------------------------------
    results_rows = [
        {"Model": "QSVM (Default Thr = 0.0)", "Threshold": 0.0, **metrics_qsvm_default},
        {"Model": f"QSVM (Tuned Thr = {tuned_threshold:+.4f})", "Threshold": tuned_threshold, **metrics_qsvm_tuned},
        {"Model": "Classical RBF-SVM (All 13 Features)", "Threshold": 0.0, **metrics_baseline_all},
        {"Model": f"Classical RBF-SVM ({best_cfg['qubits']} Features)", "Threshold": 0.0, **metrics_baseline_reduced},
    ]
    df_results = pd.DataFrame(results_rows)
    df_results.to_csv(outputs_dir / "results_table.csv", index=False)

    print("=" * 78)
    print("  TEST BENCHMARK RESULTS TABLE")
    print("=" * 78)
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC", "TN", "FP", "FN", "TP"]
    print(df_results[display_cols].to_string(index=False))
    print()

    # -----------------------------------------------------------------------
    # Step G: Generate Figures (Confusion Matrix & ROC Curves)
    # -----------------------------------------------------------------------
    # 1. Confusion Matrix for QSVM (Tuned Threshold)
    fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, qsvm_test_pred_tuned, labels=[0, 1])
    cax = ax_cm.matshow(cm, cmap="Blues", alpha=0.85)
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            label = f"{val}\n({'TN' if (i==0 and j==0) else 'FP' if (i==0 and j==1) else 'FN' if (i==1 and j==0) else 'TP'})"
            ax_cm.text(j, i, label, ha="center", va="center", fontsize=12, fontweight="bold",
                       color="black" if cm[i, j] < cm.max()/2 else "white")
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(["No Disease (0)", "Disease (1)"])
    ax_cm.set_yticklabels(["No Disease (0)", "Disease (1)"])
    ax_cm.set_xlabel("Predicted Label", fontsize=11, fontweight="bold")
    ax_cm.set_ylabel("True Label", fontsize=11, fontweight="bold")
    ax_cm.set_title(f"QSVM Confusion Matrix (Tuned Thr = {tuned_threshold:+.3f})\n"
                    f"Recall={metrics_qsvm_tuned['Recall']:.2%}, Specificity={metrics_qsvm_tuned['Specificity']:.2%}",
                    fontsize=12, pad=15)
    plt.colorbar(cax, ax=ax_cm, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig_cm.savefig(outputs_dir / "confusion_matrix_qsvm_tuned.png", dpi=300)
    plt.close(fig_cm)
    print(f"[*] Saved confusion matrix plot to: {outputs_dir / 'confusion_matrix_qsvm_tuned.png'}")

    # 2. Combined ROC Curves
    fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
    fpr_q, tpr_q, _ = roc_curve(y_test, qsvm_test_scores)
    fpr_b1, tpr_b1, _ = roc_curve(y_test, baseline_all_scores)
    fpr_b2, tpr_b2, _ = roc_curve(y_test, baseline_reduced_scores)

    ax_roc.plot(fpr_q, tpr_q, color="#1f77b4", lw=2.5,
                label=f"QSVM ({best_cfg['qubits']}q, {best_cfg['reducer'].upper()}) [AUC = {metrics_qsvm_tuned['ROC-AUC']:.4f}]")
    ax_roc.plot(fpr_b1, tpr_b1, color="#2ca02c", lw=2, linestyle="--",
                label=f"Classical RBF-SVM (All 13) [AUC = {metrics_baseline_all['ROC-AUC']:.4f}]")
    ax_roc.plot(fpr_b2, tpr_b2, color="#ff7f0e", lw=2, linestyle="-.",
                label=f"Classical RBF-SVM ({best_cfg['qubits']} feats) [AUC = {metrics_baseline_reduced['ROC-AUC']:.4f}]")
    ax_roc.plot([0, 1], [0, 1], color="grey", lw=1, linestyle=":", label="Chance (AUC = 0.5000)")

    ax_roc.set_xlim([-0.02, 1.02])
    ax_roc.set_ylim([-0.02, 1.02])
    ax_roc.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight="bold")
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Comparison\nCleveland Heart Disease Dataset", fontsize=12)
    ax_roc.legend(loc="lower right", fontsize=10, frameon=True)
    ax_roc.grid(True, alpha=0.3)
    plt.tight_layout()
    fig_roc.savefig(outputs_dir / "roc_curves_comparison.png", dpi=300)
    plt.close(fig_roc)
    print(f"[*] Saved ROC comparison plot to: {outputs_dir / 'roc_curves_comparison.png'}\n")

    # -----------------------------------------------------------------------
    # Step H: Multi-Seed Robustness Evaluation (5 Random Seeds)
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("  MULTI-SEED ROBUSTNESS EVALUATION (5 SEEDS: [42, 123, 7, 999, 2026])")
    print("  *Caveat: Hyperparameters fixed to seed-42 training split choice")
    print("=" * 78)
    seeds = [42, 123, 7, 999, 2026]
    seed_records = []

    for s in seeds:
        X_tr_s, X_te_s, y_tr_s, y_te_s = train_test_split(
            X, y, test_size=0.25, stratify=y, random_state=s
        )
        X_tr_sq, X_te_sq, _, _, _ = build_preprocessed_fold(
            X_tr_s, y_tr_s, X_te_s,
            best_cfg["reducer"], int(best_cfg["qubits"]), float(best_cfg["scale"])
        )
        st_tr = compute_statevectors(
            X_tr_sq, int(best_cfg["qubits"]), int(best_cfg["reps"]), str(best_cfg["entanglement"])
        )
        st_te = compute_statevectors(
            X_te_sq, int(best_cfg["qubits"]), int(best_cfg["reps"]), str(best_cfg["entanglement"])
        )
        K_tr_s = compute_kernel_matrix(st_tr)
        K_te_s = compute_kernel_matrix(st_te, st_tr)

        clf_s = SVC(kernel="precomputed", C=float(best_cfg["C"]))
        clf_s.fit(K_tr_s, y_tr_s)
        scores_s = clf_s.decision_function(K_te_s)
        preds_s_default = (scores_s >= 0.0).astype(int)
        preds_s_tuned = (scores_s >= tuned_threshold).astype(int)

        m_def = compute_all_metrics(y_te_s, preds_s_default, scores_s)
        m_tun = compute_all_metrics(y_te_s, preds_s_tuned, scores_s)

        seed_records.append({"seed": s, "type": "default", **m_def})
        seed_records.append({"seed": s, "type": "tuned", **m_tun})

    df_seed_records = pd.DataFrame(seed_records)
    
    summary_rows = []
    for m_type in ["default", "tuned"]:
        sub = df_seed_records[df_seed_records["type"] == m_type]
        row_dict = {"Configuration": f"QSVM ({'Default Thr' if m_type=='default' else 'Tuned Thr'})"}
        for metric_name in ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]:
            vals = sub[metric_name].values
            row_dict[metric_name] = f"{np.mean(vals):.4f} +/- {np.std(vals):.4f}"
        summary_rows.append(row_dict)

    df_seed_summary = pd.DataFrame(summary_rows)
    df_seed_summary.to_csv(outputs_dir / "multi_seed_summary.csv", index=False)
    print(df_seed_summary.to_string(index=False))
    print()

    # -----------------------------------------------------------------------
    # Step I: Sample Test Predictions & Custom Patient Verification
    # -----------------------------------------------------------------------
    print("=" * 78)
    print("  SAMPLE INFERENCE TESTING (8 Random Test Set Patients)")
    print("=" * 78)
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(X_test), size=min(8, len(X_test)), replace=False)
    sample_rows = []
    for idx in sample_indices:
        actual = y_test[idx]
        p_def = qsvm_test_pred_default[idx]
        p_tun = qsvm_test_pred_tuned[idx]
        score = qsvm_test_scores[idx]
        sample_rows.append({
            "Test_Idx": idx,
            "Actual": "Disease (1)" if actual == 1 else "No Disease (0)",
            "Score": f"{score:+.4f}",
            "Pred_Default": "Disease (1)" if p_def == 1 else "No Disease (0)",
            "Pred_Tuned": "Disease (1)" if p_tun == 1 else "No Disease (0)",
            "Status": "CORRECT" if p_tun == actual else "MISS",
        })
    df_samples = pd.DataFrame(sample_rows)
    print(df_samples.to_string(index=False))
    print()

    print("=" * 78)
    print("  CUSTOM CLINICAL PATIENT TEST")
    print("=" * 78)
    custom_patient = {
        "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0,
        "restecg": 2, "thalach": 120, "exang": 1, "oldpeak": 2.4, "slope": 2,
        "ca": 2, "thal": 7,
    }
    df_custom = pd.DataFrame([custom_patient])[feature_names]
    # Transform custom patient with fitted pipeline
    custom_s = scaler_final.transform(df_custom.values)
    custom_r = reducer_final.transform(custom_s)
    custom_q = minmax_final.transform(custom_r)
    custom_state = compute_statevectors(
        custom_q, int(best_cfg["qubits"]), int(best_cfg["reps"]), str(best_cfg["entanglement"])
    )
    custom_kernel = compute_kernel_matrix(custom_state, states_train)
    custom_score = float(qsvm_final.decision_function(custom_kernel)[0])
    custom_pred_default = int(custom_score >= 0.0)
    custom_pred_tuned = int(custom_score >= tuned_threshold)

    print("Patient Clinical Profile:")
    for k, v in custom_patient.items():
        print(f"  {k:<10s}: {v}")
    print(f"Decision Function Score   : {custom_score:+.4f}")
    print(f"Default Prediction (Thr=0): {'Disease (1)' if custom_pred_default==1 else 'No Disease (0)'}")
    print(f"Tuned Prediction   (Thr={tuned_threshold:+.3f}): {'Disease (1)' if custom_pred_tuned==1 else 'No Disease (0)'}\n")

    # -----------------------------------------------------------------------
    # Step J: Console Summary for Slide Placeholders & Academic Abstract
    # -----------------------------------------------------------------------
    t_total = time.time() - t_start
    print("=" * 78)
    print("  CONSOLE SUMMARY FOR SLIDE PLACEHOLDERS [X] & ACADEMIC ABSTRACT")
    print("=" * 78)
    print(f"""
[SLIDE BULLET 1 - BEST HYPERPARAMETERS]
- Quantum Circuit : ZZFeatureMap (reps={int(best_cfg['reps'])}, entanglement='{best_cfg['entanglement']}')
- Feature Reducer : {best_cfg['reducer'].upper()} (k={int(best_cfg['qubits'])} qubits)
- Feature Scaling : [0, {best_cfg['scale']:.4f}]
- Classifier      : SVC(C={best_cfg['C']}, kernel='precomputed')
- Threshold       : {tuned_threshold:+.4f} (tuned via OOF F2-score)

[SLIDE BULLET 2 - TEST PERFORMANCE COMPARISON]
- QSVM (Tuned Thr): Accuracy={metrics_qsvm_tuned['Accuracy']:.2%}, Recall={metrics_qsvm_tuned['Recall']:.2%}, Specificity={metrics_qsvm_tuned['Specificity']:.2%}, AUC={metrics_qsvm_tuned['ROC-AUC']:.4f}
- QSVM (Def. Thr) : Accuracy={metrics_qsvm_default['Accuracy']:.2%}, Recall={metrics_qsvm_default['Recall']:.2%}, Specificity={metrics_qsvm_default['Specificity']:.2%}, AUC={metrics_qsvm_default['ROC-AUC']:.4f}
- Classical RBF 13: Accuracy={metrics_baseline_all['Accuracy']:.2%}, Recall={metrics_baseline_all['Recall']:.2%}, Specificity={metrics_baseline_all['Specificity']:.2%}, AUC={metrics_baseline_all['ROC-AUC']:.4f}
- Classical RBF {best_cfg['qubits']}: Accuracy={metrics_baseline_reduced['Accuracy']:.2%}, Recall={metrics_baseline_reduced['Recall']:.2%}, Specificity={metrics_baseline_reduced['Specificity']:.2%}, AUC={metrics_baseline_reduced['ROC-AUC']:.4f}

[SLIDE BULLET 3 - MULTI-SEED STABILITY (5 SEEDS)]
- QSVM Accuracy   : {df_seed_summary.loc[df_seed_summary['Configuration']=='QSVM (Tuned Thr)', 'Accuracy'].values[0]}
- QSVM Recall     : {df_seed_summary.loc[df_seed_summary['Configuration']=='QSVM (Tuned Thr)', 'Recall'].values[0]}
- QSVM ROC-AUC    : {df_seed_summary.loc[df_seed_summary['Configuration']=='QSVM (Tuned Thr)', 'ROC-AUC'].values[0]}

[ABSTRACT SNIPPET]
"Using a leak-free 5-fold cross-validated grid search over feature reducers, qubit allocations,
circuit depths, and entanglement topologies on the 297-sample Cleveland dataset, the optimized
QSVM achieved an ROC-AUC of {metrics_qsvm_tuned['ROC-AUC']:.4f} and {metrics_qsvm_tuned['Accuracy']:.2%} accuracy.
By optimizing the decision boundary for clinical recall via out-of-fold F2 tuning, sensitivity
reached {metrics_qsvm_tuned['Recall']:.2%} (specificity {metrics_qsvm_tuned['Specificity']:.2%}). In comparison,
the fully-tuned classical RBF-SVM baseline achieved an ROC-AUC of {metrics_baseline_all['ROC-AUC']:.4f} and {metrics_baseline_all['Accuracy']:.2%}
accuracy, demonstrating parity between shallow quantum kernels and classical non-linear baselines
under rigorous experimental protocols without quantum advantage claims."

Total Pipeline Runtime: {t_total:.1f} seconds ({t_total/60:.2f} minutes).
""")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QSVM on UCI Heart Disease - Academic Benchmark")
    parser.add_argument("--quick", action="store_true", help="Run minimal parameter grid for fast debugging")
    parser.add_argument("--dataset-path", type=str, default=None, help="Optional path to custom 918-record CSV dataset")
    args = parser.parse_args()

    run_pipeline(quick=args.quick, dataset_path=args.dataset_path)
