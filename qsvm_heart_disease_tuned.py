"""Nested-CV quantum-kernel and classical baselines for Cleveland heart data."""
from __future__ import annotations

import argparse
from functools import partial
from itertools import combinations, product
import hashlib
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qiskit
import sklearn
from joblib import Parallel, delayed
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             fbeta_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Keep data and output paths anchored to this script's folder.
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
DEFAULT_DATA = ROOT / "heart_disease_cleveland.csv"
CATEGORICAL = {"sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"}
SEED = 42


# Read the CSV, remove invalid rows, and convert Cleveland labels to binary classes.
def load_data(dataset_path: str | None = None, target_column: str | None = None):
    """Load and validate numeric tabular data. Cleveland labels 0..4 map to 0/1."""
    path = Path(dataset_path).expanduser().resolve() if dataset_path else DEFAULT_DATA
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    raw_frame = pd.read_csv(path)
    frame = raw_frame.copy()
    if frame.empty or frame.shape[1] < 2:
        raise ValueError("Dataset must contain feature columns and one target column.")
    target = target_column or str(frame.columns[-1])
    if target not in frame.columns:
        raise ValueError(f"Target column {target!r} is not present in {path.name}.")
    y_raw = pd.to_numeric(frame.pop(target), errors="coerce")
    X_df = frame.apply(pd.to_numeric, errors="coerce")
    X_df = X_df.replace([np.inf, -np.inf], np.nan)
    valid = y_raw.notna() & np.isfinite(y_raw) & X_df.notna().all(axis=1)
    X_df, y_raw = X_df.loc[valid], y_raw.loc[valid]
    labels = np.sort(y_raw.unique())
    if len(labels) != 2 and not np.all(np.isclose(labels, np.round(labels)) & (labels >= 0) & (labels <= 4)):
        raise ValueError(f"Target must be binary or use Cleveland 0..4 labels; found {labels.tolist()}.")
    y = (y_raw.to_numpy() > 0).astype(int)
    if np.unique(y).size != 2:
        raise ValueError("Target must contain both negative and positive examples after binarization.")
    if len(X_df) < 50:
        raise ValueError(f"Only {len(X_df)} complete rows remain; at least 50 are required.")
    if X_df.columns.duplicated().any():
        raise ValueError("Feature names must be unique.")
    return X_df.reset_index(drop=True), y, path, int(len(raw_frame) - len(X_df))


# Scale numeric inputs and one-hot encode the known categorical inputs.
def make_encoder(X: pd.DataFrame) -> ColumnTransformer:
    cats = [c for c in X.columns if c.lower() in CATEGORICAL]
    nums = [c for c in X.columns if c not in cats]
    transformers = []
    if nums:
        transformers.append(("numeric", StandardScaler(), nums))
    if cats:
        transformers.append(("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cats))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


# Identify encoded columns that mutual-information scoring should treat as discrete.
def encoded_discrete_mask(encoder: ColumnTransformer) -> np.ndarray:
    """Mark one-hot outputs as discrete for mutual-information scoring."""
    names = encoder.get_feature_names_out()
    return np.array(["=" in name or name.split("_")[0].lower() in CATEGORICAL for name in names], dtype=bool)


# Fit fold-local encoding, dimensionality reduction, and quantum-angle scaling.
def fit_representation(X_train: pd.DataFrame, y_train: np.ndarray, X_eval: pd.DataFrame,
                       reducer_name: str, k: int, scale: float, seed: int):
    encoder = make_encoder(X_train).fit(X_train)
    A, B = encoder.transform(X_train), encoder.transform(X_eval)
    if k > A.shape[1]:
        raise ValueError(f"Cannot select {k} dimensions from {A.shape[1]} encoded columns.")
    if reducer_name == "mi":
        mask = encoded_discrete_mask(encoder)
        scorer = partial(mutual_info_classif, discrete_features=mask, random_state=seed)
        reducer = SelectKBest(score_func=scorer, k=k)
    else:
        reducer = PCA(n_components=k, random_state=seed)
    A = reducer.fit_transform(A, y_train) if reducer_name == "mi" else reducer.fit_transform(A)
    B = reducer.transform(B)
    scaler = MinMaxScaler(feature_range=(0.0, float(scale))).fit(A)
    return scaler.transform(A), scaler.transform(B), encoder, reducer, scaler


# Encode each row in a ZZ feature-map circuit and simulate its quantum state.
def statevectors(X: np.ndarray, n_qubits: int, reps: int, entanglement: str) -> np.ndarray:
    if X.ndim != 2 or X.shape[1] != n_qubits:
        raise ValueError(f"Expected transformed shape (n, {n_qubits}); received {X.shape}.")
    circuit = zz_feature_map(feature_dimension=n_qubits, reps=reps, entanglement=entanglement)
    params = sorted(circuit.parameters, key=lambda p: int(p.name.split("[")[1].rstrip("]")))
    rows = []
    for row in X:
        bound = circuit.assign_parameters(dict(zip(params, row)), strict=True)
        rows.append(Statevector(bound).data)
    return np.asarray(rows)


# Compute pairwise quantum-state fidelity scores.
def kernel(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    B = A if B is None else B
    return np.abs(A @ B.conj().T) ** 2


# Convert quantum states to expectation values for local Pauli measurements.
def projected_observables(states: np.ndarray, n_qubits: int, order: int = 1) -> np.ndarray:
    """Return all Pauli expectations on subsets of up to ``order`` qubits."""
    indices = np.arange(1 << n_qubits)
    features = []
    for weight in range(1, order + 1):
        for qubits in combinations(range(n_qubits), weight):
            for paulis in product("XYZ", repeat=weight):
                flip_mask = 0
                phase = np.ones(len(indices), dtype=complex)
                for qubit, pauli in zip(qubits, paulis):
                    bit = 1 << qubit
                    bits_set = (indices & bit) != 0
                    if pauli in "XY":
                        flip_mask |= bit
                    if pauli == "Y":
                        phase *= 1j * np.where(bits_set, -1.0, 1.0)
                    elif pauli == "Z":
                        phase *= np.where(bits_set, -1.0, 1.0)
                mapped = indices ^ flip_mask
                exp_value = np.sum(states[:, mapped].conj() * states * phase[None, :], axis=1)
                features.append(exp_value.real)
    return np.column_stack(features)


# Return single-qubit Pauli expectations as local state features.
def local_bloch_features(states: np.ndarray, n_qubits: int) -> np.ndarray:
    return projected_observables(states, n_qubits, order=1)


# Compute nonnegative pairwise squared Euclidean distances.
def squared_distances(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, (A * A).sum(axis=1)[:, None] +
                      (B * B).sum(axis=1)[None, :] - 2.0 * A @ B.T)


# Build train and evaluation kernels using the selected quantum-kernel definition.
def quantum_kernel_pair(states_a: np.ndarray, states_b: np.ndarray, cfg: dict,
                        n_qubits: int):
    """Construct either the fidelity or local projected quantum kernel."""
    if cfg["kernel_type"] == "fidelity":
        return kernel(states_a), kernel(states_b, states_a), None
    order = int(cfg.get("projection_order") or 1)
    features_a = projected_observables(states_a, n_qubits, order)
    features_b = projected_observables(states_b, n_qubits, order)
    d_train = squared_distances(features_a, features_a)
    upper = d_train[np.triu_indices_from(d_train, k=1)]
    median = float(np.median(upper)) if upper.size else 1.0
    gamma = float(cfg["gamma_multiplier"] / max(median, 1e-12))
    K_train = np.exp(-gamma * d_train)
    K_eval = np.exp(-gamma * squared_distances(features_b, features_a))
    return K_train, K_eval, gamma


# Rebuild a kernel from cached distances to avoid repeating distance calculations.
def cached_quantum_kernel(states_a: np.ndarray, states_b: np.ndarray,
                          distances: tuple[np.ndarray, np.ndarray, float], cfg: dict):
    if cfg["kernel_type"] == "fidelity":
        return kernel(states_a), kernel(states_b, states_a)
    d_train, d_eval, median = distances
    gamma = float(cfg["gamma_multiplier"] / max(median, 1e-12))
    return np.exp(-gamma * d_train), np.exp(-gamma * d_eval)


# Normalize evaluation decision scores using the training score distribution.
def normalized_margin(model: SVC, K_train: np.ndarray, K_eval: np.ndarray):
    train_scores = model.decision_function(K_train)
    center = float(np.mean(train_scores))
    spread = float(np.std(train_scores)) or 1.0
    return (model.decision_function(K_eval) - center) / spread, center, spread


# Enumerate the circuit, kernel, feature-reduction, and regularization search space.
def circuit_groups(quick: bool):
    reducers, qubits, reps = (["mi"], [4], [1]) if quick else (["mi", "pca"], [4, 6, 8], [1, 2])
    entanglements = ["linear"] if quick else ["linear", "full"]
    scales = [float(np.pi)] if quick else [1.0, float(np.pi)]
    c_values = [1.0, 10.0] if quick else [0.1, 1.0, 10.0, 100.0]
    configs = []
    for r in reducers:
        for k in qubits:
            for d in reps:
                for e in entanglements:
                    for s in scales:
                        core = {"reducer": r, "qubits": k, "reps": d,
                                "entanglement": e, "scale": s}
                        configs.append({**core, "kernel_type": "fidelity",
                                        "projection_order": None, "gamma_multiplier": None})
                        orders = [1] if quick else [1, 2]
                        gammas = [1.0] if quick else [0.1, 1.0, 10.0]
                        for order in orders:
                            for gamma in gammas:
                                configs.append({**core, "kernel_type": "projected",
                                                "projection_order": order,
                                                "gamma_multiplier": gamma})
    return configs, c_values


# Score one grouped candidate grid with inner-fold out-of-fold predictions.
def evaluate_group(group: dict, X: pd.DataFrame, y: np.ndarray, c_values: list[float],
                   folds: int, seed: int):
    """Reuse fold encodings across circuit depths/topologies and all C values."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    fold_data = []
    for fold, (it, iv) in enumerate(skf.split(X, y)):
        A, B, *_ = fit_representation(X.iloc[it], y[it], X.iloc[iv], group["reducer"],
                                      group["qubits"], group["scale"], seed + fold)
        maps = {}
        for reps in sorted({cfg["reps"] for cfg in group["configs"]}):
            for ent in sorted({cfg["entanglement"] for cfg in group["configs"]}):
                sa = statevectors(A, group["qubits"], reps, ent)
                sb = statevectors(B, group["qubits"], reps, ent)
                maps[(reps, ent, "fidelity", None, None)] = (kernel(sa), kernel(sb, sa))
                projected_specs = {(cfg["projection_order"], cfg["gamma_multiplier"])
                                   for cfg in group["configs"]
                                    if cfg["reps"] == reps and cfg["entanglement"] == ent
                                    and cfg["kernel_type"] == "projected"}
                for order in sorted({o for o, _ in projected_specs}):
                    obs_a = projected_observables(sa, group["qubits"], order)
                    obs_b = projected_observables(sb, group["qubits"], order)
                    d_train = squared_distances(obs_a, obs_a)
                    d_eval = squared_distances(obs_b, obs_a)
                    upper = d_train[np.triu_indices_from(d_train, k=1)]
                    median = float(np.median(upper)) if upper.size else 1.0
                    for multiplier in sorted(g for o, g in projected_specs if o == order):
                        gamma = float(multiplier / max(median, 1e-12))
                        maps[(reps, ent, "projected", order, multiplier)] = (
                            np.exp(-gamma * d_train), np.exp(-gamma * d_eval))
        fold_data.append((it, iv, y[it], y[iv], maps))

    records = []
    for cfg in group["configs"]:
        for C in c_values:
            scores = np.zeros(len(y))
            fold_auc = []
            for it, iv, yt, yv, maps in fold_data:
                Ktr, Kev = maps[(cfg["reps"], cfg["entanglement"], cfg["kernel_type"],
                                  cfg["projection_order"], cfg["gamma_multiplier"])]
                clf = SVC(kernel="precomputed", C=C).fit(Ktr, yt)
                scores[iv], _, _ = normalized_margin(clf, Ktr, Kev)
                fold_auc.append(float(roc_auc_score(yv, scores[iv])))
            records.append({**cfg, "C": float(C), "mean_cv_roc_auc": float(np.mean(fold_auc)),
                            "std_cv_roc_auc": float(np.std(fold_auc)), "oof_scores": scores})
    return records


# Select the decision threshold that maximizes F2 on out-of-fold scores.
def choose_threshold(scores: np.ndarray, y: np.ndarray):
    thresholds = np.r_[np.unique(scores), np.nextafter(np.max(scores), np.inf)]
    f2s = [fbeta_score(y, scores >= t, beta=2, zero_division=0) for t in thresholds]
    # Prefer the higher threshold on ties, which reduces false positives.
    best = max(range(len(f2s)), key=lambda i: (f2s[i], thresholds[i]))
    return float(thresholds[best]), float(f2s[best])


# Calculate classification metrics and confusion-matrix counts at one threshold.
def metric_row(y: np.ndarray, scores: np.ndarray, threshold: float):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"Accuracy": float(accuracy_score(y, pred)),
            "Precision": float(precision_score(y, pred, zero_division=0)),
            "Recall": float(recall_score(y, pred, zero_division=0)),
            "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
            "F1": float(f1_score(y, pred, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y, scores)),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp)}


# Fit the selected quantum pipeline and return scores, reusable components, and diagnostics.
def fit_quantum(Xtr, ytr, Xev, cfg, seed):
    A, B, encoder, reducer, scaler = fit_representation(
        Xtr, ytr, Xev, cfg["reducer"], cfg["qubits"], cfg["scale"], seed)
    sv_tr = statevectors(A, cfg["qubits"], cfg["reps"], cfg["entanglement"])
    sv_ev = statevectors(B, cfg["qubits"], cfg["reps"], cfg["entanglement"])
    Ktr, Kev, actual_gamma = quantum_kernel_pair(sv_tr, sv_ev, cfg, cfg["qubits"])
    clf = SVC(kernel="precomputed", C=cfg["C"]).fit(Ktr, ytr)
    scores, center, spread = normalized_margin(clf, Ktr, Kev)
    # Kernel diagnostics are on the final training Gram matrix.
    sym_err = float(np.max(np.abs(Ktr - Ktr.T)))
    eigenvalues = np.linalg.eigvalsh((Ktr + Ktr.T) / 2)
    eig_min = float(eigenvalues.min())
    positive = np.clip(eigenvalues, 0.0, None)
    probs = positive / positive.sum() if positive.sum() else positive
    effective_rank = float(np.exp(-np.sum(probs[probs > 0] * np.log(probs[probs > 0]))))
    spectral_floor = max(float(eigenvalues.max()), 1.0) * 1e-12
    resolved_positive = eigenvalues[eigenvalues > spectral_floor]
    resolved_condition = (float(resolved_positive.max() / resolved_positive.min())
                          if resolved_positive.size else float("nan"))
    return scores, (encoder, reducer, scaler, clf, sv_tr), {
        "train_kernel_symmetry_max_error": sym_err,
        "train_kernel_diagonal_max_error": float(np.max(np.abs(np.diag(Ktr) - 1.0))),
        "train_kernel_min_eigenvalue": eig_min,
        "train_kernel_effective_rank": effective_rank,
        "train_kernel_condition_number_above_1e-12_relative_floor": resolved_condition,
        "train_kernel_offdiagonal_mean": float((Ktr.sum() - np.trace(Ktr)) / max(1, Ktr.size - len(Ktr))),
        "decision_score_center": center, "decision_score_scale": spread,
        "projected_kernel_gamma": actual_gamma,
        "projection_order": cfg.get("projection_order"),
    }


# Apply a saved research model bundle to a dataframe of new examples.
def predict_frame(model_bundle: dict, frame: pd.DataFrame) -> pd.DataFrame:
    """Run research inference with a bundle saved by this script."""
    expected = model_bundle["feature_names"]
    missing = sorted(set(expected) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing input features: {missing}")
    X = frame[expected].apply(pd.to_numeric, errors="raise")
    encoder = model_bundle["encoder"]
    encoded = encoder.transform(X)
    reduced = model_bundle["reducer"].transform(encoded)
    transformed = model_bundle["minmax"].transform(reduced)
    cfg = model_bundle["config"]
    states = statevectors(transformed, cfg["qubits"], cfg["reps"], cfg["entanglement"])
    k_eval = quantum_kernel_pair(model_bundle["train_statevectors"], states,
                                 cfg, cfg["qubits"])[1]
    raw = model_bundle["classifier"].decision_function(k_eval)
    normalized = (raw - model_bundle["score_center"]) / model_bundle["score_scale"]
    threshold = model_bundle["threshold"]
    return pd.DataFrame({"decision_score": normalized, "threshold": threshold,
                         "prediction": (normalized >= threshold).astype(int)})


# Tune the full-feature RBF baseline and produce normalized validation scores per fold.
def classical_all_fold_scores(X: pd.DataFrame, y: np.ndarray, splits, seed: int, folds: int):
    scores = np.zeros(len(y))
    grid = {"svc__C": [0.1, 1.0, 10.0, 100.0],
            "svc__gamma": ["scale", "auto", 0.01, 0.1, 1.0]}
    for fold, (it, iv) in enumerate(splits):
        pipe = Pipeline([("encode", make_encoder(X.iloc[it])), ("svc", SVC(kernel="rbf"))])
        search = GridSearchCV(pipe, grid, cv=StratifiedKFold(folds, shuffle=True, random_state=seed + fold),
                              scoring="roc_auc", n_jobs=1, refit=True)
        search.fit(X.iloc[it], y[it])
        estimator = search.best_estimator_
        train_scores = estimator.decision_function(X.iloc[it])
        eval_scores = estimator.decision_function(X.iloc[iv])
        scores[iv] = (eval_scores - train_scores.mean()) / (train_scores.std() or 1.0)
    return scores


# Tune an RBF baseline using the same reduced inputs as a quantum candidate.
def classical_reduced_cv(X: pd.DataFrame, y: np.ndarray, splits, cfg: dict, seed: int, folds: int):
    """Tune an RBF SVM on the same fold-local reduced representation as QSVM."""
    grid = [(C, gamma) for C in [0.1, 1.0, 10.0, 100.0]
            for gamma in ["scale", "auto", 0.01, 0.1, 1.0]]
    oof = {p: np.zeros(len(y)) for p in grid}
    fold_auc = {p: [] for p in grid}
    for fold, (it, iv) in enumerate(splits):
        A, B, *_ = fit_representation(X.iloc[it], y[it], X.iloc[iv], cfg["reducer"],
                                      cfg["qubits"], cfg["scale"], seed + fold)
        for C, gamma in grid:
            clf = SVC(kernel="rbf", C=C, gamma=gamma).fit(A, y[it])
            train_score = clf.decision_function(A)
            val_score = clf.decision_function(B)
            normalized = (val_score - train_score.mean()) / (train_score.std() or 1.0)
            oof[(C, gamma)][iv] = normalized
            fold_auc[(C, gamma)].append(float(roc_auc_score(y[iv], normalized)))
    best_params = max(grid, key=lambda p: (np.mean(fold_auc[p]), -np.std(fold_auc[p])))
    tau, f2 = choose_threshold(oof[best_params], y)
    return best_params, oof[best_params], tau, f2


# Run nested evaluation, compare baselines, fit the review model, and write reports.
def run(quick=False, dataset_path=None, target_column=None, outer_folds=5, inner_folds=5,
        jobs=None, seed=SEED):
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    X, y, data_path, dropped = load_data(dataset_path, target_column)
    if outer_folds < 3 or inner_folds < 3:
        raise ValueError("Nested CV needs at least 3 outer and inner folds.")
    configs, c_values = circuit_groups(quick)
    # Group grids so encodings and statevectors are shared where possible.
    groups = {}
    for cfg in configs:
        key = (cfg["reducer"], cfg["qubits"], cfg["scale"])
        groups.setdefault(key, {**{k: cfg[k] for k in ("reducer", "qubits", "scale")}, "configs": []})
        groups[key]["configs"].append(cfg)
    n_jobs = jobs or min(4, len(groups), os.cpu_count() or 1)
    outer = StratifiedKFold(outer_folds, shuffle=True, random_state=seed)
    outer_rows, all_search_rows, fold_details = [], [], []
    # Tune and evaluate each candidate using only the current outer-training partition.
    for outer_i, (it, iv) in enumerate(outer.split(X, y), 1):
        print(f"Outer fold {outer_i}/{outer_folds}: inner tuning over {len(configs) * len(c_values)} settings", flush=True)
        Xtr, Xev, ytr, yev = X.iloc[it], X.iloc[iv], y[it], y[iv]
        inner_splits = list(StratifiedKFold(inner_folds, shuffle=True,
                                             random_state=seed + outer_i).split(Xtr, ytr))
        batches = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(evaluate_group)(group, Xtr, ytr, c_values, inner_folds, seed + outer_i)
            for group in groups.values())
        candidates = [r for batch in batches for r in batch]
        ranked = sorted(candidates, key=lambda r: (r["mean_cv_roc_auc"], r["std_cv_roc_auc"] * -1), reverse=True)
        best = ranked[0]
        threshold, inner_f2 = choose_threshold(best["oof_scores"], ytr)
        scores, fitted, diag = fit_quantum(Xtr, ytr, Xev, best, seed + outer_i)
        qmetrics = metric_row(yev, scores, threshold)
        # Classical all-feature baseline follows the same outer split; its threshold is 0.
        all_scores = classical_all_fold_scores(Xtr, ytr, inner_splits, seed + outer_i, inner_folds)
        all_tau, all_f2 = choose_threshold(all_scores, ytr)
        allpipe = Pipeline([("encode", make_encoder(Xtr)), ("svc", SVC(kernel="rbf"))])
        allgrid = GridSearchCV(allpipe, {"svc__C": [0.1, 1, 10, 100],
                                       "svc__gamma": ["scale", "auto", .01, .1, 1.]},
                               cv=StratifiedKFold(inner_folds, shuffle=True, random_state=seed + outer_i),
                               scoring="roc_auc", n_jobs=1).fit(Xtr, ytr)
        all_test_scores = allgrid.best_estimator_.decision_function(Xev)
        # Normalize the final SVC's scores using its training-score distribution.
        all_train_scores = allgrid.best_estimator_.decision_function(Xtr)
        all_test_scores = (all_test_scores - all_train_scores.mean()) / (all_train_scores.std() or 1.0)
        # Inner OOF scores are standardized per fold; final scores use the analogous model normalization.
        classical_all = metric_row(yev, all_test_scores, all_tau)
        reduced_params, reduced_oof, reduced_tau, reduced_f2 = classical_reduced_cv(
            Xtr, ytr, inner_splits, best, seed + outer_i, inner_folds)
        red_A, red_B, *_ = fit_representation(Xtr, ytr, Xev, best["reducer"],
                                               best["qubits"], best["scale"], seed + outer_i)
        reduced_clf = SVC(kernel="rbf", C=reduced_params[0], gamma=reduced_params[1]).fit(red_A, ytr)
        red_train_scores = reduced_clf.decision_function(red_A)
        red_test_scores = (reduced_clf.decision_function(red_B) - red_train_scores.mean()) / (red_train_scores.std() or 1.0)
        classical_reduced = metric_row(yev, red_test_scores, reduced_tau)
        row_base = {"outer_fold": outer_i, "train_n": len(it), "test_n": len(iv)}
        outer_rows.extend([{**row_base, "Model": "QSVM", **qmetrics},
                           {**row_base, "Model": "Classical RBF (all features)", **classical_all},
                           {**row_base, "Model": "Classical RBF (same reduced input)", **classical_reduced}])
        fold_details.append({"outer_fold": outer_i, "selected_config": {k: v for k, v in best.items()
                              if k not in ("oof_scores",)}, "threshold": threshold,
                            "inner_oof_f2": inner_f2, "classical_threshold": all_tau,
                            "classical_inner_oof_f2": all_f2, "classical_best_params": allgrid.best_params_,
                            "classical_reduced_threshold": reduced_tau,
                            "classical_reduced_inner_oof_f2": reduced_f2,
                            "classical_reduced_best_params": {"C": reduced_params[0], "gamma": reduced_params[1]},
                            "kernel_diagnostics": diag})
        for r in candidates:
            all_search_rows.append({"outer_fold": outer_i, **{k: v for k, v in r.items() if k != "oof_scores"}})
        proj_text = (f", projected order={best['projection_order']} gamma x={best['gamma_multiplier']}"
                     if best["kernel_type"] == "projected" else "")
        print(f"  QSVM AUC={qmetrics['ROC-AUC']:.3f}, recall={qmetrics['Recall']:.3f}, "
              f"specificity={qmetrics['Specificity']:.3f}; selected {best['reducer']}/{best['qubits']}q "
              f"reps={best['reps']} {best['entanglement']} {best['kernel_type']}{proj_text} "
              f"C={best['C']} scale={best['scale']:.3g}", flush=True)

    # Save fold metrics, candidate scores, and per-fold selection details.
    results = pd.DataFrame(outer_rows)
    results.to_csv(OUT / "nested_cv_fold_results.csv", index=False)
    pd.DataFrame(all_search_rows).to_csv(OUT / "nested_cv_search_results.csv", index=False)
    summary = results.groupby("Model")[["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]].agg(["mean", "std"])
    summary.to_csv(OUT / "nested_cv_summary.csv")
    with open(OUT / "nested_cv_details.json", "w", encoding="utf-8") as f:
        json.dump(fold_details, f, indent=2)

    # Refit a final review artifact on all data. Its settings are selected by inner CV;
    # nested outer results above remain the unbiased procedure-level estimate.
    print("Selecting final review configuration on all data...", flush=True)
    full_splits = list(StratifiedKFold(inner_folds, shuffle=True, random_state=seed).split(X, y))
    full_batches = Parallel(n_jobs=n_jobs)(delayed(evaluate_group)(group, X, y, c_values,
                                                                  inner_folds, seed)
                                          for group in groups.values())
    final_candidates = [r for batch in full_batches for r in batch]
    final_best = max(final_candidates, key=lambda r: r["mean_cv_roc_auc"])
    final_threshold, final_f2 = choose_threshold(final_best["oof_scores"], y)
    # Fit estimators and save deployable inference components only if joblib is available.
    final_scores, fitted, final_diag = fit_quantum(X, y, X, final_best, seed)
    final_config = {k: v for k, v in final_best.items() if k != "oof_scores"}
    data_sha = hashlib.sha256(data_path.read_bytes()).hexdigest()
    metadata = {"dataset_path": str(data_path), "dataset_sha256": data_sha,
                "rows_used": int(len(y)), "rows_dropped": dropped,
                "features": list(X.columns), "categorical_features_one_hot": [c for c in X.columns if c.lower() in CATEGORICAL],
                "positive_label_rule": "target > 0 maps to disease (1); target 0 maps to no disease (0)",
                "outer_folds": outer_folds, "inner_folds": inner_folds, "seed": seed,
                "quick_mode": quick, "parallel_jobs": n_jobs,
                "search_space_size_per_outer_fold": len(configs) * len(c_values),
                "best_full_data_config": final_config, "oof_threshold_normalized_margin": final_threshold,
                "inner_oof_f2": final_f2, "final_kernel_diagnostics": final_diag,
                "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__, "qiskit": qiskit.__version__,
                "elapsed_seconds": round(time.perf_counter() - started, 3)}
    (OUT / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Save preprocessing and classifier components for later research inference.
    try:
        import joblib
        joblib.dump({"encoder": fitted[0], "reducer": fitted[1], "minmax": fitted[2],
                     "classifier": fitted[3], "train_statevectors": fitted[4],
                     "feature_names": list(X.columns), "config": final_config,
                     "score_center": final_diag["decision_score_center"],
                     "score_scale": final_diag["decision_score_scale"],
                     "threshold": final_threshold}, OUT / "qsvm_review_model.joblib", compress=3)
    except Exception as exc:
        warnings.warn(f"Could not save inference artifact: {exc}")

    # Concise review report with fold-level results and explicit interpretation.
    lines = ["# Nested CV model review", "", f"Dataset: `{data_path.name}` ({len(y)} rows used; {dropped} dropped)",
             f"Outer/inner CV: {outer_folds} × {inner_folds}; seed {seed}; mode: {'quick' if quick else 'full'}", "",
             "## Outer-fold performance", "",
             "Scores below are the procedure-level estimate. Model settings and the threshold were selected using only each outer training partition.", "",
             "| Model | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, group in results.groupby("Model"):
        vals = []
        for m in ["Accuracy", "Precision", "Recall", "Specificity", "F1", "ROC-AUC"]:
            vals.append(f"{group[m].mean():.3f} ± {group[m].std(ddof=1):.3f}")
        lines.append(f"| {name} | " + " | ".join(vals) + " |")
    q = results[results["Model"] == "QSVM"]
    classical = results[results["Model"] == "Classical RBF (all features)"]
    q_auc = q["ROC-AUC"].mean()
    c_auc = classical["ROC-AUC"].mean()
    q_acc = q["Accuracy"].mean()
    c_acc = classical["Accuracy"].mean()
    configs_by_fold = []
    for d in fold_details:
        cfg = d["selected_config"]
        kernel_desc = (f"projected order {cfg['projection_order']} γ×{cfg['gamma_multiplier']}"
                       if cfg["kernel_type"] == "projected" else "fidelity")
        configs_by_fold.append(f"fold {d['outer_fold']}: {cfg['reducer']}/{cfg['qubits']}q, "
                               f"reps={cfg['reps']}, {cfg['entanglement']}, {kernel_desc}")
    lines += ["", "## Interpretation", "",
              f"The expanded 8-qubit projected QSVM had mean outer-fold ROC-AUC {q_auc:.3f} and accuracy {q_acc:.3f}; "
              f"the all-feature classical RBF-SVM scored {c_auc:.3f} AUC and {c_acc:.3f} accuracy "
              f"(QSVM minus classical: {q_auc-c_auc:+.3f} AUC, {q_acc-c_acc:+.3f} accuracy). "
              "This is a modest predictive edge in this five-fold run, not evidence of a statistically robust "
              "or computational quantum advantage; the fold-to-fold uncertainty is substantial and the final "
              "projected kernel is highly concentrated.",
              f"The QSVM threshold averaged {q['Recall'].mean():.3f} recall and "
              f"{q['Specificity'].mean():.3f} specificity; its fold-specific tradeoff varies, so these values "
              "should not be read as clinical validation.",
              "Selected quantum settings by outer fold: " + "; ".join(configs_by_fold) + "."]
    lines += ["", "## Per-fold results", "", "| Fold | Model | Accuracy | Recall | Specificity | ROC-AUC | TN | FP | FN | TP |",
              "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in results.iterrows():
        lines.append(f"| {int(r.outer_fold)} | {r.Model} | {r.Accuracy:.3f} | {r.Recall:.3f} | {r.Specificity:.3f} | {r['ROC-AUC']:.3f} | {int(r.TN)} | {int(r.FP)} | {int(r.FN)} | {int(r.TP)} |")
    lines += ["", "## Final full-data review artifact", "",
              "The separately refit artifact uses settings chosen by CV over all available data. Do not treat its training/CV scores as an independent performance estimate.",
              f"", f"- Selected settings: `{json.dumps(final_config, sort_keys=True)}`",
              f"- OOF F2 threshold on standardized margin: `{final_threshold:.6g}` (OOF F2 `{final_f2:.3f}`)",
              f"- Training-kernel diagnostics: `{json.dumps(final_diag, sort_keys=True)}`", "",
              "## Files", "", "- `nested_cv_fold_results.csv`: outer-fold metrics.",
              "- `nested_cv_search_results.csv`: inner-search scores for each outer fold.",
              "- `nested_cv_details.json`: selected settings, thresholds, and kernel diagnostics per outer fold.",
              "- `nested_cv_summary.csv`: mean and standard deviation by model.",
              "- `run_metadata.json`: dataset hash, configuration, environment, and runtime.",
              "- `qsvm_review_model.joblib`: fitted full-data inference artifact; for research review only, not clinical use.", ""]
    (OUT / "review_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nComplete in {metadata['elapsed_seconds']:.1f}s. Review: {OUT / 'review_report.md'}", flush=True)


# Parse command-line options and start the requested experiment.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="Use a small search space for a quick pipeline run")
    parser.add_argument("--dataset-path", help="CSV path; defaults to the bundled Cleveland dataset")
    parser.add_argument("--target-column", help="Target column name; defaults to the final CSV column")
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--jobs", type=int, default=None, help="Parallel circuit-group workers")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    run(args.quick, args.dataset_path, args.target_column, args.outer_folds,
        args.inner_folds, args.jobs, args.seed)


# Run the CLI only when this file is executed directly.
if __name__ == "__main__":
    main()
