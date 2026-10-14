"""
QSVM on UCI Heart Disease (id=45)

Install:
    pip install qiskit qiskit-machine-learning ucimlrepo scikit-learn pandas numpy
"""
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report)

from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC

# ----------------------------- Config -----------------------------
N_QUBITS = 4          # PCA components = qubits (keep small: simulation cost grows fast)
REPS = 2              # feature map depth
MAX_SAMPLES = 200     # subsample for speed; set None to use all (~297 rows)
TEST_SIZE = 0.25
SEED = 42

# ----------------------------- Data -------------------------------
import os
if os.path.exists("heart_disease_cleveland.csv"):
    df_raw = pd.read_csv("heart_disease_cleveland.csv").dropna()
    target_col = df_raw.columns[-1]
    df = df_raw.copy()
    df[target_col] = (df[target_col] > 0).astype(int)
else:
    heart = fetch_ucirepo(id=45)
    X = heart.data.features.copy()
    y = heart.data.targets.copy()
    df = pd.concat([X, y], axis=1).dropna()          # 6 rows have missing values (ca, thal)
    target_col = y.columns[0]
    df[target_col] = (df[target_col] > 0).astype(int)  # 0 = no disease, 1 = disease

if MAX_SAMPLES and len(df) > MAX_SAMPLES:
    df = df.groupby(target_col, group_keys=False).apply(
        lambda g: g.sample(int(MAX_SAMPLES * len(g) / len(df)), random_state=SEED))

feature_names = [c for c in df.columns if c != target_col]
X = df[feature_names].values
y = df[target_col].values
print(f"Samples: {len(df)} | Features: {len(feature_names)} | "
      f"Class balance: {np.bincount(y)}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)

# Preprocessing: standardize -> PCA to N_QUBITS -> scale to [0, pi] for the feature map
scaler = StandardScaler().fit(X_train)
pca = PCA(n_components=N_QUBITS, random_state=SEED).fit(scaler.transform(X_train))
minmax = MinMaxScaler(feature_range=(0, np.pi))


def preprocess(data, fit=False):
    z = pca.transform(scaler.transform(data))
    return minmax.fit_transform(z) if fit else minmax.transform(z)


Xq_train = preprocess(X_train, fit=True)
Xq_test = preprocess(X_test)
print(f"PCA explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# ----------------------------- QSVM -------------------------------
feature_map = ZZFeatureMap(feature_dimension=N_QUBITS, reps=REPS, entanglement="linear")
qkernel = FidelityQuantumKernel(feature_map=feature_map)
qsvc = QSVC(quantum_kernel=qkernel)

print("\nTraining QSVM (computing quantum kernel matrix)...")
qsvc.fit(Xq_train, y_train)

y_pred = qsvc.predict(Xq_test)
y_score = qsvc.decision_function(Xq_test)

# ----------------------------- Metrics ----------------------------
def report(name, y_true, y_hat, y_sc):
    print(f"\n===== {name} =====")
    print(f"Accuracy : {accuracy_score(y_true, y_hat):.4f}")
    print(f"Precision: {precision_score(y_true, y_hat):.4f}")
    print(f"Recall   : {recall_score(y_true, y_hat):.4f}")
    print(f"F1-score : {f1_score(y_true, y_hat):.4f}")
    print(f"ROC-AUC  : {roc_auc_score(y_true, y_sc):.4f}")
    print("Confusion matrix:\n", confusion_matrix(y_true, y_hat))
    print(classification_report(y_true, y_hat, target_names=["No disease", "Disease"]))


report("QSVM", y_test, y_pred, y_score)

# Classical baseline on the same features for comparison
svc = SVC(kernel="rbf").fit(Xq_train, y_train)
report("Classical SVM (RBF) baseline", y_test, svc.predict(Xq_test),
       svc.decision_function(Xq_test))

# ------------------------ Sample data testing ---------------------
LABELS = {0: "No disease", 1: "Disease"}

print("\n===== Sample predictions from the test set =====")
rng = np.random.default_rng(SEED)
idx = rng.choice(len(Xq_test), size=min(8, len(Xq_test)), replace=False)
for i in idx:
    p = qsvc.predict(Xq_test[i:i + 1])[0]
    flag = "OK " if p == y_test[i] else "ERR"
    print(f"[{flag}] actual={LABELS[y_test[i]]:<10} predicted={LABELS[p]:<10}")

print("\n===== Custom patient prediction =====")
# Order: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
new_patient = {
    "age": 58, "sex": 1, "cp": 4, "trestbps": 140, "chol": 260, "fbs": 0,
    "restecg": 2, "thalach": 120, "exang": 1, "oldpeak": 2.4, "slope": 2,
    "ca": 2, "thal": 7,
}
new_df = pd.DataFrame([new_patient])[feature_names]   # enforce column order
new_q = preprocess(new_df.values)
pred = qsvc.predict(new_q)[0]
print(new_patient)
print("Prediction:", LABELS[pred])
