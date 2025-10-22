import io
import os
import time
import math
import base64
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from hashlib import sha256
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pennylane as qml

st.set_page_config(page_title="Quantum File Tamper Detection Simulator", layout="wide")

RNG = np.random.default_rng(42)

# ---------------------------
# file features
# ---------------------------

def file_entropy(byte_arr: bytes) -> float:
    if len(byte_arr) == 0:
        return 0.0
    counts = np.bincount(np.frombuffer(byte_arr, dtype=np.uint8), minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))


def byte_histogram_groups(byte_arr: bytes, groups=8):
    # 256 bins grouped into `groups` larger bins
    counts = np.bincount(np.frombuffer(byte_arr, dtype=np.uint8), minlength=256)
    group_size = 256 // groups
    grouped = counts.reshape(groups, group_size).sum(axis=1)
    return grouped.astype(float)


def extract_file_features(byte_arr: bytes):
    """Return a feature vector for a file's raw bytes."""
    size = len(byte_arr)
    ent = file_entropy(byte_arr)
    uniq = len(np.unique(np.frombuffer(byte_arr, dtype=np.uint8)))
    groups = byte_histogram_groups(byte_arr, groups=8)  # 8-bin histogram
    # normalize some features to reasonable scales
    feat = np.concatenate(([size, ent, uniq], groups))
    feat = feat.astype(float)
    return feat

# ---------------------------
# Tamper simulation
# ---------------------------

def tamper_bytes(byte_arr: bytes, intensity: float = 0.01) -> bytes:
    """Randomly flip bytes in the file. intensity = fraction of bytes changed."""
    if len(byte_arr) == 0:
        return byte_arr
    arr = bytearray(byte_arr)
    n = len(arr)
    n_changes = max(1, int(n * float(np.clip(intensity, 0.0, 1.0))))
    idx = RNG.choice(n, size=n_changes, replace=False)
    for i in idx:
        arr[i] = RNG.integers(0, 256)
    return bytes(arr)

# ---------------------------
# Quantum embedding & kernel
# ---------------------------

def build_qnode(n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x):
        # x expected length == n_qubits
        # map features -> angles in [0, pi]
        for i in range(n_qubits):
            angle = float(x[i])
            # keep angles bounded
            qml.RY(angle, wires=i)
        # entangle
        for i in range(n_qubits - 1):
            qml.CNOT(wires=[i, i + 1])
        # return expectation values as an embedding
        return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

    return circuit


def quantum_embedding_vector(circuit, x, n_qubits):
    x_pad = np.pad(x, (0, max(0, n_qubits - len(x))))[:n_qubits]
    return np.array(circuit(x_pad))


def compute_quantum_kernel_matrix(circuit, X1, X2, n_qubits):
    E1 = np.array([quantum_embedding_vector(circuit, x, n_qubits) for x in X1])
    E2 = np.array([quantum_embedding_vector(circuit, x, n_qubits) for x in X2])
    K = E1 @ E2.T
    norms1 = np.linalg.norm(E1, axis=1)
    norms2 = np.linalg.norm(E2, axis=1)
    K = K / (norms1[:, None] * norms2[None, :] + 1e-12)
    return K

# ---------------------------
# App UI and Flow
# ---------------------------

st.title("Quantum File Tamper Detection Simulator 🕵️‍♀️⚛️")
st.write("Upload files or auto-generate samples. Compare Classical SVM vs Quantum-kernel SVM for tamper detection.")

with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Mode", ["Upload Files", "Synthetic Batch"])
    model_choice = st.selectbox("Model", ["Classical SVM", "Quantum SVM"], index=1)
    n_qubits = st.slider("Quantum qubits (for embedding)", 2, 8, 4)
    tamper_intensity = st.slider("Tamper intensity (fraction of bytes changed)", 0.0, 0.2, 0.02, step=0.01)
    synth_count = st.slider("Synthetic files (when in Synthetic Batch)", 10, 400, 80, step=10)
    run_button = st.button("Run Detection")


def read_uploaded_files(uploaded_files):
    files = []
    for f in uploaded_files:
        try:
            content = f.read()
            files.append((f.name, content))
        except Exception as e:
            st.warning(f"Could not read {f.name}: {e}")
    return files


def make_synthetic_files(n):
    items = []
    for i in range(n):
        # create pseudo-files: random bytes with repeatable RNG
        size = RNG.integers(256, 4096)
        b = RNG.integers(0, 256, size=size, dtype=np.uint8).tobytes()
        items.append((f"synthetic_{i}.bin", b))
    return items


# Container for results
results_container = st.container()

if run_button:
    files = []
    if mode == "Upload Files":
        uploaded = st.file_uploader("Upload one or more files", accept_multiple_files=True)
        if not uploaded:
            st.error("Please upload at least one file or switch to Synthetic Batch mode.")
        else:
            files = read_uploaded_files(uploaded)
    else:
        files = make_synthetic_files(synth_count)

    if files:
        # Build dataset: for each file, create original + tampered copy
        rows = []
        for name, content in files:
            feat_orig = extract_file_features(content)
            # create tampered copy
            tampered = tamper_bytes(content, intensity=tamper_intensity)
            feat_tam = extract_file_features(tampered)
            # store rows (features, label, name)
            rows.append((name + "::orig", feat_orig, 0))
            rows.append((name + "::tamp", feat_tam, 1))

        df = pd.DataFrame([np.concatenate(([r[0]], r[1], [r[2]])) for r in rows])
        # build columns
        feat_len = len(rows[0][1])
        cols = ["file_id"] + [f"f{i}" for i in range(feat_len)] + ["label"]
        df.columns = cols

        st.subheader("Dataset preview")
        st.write(f"Total samples: {len(df)} (orig + tampered duplicates)")
        st.dataframe(df.head(10))

        # Prepare X,y
        X = np.vstack(df[[f"f{i}" for i in range(feat_len)]].astype(float).values)
        y = df["label"].astype(int).values

        # Scale features (note: quantum embedding will re-map them but scaling helps)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        # Optional: reduce dims to n_qubits by simple projection (PCA-like via SVD)
        if Xs.shape[1] > n_qubits:
            # project to first n_qubits principal directions using SVD (cheap)
            U, S, Vt = np.linalg.svd(Xs - Xs.mean(axis=0), full_matrices=False)
            Xp = (Xs - Xs.mean(axis=0)) @ Vt.T[:, :n_qubits]
        else:
            Xp = Xs

        X_train, X_test, y_train, y_test = train_test_split(Xp, y, test_size=0.25, stratify=y, random_state=42)

        if model_choice == "Classical SVM":
            clf = SVC(kernel='rbf', probability=True)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            extra_info = None
        else:
            st.info("Computing quantum kernel (simulator). This may take some time for many samples.")
            circuit = build_qnode(n_qubits)
            K_train = compute_quantum_kernel_matrix(circuit, X_train[:, :n_qubits], X_train[:, :n_qubits], n_qubits)
            K_test = compute_quantum_kernel_matrix(circuit, X_test[:, :n_qubits], X_train[:, :n_qubits], n_qubits)
            clf = SVC(kernel='precomputed')
            clf.fit(K_train, y_train)
            y_pred = clf.predict(K_test)
            extra_info = dict(K_train=K_train, K_test=K_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        with results_container:
            st.subheader("Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{acc*100:.2f}%")
            c2.metric("Precision", f"{prec*100:.2f}%")
            c3.metric("Recall", f"{rec*100:.2f}%")
            c4.metric("F1 score", f"{f1*100:.2f}%")

            st.write("### Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            im = ax.imshow(cm, cmap='Blues')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, int(cm[i, j]), ha='center', va='center', color='black')
            st.pyplot(fig)

            if extra_info is not None:
                st.write("### Quantum Kernel (train) heatmap")
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                im2 = ax2.imshow(extra_info['K_train'], aspect='auto')
                plt.colorbar(im2, ax=ax2)
                ax2.set_title('Quantum Kernel (train samples)')
                st.pyplot(fig2)

            st.success("Detection complete. Remember: this is a teaching prototype, not a validated forensic tool.")

else:
    st.info("Configure options in the sidebar and click 'Run Detection' to begin.")
