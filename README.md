# quantum-file-tamper-detection

Quantum File Tamper Detection Simulator is an interactive Streamlit-based teaching prototype that demonstrates how quantum machine learning (QML) concepts can be applied to digital forensics and file tamper detection.
The simulator compares a Classical SVM model with a Quantum-kernel SVM (using PennyLane) for detecting whether a file has been tampered with — based on simple forensic features extracted from its raw bytes.

🚀 Features

✅ Upload or auto-generate synthetic files (binary, text, or image)

✅ Extract forensic features such as entropy, byte histogram, and unique byte count

✅ Simulate tampering by flipping random bytes in files

✅ Train & compare:

 🧩 Classical SVM (RBF kernel)
 
 ⚛️ Quantum SVM (Quantum kernel embedding using PennyLane)
 
✅ Visualize confusion matrix and quantum kernel heatmap

✅ Simple, fast, and educational prototype for quantum forensics

🧰 Requirements

Install all dependencies using pip:

-pip install streamlit pennylane scikit-learn matplotlib numpy pandas

▶️ How to Run

Save the script as Quantum_File_Tamper_Detection_Simulator.py

-streamlit run Quantum_File_Tamper_Detection_Simulator.py

The app will open in your browser (usually at http://localhost:8501)

🧪 App Workflow

1. Configuration (Sidebar)
   
Choose Mode:

Upload Files: Upload your own files

Synthetic Batch: Auto-generate random pseudo-files

Choose Model:

Classical SVM or Quantum SVM

Adjust:

Number of qubits (2–8)

Tamper intensity (fraction of bytes changed)

Number of synthetic samples

2. Run Detection
   
The app extracts features, simulates tampering, trains models, and evaluates results.

4. Results Display
   
Accuracy, Precision, Recall, F1 Score

Confusion Matrix

Quantum Kernel Heatmap (if Quantum SVM used)

⚛️ Quantum Kernel Overview

The Quantum SVM uses a custom embedding circuit:

Each feature is encoded as a rotation angle (RY) on a qubit.

Qubits are entangled using CNOT gates.

The circuit’s expectation values form the quantum embedding vector.

A quantum kernel matrix is computed using cosine similarity of embeddings.

This allows non-linear mapping of classical features into quantum space.

📚 Notes

This project is for educational and research purposes only.

Not suitable for production forensic workflows.

For real-world applications, use verified forensic libraries and secure evidence-handling procedures.

