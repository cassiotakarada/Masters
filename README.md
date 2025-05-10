# MedMNIST Neural Network Complexity Analysis

This project analyzes the complexity of convolutional neural networks trained on MedMNIST datasets using entropy-based metrics.

## 📁 Structure

- `medmnist-vs2.py` — Trains models with various configurations and logs metrics using TensorBoard. Saves weights and summary CSV in structured folders.
- `export_tensorboard_scalars.py` — Extracts scalar metrics from TensorBoard logs and exports them to CSV for the specified run folder.
- `analyze_model_weights.py` — Computes LMC (Lopez-Ruiz–Mancini–Calbet) and Sample Entropy for each weight tensor in the saved models.
- `results/` — Contains folders like `1_3/`, `2_4/`, etc., each with model weights, TensorBoard logs, scalar CSVs, and entropy results.

## 🚀 How to Run

### 1. Train and Save Models
```bash
python medmnist-vs2.py
```
You will be prompted to enter:
- The run ID (e.g., `1`)
- The number of epochs (e.g., `3`)

This will create a folder `results/1_3` containing weights and training metadata.

### 2. Export TensorBoard Scalars
```bash
python export_tensorboard_scalars.py
```
You will be prompted to enter the folder (e.g., `1_3`) and it will export `tensorboard_scalars.csv` into that folder.

### 3. Analyze Model Weights
```bash
python analyze_model_weights.py
```
Enter the same folder name (e.g., `1_3`) to compute entropy and complexity metrics. A new file `weights_entropy_results.csv` will be saved there.

## 📊 Metrics
- **LMC Complexity** — Quantifies order-disorder balance using Shannon entropy and disequilibrium.
- **Sample Entropy** — Measures the regularity and unpredictability of weights.

## 📚 Requirements
Install dependencies:
```bash
pip install torch torchvision pandas matplotlib seaborn scipy tensorboard
```

## 📌 Notes
- Compatible with both 2D and 3D MedMNIST datasets
- Supports ResNet and simple CNN architectures
- Saves all outputs in `results/{id}_{epochs}/`

---

📬 Let me know if you'd like to extend this with more entropy types, model architectures, or comparisons between dataset sizes.
