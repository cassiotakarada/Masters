# 🧠 MedMNIST Neural Network Complexity Analysis

This project analyzes the complexity of convolutional neural networks (CNNs) trained on [MedMNIST](https://medmnist.com/) datasets using entropy-based metrics. It evaluates models using both **Sample Entropy** and **LMC Complexity**, exports **TensorBoard** scalars, and supports inference-based neuron analysis.

---

## 📦 Project Setup

### 1. ✅ Prerequisites
- Python 3.8+
- pip

### 2. 🔧 Create a virtual environment
```bash
python -m venv myenv
source myenv/bin/activate      # Linux/macOS
myenv\Scripts\activate         # Windows
```

### 3. 📦 Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 How to Use

Run everything using the interactive tool:
```bash
python run_tool.py
```
You will see a menu:
```
1. Extract model weights after training
2. Export TensorBoard scalars
3. Analyze inference outputs
4. Analyze weights complexity (LMC & Sample Entropy)
```

### 1️⃣ Extract Model Weights (Training)
```bash
# inside tool or manually
python medmnist_weights_extraction.py
```
- Prompts for number of epochs
- Automatically creates a folder like `results/1_3` containing:
  - CNN/ResNet model weights (`*.pth`)
  - Inference results (`*_inference_outputs.pt`, `*_inference_mean.json`)
  - Summary of run (`summary.csv`)

TensorBoard logs are saved in:
- `runs/pathmnist_28_1_3/`, `runs/organmnist3d_64_1_3/`, etc.

### 2️⃣ Export TensorBoard Scalars
```bash
python export_tensorboard_scalars.py
```
- Prompts for folder like `1_3`
- Extracts scalars and saves as:
  - `results/1_3/tensorboard_scalars.csv`

### 3️⃣ Analyze Inference Outputs
```bash
python analyze_inference_outputs.py
```
- Prompts for folder like `1_3`
- Outputs:
  - `inference_summary.csv` with mean activations
  - Mean barplot: `*_mean_plot.png`
  - Optional neuron heatmaps: `*_heatmap.png`

### 4️⃣ Analyze Weight Complexity
```bash
python data_analysis_summary.py
```
- Prompts for folder like `1_3`
- Computes entropy metrics:
  - `weights_entropy_results.csv`
  - Plots: LMC barplot, SampEn barplot, LMC by layer type

---

## 📁 Folder Structure
```
project/
├── results/
│   └── 1_3/
│       ├── *.pth (model weights)
│       ├── summary.csv
│       ├── tensorboard_scalars.csv
│       ├── inference_summary.csv
│       ├── *_inference_mean.json
│       ├── *_inference_outputs.pt
│       ├── *_mean_plot.png / *_heatmap.png
│       └── weights_entropy_results.csv
│
├── runs/
│   └── pathmnist_28_1_3/ (TensorBoard event files)
│   └── organmnist3d_64_1_3/ (etc.)
│
├── run_tool.py
├── medmnist_weights_extraction.py
├── export_tensorboard_scalars.py
├── analyze_inference_outputs.py
├── data_analysis_summary.py
└── requirements.txt
```

---

## 📊 Metrics Explained
| Metric            | Description                                              |
|------------------|----------------------------------------------------------|
| **Sample Entropy** | Measures unpredictability/irregularity of weights        |
| **LMC Complexity** | Product of Shannon entropy and disequilibrium            |
| **Mean Inference** | Average neuron activation after running inference         |

---

## 🧩 Notes
- Works with both **2D (PathMNIST)** and **3D (OrganMNIST3D)** datasets
- Supports both ResNet and simple CNN architectures
- All outputs are saved under `results/{run_id}_{epochs}/`
- Use `tensorboard --logdir runs` to visualize training logs

---

🙋‍♂️ Feel free to fork, contribute or suggest ideas (like comparing architectures, new entropy metrics, or UI dashboards).
