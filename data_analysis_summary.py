import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from scipy.spatial.distance import cdist

results_root = "results"
graph_output_root = os.path.join(results_root, "LMC and SamEn Graphs")
os.makedirs(graph_output_root, exist_ok=True)

palette = {
    "Convolutional": "#1f77b4",
    "Linear": "#2ca02c",
    "Embedding": "#d62728",
    "BatchNorm": "#ff7f0e",
    "Activation": "#9467bd",
    "Pooling": "#8c564b",
    "Flatten": "#e377c2",
    "Other": "#7f7f7f"
}

def compute_normalized_histogram(data, bins=100):
    hist, _ = np.histogram(data, bins=bins, density=True)
    return hist / np.sum(hist)

def shannon_entropy_from_hist(hist):
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def disequilibrium_from_hist(hist):
    uniform = np.ones_like(hist) / len(hist)
    return np.sum((hist - uniform) ** 2)

def lmc_complexity(data, bins=100):
    hist = compute_normalized_histogram(data, bins)
    ent = shannon_entropy_from_hist(hist)
    dis = disequilibrium_from_hist(hist)
    return ent * dis

def sample_entropy(U, m=2, r=None):
    U = np.asarray(U)
    N = len(U)
    if r is None:
        r = 0.2 * np.std(U)
    if N <= m + 1:
        return np.nan
    try:
        xmi = np.array([U[i:i + m] for i in range(N - m)])
        xmj = np.array([U[i:i + m + 1] for i in range(N - m - 1)])
        dist_m = cdist(xmi, xmi, metric='chebyshev')
        dist_m1 = cdist(xmj, xmj, metric='chebyshev')
        count_m = np.sum(dist_m <= r) - len(xmi)
        count_m1 = np.sum(dist_m1 <= r) - len(xmj)
        if count_m == 0 or count_m1 == 0:
            return np.nan
        return -np.log(count_m1 / count_m)
    except:
        return np.nan

for folder in os.listdir(results_root):
    folder_path = os.path.join(results_root, folder)
    if not os.path.isdir(folder_path) or "_" not in folder:
        continue

    # 🗂️ Extrai número de épocas da pasta (ex: "organmnist3d_64_50" -> 50)
    try:
        epochs = folder.split("_")[-1]
        if not epochs.isdigit():
            epochs = "unknown"
    except:
        epochs = "unknown"

    save_dir = os.path.join(graph_output_root, folder)
    os.makedirs(save_dir, exist_ok=True)

    for file in os.listdir(folder_path):
        if not (file.endswith(".pth") and file.startswith(("aft_", "bef_", "initial_"))):
            continue

        print(f"Processing: {file}")

        weight_path = os.path.join(folder_path, file)
        basename = file.replace(".pth", "")
        param_type_file = f"{basename}_param_types.json"
        param_type_path = os.path.join(folder_path, param_type_file)

        if not os.path.exists(param_type_path):
            print(f"⚠️ Missing param_types.json for {file}")
            continue

        try:
            model_weights = torch.load(weight_path, map_location="cpu")
        except Exception as e:
            print(f"❌ Error loading {file}: {e}")
            continue

        with open(param_type_path, "r") as f:
            param_types = json.load(f)

        records = []
        for name, weight in model_weights.items():
            if not isinstance(weight, torch.Tensor) or any(x in name for x in ["bias", "running_var", "running_mean"]):
                continue
            flat = weight.detach().cpu().numpy().flatten()
            if flat.size == 0:
                continue
            sample = flat[:10000] if len(flat) > 10000 else flat
            lmc = lmc_complexity(sample)
            sampen = sample_entropy(sample)
            layer_type = param_types.get(name, "Other")
            records.append({
                "layer": name,
                "layer_type": layer_type,
                "LMC": lmc,
                "SampEn": sampen,
                "layer_label": f"{layer_type} | {name}"
            })

        df = pd.DataFrame(records).dropna()
        if df.empty:
            print(f"⚠️ No valid data in {file}")
            continue

        df = df.sort_values(by="layer")

        # 📸 Nome de saída agora inclui número de épocas
        lmc_output = os.path.join(save_dir, f"{basename}_LMC_{epochs}.png")
        sampen_output = os.path.join(save_dir, f"{basename}_SampEn_{epochs}.png")

        if not os.path.exists(lmc_output):
            plt.figure(figsize=(16, 6))
            sns.barplot(x="layer_label", y="LMC", hue="layer_type", data=df, palette=palette, dodge=False)
            plt.xticks(rotation=90)
            plt.title(f"LMC Complexity - {basename} ({epochs} epochs)")
            plt.tight_layout()
            plt.savefig(lmc_output)
            plt.close()
            print(f"✅ Saved: {lmc_output}")

        if not os.path.exists(sampen_output):
            plt.figure(figsize=(16, 6))
            sns.barplot(x="layer_label", y="SampEn", hue="layer_type", data=df, palette=palette, dodge=False)
            plt.xticks(rotation=90)
            plt.title(f"Sample Entropy - {basename} ({epochs} epochs)")
            plt.tight_layout()
            plt.savefig(sampen_output)
            plt.close()
            print(f"✅ Saved: {sampen_output}")
