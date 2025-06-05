import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import entropy
from scipy.spatial.distance import cdist

# --------------------- Layer Type Inference ---------------------
def infer_layer_type(layer_name):
    name = layer_name.lower()
    if "conv" in name or any(k in name for k in ["0", "3", "6"]):  # heuristics for nn.Sequential
        return "Convolutional"
    elif "bn" in name or "batchnorm" in name or "running" in name:
        return "BatchNorm"
    elif "relu" in name:
        return "Activation"
    elif "pool" in name:
        return "Pooling"
    elif "linear" in name or "fc" in name:
        return "Linear"
    elif "flatten" in name:
        return "Flatten"
    else:
        return "Other"

# --------------------- Normalization ---------------------
def compute_normalized_histogram(data, bins=100):
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist / np.sum(hist)
    return hist

# --------------------- LMC Complexity ---------------------
def shannon_entropy_from_hist(hist):
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def disequilibrium_from_hist(hist):
    uniform = np.ones_like(hist) / len(hist)
    return np.sum((hist - uniform) ** 2)

def lmc_complexity(data, bins=100):
    hist = compute_normalized_histogram(data, bins=bins)
    ent = shannon_entropy_from_hist(hist)
    dis = disequilibrium_from_hist(hist)
    return ent * dis

# --------------------- Sample Entropy ---------------------
def sample_entropy(U, m=2, r=None):
    U = np.asarray(U)
    N = len(U)
    if r is None:
        r = 0.2 * np.std(U)
    if N <= m + 1:
        return np.nan

    def _create_sequences(data, m):
        return np.array([data[i:i + m] for i in range(len(data) - m + 1)])

    try:
        xmi = _create_sequences(U, m)
        xmj = _create_sequences(U, m + 1)

        dist_m = cdist(xmi, xmi, metric='chebyshev')
        dist_m1 = cdist(xmj, xmj, metric='chebyshev')

        count_m = np.sum(dist_m <= r) - len(xmi)
        count_m1 = np.sum(dist_m1 <= r) - len(xmj)

        if count_m == 0 or count_m1 == 0:
            return np.nan

        return -np.log(count_m1 / count_m)
    except:
        return np.nan

# ------------------- Folder Selection -------------------
folder_name = input("📂 Enter the folder name (e.g., 1_3): ").strip()
weights_dir = os.path.join("results", folder_name)
if not os.path.exists(weights_dir):
    print(f"❌ Folder not found: {weights_dir}")
    exit()

# ------------------- Analyze Weights -------------------
results = []
for file in os.listdir(weights_dir):
    if file.endswith("_weights.pth"):
        path = os.path.join(weights_dir, file)
        print(f"🔍 Analyzing {file}...")
        try:
            model_weights = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"⚠️ Skipping {file} due to error: {e}")
            continue

        for name, weight in model_weights.items():
            if not isinstance(weight, torch.Tensor):
                continue
            if "bias" in name.lower():
                continue
            flat = weight.detach().cpu().numpy().flatten()
            if flat.size == 0:
                continue
            flat_sample = flat[:10000] if len(flat) > 10000 else flat
            lmc = lmc_complexity(flat_sample)
            sampen = sample_entropy(flat_sample)
            layer_type = infer_layer_type(name)

            results.append({
                "file": file,
                "layer": name,
                "layer_type": layer_type,
                "shape": weight.shape,
                "LMC": lmc,
                "SampEn": sampen,
                "layer_label": f"{layer_type} | {name}"
            })

# ------------------- DataFrame Output -------------------
df = pd.DataFrame(results)
print("\n📊 Model Complexity Analysis Summary:\n")
print(df[["file", "layer", "layer_type", "shape", "LMC", "SampEn"]])

# ------------------- Ask for Specific Plot -------------------
unique_files = df["file"].unique()
menu = {str(i+1): f for i, f in enumerate(unique_files)}
menu[str(len(unique_files)+1)] = "all"

print("\nAvailable model files:")
for key, val in menu.items():
    print(f"{key}: {val}")

choice = input("\n🔎 Select the model(s) to plot by number (e.g., 1) or 'all': ").strip()

if choice != str(len(unique_files)+1):
    selected_file = menu.get(choice)
    if selected_file:
        df = df[df["file"] == selected_file]
    else:
        print("❌ Invalid choice.")
        exit()

# ------------------- Plotting -------------------
df = df.sort_values(by="layer")
df = df.dropna(subset=["LMC", "SampEn"])

# 🎨 Custom color palette for layer types
palette = {
    "Convolutional": "#1f77b4",
    "BatchNorm": "#ff7f0e",
    "Activation": "#2ca02c",
    "Pooling": "#9467bd",
    "Linear": "#8c564b",
    "Flatten": "#e377c2",
    "Other": "#7f7f7f"
}

plt.figure(figsize=(14, 6))
sns.barplot(x="layer_label", y="LMC", hue="layer_type", data=df, palette=palette, dodge=False)
plt.xticks(rotation=90)
plt.title("📊 LMC Complexity by Layer Type")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(x="layer_label", y="SampEn", hue="layer_type", data=df, palette=palette, dodge=False)
plt.xticks(rotation=90)
plt.title("📊 Sample Entropy by Layer Type")
plt.tight_layout()
plt.show()

# ------------------- Save CSV -------------------
output_path = os.path.join(weights_dir, "weights_entropy_results.csv")
df.to_csv(output_path, index=False)
print(f"✅ Saved complexity results to: {output_path}")

# ------------------- Group Summary by Layer Type -------------------
group_summary = df.groupby("layer_type")[["LMC", "SampEn"]].agg(["mean", "std", "count", "min", "max"])
group_summary.columns = ['_'.join(col).strip() for col in group_summary.columns.values]
summary_path = os.path.join(weights_dir, "layer_type_summary.csv")
group_summary.to_csv(summary_path)
print(f"📄 Layer-type summary saved to: {summary_path}")
