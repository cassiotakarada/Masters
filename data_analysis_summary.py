import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from scipy.stats import entropy
from scipy.spatial.distance import cdist

def compute_normalized_histogram(data, bins=100):
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist / np.sum(hist)
    return hist

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

folder_name = input("📂 Enter the folder name (e.g., 1_3): ").strip()
weights_dir = os.path.join("results", folder_name)
if not os.path.exists(weights_dir):
    print(f"❌ Folder not found: {weights_dir}")
    exit()

results = []
param_type_cache = {}

for file in os.listdir(weights_dir):
    if file.endswith("_weights.pth"):
        basename = file.replace("_weights.pth", "")
        json_file = os.path.join(weights_dir, f"{basename}_param_types.json")
        if not os.path.isfile(json_file):
            print(f"⚠️ No param_types.json for {file}, skipping...")
            continue

        with open(json_file, "r") as f:
            param_type_cache[file] = json.load(f)

        path = os.path.join(weights_dir, file)
        print(f"🔍 Analyzing {file}...")
        try:
            model_weights = torch.load(path, map_location="cpu")
        except Exception as e:
            print(f"⚠️ Skipping {file} due to error: {e}")
            continue

        param_types = param_type_cache[file]
        epoch_match = file.split("_e")[-1].replace("_weights.pth", "") if "_e" in file else "final"

        for name, weight in model_weights.items():
            if not isinstance(weight, torch.Tensor):
                continue
            if any(skip in name.lower() for skip in ["bias", "running_var", "running_mean"]):
                continue
            flat = weight.detach().cpu().numpy().flatten()
            if flat.size == 0:
                continue
            flat_sample = flat[:10000] if len(flat) > 10000 else flat
            lmc = lmc_complexity(flat_sample)
            sampen = sample_entropy(flat_sample)

            layer_type = param_types.get(name, "Other")
            results.append({
                "file": file,
                "dataset": basename,
                "epoch": epoch_match,
                "layer": name,
                "layer_type": layer_type,
                "shape": weight.shape,
                "LMC": lmc,
                "SampEn": sampen,
                "layer_label": f"{layer_type} | {name} | Epoch {epoch_match}"
            })

df = pd.DataFrame(results)
print("\n📊 Model Complexity Analysis Summary:\n")
print(df[["file", "layer", "layer_type", "shape", "LMC", "SampEn"]])

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

df = df.dropna(subset=["LMC", "SampEn"])
df = df.sort_values(by="layer")

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

title_suffix = f" ({folder_name})" if choice == str(len(unique_files)+1) else ""

plt.figure(figsize=(14, 6))
sns.barplot(x="layer_label", y="LMC", hue="layer_type", data=df, palette=palette, dodge=False)
plt.xticks(rotation=90)
plt.title(f"📊 LMC Complexity by Layer Type{title_suffix}")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
sns.barplot(x="layer_label", y="SampEn", hue="layer_type", data=df, palette=palette, dodge=False)
plt.xticks(rotation=90)
plt.title(f"📊 Sample Entropy by Layer Type{title_suffix}")
plt.tight_layout()
plt.show()

output_path = os.path.join(weights_dir, "weights_entropy_results.csv")
df.to_csv(output_path, index=False)
print(f"✅ Saved complexity results to: {output_path}")

group_summary = df.groupby("layer_type")[["LMC", "SampEn"]].agg(["mean", "std", "count", "min", "max"])
group_summary.columns = ['_'.join(col).strip() for col in group_summary.columns.values]
summary_path = os.path.join(weights_dir, "layer_type_summary.csv")
group_summary.to_csv(summary_path)
print(f"📄 Layer-type summary saved to: {summary_path}")
