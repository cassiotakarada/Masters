import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.stats import entropy
from scipy.spatial.distance import cdist

# --------------------- Configuration ---------------------
def infer_layer_type(layer_name):
    name = layer_name.lower()
    if "conv" in name:
        return "Convolutional"
    elif "bn" in name or "batchnorm" in name:
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
        return "Unknown"

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

# ------------------- Ask Folder and Analyze Weights -------------------
folder_name = input("📂 Enter the folder name (e.g., 1_3): ").strip()
weights_dir = os.path.join("results", folder_name)
if not os.path.exists(weights_dir):
    print(f"❌ Folder not found: {weights_dir}")
    exit()

results = []
for file in os.listdir(weights_dir):
    if file.endswith("_weights.pth"):
        path = os.path.join(weights_dir, file)
        print(f"Analyzing {file}...")
        model_weights = torch.load(path, map_location="cpu")

        for name, weight in model_weights.items():
            if not isinstance(weight, torch.Tensor):
                continue
            weight_np = weight.cpu().detach().numpy()
            flat = weight_np.flatten()

            flat_sample = flat[:10000] if len(flat) > 10000 else flat

            lmc = lmc_complexity(flat_sample)
            sampen = sample_entropy(flat_sample)

            results.append({
                "file": file,
                "layer": name,
                "layer_type": infer_layer_type(name),
                "shape": weight_np.shape,
                "LMC": lmc,
                "SampEn": sampen
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
plt.figure(figsize=(12, 5))
sns.barplot(x="layer", y="LMC", hue="file", data=df)
plt.xticks(rotation=90)
plt.title("LMC Complexity by Layer")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
sns.barplot(x="layer", y="SampEn", hue="file", data=df)
plt.xticks(rotation=90)
plt.title("Sample Entropy by Layer")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
sns.boxplot(x="layer_type", y="LMC", data=df)
plt.xticks(rotation=45)
plt.title("LMC by Layer Type")
plt.tight_layout()
plt.show()

# ------------------- Save CSV -------------------
output_path = os.path.join(weights_dir, "weights_entropy_results.csv")
df.to_csv(output_path, index=False)
print(f"✅ Saved complexity results to: {output_path}")

# ------------------- Group Summary by Layer Type -------------------
group_summary = df.groupby("layer_type")[["LMC", "SampEn"]].agg(["mean", "std", "count", "min", "max"])
group_summary.columns = ['_'.join(col).strip() for col in group_summary.columns.values]  # flatten MultiIndex

summary_path = os.path.join(weights_dir, "layer_type_summary.csv")
group_summary.to_csv(summary_path)
print(f"📄 Layer-type summary saved to: {summary_path}")
