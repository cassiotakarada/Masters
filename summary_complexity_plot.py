import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

results_root = "results"
global_output_root = os.path.join(results_root, "global_lmc_sampen")

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

# Organize by dataset
datasets = {}

for folder in os.listdir(results_root):
    if not "_" in folder:
        continue

    run_path = os.path.join(results_root, folder)
    if not os.path.isdir(run_path):
        continue

    for file in os.listdir(run_path):
        if not file.endswith(".pth"):
            continue

        status = "initial" if "initial_" in file else "bef" if "bef_" in file else "aft"
        dataset_key = file.replace("initial_", "").replace("bef_", "").replace("aft_", "").replace(".pth", "")

        weight_path = os.path.join(run_path, file)
        param_path = os.path.join(run_path, f"{file.replace('.pth', '')}_param_types.json")
        if not os.path.exists(param_path):
            param_path = os.path.join(run_path, f"{dataset_key}_param_types.json")
        if not os.path.exists(param_path):
            print(f"⚠️ Missing param_types.json for {file}")
            continue

        datasets.setdefault(dataset_key, {})[status] = {
            "weight": weight_path,
            "param": param_path,
            "folder": folder
        }

# Process each dataset
for dataset, entries in datasets.items():
    print(f"📦 Processing: {dataset}")
    output_folder = os.path.join(global_output_root, dataset)
    os.makedirs(output_folder, exist_ok=True)

    summary = []

    for status in ["initial", "bef", "aft"]:
        if status not in entries:
            continue
        try:
            weights = torch.load(entries[status]["weight"], map_location="cpu")
            with open(entries[status]["param"], "r") as f:
                param_types = json.load(f)
        except Exception as e:
            print(f"❌ Failed loading {dataset} [{status}]: {e}")
            continue

        lmc_vals, sampen_vals, entropy_vals, diseq_vals = [], [], [], []

        for name, tensor in weights.items():
            if not isinstance(tensor, torch.Tensor): continue
            if any(skip in name for skip in ["bias", "running_var", "running_mean"]): continue
            flat = tensor.detach().cpu().numpy().flatten()
            sample = flat[:10000] if len(flat) > 10000 else flat
            hist = compute_normalized_histogram(sample)
            ent = shannon_entropy_from_hist(hist)
            dis = disequilibrium_from_hist(hist)
            lmc = ent * dis
            sampen = sample_entropy(sample)
            if np.isfinite(ent) and np.isfinite(dis) and np.isfinite(lmc) and np.isfinite(sampen):
                entropy_vals.append(ent)
                diseq_vals.append(dis)
                lmc_vals.append(lmc)
                sampen_vals.append(sampen)

        if not lmc_vals: continue
        summary.append({
            "status": status,
            "LMC_mean": np.mean(lmc_vals),
            "Entropy_mean": np.mean(entropy_vals),
            "Desequilibrium_mean": np.mean(diseq_vals),
            "SampEn_mean": np.mean(sampen_vals)
        })

    if not summary:
        print(f"⚠️ No data extracted for {dataset}")
        continue

    df = pd.DataFrame(summary)
    df.to_csv(os.path.join(output_folder, "summary.csv"), index=False)

    # Plot per dataset
    plt.figure(figsize=(8, 6))
    plt.plot(df["status"], df["LMC_mean"], label="Complexidade (LMC)", marker="o", color="cyan")
    plt.plot(df["status"], df["Entropy_mean"], label="Entropia", marker="o", color="red")
    plt.plot(df["status"], df["Desequilibrium_mean"], label="Desequilíbrio", marker="o", color="gold")
    plt.title(f"📈 Complexidade por Estágio - {dataset}")
    plt.xlabel("Estágio do Treinamento")
    plt.ylabel("Valor Médio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"{dataset}_complexity_plot.png"))
    plt.close()
    print(f"✅ Saved: {dataset}_complexity_plot.png")
