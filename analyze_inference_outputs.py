import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------ Ask User for Folder ------------------------
folder_name = input("📂 Enter the run folder name (e.g., 1_3): ").strip()
results_dir = os.path.join("results", folder_name)

if not os.path.exists(results_dir):
    print(f"❌ Folder not found: {results_dir}")
    exit()

# ------------------------ Collect and Plot ------------------------
mean_records = []

for file in os.listdir(results_dir):
    if file.endswith("_inference_mean.json"):
        base_name = file.replace("_inference_mean.json", "")
        json_path = os.path.join(results_dir, file)
        pt_path = os.path.join(results_dir, f"{base_name}_inference_outputs.pt")

        with open(json_path, 'r') as f:
            layer_means = json.load(f)

        for layer, value in layer_means.items():
            mean_records.append({
                "model": base_name,
                "layer": layer,
                "mean_activation": value
            })

        # Plot mean activations per layer
        plt.figure(figsize=(10, 4))
        layers = list(layer_means.keys())
        means = list(layer_means.values())
        sns.barplot(x=layers, y=means)
        plt.xticks(rotation=90)
        plt.title(f"🔍 Mean Activations - {base_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{base_name}_mean_plot.png"))
        plt.close()

        # Optional: plot a sample heatmap of inference outputs (first batch only)
        if os.path.exists(pt_path):
            activations = torch.load(pt_path, map_location="cpu")
            for layer, output in activations.items():
                sample = output[0]  # first sample in batch
                if isinstance(sample, torch.Tensor) and sample.ndim >= 2:
                    plt.figure(figsize=(5, 4))
                    if sample.ndim == 3:
                        sample = sample.mean(0)  # average over channels
                    sns.heatmap(sample.numpy(), cmap="viridis")
                    plt.title(f"🔥 Heatmap: {base_name} - {layer}")
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f"{base_name}_{layer}_heatmap.png"))
                    plt.close()

# ------------------------ Save CSV Summary ------------------------
df = pd.DataFrame(mean_records)
output_csv = os.path.join(results_dir, "inference_summary.csv")
df.to_csv(output_csv, index=False)
print(f"✅ Saved mean inference summary to: {output_csv}")
