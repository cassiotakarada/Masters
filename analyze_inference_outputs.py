import os
import json
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------ Ask User for Folder ------------------------
folder_name = input("\U0001F4C2 Enter the run folder name (e.g., 1_3): ").strip()
results_dir = os.path.join("results", folder_name)

if not os.path.exists(results_dir):
    print(f"❌ Folder not found: {results_dir}")
    exit()

# ------------------------ Collect and Plot ------------------------
mean_records = []

for file in os.listdir(results_dir):
    if file.endswith("_inference_mean.json"):
        model_name = file.replace("_inference_mean.json", "")
        json_path = os.path.join(results_dir, file)
        pt_path = os.path.join(results_dir, f"{model_name}_inference_outputs.pt")

        with open(json_path, 'r') as f:
            values = json.load(f)

        # Ensure values is a flat list
        if isinstance(values, dict):
            values = list(values.values())
        elif not isinstance(values, list):
            print(f"⚠️ Unexpected JSON format in {json_path}")
            continue

        # Save to summary list
        for i, value in enumerate(values):
            mean_records.append({
                "model": model_name,
                "output_unit": i,
                "mean_activation": value
            })

        # Plot mean activations per unit
        plt.figure(figsize=(10, 4))
        sns.barplot(x=list(range(len(values))), y=values)
        plt.xticks(rotation=90)
        plt.title(f"🔍 Mean Activations - {model_name}")
        plt.xlabel("Output Unit")
        plt.ylabel("Mean Activation")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{model_name}_mean_plot.png"))
        plt.show()

        # Optional: plot a sample heatmap of inference outputs (first batch only)
        if os.path.exists(pt_path):
            activations = torch.load(pt_path, map_location="cpu")
            if isinstance(activations, torch.Tensor):
                sample = activations[0]
                if sample.ndim >= 2:
                    plt.figure(figsize=(5, 4))
                    if sample.ndim == 3:
                        sample = sample.mean(0)
                    sns.heatmap(sample.numpy(), cmap="viridis")
                    plt.title(f"🔥 Heatmap: {model_name} - Output[0]")
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f"{model_name}_heatmap.png"))
                    plt.show()

# ------------------------ Save CSV Summary ------------------------
df = pd.DataFrame(mean_records)
output_csv = os.path.join(results_dir, "inference_summary.csv")
df.to_csv(output_csv, index=False)
print(f"✅ Saved mean inference summary to: {output_csv}")
