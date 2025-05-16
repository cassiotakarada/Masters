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
        model_name = file.replace("_inference_mean.json", "")
        json_path = os.path.join(results_dir, file)
        pt_path = os.path.join(results_dir, f"{model_name}_inference_outputs.pt")

        # Load JSON mean activations
        with open(json_path, 'r') as f:
            values = json.load(f)

        for idx, val in enumerate(values):
            mean_records.append({
                "model": model_name,
                "unit": f"output_{idx}",
                "mean_activation": val
            })

        # Plot mean activations (bar chart)
        plt.figure(figsize=(10, 4))
        sns.barplot(x=list(range(len(values))), y=values)
        plt.title(f"🔍 Mean Activations - {model_name}")
        plt.xlabel("Output Unit")
        plt.ylabel("Mean Activation")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{model_name}_mean_plot.png"))
        plt.show()

        # Optional: visualize sample output from .pt file
        if os.path.exists(pt_path):
            output_tensor = torch.load(pt_path, map_location="cpu")  # shape: [N, C]
            if isinstance(output_tensor, torch.Tensor) and output_tensor.ndim == 2:
                sample = output_tensor[0].numpy()
                plt.figure(figsize=(6, 3))
                sns.heatmap(sample.reshape(1, -1), cmap="viridis", cbar=True)
                plt.title(f"🔥 Output Activations Heatmap - {model_name}")
                plt.tight_layout()
                plt.savefig(os.path.join(results_dir, f"{model_name}_output_heatmap.png"))
                plt.show()

# ------------------------ Save CSV Summary ------------------------
df = pd.DataFrame(mean_records)
output_csv = os.path.join(results_dir, "inference_summary.csv")
df.to_csv(output_csv, index=False)
print(f"✅ Saved mean inference summary to: {output_csv}")
