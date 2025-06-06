import os
import pandas as pd

# Solicita o nome da pasta
folder_name = input("📂 Enter the folder name (e.g., 1_3): ").strip()
folder_path = os.path.join("results", folder_name)

if not os.path.exists(folder_path):
    print(f"❌ Folder not found: {folder_path}")
    exit()

# Coleta todos os arquivos de métricas
metrics_files = [f for f in os.listdir(folder_path) if f.endswith("_training_metrics.csv")]

if not metrics_files:
    print("⚠️ No training metrics CSV files found in the folder.")
    exit()

summary_data = []

for metrics_file in metrics_files:
    csv_path = os.path.join(folder_path, metrics_file)
    df = pd.read_csv(csv_path)

    if {"epoch", "train_loss", "test_loss", "test_accuracy"}.issubset(df.columns):
        best_epoch_row = df.loc[df["test_loss"].idxmin()]
        summary_data.append({
            "model": metrics_file.replace("_training_metrics.csv", ""),
            "best_epoch": int(best_epoch_row["epoch"]),
            "train_loss": round(best_epoch_row["train_loss"], 4),
            "test_loss": round(best_epoch_row["test_loss"], 4),
            "test_accuracy": round(best_epoch_row["test_accuracy"], 4),
        })

# Cria DataFrame e exibe
summary_df = pd.DataFrame(summary_data)
print("\n📈 Best Epoch Summary:")
print(summary_df.to_string(index=False))

# Salva em CSV
output_path = os.path.join(folder_path, "best_epoch_summary.csv")
summary_df.to_csv(output_path, index=False)
print(f"\n✅ Saved summary to: {output_path}")
