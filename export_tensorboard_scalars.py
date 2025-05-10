import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Ask user for the run folder (e.g., 1_3)
run_folder = input("📂 Enter the run folder name (e.g., 1_3): ").strip()

# Path to TensorBoard logs
log_dir = "runs"
target_runs = [r for r in os.listdir(log_dir) if run_folder in r]

if not target_runs:
    print(f"⚠️ No matching runs found for folder '{run_folder}'")
    exit()

# Prepare scalar storage
all_scalars = []

for run in target_runs:
    run_path = os.path.join(log_dir, run)
    event_acc = EventAccumulator(run_path)

    try:
        event_acc.Reload()
    except Exception as e:
        print(f"❌ Failed to load {run_path}: {e}")
        continue

    tags = event_acc.Tags().get("scalars", [])
    for tag in tags:
        scalars = event_acc.Scalars(tag)
        for s in scalars:
            all_scalars.append({
                "run": run,
                "tag": tag,
                "step": s.step,
                "value": s.value,
                "wall_time": s.wall_time
            })

# Export to CSV inside the correct results folder
df = pd.DataFrame(all_scalars)
results_dir = os.path.join("results", run_folder)
os.makedirs(results_dir, exist_ok=True)
output_path = os.path.join(results_dir, "tensorboard_scalars.csv")

if df.empty:
    print("⚠️ No scalar data extracted.")
else:
    df.to_csv(output_path, index=False)
    print(f"✅ Exported scalar data to: {output_path}")
