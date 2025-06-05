import os
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = "runs"
available_runs = sorted(os.listdir(log_dir))

if not available_runs:
    print("⚠️ No TensorBoard runs found in the 'runs/' folder.")
    exit()

print(f"📦 Found {len(available_runs)} run folders. Extracting scalars...\n")

for run_folder in available_runs:
    run_path = os.path.join(log_dir, run_folder)
    event_acc = EventAccumulator(run_path)

    try:
        event_acc.Reload()
    except Exception as e:
        print(f"❌ Failed to load {run_path}: {e}")
        continue

    tags = event_acc.Tags().get("scalars", [])
    all_scalars = []

    for tag in tags:
        scalars = event_acc.Scalars(tag)
        for s in scalars:
            all_scalars.append({
                "run": run_folder,
                "tag": tag,
                "step": s.step,
                "value": s.value,
                "wall_time": s.wall_time
            })

    df = pd.DataFrame(all_scalars)
    results_dir = os.path.join("results", run_folder)
    os.makedirs(results_dir, exist_ok=True)
    output_path = os.path.join(results_dir, "tensorboard_scalars.csv")

    if df.empty:
        print(f"⚠️ No scalar data extracted from: {run_folder}")
    else:
        df.to_csv(output_path, index=False)
        print(f"✅ Exported scalars from '{run_folder}' to: {output_path}")

print("\n🏁 All runs processed.")
