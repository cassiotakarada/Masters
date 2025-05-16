import os

options = {
    "1": ("Extract model weights after training", "medmnist_weights_extraction.py"),
    "2": ("Export TensorBoard scalars", "export_tensorboard_scalars.py"),
    "3": ("Analyze inference outputs", "analyze_inference_outputs.py"),
    "4": ("Analyze weights complexity (LMC & Sample Entropy)", "data_analysis_summary.py")
}

print("🧠 Select a script to run:\n")
for key, (desc, _) in options.items():
    print(f"{key}. {desc}")

choice = input("\n🔢 Your choice: ").strip()

if choice in options:
    _, script = options[choice]
    print(f"\n🚀 Running: {script}...\n")
    os.system(f"python {script}")
else:
    print("❌ Invalid choice.")
