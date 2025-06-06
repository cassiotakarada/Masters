import os

options = {
    "1": ("Extract model weights after training", "medmnist_weights_extraction.py"),
    "2": ("Export TensorBoard scalars", "export_tensorboard_scalars.py"),
    "3": ("Run TensorBoard", "tensorboard --logdir=runs --host=0.0.0.0 --port=6006"),
    "4": ("Analyze inference outputs", "analyze_inference_outputs.py"),
    "5": ("Analyze weights complexity (LMC & Sample Entropy)", "data_analysis_summary.py"),
    "6": ("Show best training epochs summary", "best_epochs.py"),
    "7": ("Activate virtual environment", "echo '⚠️ Please run this manually: source venv/bin/activate'")
}

print("🧠 Select a script to run:\n")
for key, (desc, _) in options.items():
    print(f"{key}. {desc}")

choice = input("\n🔢 Your choice: ").strip()

if choice in options:
    description, command = options[choice]
    print(f"\n🚀 Running: {description}...\n")
    if choice == "3":  # Run TensorBoard
        os.system(command)
    elif choice == "7":  # Just prints echo
        os.system(command)
    else:
        os.system(f"python {command}")
else:
    print("❌ Invalid choice.")
