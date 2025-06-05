import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from medmnist import INFO
import medmnist
from sklearn.metrics import accuracy_score
import numpy as np
import random
import time
from torch.utils.tensorboard import SummaryWriter
import os
import pandas as pd
import gc
import re
import json

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

# Simple CNN for 2D
def get_simple_cnn(in_channels, num_classes):
    return nn.Sequential(
        nn.Conv2d(in_channels, 16, kernel_size=3), nn.BatchNorm2d(16), nn.ReLU(),
        nn.Conv2d(16, 16, kernel_size=3), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3), nn.BatchNorm2d(64), nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(64 * 4 * 4, 128), nn.ReLU(),
        nn.Linear(128, 128), nn.ReLU(),
        nn.Linear(128, num_classes)
    )

# Ask user for number of epochs
NUM_EPOCHS = int(input("🔢 Enter number of epochs: ").strip())

# Auto-incremental experiment ID
def get_next_id(results_dir="results"):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        return 1
    existing_ids = []
    pattern = re.compile(r"^(\d+)_\d+$")
    for name in os.listdir(results_dir):
        match = pattern.match(name)
        if match:
            existing_ids.append(int(match.group(1)))
    return max(existing_ids, default=0) + 1

# Counter and save folder
global_id_counter = get_next_id()
folder_name = f"{global_id_counter}_{NUM_EPOCHS}"
save_dir = os.path.join("results", folder_name)
os.makedirs(save_dir, exist_ok=True)

all_results = []

def identify_layer_type(layer):
    if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Conv3d):
        return "Convolutional"
    elif isinstance(layer, nn.BatchNorm2d) or isinstance(layer, nn.BatchNorm3d):
        return "BatchNorm"
    elif isinstance(layer, nn.ReLU):
        return "Activation"
    elif isinstance(layer, nn.MaxPool2d) or isinstance(layer, nn.AdaptiveAvgPool3d):
        return "Pooling"
    elif isinstance(layer, nn.Linear):
        return "Linear"
    elif isinstance(layer, nn.Flatten):
        return "Flatten"
    else:
        return layer.__class__.__name__

def run_inference(model, dataloader):
    model.eval()
    outputs = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.float().to(device)
            output = model(inputs)
            outputs.append(output.cpu())
    all_outputs = torch.cat(outputs, dim=0)
    mean_activation = torch.mean(all_outputs, dim=0)
    return mean_activation.tolist(), all_outputs.numpy()

def run_experiment(data_flag, size=28, is_3d=False, use_resnet=False):
    print(f"\n▶️ Running: {data_flag.upper()} | Size: {size} | 3D: {is_3d} | ResNet: {use_resnet}")
    set_seed()

    info = INFO[data_flag]
    n_channels = info['n_channels']
    n_classes = len(info['label'])
    DataClass = getattr(medmnist, info['python_class'])

    transform = None if is_3d else transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])
    kwargs = {'split': 'train', 'transform': transform, 'download': True}
    if size != 28:
        kwargs['size'] = size

    train_dataset = DataClass(**kwargs)
    test_dataset = DataClass(split='test', transform=transform, download=True, size=size if size != 28 else None)

    if is_3d and size >= 64:
        batch_size = 8
        test_batch_size = 16
    else:
        batch_size = 128
        test_batch_size = 256

    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False)

    if use_resnet:
        model = torchvision.models.resnet18(num_classes=n_classes)
        if n_channels != 3:
            model.conv1 = nn.Conv2d(n_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    elif is_3d:
        model = nn.Sequential(
            nn.Conv3d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32), nn.ReLU(),
            nn.AdaptiveAvgPool3d((1, 1, 1)), nn.Flatten(),
            nn.Linear(32, n_classes)
        )
    else:
        model = get_simple_cnn(n_channels, n_classes)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    weights_filename = f"{data_flag}_{size}_weights.pth"
    save_path = os.path.join(save_dir, weights_filename)

    writer = SummaryWriter(log_dir=os.path.join("runs", f"{data_flag}_{size}_{folder_name}"))

    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0
        for inputs, targets in train_loader:
            inputs = inputs.float().to(device)
            targets = targets.squeeze().long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            for name, param in model.named_parameters():
                writer.add_histogram(f"Weights/{name}", param, epoch)
                if param.grad is not None:
                    writer.add_histogram(f"Gradients/{name}", param.grad, epoch)

        avg_train_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        if epoch == 0 and not is_3d:
            img_grid = torchvision.utils.make_grid(inputs[:16])
            writer.add_image("Input/sample", img_grid, epoch)

        model.eval()
        test_loss = 0.0
        y_true, y_pred = [], []
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.float().to(device)
                targets = targets.squeeze().long().to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(predicted.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        writer.add_scalar("Loss/test", test_loss / len(test_loader), epoch)
        writer.add_scalar("Accuracy/test", acc, epoch)

        for param_group in optimizer.param_groups:
            writer.add_scalar("LearningRate", param_group["lr"], epoch)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    writer.add_scalar("Params", total_params, 0)
    writer.add_text("Experiment Info", f"Dataset: {data_flag}\nSize: {size}\nResNet: {use_resnet}")
    writer.close()

    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved weights to: {save_path}")

    torch.cuda.empty_cache()
    gc.collect()

    layer_types = [(name, identify_layer_type(layer)) for name, layer in model.named_modules() if name]
    inference_mean, inference_outputs = run_inference(model, test_loader)

    with open(os.path.join(save_dir, f"{data_flag}_{size}_inference_mean.json"), "w") as f:
        json.dump({f"neuron_{i}": val for i, val in enumerate(inference_mean)}, f)

    with open(os.path.join(save_dir, f"{data_flag}_{size}_layer_types.json"), "w") as f:
        json.dump(layer_types, f, indent=2)

    torch.save({"layer": torch.tensor(inference_outputs)}, os.path.join(save_dir, f"{data_flag}_{size}_inference_outputs.pt"))

    result = {
        "dataset": data_flag,
        "size": size,
        "3d": is_3d,
        "resnet": use_resnet,
        "params": total_params,
        "accuracy": acc,
        "weights_file": weights_filename,
        "layers": layer_types,
        "inference_mean": inference_mean,
        "inference_outputs": inference_outputs.tolist()
    }
    all_results.append(result)
    return result

experiments = [
    ('pathmnist', 28, False, False),
    ('pathmnist', 224, False, True),
    ('organmnist3d', 28, True, False),
    ('organmnist3d', 64, True, False)
]

results = [run_experiment(*args) for args in experiments]
df = pd.DataFrame([{k: v for k, v in r.items() if k not in ['layers', 'inference_outputs']} for r in all_results])
df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
print(f"✅ Results saved to {os.path.join(save_dir, 'summary.csv')}")
