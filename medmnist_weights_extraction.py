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
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

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

NUM_EPOCHS = int(input("\U0001f522 Enter number of epochs: ").strip())

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

global_id_counter = get_next_id()
folder_name = f"{global_id_counter}_{NUM_EPOCHS}"
save_dir = os.path.join("results", folder_name)
os.makedirs(save_dir, exist_ok=True)

all_results = []

def identify_layer_type(layer):
    if isinstance(layer, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return "Convolutional"
    elif isinstance(layer, nn.Linear):
        return "Linear"
    elif isinstance(layer, nn.Embedding):
        return "Embedding"
    elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm3d)):
        return "BatchNorm"
    elif isinstance(layer, nn.ReLU):
        return "Activation"
    elif isinstance(layer, (nn.MaxPool2d, nn.AdaptiveAvgPool3d)):
        return "Pooling"
    elif isinstance(layer, nn.Flatten):
        return "Flatten"
    else:
        return "Other"

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

    transform = None if is_3d else transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])
    kwargs = {'split': 'train', 'transform': transform, 'download': True}
    if size != 28:
        kwargs['size'] = size

    train_dataset = DataClass(**kwargs)
    test_dataset = DataClass(split='test', transform=transform, download=True, size=size if size != 28 else None)

    batch_size = 8 if is_3d and size >= 64 else 128
    test_batch_size = 16 if is_3d and size >= 64 else 256

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

    writer = SummaryWriter(log_dir=os.path.join("runs", f"{data_flag}_{size}_{folder_name}"))

    train_losses = []
    test_losses = []
    test_accuracies = []

    best_test_loss = float("inf")
    best_model_state = None
    best_epoch = 0
    overfit_epoch = None

    torch.save(model.state_dict(), os.path.join(save_dir, f"initial_weights_{data_flag}_{size}.pth"))

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

        avg_train_loss = running_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

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

        avg_test_loss = test_loss / len(test_loader)
        acc = accuracy_score(y_true, y_pred)
        test_losses.append(avg_test_loss)
        test_accuracies.append(acc)

        writer.add_scalar("Loss/test", avg_test_loss, epoch)
        writer.add_scalar("Accuracy/test", acc, epoch)

        if avg_test_loss < best_test_loss:
            best_test_loss = avg_test_loss
            best_model_state = model.state_dict()
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, f"bef_weights_{data_flag}_{size}.pth"))
            overfit_epoch = None
        elif overfit_epoch is None:
            overfit_epoch = epoch
            torch.save(model.state_dict(), os.path.join(save_dir, f"aft_weights_{data_flag}_{size}.pth"))

    writer.close()

    if best_model_state:
        torch.save(best_model_state, os.path.join(save_dir, f"{data_flag}_{size}_weights.pth"))

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    if overfit_epoch is not None:
        plt.axvline(x=overfit_epoch, color='red', linestyle='--', label=f'Overfit @ Epoch {overfit_epoch+1}')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Training vs Test Loss - {data_flag}_{size}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{data_flag}_{size}_loss_plot.png"))
    plt.close()

    param_type_mapping = {}
    for module_name, module in model.named_modules():
        for param_name, _ in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            param_type_mapping[full_name] = identify_layer_type(module)

    # Save original and per-version param_types
    with open(os.path.join(save_dir, f"{data_flag}_{size}_param_types.json"), "w") as f:
        json.dump(param_type_mapping, f, indent=2)

    prefixes = ["initial", "bef", "aft"]
    for prefix in prefixes:
        versioned_name = f"{prefix}_weights_{data_flag}_{size}_param_types.json"
        with open(os.path.join(save_dir, versioned_name), "w") as f:
            json.dump(param_type_mapping, f, indent=2)

    inference_mean, inference_outputs = run_inference(model, test_loader)

    with open(os.path.join(save_dir, f"{data_flag}_{size}_inference_mean.json"), "w") as f:
        json.dump({f"neuron_{i}": val for i, val in enumerate(inference_mean)}, f)

    torch.save({"layer": torch.tensor(inference_outputs)},
               os.path.join(save_dir, f"{data_flag}_{size}_inference_outputs.pt"))

    result = {
        "dataset": data_flag,
        "size": size,
        "3d": is_3d,
        "resnet": use_resnet,
        "params": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "accuracy": acc,
        "weights_file": f"{data_flag}_{size}_weights.pth",
        "inference_mean": inference_mean
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
df = pd.DataFrame(results)
df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
print(f"✅ Results saved to {os.path.join(save_dir, 'summary.csv')}")
