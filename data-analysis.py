import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from scipy.spatial.distance import cdist

# ------------------- Load Model Weights -------------------
model_weights = torch.load("/home/users/u7594034/Área de trabalho/Masters Project/model_weights.pth", map_location="cpu")

# ------------------- Utility Functions -------------------

def shannon_entropy(data, bins=100):
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist[hist > 0]
    return -np.sum(hist * np.log2(hist))

def disequilibrium(data, bins=100):
    hist, _ = np.histogram(data, bins=bins, density=True)
    hist = hist / np.sum(hist)
    uniform = np.ones_like(hist) / len(hist)
    return np.sum((hist - uniform) ** 2)

def lmc_complexity(data):
    ent = shannon_entropy(data)
    dis = disequilibrium(data)
    return ent * dis

def sample_entropy(U, m=2, r=None):
    U = np.asarray(U)
    N = len(U)
    if r is None:
        r = 0.2 * np.std(U)
    if N <= m + 1:
        return np.nan

    def _create_sequences(data, m):
        return np.array([data[i:i + m] for i in range(len(data) - m + 1)])

    try:
        xmi = _create_sequences(U, m)
        xmj = _create_sequences(U, m + 1)

        dist_m = cdist(xmi, xmi, metric='chebyshev')
        dist_m1 = cdist(xmj, xmj, metric='chebyshev')

        count_m = np.sum(dist_m <= r) - len(xmi)
        count_m1 = np.sum(dist_m1 <= r) - len(xmj)

        if count_m == 0 or count_m1 == 0:
            return np.nan

        return -np.log(count_m1 / count_m)
    except:
        return np.nan

def sample_entropy_2d(image, m=2, r=None):
    if r is None:
        r = 0.2 * np.std(image)
    N, M = image.shape
    if N <= m or M <= m:
        return np.nan

    def _phi(m):
        count = 0
        for i in range(N - m):
            for j in range(M - m):
                template = image[i:i + m, j:j + m]
                for di in range(i + 1, N - m):
                    for dj in range(j + 1, M - m):
                        window = image[di:di + m, dj:dj + m]
                        if np.max(np.abs(template - window)) < r:
                            count += 1
        denom = (N - m) * (M - m)
        return count / denom if denom > 0 else 1e-10

    try:
        return -np.log(_phi(m + 1) / _phi(m))
    except:
        return np.nan

def multiscale_entropy(data, max_scale=5):
    mse = []
    for scale in range(1, max_scale + 1):
        if len(data) < scale:
            break
        coarse = [np.mean(data[i:i+scale]) for i in range(0, len(data)-scale+1, scale)]
        mse.append(sample_entropy(coarse))
    return mse

# ------------------- Analyze All Weights -------------------
results = []

for name, weight in model_weights.items():
    if not isinstance(weight, torch.Tensor):
        continue
    weight_np = weight.cpu().detach().numpy()
    flat = weight_np.flatten()

    # Limit large arrays
    if len(flat) > 10000:
        flat_sample = flat[:10000]
    else:
        flat_sample = flat

    lmc = lmc_complexity(flat_sample)
    sampen = sample_entropy(flat_sample)
    mse = multiscale_entropy(flat_sample)

    # Sample Entropy 2D
    if weight_np.ndim >= 2:
        shape = weight_np.shape
        weight_2d = weight_np.reshape(-1, shape[-1]) if weight_np.ndim == 2 else weight_np[0, 0]
        if weight_2d.ndim == 2:
            sampen2d = sample_entropy_2d(weight_2d)
        else:
            sampen2d = np.nan
    else:
        sampen2d = np.nan

    results.append({
        "layer": name,
        "shape": weight_np.shape,
        "LMC": lmc,
        "SampEn": sampen,
        "SampEn2D": sampen2d,
        "MSE": mse
    })

# ------------------- DataFrame Output -------------------
df = pd.DataFrame(results)
print("\n📊 Model Complexity Analysis:\n")
print(df[["layer", "shape", "LMC", "SampEn", "SampEn2D"]])

# ------------------- Plot LMC and SampEn -------------------
plt.figure(figsize=(12, 5))
sns.barplot(x="layer", y="LMC", data=df)
plt.xticks(rotation=90)
plt.title("LMC Complexity by Layer")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))
sns.barplot(x="layer", y="SampEn", data=df)
plt.xticks(rotation=90)
plt.title("Sample Entropy by Layer")
plt.tight_layout()
plt.show()

# ------------------- Optional: Plot Multiscale Entropy -------------------
plt.figure(figsize=(12, 5))
for i, row in df.iterrows():
    if isinstance(row["MSE"], list) and all(np.isfinite(row["MSE"])):
        plt.plot(row["MSE"], label=row["layer"])
plt.xlabel("Scale")
plt.ylabel("MSE")
plt.title("Multiscale Entropy (MSE) per Layer")
plt.legend(loc="upper right", fontsize="small", ncol=2)
plt.tight_layout()
plt.show()
