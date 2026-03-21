import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

np.random.seed(42)

# ==============================
# 1. Generate Light Curve
# ==============================
def generate_light_curve(has_planet):
    brightness = np.ones(300)

    # Realistic dips
    if has_planet:
        num_dips = np.random.randint(1, 3)
        for _ in range(num_dips):
            dip_center = np.random.randint(50, 250)
            dip_width = np.random.uniform(1, 3)
            dip_depth = np.random.uniform(0.002, 0.008)

            dip = dip_depth * np.exp(-(np.arange(300) - dip_center)**2 / (2 * dip_width**2))
            brightness -= dip

    # Fake dips (noise)
    else:
        if np.random.rand() < 0.3:
            dip_center = np.random.randint(50, 250)
            dip_width = np.random.uniform(1, 3)
            dip_depth = np.random.uniform(0.001, 0.005)

            dip = dip_depth * np.exp(-(np.arange(300) - dip_center)**2 / (2 * dip_width**2))
            brightness -= dip

    # Stellar variability
    variation = 0.004 * np.sin(np.linspace(0, 20, 300))
    brightness += variation

    # Noise
    noise = np.random.normal(0, 0.005, 300)
    brightness += noise

    return brightness


# ==============================
# 2. Create Dataset
# ==============================
def create_dataset(n_samples=2000):
    X = []
    y = []

    for _ in range(n_samples):
        has_planet = np.random.choice([0, 1])
        curve = generate_light_curve(has_planet)

        X.append(curve)
        y.append(has_planet)

    return np.array(X), np.array(y)


# ==============================
# 3. CNN Model
# ==============================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 32, 5)
        self.conv2 = nn.Conv1d(32, 64, 5)
        self.conv3 = nn.Conv1d(64, 128, 5)
        self.pool = nn.MaxPool1d(2)

        self.fc1 = nn.LazyLinear(128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)  # no sigmoid here

        return x


# ==============================
# 4. Train Model
# ==============================
def train_pytorch(X, y):

    # Normalize safely (numpy)
    mean = np.mean(X, axis=1, keepdims=True)
    std = np.std(X, axis=1, keepdims=True)
    std[std == 0] = 1e-8

    X = (X - mean) / std

    print("Any NaN in X:", np.isnan(X).any())

    # Convert to torch
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # Shuffle
    perm = torch.randperm(X.size(0))
    X = X[perm]
    y = y[perm]

    model = SimpleCNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(50):
        outputs = model(X)
        loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    # Accuracy
    with torch.no_grad():
        preds = (torch.sigmoid(model(X)) > 0.5).float()
        acc = (preds == y).float().mean()

    print(f"\n🚀 PyTorch Accuracy: {acc.item():.2f}")


# ==============================
# 5. Plot Sample
# ==============================
def plot_sample():
    plt.plot(generate_light_curve(1), label="Planet")
    plt.plot(generate_light_curve(0), label="No Planet")
    plt.legend()
    plt.title("Sample Light Curves")
    plt.show()


# ==============================
# 6. Main
# ==============================
def main():
    print("🌌 Generating dataset...")
    X, y = create_dataset()

    print("🤖 Training CNN model...")
    train_pytorch(X, y)

    print("📈 Showing sample curves...")
    plot_sample()


if __name__ == "__main__":
    main()