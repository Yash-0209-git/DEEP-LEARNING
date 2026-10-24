
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 1e-3

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


class CrackDataset(Dataset):
    def __init__(self, split):
        self.image_dir = DATA / "images" / split
        self.mask_dir = DATA / "masks" / split

        self.images = sorted(
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and (self.mask_dir / f"{p.stem}.png").exists()
        )

        self.transform = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        mask_path = self.mask_dir / f"{path.stem}.png"

        image = Image.open(path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)

        mask = mask.resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.NEAREST
        )
        mask = np.asarray(mask)

        # A positive image has a nontrivial amount of annotated crack area.
        label = int((mask > 0).mean() >= 0.001)

        return image, torch.tensor(label, dtype=torch.long)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


train_ds = CrackDataset("train")
val_ds = CrackDataset("val")
test_ds = CrackDataset("test")

print("Train:", len(train_ds))
print("Validation:", len(val_ds))
print("Test:", len(test_ds))

if not train_ds or not val_ds or not test_ds:
    raise RuntimeError("A dataset split is empty. Check the image and mask folders.")

train_labels = np.array([label.item() for _, label in train_ds])
counts = np.bincount(train_labels, minlength=2)
print("Training class counts [NO CRACK, CRACK]:", counts)

# Account for class imbalance if both classes are present.
class_weights = len(train_labels) / (
    2 * np.maximum(counts, 1)
)
weights = torch.tensor(
    class_weights, dtype=torch.float32, device=device
)

train_loader = DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
)
val_loader = DataLoader(
    val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)
test_loader = DataLoader(
    test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
)

model = SmallCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

best_val_loss = float("inf")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)

    train_loss = total_loss / len(train_ds)

    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            val_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()

    val_loss /= len(val_ds)
    val_accuracy = correct / len(val_ds)

    print(
        f"Epoch {epoch + 1}: "
        f"train_loss={train_loss:.4f}, "
        f"val_loss={val_loss:.4f}, "
        f"val_accuracy={val_accuracy:.4f}"
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), MODEL_DIR / "baseline_cnn.pth")

# Evaluate the selected model on the held-out test set.
model.load_state_dict(
    torch.load(
        MODEL_DIR / "baseline_cnn.pth",
        map_location=device,
        weights_only=True,
    )
)
model.eval()

all_labels = []
all_preds = []

with torch.no_grad():
    for images, labels in test_loader:
        logits = model(images.to(device))
        predictions = logits.argmax(1).cpu().numpy()

        all_labels.extend(labels.numpy())
        all_preds.extend(predictions)

print("\nTest classification report:")
print(classification_report(
    all_labels,
    all_preds,
    labels=[0, 1],
    target_names=["NO CRACK", "CRACK"],
    zero_division=0,
))

print("Confusion matrix:")
print(confusion_matrix(all_labels, all_preds, labels=[0, 1]))

print("\nSaved baseline model:", MODEL_DIR / "baseline_cnn.pth")
