import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ai.models.dr_classifier import DRClassifier
from ai.datasets.dr_dataset import IDRiDDataset
from ai.preprocessing.augmentation import get_validation_transforms


# ==========================================================
# SETTINGS
# ==========================================================

IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_CLASSES = 5

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "checkpoints"
    / "dr_model_512.pth"
)

VALIDATION_IMAGE_DIR = (
    PROJECT_ROOT
    / "ai"
    / "datasets"
    / "processed"
    / "validation"
)

VALIDATION_LABEL_FILE = (
    PROJECT_ROOT
    / "ai"
    / "datasets"
    / "processed"
    / "validation_labels.csv"
)

TEMPERATURE_PATH = (
    PROJECT_ROOT
    / "ai"
    / "checkpoints"
    / "temperature.pt"
)


# ==========================================================
# DEVICE
# ==========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# ==========================================================
# TEMPERATURE SCALING
# ==========================================================

class TemperatureScaler(nn.Module):
    """
    Temperature scaling for calibrating neural-network
    logits.

    A single temperature parameter is learned using
    validation data.

    Temperature > 1 generally softens predictions.
    Temperature < 1 generally sharpens predictions.
    """

    def __init__(self):
        super().__init__()

        self.temperature = nn.Parameter(
            torch.ones(1) * 1.0
        )

    def forward(self, logits):
        return logits / self.temperature


# ==========================================================
# LOAD VALIDATION DATA
# ==========================================================

print()
print("=" * 60)
print("LOADING VALIDATION DATA")
print("=" * 60)

validation_dataset = IDRiDDataset(
    image_dir=VALIDATION_IMAGE_DIR,
    label_file=VALIDATION_LABEL_FILE,
    transform=get_validation_transforms(IMAGE_SIZE)
)

validation_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    "Validation samples:",
    len(validation_dataset)
)


# ==========================================================
# LOAD 512x512 MODEL
# ==========================================================

model = DRClassifier(
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

print(
    "Loaded model:",
    MODEL_PATH
)


# ==========================================================
# COLLECT VALIDATION LOGITS
# ==========================================================

all_logits = []
all_labels = []

print()
print("Collecting validation predictions...")

with torch.no_grad():

    for images, labels in validation_loader:

        images = images.to(device)

        logits = model(images)

        all_logits.append(
            logits.cpu()
        )

        all_labels.append(
            labels
        )


logits = torch.cat(
    all_logits
)

labels = torch.cat(
    all_labels
)


# ==========================================================
# CALIBRATION
# ==========================================================

print()
print("=" * 60)
print("TEMPERATURE SCALING")
print("=" * 60)

temperature_scaler = TemperatureScaler()

temperature_scaler = temperature_scaler.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = optim.LBFGS(
    [temperature_scaler.temperature],
    lr=0.01,
    max_iter=100
)

logits = logits.to(device)
labels = labels.to(device)


def closure():

    optimizer.zero_grad()

    scaled_logits = temperature_scaler(
        logits
    )

    loss = criterion(
        scaled_logits,
        labels
    )

    loss.backward()

    return loss


optimizer.step(closure)


# ==========================================================
# SAVE TEMPERATURE
# ==========================================================

temperature = (
    temperature_scaler.temperature
    .detach()
    .cpu()
    .item()
)

torch.save(
    {
        "temperature": temperature
    },
    TEMPERATURE_PATH
)


# ==========================================================
# COMPARE BEFORE / AFTER
# ==========================================================

with torch.no_grad():

    original_probabilities = torch.softmax(
        logits,
        dim=1
    )

    calibrated_logits = (
        logits / temperature
    )

    calibrated_probabilities = torch.softmax(
        calibrated_logits,
        dim=1
    )

    original_predictions = torch.argmax(
        original_probabilities,
        dim=1
    )

    calibrated_predictions = torch.argmax(
        calibrated_probabilities,
        dim=1
    )

    original_accuracy = (
        original_predictions == labels
    ).float().mean().item()

    calibrated_accuracy = (
        calibrated_predictions == labels
    ).float().mean().item()


# ==========================================================
# OUTPUT
# ==========================================================

print()
print(
    f"Learned temperature : {temperature:.4f}"
)

print(
    f"Original accuracy   : {original_accuracy:.4f}"
)

print(
    f"Calibrated accuracy : {calibrated_accuracy:.4f}"
)

print()
print(
    "Temperature scaling changes probability calibration."
)

print(
    "It does not retrain or modify the DR classifier."
)

print()
print("Temperature saved to:")
print(TEMPERATURE_PATH)

print()
print("=" * 60)
print("CALIBRATION COMPLETE")
print("=" * 60)