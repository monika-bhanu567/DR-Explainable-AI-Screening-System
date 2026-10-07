import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.models.dr_classifier import DRClassifier
from ai.datasets.dr_dataset import IDRiDDataset
from ai.preprocessing.augmentation import get_validation_transforms


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_CLASSES = 5

MODEL_PATH = PROJECT_ROOT / "ai" / "checkpoints" / "dr_model_512.pth"
TEMPERATURE_PATH = PROJECT_ROOT / "ai" / "checkpoints" / "temperature.pt"

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


# ---------------------------------------------------------
# DEVICE
# ---------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)


# ---------------------------------------------------------
# LOAD DATASET
# ---------------------------------------------------------

print()
print("=" * 60)
print("LOADING VALIDATION DATA")
print("=" * 60)

val_dataset = IDRiDDataset(
    image_dir=VALIDATION_IMAGE_DIR,
    label_file=VALIDATION_LABEL_FILE,
    transform=get_validation_transforms(IMAGE_SIZE)
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Validation samples:", len(val_dataset))


# ---------------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------------

model = DRClassifier(num_classes=NUM_CLASSES)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(checkpoint)

model = model.to(device)
model.eval()

print("Loaded model:", MODEL_PATH)


# ---------------------------------------------------------
# LOAD TEMPERATURE
# ---------------------------------------------------------

temperature_data = torch.load(
    TEMPERATURE_PATH,
    map_location="cpu"
)

temperature = float(temperature_data["temperature"])

print("Loaded temperature:", round(temperature, 4))


# ---------------------------------------------------------
# COLLECT LOGITS + LABELS
# ---------------------------------------------------------

print()
print("Collecting validation predictions...")

all_logits = []
all_labels = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())


logits = torch.cat(all_logits)
labels = torch.cat(all_labels)

print("Collected predictions:", len(labels))


# ---------------------------------------------------------
# PROBABILITIES
# ---------------------------------------------------------

# Original model probabilities
original_probabilities = torch.softmax(logits, dim=1)


# Calibrated probabilities
calibrated_probabilities = torch.softmax(
    logits / temperature,
    dim=1
)


# ---------------------------------------------------------
# NLL
# ---------------------------------------------------------

criterion = nn.CrossEntropyLoss()

original_nll = criterion(
    logits,
    labels
).item()

calibrated_nll = criterion(
    logits / temperature,
    labels
).item()


# ---------------------------------------------------------
# ECE FUNCTION
# ---------------------------------------------------------

def calculate_ece(probabilities, labels, num_bins=10):

    confidences, predictions = torch.max(
        probabilities,
        dim=1
    )

    accuracies = predictions.eq(labels)

    ece = 0.0

    bin_boundaries = torch.linspace(
        0.0,
        1.0,
        num_bins + 1
    )

    for i in range(num_bins):

        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]

        if i == 0:
            mask = (
                (confidences >= lower)
                & (confidences <= upper)
            )
        else:
            mask = (
                (confidences > lower)
                & (confidences <= upper)
            )

        if mask.sum() == 0:
            continue

        bin_accuracy = accuracies[mask].float().mean()
        bin_confidence = confidences[mask].mean()

        bin_fraction = mask.float().mean()

        ece += torch.abs(
            bin_accuracy - bin_confidence
        ) * bin_fraction

    return float(ece)


# ---------------------------------------------------------
# ECE
# ---------------------------------------------------------

original_ece = calculate_ece(
    original_probabilities,
    labels
)

calibrated_ece = calculate_ece(
    calibrated_probabilities,
    labels
)


# ---------------------------------------------------------
# ACCURACY
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# RESULTS
# ---------------------------------------------------------

print()
print("=" * 60)
print("CALIBRATION EVALUATION")
print("=" * 60)

print()
print("Temperature")
print("-" * 60)
print(f"Learned temperature : {temperature:.4f}")

print()
print("ACCURACY")
print("-" * 60)
print(f"Original accuracy   : {original_accuracy:.4f}")
print(f"Calibrated accuracy : {calibrated_accuracy:.4f}")

print()
print("NEGATIVE LOG-LIKELIHOOD (NLL)")
print("-" * 60)
print(f"Original NLL        : {original_nll:.4f}")
print(f"Calibrated NLL      : {calibrated_nll:.4f}")

print()
print("EXPECTED CALIBRATION ERROR (ECE)")
print("-" * 60)
print(f"Original ECE        : {original_ece:.4f}")
print(f"Calibrated ECE      : {calibrated_ece:.4f}")


# ---------------------------------------------------------
# INTERPRETATION
# ---------------------------------------------------------

print()
print("=" * 60)
print("INTERPRETATION")
print("=" * 60)

if calibrated_nll < original_nll:
    print("✓ NLL improved after temperature scaling.")
else:
    print("• NLL did not improve after temperature scaling.")

if calibrated_ece < original_ece:
    print("✓ ECE improved after temperature scaling.")
else:
    print("• ECE did not improve after temperature scaling.")

if abs(
    original_accuracy - calibrated_accuracy
) < 1e-6:
    print("✓ Classification accuracy remained unchanged.")

print()
print("Lower NLL and lower ECE indicate better probability calibration.")
print()
print("NOTE:")
print("This evaluation uses the same validation set used to")
print("fit the temperature. It should therefore be treated as")
print("a prototype calibration evaluation, not an independent")
print("clinical validation result.")