import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score
)

from ai.models.dr_classifier import DRClassifier
from ai.datasets.dr_dataset import IDRiDDataset
from ai.preprocessing.augmentation import get_validation_transforms


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROCESSED_DIR = PROJECT_ROOT / "ai" / "datasets" / "processed"

VALIDATION_IMAGE_DIR = PROCESSED_DIR / "validation"
VALIDATION_LABEL_FILE = PROCESSED_DIR / "validation_labels.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "checkpoints"
    / "dr_model_512.pth"
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_SIZE = 512
BATCH_SIZE = 4
NUM_CLASSES = 5

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# VALIDATION DATASET
# --------------------------------------------------

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

print("Validation samples:", len(validation_dataset))
print("Input size:", f"{IMAGE_SIZE}x{IMAGE_SIZE}")


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

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


# --------------------------------------------------
# GET PREDICTIONS
# --------------------------------------------------

all_labels = []
all_predictions = []


with torch.no_grad():

    for images, labels in validation_loader:

        images = images.to(device)

        outputs = model(images)

        _, predictions = torch.max(
            outputs,
            1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# --------------------------------------------------
# CALCULATE METRICS
# --------------------------------------------------

accuracy = np.mean(
    np.array(all_labels)
    == np.array(all_predictions)
)

balanced_accuracy = balanced_accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)


# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print()
print("=" * 60)
print("512x512 DR MODEL EVALUATION")
print("=" * 60)

print(
    f"Validation samples  : {len(all_labels)}"
)

print(
    f"Accuracy            : {accuracy:.4f}"
)

print(
    f"Balanced Accuracy   : {balanced_accuracy:.4f}"
)

print(
    f"Macro F1 Score      : {macro_f1:.4f}"
)


# --------------------------------------------------
# CLASSIFICATION REPORT
# --------------------------------------------------

print()
print("CLASSIFICATION REPORT")
print("-" * 60)

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=CLASS_NAMES,
        zero_division=0
    )
)


# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)

print()
print("CONFUSION MATRIX")
print("-" * 60)

print(cm)

print()
print("=" * 60)
print("512x512 EVALUATION COMPLETE")
print("=" * 60)