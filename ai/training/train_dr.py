import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler

from ai.models.dr_classifier import DRClassifier
from ai.datasets.dr_dataset import IDRiDDataset
from ai.preprocessing.augmentation import (
    get_train_transforms,
    get_validation_transforms
)


# --------------------------------------------------
# PATHS
# --------------------------------------------------

PROCESSED_DIR = PROJECT_ROOT / "ai" / "datasets" / "processed"

TRAIN_IMAGE_DIR = PROCESSED_DIR / "train"
VALIDATION_IMAGE_DIR = PROCESSED_DIR / "validation"

TRAIN_LABEL_FILE = PROCESSED_DIR / "train_labels.csv"
VALIDATION_LABEL_FILE = PROCESSED_DIR / "validation_labels.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "ai" / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.0001
NUM_CLASSES = 5


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Using device:", device)


# --------------------------------------------------
# DATASETS
# --------------------------------------------------

train_dataset = IDRiDDataset(
    image_dir=TRAIN_IMAGE_DIR,
    label_file=TRAIN_LABEL_FILE,
    transform=get_train_transforms(IMAGE_SIZE)
)

validation_dataset = IDRiDDataset(
    image_dir=VALIDATION_IMAGE_DIR,
    label_file=VALIDATION_LABEL_FILE,
    transform=get_validation_transforms(IMAGE_SIZE)
)


# --------------------------------------------------
# HANDLE CLASS IMBALANCE
# WeightedRandomSampler
# --------------------------------------------------

labels = train_dataset.labels["Retinopathy grade"].astype(int).tolist()

class_counts = torch.bincount(
    torch.tensor(labels),
    minlength=NUM_CLASSES
)

print()
print("TRAINING CLASS COUNTS")
print("------------------------------")

for class_id in range(NUM_CLASSES):
    print(
        f"Class {class_id}: "
        f"{class_counts[class_id].item()}"
    )


# Calculate weight for each class
class_weights = 1.0 / class_counts.float()

# Assign each training image the weight of its class
sample_weights = torch.tensor(
    [class_weights[label] for label in labels],
    dtype=torch.double
)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# --------------------------------------------------
# DATA LOADERS
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print()
print("Training samples:", len(train_dataset))
print("Validation samples:", len(validation_dataset))


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = DRClassifier(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# --------------------------------------------------
# LOSS FUNCTION
# --------------------------------------------------

# IMPORTANT:
# We are using WeightedRandomSampler for imbalance.
# Therefore we use NORMAL CrossEntropyLoss here.

criterion = nn.CrossEntropyLoss()


# --------------------------------------------------
# OPTIMIZER
# --------------------------------------------------

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

best_validation_loss = float("inf")

print()
print("=" * 60)
print("STARTING IMPROVED TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    # ----------------------------------------------
    # TRAINING
    # ----------------------------------------------

    model.train()

    running_train_loss = 0.0
    correct_train = 0
    total_train = 0

    for images, labels_batch in train_loader:

        images = images.to(device)
        labels_batch = labels_batch.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels_batch
        )

        loss.backward()

        optimizer.step()

        running_train_loss += loss.item()

        _, predicted = torch.max(
            outputs,
            1
        )

        total_train += labels_batch.size(0)

        correct_train += (
            predicted == labels_batch
        ).sum().item()


    train_loss = (
        running_train_loss / len(train_loader)
    )

    train_accuracy = (
        correct_train / total_train
    )


    # ----------------------------------------------
    # VALIDATION
    # ----------------------------------------------

    model.eval()

    running_validation_loss = 0.0
    correct_validation = 0
    total_validation = 0

    with torch.no_grad():

        for images, labels_batch in validation_loader:

            images = images.to(device)
            labels_batch = labels_batch.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels_batch
            )

            running_validation_loss += loss.item()

            _, predicted = torch.max(
                outputs,
                1
            )

            total_validation += labels_batch.size(0)

            correct_validation += (
                predicted == labels_batch
            ).sum().item()


    validation_loss = (
        running_validation_loss / len(validation_loader)
    )

    validation_accuracy = (
        correct_validation / total_validation
    )


    # ----------------------------------------------
    # PRINT RESULTS
    # ----------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Val Loss: {validation_loss:.4f} "
        f"| Val Acc: {validation_accuracy:.4f}"
    )


    # ----------------------------------------------
    # SAVE BEST MODEL
    # ----------------------------------------------

    if validation_loss < best_validation_loss:

        best_validation_loss = validation_loss

        model_path = (
            CHECKPOINT_DIR / "dr_model.pth"
        )

        torch.save(
            model.state_dict(),
            model_path
        )

        print(
            f"  ✓ Best model saved: {model_path}"
        )


print()
print("=" * 60)
print("IMPROVED TRAINING COMPLETE!")
print("=" * 60)

print(
    "Best model saved in:"
)

print(
    CHECKPOINT_DIR / "dr_model.pth"
)