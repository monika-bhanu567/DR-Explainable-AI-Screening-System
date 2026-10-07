from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from ai.datasets.dr_dataset import IDRiDDataset
from ai.preprocessing.augmentation import (
    get_train_transforms,
    get_validation_transforms
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "ai" / "datasets" / "processed"

TRAIN_IMAGE_DIR = PROCESSED_DIR / "train"
VALIDATION_IMAGE_DIR = PROCESSED_DIR / "validation"

TRAIN_LABEL_FILE = PROCESSED_DIR / "train_labels.csv"
VALIDATION_LABEL_FILE = PROCESSED_DIR / "validation_labels.csv"


train_dataset = IDRiDDataset(
    image_dir=TRAIN_IMAGE_DIR,
    label_file=TRAIN_LABEL_FILE,
    transform=get_train_transforms()
)

validation_dataset = IDRiDDataset(
    image_dir=VALIDATION_IMAGE_DIR,
    label_file=VALIDATION_LABEL_FILE,
    transform=get_validation_transforms()
)


print("DATASET TEST")
print("------------------------------")

print("Training samples   :", len(train_dataset))
print("Validation samples :", len(validation_dataset))


image, label = train_dataset[0]

print()
print("First training sample:")
print("Image shape :", image.shape)
print("Label       :", label.item())
print("Image type  :", image.dtype)

print()
print("Dataset test successful!")