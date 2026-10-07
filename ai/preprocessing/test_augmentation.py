from pathlib import Path
import cv2

from augmentation import get_train_transforms


# --------------------------------------------------
# PATH
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_PATH = (
    PROJECT_ROOT
    / "ai"
    / "datasets"
    / "processed"
    / "train"
    / "IDRiD_001.jpg"
)


# --------------------------------------------------
# READ IMAGE
# --------------------------------------------------

image = cv2.imread(str(IMAGE_PATH))

if image is None:
    raise FileNotFoundError(
        f"Could not read image:\n{IMAGE_PATH}"
    )


# OpenCV uses BGR.
# Albumentations works with NumPy arrays.
image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# --------------------------------------------------
# APPLY AUGMENTATION
# --------------------------------------------------

transform = get_train_transforms(
    image_size=224
)

augmented = transform(
    image=image
)["image"]


# --------------------------------------------------
# SAVE RESULT
# --------------------------------------------------

output_path = (
    PROJECT_ROOT
    / "ai"
    / "datasets"
    / "processed"
    / "train"
    / "IDRiD_001_augmented.jpg"
)


augmented = cv2.cvtColor(
    augmented,
    cv2.COLOR_RGB2BGR
)

cv2.imwrite(
    str(output_path),
    augmented
)

print("Augmentation successful!")
print(f"Saved to: {output_path}")