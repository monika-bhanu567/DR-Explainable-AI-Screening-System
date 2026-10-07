from pathlib import Path
import pandas as pd


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IDRID_DIR = (
    PROJECT_ROOT
    / "ai"
    / "datasets"
    / "raw"
    / "IDRiD"
)


# ============================================================
# PRINT BASIC INFORMATION
# ============================================================

print("=" * 60)
print("IDRiD DATASET INSPECTION")
print("=" * 60)

print("\nIDRiD folder:")
print(IDRID_DIR)


# ============================================================
# CHECK IDRiD FOLDER
# ============================================================

if not IDRID_DIR.exists():
    raise FileNotFoundError(
        f"\nIDRiD folder not found:\n{IDRID_DIR}"
    )


# ============================================================
# DISEASE GRADING FOLDER
# ============================================================

# Your downloaded dataset contains:
#
# IDRiD
# └── B_Disease_Grading
#     └── B. Disease Grading
#
# So we use the nested folder here.

DISEASE_GRADING_DIR = (
    IDRID_DIR
    / "B_Disease_Grading"
    / "B. Disease Grading"
)


if not DISEASE_GRADING_DIR.exists():
    raise FileNotFoundError(
        f"\nDisease Grading folder not found:\n"
        f"{DISEASE_GRADING_DIR}"
    )


print("\nDisease Grading folder:")
print(DISEASE_GRADING_DIR)


# ============================================================
# GROUNDTRUTH / LABEL FILE
# ============================================================

GROUNDTRUTH_DIR = (
    DISEASE_GRADING_DIR
    / "2. Groundtruths"
)


if not GROUNDTRUTH_DIR.exists():
    raise FileNotFoundError(
        f"\nGroundtruth folder not found:\n"
        f"{GROUNDTRUTH_DIR}"
    )


training_csvs = list(
    GROUNDTRUTH_DIR.glob("*Training*Labels*.csv")
)


if not training_csvs:
    raise FileNotFoundError(
        f"\nTraining label CSV not found in:\n"
        f"{GROUNDTRUTH_DIR}"
    )


TRAIN_LABELS = training_csvs[0]


print("\nTraining label file:")
print(TRAIN_LABELS)


# ============================================================
# READ TRAINING LABELS
# ============================================================

df = pd.read_csv(TRAIN_LABELS)


print("\nCSV columns:")
print(df.columns.tolist())


# ============================================================
# BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 60)
print("BASIC DATASET INFORMATION")
print("=" * 60)

print(f"\nNumber of training records: {len(df)}")


print("\nFirst 10 records:")
print(
    df.head(10).to_string(index=False)
)


# ============================================================
# DR CLASS DISTRIBUTION
# ============================================================

grade_column = "Retinopathy grade"


if grade_column not in df.columns:
    raise KeyError(
        f"\nColumn '{grade_column}' was not found.\n"
        f"Available columns: {df.columns.tolist()}"
    )


class_names = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR"
}


print("\n" + "=" * 60)
print("DR CLASS DISTRIBUTION")
print("=" * 60)


class_counts = (
    df[grade_column]
    .value_counts()
    .sort_index()
)


for grade, count in class_counts.items():

    name = class_names.get(
        grade,
        "Unknown"
    )

    print(
        f"Class {grade} ({name}): {count}"
    )


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)


print(
    df[
        ["Image name", "Retinopathy grade"]
    ].isnull().sum()
)


# ============================================================
# DUPLICATE IMAGE NAMES
# ============================================================

print("\nDuplicate image names:")

print(
    df["Image name"].duplicated().sum()
)


# ============================================================
# TRAINING IMAGE FOLDER
# ============================================================

IMAGE_DIR = (
    DISEASE_GRADING_DIR
    / "1. Original Images"
    / "a. Training Set"
)


if not IMAGE_DIR.exists():
    raise FileNotFoundError(
        f"\nTraining image folder not found:\n"
        f"{IMAGE_DIR}"
    )


print("\n" + "=" * 60)
print("TRAINING IMAGE CHECK")
print("=" * 60)


print("\nTraining image folder:")
print(IMAGE_DIR)


# ============================================================
# COUNT TRAINING IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}


images = [
    file
    for file in IMAGE_DIR.iterdir()
    if file.is_file()
    and file.suffix in image_extensions
]


print(
    f"\nTraining images found: {len(images)}"
)


# ============================================================
# CHECK IMAGE ↔ LABEL MATCHING
# ============================================================

image_names = {
    file.stem
    for file in images
}


label_names = {
    str(name).strip()
    for name in df["Image name"]
}


missing_images = (
    label_names - image_names
)


missing_labels = (
    image_names - label_names
)


print("\n" + "=" * 60)
print("IMAGE ↔ LABEL MATCHING")
print("=" * 60)


print(
    f"\nLabels in CSV: {len(label_names)}"
)

print(
    f"Images in folder: {len(image_names)}"
)

print(
    f"Labels without images: {len(missing_images)}"
)

print(
    f"Images without labels: {len(missing_labels)}"
)


# ============================================================
# FINAL RESULT
# ============================================================

print("\n" + "=" * 60)
print("INSPECTION COMPLETE")
print("=" * 60)