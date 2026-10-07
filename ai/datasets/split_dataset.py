from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import shutil


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ============================================================
# IDRiD PATHS
# ============================================================

IDRID_DIR = (
    PROJECT_ROOT
    / "ai"
    / "datasets"
    / "raw"
    / "IDRiD"
    / "B_Disease_Grading"
    / "B. Disease Grading"
)


LABEL_FILE = (
    IDRID_DIR
    / "2. Groundtruths"
    / "a. IDRiD_Disease Grading_Training Labels.csv"
)


PROCESSED_DIR = (
    PROJECT_ROOT
    / "ai"
    / "datasets"
    / "processed"
)


ALL_IMAGES_DIR = PROCESSED_DIR / "all"
TRAIN_DIR = PROCESSED_DIR / "train"
VAL_DIR = PROCESSED_DIR / "validation"


# ============================================================
# CHECK SOURCE
# ============================================================

if not ALL_IMAGES_DIR.exists():
    raise FileNotFoundError(
        f"Processed images folder not found:\n{ALL_IMAGES_DIR}\n\n"
        "Run process_dataset.py first."
    )


# ============================================================
# READ LABELS
# ============================================================

print("=" * 60)
print("IDRiD TRAIN / VALIDATION SPLIT")
print("=" * 60)


df = pd.read_csv(LABEL_FILE)

df = df[
    ["Image name", "Retinopathy grade"]
].copy()


df["Image name"] = (
    df["Image name"]
    .astype(str)
    .str.strip()
)


print(f"\nTotal labeled images: {len(df)}")


# ============================================================
# STRATIFIED SPLIT
# ============================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["Retinopathy grade"]
)


print(f"Training images:   {len(train_df)}")
print(f"Validation images: {len(val_df)}")


# ============================================================
# CREATE FOLDERS
# ============================================================

TRAIN_DIR.mkdir(
    parents=True,
    exist_ok=True
)

VAL_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# CLEAR ONLY TRAIN / VALIDATION
# ============================================================

for folder in [TRAIN_DIR, VAL_DIR]:

    for file in folder.iterdir():

        if file.is_file():
            file.unlink()


# ============================================================
# COPY TRAINING IMAGES
# ============================================================

train_success = 0
train_missing = 0


for _, row in train_df.iterrows():

    image_name = row["Image name"]

    source = ALL_IMAGES_DIR / f"{image_name}.jpg"

    destination = TRAIN_DIR / source.name

    if source.exists():

        shutil.copy2(
            source,
            destination
        )

        train_success += 1

    else:

        train_missing += 1


# ============================================================
# COPY VALIDATION IMAGES
# ============================================================

val_success = 0
val_missing = 0


for _, row in val_df.iterrows():

    image_name = row["Image name"]

    source = ALL_IMAGES_DIR / f"{image_name}.jpg"

    destination = VAL_DIR / source.name

    if source.exists():

        shutil.copy2(
            source,
            destination
        )

        val_success += 1

    else:

        val_missing += 1


# ============================================================
# SAVE LABEL FILES
# ============================================================

train_labels_file = (
    PROCESSED_DIR / "train_labels.csv"
)

validation_labels_file = (
    PROCESSED_DIR / "validation_labels.csv"
)


train_df.to_csv(
    train_labels_file,
    index=False
)


val_df.to_csv(
    validation_labels_file,
    index=False
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 60)

print(
    train_df["Retinopathy grade"]
    .value_counts()
    .sort_index()
)


print("\n" + "=" * 60)
print("VALIDATION CLASS DISTRIBUTION")
print("=" * 60)

print(
    val_df["Retinopathy grade"]
    .value_counts()
    .sort_index()
)


# ============================================================
# FINAL CHECK
# ============================================================

print("\n" + "=" * 60)
print("FINAL CHECK")
print("=" * 60)

print(
    f"\nProcessed master images: "
    f"{len(list(ALL_IMAGES_DIR.glob('*.jpg')))}"
)

print(
    f"Training images copied:   {train_success}"
)

print(
    f"Training images missing:  {train_missing}"
)

print(
    f"Validation images copied: {val_success}"
)

print(
    f"Validation images missing: {val_missing}"
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 60)
print("SPLIT COMPLETE")
print("=" * 60)