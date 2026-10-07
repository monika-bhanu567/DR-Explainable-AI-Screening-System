from pathlib import Path
import sys

# Allow Python to find the preprocessing module
PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.append(str(PROJECT_ROOT))

from ai.preprocessing.image_preprocessing import preprocess_image


# ============================================================
# PATHS
# ============================================================

INPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "datasets"
    / "raw"
    / "IDRiD"
    / "B_Disease_Grading"
    / "B. Disease Grading"
    / "1. Original Images"
    / "a. Training Set"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "datasets"
    / "processed"
    / "all"
)


# ============================================================
# CHECK INPUT FOLDER
# ============================================================

if not INPUT_DIR.exists():
    raise FileNotFoundError(
        f"Training image folder not found:\n{INPUT_DIR}"
    )


# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# FIND IMAGES
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".JPG",
    ".JPEG",
    ".PNG"
}

images = sorted(
    [
        file
        for file in INPUT_DIR.iterdir()
        if file.is_file()
        and file.suffix in image_extensions
    ]
)


print("=" * 60)
print("IDRiD BATCH PREPROCESSING")
print("=" * 60)

print(f"\nInput images : {len(images)}")
print(f"Output folder: {OUTPUT_DIR}")


# ============================================================
# PROCESS IMAGES
# ============================================================

successful = 0
failed = 0

for index, image_path in enumerate(images, start=1):

    output_path = OUTPUT_DIR / image_path.name

    print(
        f"\nProcessing {index}/{len(images)}: "
        f"{image_path.name}"
    )

    try:

        preprocess_image(
            str(image_path),
            str(output_path)
        )

        successful += 1

    except Exception as error:

        failed += 1

        print(
            f"FAILED: {image_path.name}"
        )

        print(
            f"Error: {error}"
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("BATCH PREPROCESSING COMPLETE")
print("=" * 60)

print(f"\nTotal images : {len(images)}")
print(f"Successful   : {successful}")
print(f"Failed       : {failed}")

print("\nProcessed images saved in:")

print(OUTPUT_DIR)