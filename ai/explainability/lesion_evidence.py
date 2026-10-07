from pathlib import Path
import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IDRID_DIR = PROJECT_ROOT / "ai" / "datasets" / "raw" / "IDRiD"

SEGMENTATION_DIR = (
    IDRID_DIR
    / "A_Segmentation"
    / "A. Segmentation"
)

OUTPUT_DIR = PROJECT_ROOT / "ai" / "outputs" / "lesions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


LESION_TYPES = {
    "MA": "Microaneurysms",
    "HE": "Hemorrhages",
    "EX": "Hard Exudates",
}


def find_file(filename):
    """Search recursively for a file."""

    matches = list(SEGMENTATION_DIR.rglob(filename))

    if matches:
        return matches[0]

    return None


def create_lesion_evidence(image_id):

    # Find the original segmentation image anywhere
    image_path = find_file(f"{image_id}.jpg")

    if image_path is None:
        print(f"Image not found: {image_id}.jpg")
        return

    print(f"Image found: {image_path}")

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image.")
        return

    overlay = image.copy()

    found_annotations = []

    for lesion_code, lesion_name in LESION_TYPES.items():

        mask_filename = f"{image_id}_{lesion_code}.tif"

        mask_path = find_file(mask_filename)

        if mask_path is None:
            print(f"{lesion_name}: mask not found")
            continue

        print(f"{lesion_name} mask: {mask_path}")

        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print(f"Could not read mask: {mask_path}")
            continue

        # Make mask same size as image
        if mask.shape[:2] != image.shape[:2]:

            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

        binary_mask = mask > 0

        # Highlight annotated regions
        overlay[binary_mask] = (
            overlay[binary_mask] * 0.4
            + np.array([255, 255, 255]) * 0.6
        ).astype(np.uint8)

        pixel_count = int(np.sum(binary_mask))

        found_annotations.append(
            f"{lesion_name}: {pixel_count} pixels"
        )

    # Blend original and annotation overlay
    result = cv2.addWeighted(
        image,
        0.5,
        overlay,
        0.5,
        0
    )

    output_path = OUTPUT_DIR / f"{image_id}_lesion_evidence.jpg"

    cv2.imwrite(str(output_path), result)

    print()
    print("=" * 50)
    print("LESION EVIDENCE")
    print("=" * 50)

    print(f"Image: {image_id}")

    if found_annotations:

        print("\nAnnotations found:")

        for annotation in found_annotations:
            print("-", annotation)

    else:

        print("\nNo lesion annotations found.")

    print("\nSaved to:")
    print(output_path)


if __name__ == "__main__":

    create_lesion_evidence("IDRiD_06")