from pathlib import Path
import cv2
import numpy as np


# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_CAM_PATH = (
    PROJECT_ROOT
    / "ai"
    / "outputs"
    / "heatmaps"
    / "IDRiD_006_gradcam_plus_plus.npy"
)

SEGMENTATION_DIR = (
    PROJECT_ROOT
    / "ai"
    / "datasets"
    / "raw"
    / "IDRiD"
    / "A_Segmentation"
    / "A. Segmentation"
)


IMAGE_ID = "IDRiD_06"


LESION_TYPES = {
    "MA": "Microaneurysms",
    "HE": "Hemorrhages",
    "EX": "Hard Exudates",
}


# ---------------------------------------------------------
# FIND FILE
# ---------------------------------------------------------

def find_file(filename):

    matches = list(
        SEGMENTATION_DIR.rglob(filename)
    )

    if matches:
        return matches[0]

    return None


# ---------------------------------------------------------
# LOAD RAW GRAD-CAM++
# ---------------------------------------------------------

if not RAW_CAM_PATH.exists():

    print("ERROR: Raw Grad-CAM++ file not found:")
    print(RAW_CAM_PATH)

    raise SystemExit


cam = np.load(
    str(RAW_CAM_PATH)
)


print()
print("=" * 60)
print("GRAD-CAM++ XAI EVALUATION")
print("=" * 60)

print(
    "Raw CAM shape:",
    cam.shape
)


# ---------------------------------------------------------
# NORMALIZE CAM
# ---------------------------------------------------------

cam = cam.astype(np.float32)

cam -= cam.min()

if cam.max() > 0:
    cam /= cam.max()


# ---------------------------------------------------------
# FIND ORIGINAL IMAGE
# ---------------------------------------------------------

image_path = find_file(
    f"{IMAGE_ID}.jpg"
)


if image_path is None:

    print(
        f"ERROR: Could not find {IMAGE_ID}.jpg"
    )

    raise SystemExit


image = cv2.imread(
    str(image_path)
)


if image is None:

    print("ERROR: Could not read original image.")

    raise SystemExit


# ---------------------------------------------------------
# RESIZE CAM TO ORIGINAL IMAGE
# ---------------------------------------------------------

cam_resized = cv2.resize(
    cam,
    (
        image.shape[1],
        image.shape[0]
    ),
    interpolation=cv2.INTER_LINEAR
)


# ---------------------------------------------------------
# CREATE ATTENTION MASK
# ---------------------------------------------------------

# Top 20% of CAM activation
# is treated as high-attention region.

attention_threshold = np.percentile(
    cam_resized,
    80
)


attention_mask = (
    cam_resized >= attention_threshold
)


print(
    f"Attention threshold: "
    f"{attention_threshold:.4f}"
)

print()


# ---------------------------------------------------------
# EVALUATE LESIONS
# ---------------------------------------------------------

all_lesions = np.zeros(
    attention_mask.shape,
    dtype=bool
)


results = []


for lesion_code, lesion_name in LESION_TYPES.items():

    filename = (
        f"{IMAGE_ID}_{lesion_code}.tif"
    )

    mask_path = find_file(
        filename
    )


    if mask_path is None:

        print(
            f"{lesion_name}: mask not found"
        )

        continue


    mask = cv2.imread(
        str(mask_path),
        cv2.IMREAD_GRAYSCALE
    )


    if mask is None:

        print(
            f"{lesion_name}: "
            "could not read mask"
        )

        continue


    # Resize ground-truth mask
    mask = cv2.resize(
        mask,
        (
            image.shape[1],
            image.shape[0]
        ),
        interpolation=cv2.INTER_NEAREST
    )


    lesion_mask = (
        mask > 0
    )


    all_lesions[
        lesion_mask
    ] = True


    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------

    intersection = np.logical_and(
        attention_mask,
        lesion_mask
    ).sum()


    union = np.logical_or(
        attention_mask,
        lesion_mask
    ).sum()


    lesion_pixels = (
        lesion_mask.sum()
    )


    attention_pixels = (
        attention_mask.sum()
    )


    # IoU
    if union > 0:

        iou = (
            intersection /
            union
        )

    else:

        iou = 0.0


    # Percentage of lesions covered
    if lesion_pixels > 0:

        coverage = (
            intersection /
            lesion_pixels
        )

    else:

        coverage = 0.0


    # Percentage of attention inside lesions
    if attention_pixels > 0:

        precision = (
            intersection /
            attention_pixels
        )

    else:

        precision = 0.0


    results.append(
        (
            lesion_name,
            lesion_pixels,
            intersection,
            iou,
            coverage,
            precision
        )
    )


# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------

for (
    lesion_name,
    lesion_pixels,
    intersection,
    iou,
    coverage,
    precision
) in results:

    print(
        lesion_name
    )

    print(
        f"  Lesion pixels       : "
        f"{lesion_pixels}"
    )

    print(
        f"  Intersection        : "
        f"{intersection}"
    )

    print(
        f"  IoU                 : "
        f"{iou:.4f}"
    )

    print(
        f"  Lesion coverage     : "
        f"{coverage:.4f}"
    )

    print(
        f"  Attention precision : "
        f"{precision:.4f}"
    )

    print()


# ---------------------------------------------------------
# COMBINED LESION METRICS
# ---------------------------------------------------------

combined_intersection = np.logical_and(
    attention_mask,
    all_lesions
).sum()


combined_union = np.logical_or(
    attention_mask,
    all_lesions
).sum()


combined_lesion_pixels = (
    all_lesions.sum()
)


if combined_union > 0:

    combined_iou = (
        combined_intersection /
        combined_union
    )

else:

    combined_iou = 0.0


if combined_lesion_pixels > 0:

    combined_coverage = (
        combined_intersection /
        combined_lesion_pixels
    )

else:

    combined_coverage = 0.0


# ---------------------------------------------------------
# FINAL RESULT
# ---------------------------------------------------------

print("=" * 60)
print("COMBINED LESION RESULT")
print("=" * 60)

print(
    f"Combined lesion pixels : "
    f"{combined_lesion_pixels}"
)

print(
    f"Intersection           : "
    f"{combined_intersection}"
)

print(
    f"Combined IoU           : "
    f"{combined_iou:.4f}"
)

print(
    f"Lesion coverage        : "
    f"{combined_coverage:.4f}"
)

print()
print(
    "Evaluation complete."
)