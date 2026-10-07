from pathlib import Path
import cv2


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

HEATMAP_PATH = (
    PROJECT_ROOT
    / "ai"
    / "outputs"
    / "heatmaps"
    / "IDRiD_006_gradcam_plus_plus.jpg"
)

LESION_PATH = (
    PROJECT_ROOT
    / "ai"
    / "outputs"
    / "lesions"
    / "IDRiD_06_lesion_evidence.jpg"
)

OUTPUT_DIR = PROJECT_ROOT / "ai" / "outputs" / "comparisons"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "IDRiD_06_xai_comparison.jpg"


# ---------------------------------------------------------
# LOAD IMAGES
# ---------------------------------------------------------

gradcam = cv2.imread(str(HEATMAP_PATH))
lesion = cv2.imread(str(LESION_PATH))


if gradcam is None:
    print("ERROR: Grad-CAM++ image not found:")
    print(HEATMAP_PATH)
    raise SystemExit


if lesion is None:
    print("ERROR: Lesion evidence image not found:")
    print(LESION_PATH)
    raise SystemExit


# ---------------------------------------------------------
# RESIZE
# ---------------------------------------------------------

height = min(gradcam.shape[0], lesion.shape[0])

gradcam_width = int(
    gradcam.shape[1] * height / gradcam.shape[0]
)

lesion_width = int(
    lesion.shape[1] * height / lesion.shape[0]
)

gradcam = cv2.resize(
    gradcam,
    (gradcam_width, height)
)

lesion = cv2.resize(
    lesion,
    (lesion_width, height)
)


# ---------------------------------------------------------
# ADD TITLES
# ---------------------------------------------------------

title_height = 70

gradcam_panel = cv2.copyMakeBorder(
    gradcam,
    title_height,
    0,
    0,
    0,
    cv2.BORDER_CONSTANT,
    value=(30, 30, 30)
)

lesion_panel = cv2.copyMakeBorder(
    lesion,
    title_height,
    0,
    0,
    0,
    cv2.BORDER_CONSTANT,
    value=(30, 30, 30)
)


cv2.putText(
    gradcam_panel,
    "Grad-CAM++ Model Explanation",
    (20, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


cv2.putText(
    lesion_panel,
    "IDRiD Ground-Truth Lesion Evidence",
    (20, 45),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)


# ---------------------------------------------------------
# COMBINE SIDE-BY-SIDE
# ---------------------------------------------------------

comparison = cv2.hconcat(
    [gradcam_panel, lesion_panel]
)


# ---------------------------------------------------------
# SAVE
# ---------------------------------------------------------

cv2.imwrite(
    str(OUTPUT_PATH),
    comparison
)


print()
print("=" * 60)
print("XAI COMPARISON CREATED")
print("=" * 60)

print("Grad-CAM++:")
print(HEATMAP_PATH)

print()
print("Lesion Ground Truth:")
print(LESION_PATH)

print()
print("Comparison:")
print(OUTPUT_PATH)