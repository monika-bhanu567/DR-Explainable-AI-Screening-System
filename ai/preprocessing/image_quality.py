import cv2
import numpy as np
from pathlib import Path


def check_image_quality(image_path):
    """
    Basic rule-based quality assessment for fundus images.

    Checks:
    1. Blur
    2. Brightness
    3. Contrast
    4. Retinal area
    """

    image_path = Path(image_path)

    # ---------------------------------------------
    # Read image
    # ---------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # ---------------------------------------------
    # Convert to grayscale
    # ---------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # ---------------------------------------------
    # 1. BLUR DETECTION
    # ---------------------------------------------

    blur_score = cv2.Laplacian(
        gray,
        cv2.CV_64F
    ).var()

    # ---------------------------------------------
    # 2. BRIGHTNESS
    # ---------------------------------------------

    brightness = np.mean(gray)

    # ---------------------------------------------
    # 3. CONTRAST
    # ---------------------------------------------

    contrast = np.std(gray)

    # ---------------------------------------------
    # 4. RETINAL AREA
    # ---------------------------------------------

    # Pixels that are not almost black
    retinal_pixels = np.sum(gray > 10)

    total_pixels = gray.size

    retinal_area_ratio = (
        retinal_pixels / total_pixels
    )

    # ---------------------------------------------
    # QUALITY CHECKS
    # ---------------------------------------------

    issues = []

    # Blur threshold
    if blur_score < 50:
        issues.append("Image is blurry")

    # Brightness thresholds
    if brightness < 30:
        issues.append("Image is too dark")

    elif brightness > 220:
        issues.append("Image is too bright")

    # Contrast threshold
    if contrast < 25:
        issues.append("Low contrast")

    # Retinal area threshold
    if retinal_area_ratio < 0.30:
        issues.append("Insufficient retinal area")

    # ---------------------------------------------
    # FINAL QUALITY
    # ---------------------------------------------

    if len(issues) == 0:
        quality = "good"
    else:
        quality = "poor"

    # ---------------------------------------------
    # RETURN RESULT
    # ---------------------------------------------

    return {
        "quality": quality,
        "blur_score": round(float(blur_score), 2),
        "brightness": round(float(brightness), 2),
        "contrast": round(float(contrast), 2),
        "retinal_area_ratio": round(
            float(retinal_area_ratio),
            3
        ),
        "issues": issues
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    test_image = (
        PROJECT_ROOT
        / "ai"
        / "datasets"
        / "processed"
        / "validation"
        / "IDRiD_006.jpg"
    )

    print()
    print("=" * 60)
    print("FUNDUS IMAGE QUALITY ASSESSMENT")
    print("=" * 60)

    print("Image:", test_image)

    result = check_image_quality(test_image)

    print()
    print("QUALITY RESULT")
    print("-" * 60)

    print(
        "Quality              :",
        result["quality"].upper()
    )

    print(
        "Blur score           :",
        result["blur_score"]
    )

    print(
        "Brightness           :",
        result["brightness"]
    )

    print(
        "Contrast             :",
        result["contrast"]
    )

    print(
        "Retinal area ratio   :",
        result["retinal_area_ratio"]
    )

    print(
        "Issues               :",
        result["issues"]
    )

    print()
    print("=" * 60)
    print("QUALITY TEST COMPLETE")
    print("=" * 60)