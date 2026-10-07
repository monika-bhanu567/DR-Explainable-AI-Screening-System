from pathlib import Path

import cv2
import numpy as np


# --------------------------------------------------
# MACULAR PROXIMITY RISK
# --------------------------------------------------

def analyze_exudate_macular_proximity(
    image_path,
    exudate_mask_path,
    proximity_threshold=150
):
    """
    Analyze the proximity of hard exudates to the
    estimated macular region.

    This is a prototype decision-support signal.
    It is NOT a clinically validated medical rule.

    Parameters
    ----------
    image_path : str or Path
        Original retinal fundus image.

    exudate_mask_path : str or Path
        IDRiD hard-exudate annotation mask.

    proximity_threshold : int
        Pixel distance used to determine whether
        hard exudates are close to the estimated
        macular region.
    """

    image_path = Path(image_path)
    exudate_mask_path = Path(exudate_mask_path)

    # --------------------------------------------------
    # 1. LOAD IMAGE
    # --------------------------------------------------

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {image_path}"
        )

    height, width = image.shape[:2]

    # --------------------------------------------------
    # 2. LOAD HARD EXUDATE MASK
    # --------------------------------------------------

    exudate_mask = cv2.imread(
        str(exudate_mask_path),
        cv2.IMREAD_GRAYSCALE
    )

    if exudate_mask is None:
        raise FileNotFoundError(
            f"Could not load exudate mask: "
            f"{exudate_mask_path}"
        )

    # Resize mask if dimensions differ
    if exudate_mask.shape != (height, width):
        exudate_mask = cv2.resize(
            exudate_mask,
            (width, height),
            interpolation=cv2.INTER_NEAREST
        )

    # Convert to binary
    exudate_binary = (
        exudate_mask > 0
    ).astype(np.uint8)

    exudate_pixels = int(
        np.sum(exudate_binary)
    )

    # --------------------------------------------------
    # 3. ESTIMATE RETINAL REGION
    # --------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    retinal_region = (
        gray > 10
    ).astype(np.uint8)

    # --------------------------------------------------
    # 4. ESTIMATE MACULAR REGION
    # --------------------------------------------------
    #
    # For this prototype we estimate the macula as
    # a region approximately 35% of the retinal width
    # away from the optic-disc side.
    #
    # This is an engineering approximation, NOT a
    # clinically validated anatomical localization.
    # --------------------------------------------------

    retinal_pixels = np.column_stack(
        np.where(retinal_region > 0)
    )

    if len(retinal_pixels) == 0:
        raise ValueError(
            "Could not identify retinal region."
        )

    y_min, x_min = retinal_pixels.min(axis=0)
    y_max, x_max = retinal_pixels.max(axis=0)

    retinal_width = x_max - x_min
    retinal_height = y_max - y_min

    macula_center_x = int(
        x_min + 0.65 * retinal_width
    )

    macula_center_y = int(
        y_min + 0.50 * retinal_height
    )

    # --------------------------------------------------
    # 5. CREATE ESTIMATED MACULAR REGION
    # --------------------------------------------------

    macula_radius = int(
        min(retinal_width, retinal_height) * 0.08
    )

    macula_mask = np.zeros(
        (height, width),
        dtype=np.uint8
    )

    cv2.circle(
        macula_mask,
        (
            macula_center_x,
            macula_center_y
        ),
        macula_radius,
        1,
        -1
    )

    # --------------------------------------------------
    # 6. CALCULATE DISTANCE FROM EXUDATES
    # --------------------------------------------------

    if exudate_pixels == 0:

        return {
            "status": "NO_EXUDATES_DETECTED",
            "risk_level": "LOW",
            "prototype_estimation": True,
            "exudate_pixels": 0,
            "minimum_distance_pixels": None,
            "macular_proximity": False,
            "macula_center": {
                "x": macula_center_x,
                "y": macula_center_y
            },
            "proximity_threshold": proximity_threshold,
            "note": (
                "No hard-exudate pixels were present "
                "in the supplied annotation mask."
            )
        }

    # Distance transform from macular region
    distance_map = cv2.distanceTransform(
        1 - macula_mask,
        cv2.DIST_L2,
        5
    )

    exudate_distances = distance_map[
        exudate_binary > 0
    ]

    minimum_distance = float(
        np.min(exudate_distances)
    )

    # --------------------------------------------------
    # 7. DETERMINE PROXIMITY
    # --------------------------------------------------

    is_near_macula = (
        minimum_distance <= proximity_threshold
    )

    if is_near_macula:

        risk_level = "ELEVATED"

        status = "EXUDATES_NEAR_MACULAR_REGION"

        note = (
            "Hard-exudate annotation pixels are "
            "within the prototype proximity threshold "
            "of the estimated macular region."
        )

    else:

        risk_level = "LOW"

        status = "EXUDATES_NOT_NEAR_MACULAR_REGION"

        note = (
            "Hard-exudate annotation pixels are "
            "outside the prototype proximity threshold "
            "from the estimated macular region."
        )

    # --------------------------------------------------
    # 8. RETURN RESULT
    # --------------------------------------------------

    return {
        "status": status,
        "risk_level": risk_level,
        "prototype_estimation": True,
        "exudate_pixels": exudate_pixels,
        "minimum_distance_pixels": round(
            minimum_distance,
            2
        ),
        "macular_proximity": bool(
            is_near_macula
        ),
        "macula_center": {
            "x": macula_center_x,
            "y": macula_center_y
        },
        "proximity_threshold": proximity_threshold,
        "note": note
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    PROJECT_ROOT = Path(
        __file__
    ).resolve().parents[3]

    test_image = (
        PROJECT_ROOT
        / "ai"
        / "datasets"
        / "raw"
        / "IDRiD"
        / "A_Segmentation"
        / "A. Segmentation"
        / "1. Original Images"
        / "a. Training Set"
        / "IDRiD_06.jpg"
    )

    test_exudate_mask = (
        PROJECT_ROOT
        / "ai"
        / "datasets"
        / "raw"
        / "IDRiD"
        / "A_Segmentation"
        / "A. Segmentation"
        / "2. All Segmentation Groundtruths"
        / "a. Training Set"
        / "3. Hard Exudates"
        / "IDRiD_06_EX.tif"
    )

    print()
    print("=" * 60)
    print("MACULAR / EXUDATE PROXIMITY TEST")
    print("=" * 60)

    result = analyze_exudate_macular_proximity(
        test_image,
        test_exudate_mask
    )

    print()

    for key, value in result.items():
        print(f"{key}: {value}")

    print()
    print("=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)