import cv2
import numpy as np


# --------------------------------------------------
# IMAGE REGISTRATION
# --------------------------------------------------

def align_images(previous_image, current_image):
    """
    Align the current retinal image to the previous image
    using ORB feature matching.

    This is a prototype registration method and is not
    clinically validated.
    """

    previous_gray = cv2.cvtColor(
        previous_image,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_image,
        cv2.COLOR_BGR2GRAY
    )


    orb = cv2.ORB_create(
        nfeatures=1000
    )


    keypoints_previous, descriptors_previous = (
        orb.detectAndCompute(
            previous_gray,
            None
        )
    )

    keypoints_current, descriptors_current = (
        orb.detectAndCompute(
            current_gray,
            None
        )
    )


    if (
        descriptors_previous is None
        or descriptors_current is None
    ):

        return current_image, False


    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True
    )


    matches = matcher.match(
        descriptors_previous,
        descriptors_current
    )


    matches = sorted(
        matches,
        key=lambda x: x.distance
    )


    if len(matches) < 10:

        return current_image, False


    # Use best matches
    good_matches = matches[:50]


    points_previous = np.float32([
        keypoints_previous[m.queryIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)


    points_current = np.float32([
        keypoints_current[m.trainIdx].pt
        for m in good_matches
    ]).reshape(-1, 1, 2)


    transformation, mask = cv2.findHomography(
        points_current,
        points_previous,
        cv2.RANSAC,
        5.0
    )


    if transformation is None:

        return current_image, False


    height, width = previous_image.shape[:2]


    aligned = cv2.warpPerspective(
        current_image,
        transformation,
        (width, height)
    )


    return aligned, True


# --------------------------------------------------
# IMAGE CHANGE
# --------------------------------------------------

def calculate_change(
    previous_image,
    current_image
):
    """
    Calculate structural image change after alignment.

    Returns the percentage of pixels showing
    significant visual change.
    """

    previous_gray = cv2.cvtColor(
        previous_image,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_image,
        cv2.COLOR_BGR2GRAY
    )


    difference = cv2.absdiff(
        previous_gray,
        current_gray
    )


    # Remove small noise
    difference = cv2.GaussianBlur(
        difference,
        (5, 5),
        0
    )


    # Significant change threshold
    change_mask = (
        difference > 30
    )


    change_percentage = (
        np.mean(change_mask) * 100
    )


    return {
        "change_percentage": float(
            change_percentage
        ),

        "change_mask": change_mask
    }


# --------------------------------------------------
# PROGRESSION CLASSIFICATION
# --------------------------------------------------

def determine_progression(
    previous_grade,
    current_grade,
    change_percentage
):
    """
    Determine a prototype progression category.

    This is NOT a clinical progression criterion.
    """

    grade_levels = {

        "No DR": 0,

        "Mild DR": 1,

        "Moderate DR": 2,

        "Severe DR": 3,

        "Proliferative DR": 4
    }


    previous_level = grade_levels[
        previous_grade
    ]

    current_level = grade_levels[
        current_grade
    ]


    # Grade worsened
    if current_level > previous_level:

        return {
            "status": "PROGRESSING",
            "reason": (
                "DR severity increased "
                "between scans."
            )
        }


    # Grade improved
    if current_level < previous_level:

        return {
            "status": "IMPROVED",
            "reason": (
                "DR severity decreased "
                "between scans."
            )
        }


    # Same grade but substantial visual change
    if change_percentage >= 20:

        return {
            "status": "POSSIBLE CHANGE",
            "reason": (
                "DR grade is unchanged, but "
                "substantial retinal image change "
                "was detected."
            )
        }


    return {
        "status": "STABLE",
        "reason": (
            "No major change detected "
            "between scans."
        )
    }


# --------------------------------------------------
# COMPLETE COMPARISON
# --------------------------------------------------

def compare_scans(
    previous_path,
    current_path,
    previous_grade,
    current_grade
):
    """
    Compare two retinal scans.
    """

    previous_image = cv2.imread(
        str(previous_path)
    )

    current_image = cv2.imread(
        str(current_path)
    )


    if previous_image is None:

        raise FileNotFoundError(
            f"Could not read previous image: "
            f"{previous_path}"
        )


    if current_image is None:

        raise FileNotFoundError(
            f"Could not read current image: "
            f"{current_path}"
        )


    # Resize current image if necessary
    current_image = cv2.resize(
        current_image,
        (
            previous_image.shape[1],
            previous_image.shape[0]
        )
    )


    # Align images
    aligned_image, aligned = align_images(
        previous_image,
        current_image
    )


    # Calculate visual change
    change_result = calculate_change(
        previous_image,
        aligned_image
    )


    # Determine progression
    progression = determine_progression(
        previous_grade,
        current_grade,
        change_result["change_percentage"]
    )


    return {
        "previous_grade": previous_grade,

        "current_grade": current_grade,

        "registration_success": aligned,

        "change_percentage": (
            change_result["change_percentage"]
        ),

        "progression_status": (
            progression["status"]
        ),

        "reason": progression["reason"]
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    from pathlib import Path


    PROJECT_ROOT = (
        Path(__file__).resolve().parents[2]
    )


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
    print("LONGITUDINAL PROGRESSION TEST")
    print("=" * 60)


    # Demo: compare an image with itself.
    # This verifies the registration/change pipeline.
    result = compare_scans(

        previous_path=test_image,

        current_path=test_image,

        previous_grade="Moderate DR",

        current_grade="Severe DR"
    )


    print()
    print("PREVIOUS GRADE")
    print(
        result["previous_grade"]
    )


    print()
    print("CURRENT GRADE")
    print(
        result["current_grade"]
    )


    print()
    print("REGISTRATION")
    print(
        "Successful:",
        result["registration_success"]
    )


    print()
    print("IMAGE CHANGE")
    print(
        f"{result['change_percentage']:.2f}%"
    )


    print()
    print("PROGRESSION")
    print(
        result["progression_status"]
    )


    print()
    print(
        "Reason:",
        result["reason"]
    )


    print()
    print("=" * 60)
    print("PROGRESSION TEST COMPLETE")
    print("=" * 60)