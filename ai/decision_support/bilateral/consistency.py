from datetime import datetime


# ==========================================================
# DR SEVERITY LEVELS
# ==========================================================

DR_LEVELS = {
    "No DR": 0,
    "Mild DR": 1,
    "Moderate DR": 2,
    "Severe DR": 3,
    "Proliferative DR": 4
}


# ==========================================================
# BILATERAL CONSISTENCY ANALYSIS
# ==========================================================

def analyze_bilateral_consistency(
    left_eye_grade,
    right_eye_grade,
    left_eye_score=None,
    right_eye_score=None
):
    """
    Compare the predicted DR severity of the left and
    right eyes.

    This feature identifies inter-eye severity differences
    that may require additional review.

    It does NOT assume that both eyes must have the same
    DR grade and is NOT a clinical diagnostic rule.
    """

    # ------------------------------------------------------
    # Validate grades
    # ------------------------------------------------------

    if left_eye_grade not in DR_LEVELS:
        raise ValueError(
            f"Unknown left-eye grade: {left_eye_grade}"
        )

    if right_eye_grade not in DR_LEVELS:
        raise ValueError(
            f"Unknown right-eye grade: {right_eye_grade}"
        )


    # ------------------------------------------------------
    # Get severity levels
    # ------------------------------------------------------

    left_level = DR_LEVELS[left_eye_grade]

    right_level = DR_LEVELS[right_eye_grade]


    # ------------------------------------------------------
    # Calculate severity difference
    # ------------------------------------------------------

    severity_difference = abs(
        left_level - right_level
    )


    # ------------------------------------------------------
    # Determine consistency
    # ------------------------------------------------------

    if severity_difference == 0:

        consistency_status = "CONSISTENT"

        review_required = False

        recommendation = (
            "Left and right eye DR severity "
            "levels are consistent."
        )

    elif severity_difference == 1:

        consistency_status = "INTER_EYE_DIFFERENCE"

        review_required = True

        recommendation = (
            "A one-level difference in DR severity "
            "was detected between the eyes. "
            "Additional review is recommended."
        )

    else:

        consistency_status = "SIGNIFICANT_INTER_EYE_DIFFERENCE"

        review_required = True

        recommendation = (
            "A significant difference in DR severity "
            "was detected between the eyes. "
            "Manual review is recommended."
        )


    # ------------------------------------------------------
    # Determine more severe eye
    # ------------------------------------------------------

    if left_level > right_level:

        more_severe_eye = "LEFT"

    elif right_level > left_level:

        more_severe_eye = "RIGHT"

    else:

        more_severe_eye = "NONE"


    # ------------------------------------------------------
    # Result
    # ------------------------------------------------------

    result = {

        "analysis_timestamp":
            datetime.now().isoformat(),

        "left_eye": {
            "grade": left_eye_grade,
            "severity_level": left_level,
            "model_score": left_eye_score
        },

        "right_eye": {
            "grade": right_eye_grade,
            "severity_level": right_level,
            "model_score": right_eye_score
        },

        "severity_difference":
            severity_difference,

        "more_severe_eye":
            more_severe_eye,

        "consistency_status":
            consistency_status,

        "review_required":
            review_required,

        "recommendation":
            recommendation
    }

    return result


# ==========================================================
# TESTS
# ==========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("BILATERAL-EYE CONSISTENCY TEST")
    print("=" * 60)


    # ------------------------------------------------------
    # TEST 1 — SAME GRADE
    # ------------------------------------------------------

    print()
    print("TEST 1 — CONSISTENT EYES")
    print("-" * 60)

    result_1 = analyze_bilateral_consistency(

        left_eye_grade="Moderate DR",

        right_eye_grade="Moderate DR",

        left_eye_score=0.72,

        right_eye_score=0.69
    )

    print(
        "Left eye       :",
        result_1["left_eye"]["grade"]
    )

    print(
        "Right eye      :",
        result_1["right_eye"]["grade"]
    )

    print(
        "Severity diff  :",
        result_1["severity_difference"]
    )

    print(
        "Status         :",
        result_1["consistency_status"]
    )

    print(
        "Review required:",
        result_1["review_required"]
    )


    # ------------------------------------------------------
    # TEST 2 — ONE LEVEL DIFFERENCE
    # ------------------------------------------------------

    print()
    print("TEST 2 — ONE-LEVEL DIFFERENCE")
    print("-" * 60)

    result_2 = analyze_bilateral_consistency(

        left_eye_grade="Moderate DR",

        right_eye_grade="Severe DR",

        left_eye_score=0.72,

        right_eye_score=0.58
    )

    print(
        "Left eye       :",
        result_2["left_eye"]["grade"]
    )

    print(
        "Right eye      :",
        result_2["right_eye"]["grade"]
    )

    print(
        "Severity diff  :",
        result_2["severity_difference"]
    )

    print(
        "More severe eye:",
        result_2["more_severe_eye"]
    )

    print(
        "Status         :",
        result_2["consistency_status"]
    )

    print(
        "Review required:",
        result_2["review_required"]
    )


    # ------------------------------------------------------
    # TEST 3 — LARGE DIFFERENCE
    # ------------------------------------------------------

    print()
    print("TEST 3 — SIGNIFICANT DIFFERENCE")
    print("-" * 60)

    result_3 = analyze_bilateral_consistency(

        left_eye_grade="No DR",

        right_eye_grade="Proliferative DR",

        left_eye_score=0.91,

        right_eye_score=0.76
    )

    print(
        "Left eye       :",
        result_3["left_eye"]["grade"]
    )

    print(
        "Right eye      :",
        result_3["right_eye"]["grade"]
    )

    print(
        "Severity diff  :",
        result_3["severity_difference"]
    )

    print(
        "More severe eye:",
        result_3["more_severe_eye"]
    )

    print(
        "Status         :",
        result_3["consistency_status"]
    )

    print(
        "Review required:",
        result_3["review_required"]
    )


    # ------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------

    print()
    print("=" * 60)
    print("BILATERAL CONSISTENCY TEST COMPLETE")
    print("=" * 60)