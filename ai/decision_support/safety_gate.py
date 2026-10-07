from typing import Optional


def evaluate_safety_gate(
    image_quality: str,
    reliability: Optional[str] = None,
    dr_grade: Optional[str] = None,
    bilateral_consistency: Optional[dict] = None
):
    """
    AI Safety Gate for diabetic retinopathy screening.

    The safety gate decides whether an AI result can proceed
    to AI-assisted triage or requires human intervention.

    This is a prototype decision-support mechanism and
    is NOT a clinically validated protocol.
    """

    # --------------------------------------------------
    # 1. IMAGE QUALITY GATE
    # --------------------------------------------------

    if image_quality.lower() != "good":
        return {
            "status": "BLOCKED",
            "safety_level": "HIGH_RISK",
            "action": "RECAPTURE",
            "reason": (
                "Image quality is insufficient for reliable "
                "AI screening."
            ),
            "ai_result_allowed": False,
            "doctor_review_required": False
        }

    # --------------------------------------------------
    # 2. RELIABILITY GATE
    # --------------------------------------------------

    if reliability == "LOW":
        return {
            "status": "BLOCKED",
            "safety_level": "HIGH_RISK",
            "action": "MANUAL REVIEW",
            "reason": (
                "AI prediction reliability is low. "
                "Manual review is required."
            ),
            "ai_result_allowed": False,
            "doctor_review_required": True
        }

    # --------------------------------------------------
    # 3. BILATERAL EYE CONSISTENCY
    # --------------------------------------------------

    if bilateral_consistency is not None:

        consistency_status = bilateral_consistency.get(
            "consistency_status"
        )

        severity_difference = bilateral_consistency.get(
            "severity_difference", 0
        )

        # Significant difference between eyes
        if consistency_status == "SIGNIFICANT_INTER_EYE_DIFFERENCE":
            return {
                "status": "REVIEW_REQUIRED",
                "safety_level": "HIGH_RISK",
                "action": "DOCTOR REVIEW",
                "reason": (
                    "A significant difference in DR severity "
                    "was detected between the two eyes. "
                    "Doctor review is required."
                ),
                "ai_result_allowed": True,
                "doctor_review_required": True,
                "bilateral_flag": True,
                "severity_difference": severity_difference
            }

        # One-level difference between eyes
        if consistency_status == "INTER_EYE_DIFFERENCE":
            return {
                "status": "REVIEW_REQUIRED",
                "safety_level": "MEDIUM_RISK",
                "action": "DOCTOR REVIEW",
                "reason": (
                    "A difference in DR severity was detected "
                    "between the two eyes. Additional doctor "
                    "review is recommended."
                ),
                "ai_result_allowed": True,
                "doctor_review_required": True,
                "bilateral_flag": True,
                "severity_difference": severity_difference
            }

    # --------------------------------------------------
    # 4. MEDIUM RELIABILITY
    # --------------------------------------------------

    if reliability == "MEDIUM":
        return {
            "status": "REVIEW_REQUIRED",
            "safety_level": "MEDIUM_RISK",
            "action": "DOCTOR REVIEW",
            "reason": (
                "AI prediction has medium reliability. "
                "Doctor review is recommended before "
                "final clinical decision."
            ),
            "ai_result_allowed": True,
            "doctor_review_required": True
        }

    # --------------------------------------------------
    # 5. HIGH RELIABILITY
    # --------------------------------------------------

    if reliability == "HIGH":
        return {
            "status": "PASSED",
            "safety_level": "LOW_RISK",
            "action": "AI-ASSISTED TRIAGE",
            "reason": (
                "Image quality is good and AI prediction "
                "has high reliability."
            ),
            "ai_result_allowed": True,
            "doctor_review_required": False
        }

    # --------------------------------------------------
    # 6. UNKNOWN / INVALID RELIABILITY
    # --------------------------------------------------

    return {
        "status": "BLOCKED",
        "safety_level": "HIGH_RISK",
        "action": "MANUAL REVIEW",
        "reason": (
            "AI reliability status is unavailable. "
            "Manual review is required."
        ),
        "ai_result_allowed": False,
        "doctor_review_required": True
    }


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("AI SAFETY GATE TEST")
    print("=" * 60)

    # --------------------------------------------------
    # TEST 1: MEDIUM RELIABILITY
    # --------------------------------------------------

    result = evaluate_safety_gate(
        image_quality="good",
        reliability="MEDIUM",
        dr_grade="Moderate DR"
    )

    print()
    print("TEST 1: MEDIUM RELIABILITY")
    print("-" * 60)
    print(result)

    # --------------------------------------------------
    # TEST 2: POOR IMAGE
    # --------------------------------------------------

    result = evaluate_safety_gate(
        image_quality="poor",
        reliability=None,
        dr_grade=None
    )

    print()
    print("TEST 2: POOR IMAGE")
    print("-" * 60)
    print(result)

    # --------------------------------------------------
    # TEST 3: LOW RELIABILITY
    # --------------------------------------------------

    result = evaluate_safety_gate(
        image_quality="good",
        reliability="LOW",
        dr_grade="Moderate DR"
    )

    print()
    print("TEST 3: LOW RELIABILITY")
    print("-" * 60)
    print(result)

    # --------------------------------------------------
    # TEST 4: HIGH RELIABILITY
    # --------------------------------------------------

    result = evaluate_safety_gate(
        image_quality="good",
        reliability="HIGH",
        dr_grade="No DR"
    )

    print()
    print("TEST 4: HIGH RELIABILITY")
    print("-" * 60)
    print(result)

    # --------------------------------------------------
    # TEST 5: BILATERAL CONSISTENT
    # --------------------------------------------------

    bilateral_result = {
        "consistency_status": "CONSISTENT",
        "severity_difference": 0
    }

    result = evaluate_safety_gate(
        image_quality="good",
        reliability="HIGH",
        dr_grade="Moderate DR",
        bilateral_consistency=bilateral_result
    )

    print()
    print("TEST 5: BILATERAL CONSISTENT")
    print("-" * 60)
    print(result)

    # --------------------------------------------------
    # TEST 6: ONE-LEVEL BILATERAL DIFFERENCE
    # --------------------------------------------------

    bilateral_result = {
        "consistency_status": "INTER_EYE_DIFFERENCE",
        "severity_difference": 1
    }

    result = evaluate_safety_gate(
        image_quality="good",
        reliability="HIGH",
        dr_grade="Severe DR",
        bilateral_consistency=bilateral_result
    )

    print()
    print("TEST 6: ONE-LEVEL BILATERAL DIFFERENCE")
    print("-" * 60)
    print(result)

    # --------------------------------------------------
    # TEST 7: SIGNIFICANT BILATERAL DIFFERENCE
    # --------------------------------------------------

    bilateral_result = {
        "consistency_status": "SIGNIFICANT_INTER_EYE_DIFFERENCE",
        "severity_difference": 2
    }

    result = evaluate_safety_gate(
        image_quality="good",
        reliability="HIGH",
        dr_grade="PDR",
        bilateral_consistency=bilateral_result
    )

    print()
    print("TEST 7: SIGNIFICANT BILATERAL DIFFERENCE")
    print("-" * 60)
    print(result)

    # --------------------------------------------------
    # TEST 8: LOW RELIABILITY + BILATERAL DIFFERENCE
    # --------------------------------------------------

    bilateral_result = {
        "consistency_status": "SIGNIFICANT_INTER_EYE_DIFFERENCE",
        "severity_difference": 2
    }

    result = evaluate_safety_gate(
        image_quality="good",
        reliability="LOW",
        dr_grade="PDR",
        bilateral_consistency=bilateral_result
    )

    print()
    print("TEST 8: LOW RELIABILITY + BILATERAL DIFFERENCE")
    print("-" * 60)
    print(result)

    print()
    print("=" * 60)
    print("AI SAFETY GATE TEST COMPLETE")
    print("=" * 60)