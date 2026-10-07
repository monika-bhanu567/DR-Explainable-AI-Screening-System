from pathlib import Path
import sys


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# DR SEVERITY LEVELS
# --------------------------------------------------

SEVERITY_LEVELS = {
    "No DR": 0,
    "Mild DR": 1,
    "Moderate DR": 2,
    "Severe DR": 3,
    "Proliferative DR": 4
}


# --------------------------------------------------
# TRIAGE FUNCTION
# --------------------------------------------------

def determine_triage(
    dr_grade,
    reliability,
    image_quality
):
    """
    Determine screening priority using:

    - DR severity
    - AI reliability
    - image quality

    This is a prototype decision-support rule
    and is NOT a clinical protocol.
    """

    # --------------------------------------------------
    # IMAGE QUALITY SAFETY GATE
    # --------------------------------------------------

    if image_quality.lower() != "good":

        return {
            "priority": "RECAPTURE",
            "level": "ORANGE",
            "action": (
                "Image quality is insufficient. "
                "Capture another retinal image."
            ),
            "doctor_review": False
        }


    # --------------------------------------------------
    # LOW RELIABILITY SAFETY GATE
    # --------------------------------------------------

    if reliability.upper() == "LOW":

        return {
            "priority": "MANUAL REVIEW",
            "level": "ORANGE",
            "action": (
                "AI reliability is low. "
                "Manual clinical review is required."
            ),
            "doctor_review": True
        }


    # --------------------------------------------------
    # SEVERE / PDR
    # --------------------------------------------------

    if dr_grade == "Proliferative DR":

        return {
            "priority": "URGENT",
            "level": "RED",
            "action": (
                "High-severity DR detected. "
                "Urgent specialist review recommended."
            ),
            "doctor_review": True
        }


    if dr_grade == "Severe DR":

        return {
            "priority": "HIGH",
            "level": "RED",
            "action": (
                "Severe DR detected. "
                "Priority doctor review recommended."
            ),
            "doctor_review": True
        }


    # --------------------------------------------------
    # MODERATE DR
    # --------------------------------------------------

    if dr_grade == "Moderate DR":

        return {
            "priority": "DOCTOR REVIEW",
            "level": "YELLOW",
            "action": (
                "Moderate DR detected. "
                "Doctor review recommended."
            ),
            "doctor_review": True
        }


    # --------------------------------------------------
    # MILD DR
    # --------------------------------------------------

    if dr_grade == "Mild DR":

        return {
            "priority": "FOLLOW-UP",
            "level": "YELLOW",
            "action": (
                "Mild DR detected. "
                "Follow-up screening recommended."
            ),
            "doctor_review": False
        }


    # --------------------------------------------------
    # NO DR
    # --------------------------------------------------

    if dr_grade == "No DR":

        return {
            "priority": "ROUTINE",
            "level": "GREEN",
            "action": (
                "No diabetic retinopathy detected. "
                "Routine follow-up recommended."
            ),
            "doctor_review": False
        }


    # --------------------------------------------------
    # UNKNOWN RESULT
    # --------------------------------------------------

    return {
        "priority": "MANUAL REVIEW",
        "level": "ORANGE",
        "action": (
            "Unable to determine a safe triage category. "
            "Manual review required."
        ),
        "doctor_review": True
    }


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("AI DECISION SUPPORT / TRIAGE TEST")
    print("=" * 60)


    # Current example from our pipeline
    dr_grade = "Moderate DR"
    reliability = "MEDIUM"
    image_quality = "good"


    result = determine_triage(
        dr_grade,
        reliability,
        image_quality
    )


    print()
    print("INPUT")
    print("-" * 60)

    print(
        "DR grade       :",
        dr_grade
    )

    print(
        "Reliability    :",
        reliability
    )

    print(
        "Image quality  :",
        image_quality.upper()
    )


    print()
    print("TRIAGE RESULT")
    print("-" * 60)

    print(
        "Priority       :",
        result["priority"]
    )

    print(
        "Level          :",
        result["level"]
    )

    print(
        "Doctor review  :",
        result["doctor_review"]
    )

    print(
        "Action         :",
        result["action"]
    )


    print()
    print("=" * 60)
    print("TRIAGE TEST COMPLETE")
    print("=" * 60)