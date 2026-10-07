import sys
from pathlib import Path
from datetime import datetime


# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# DR SEVERITY LEVELS
# --------------------------------------------------

DR_LEVELS = {
    "No DR": 0,
    "Mild DR": 1,
    "Moderate DR": 2,
    "Severe DR": 3,
    "Proliferative DR": 4
}


# --------------------------------------------------
# CREATE DISAGREEMENT AUDIT
# --------------------------------------------------

def create_disagreement_audit(
    ai_prediction,
    ai_score,
    ai_reliability,
    doctor_grade,
    doctor_decision,
    doctor_note=""
):
    """
    Compare the AI prediction with the doctor's final grade.

    This is an audit mechanism for identifying
    AI-doctor disagreements and hard examples.

    It is NOT a clinical decision rule.
    """

    if ai_prediction not in DR_LEVELS:
        raise ValueError(
            f"Unknown AI prediction: {ai_prediction}"
        )

    if doctor_grade not in DR_LEVELS:
        raise ValueError(
            f"Unknown doctor grade: {doctor_grade}"
        )

    valid_decisions = {
        "confirm",
        "override"
    }

    if doctor_decision.lower() not in valid_decisions:
        raise ValueError(
            "Doctor decision must be "
            "'confirm' or 'override'."
        )

    ai_level = DR_LEVELS[ai_prediction]
    doctor_level = DR_LEVELS[doctor_grade]

    severity_difference = abs(
        doctor_level - ai_level
    )

    disagreement = (
        ai_prediction != doctor_grade
    )

    if disagreement:
        audit_status = "DISAGREEMENT"
    else:
        audit_status = "AGREEMENT"

    # --------------------------------------------------
    # CLASSIFY DISAGREEMENT
    # --------------------------------------------------

    if not disagreement:

        disagreement_type = "NONE"

    elif doctor_level > ai_level:

        disagreement_type = "DOCTOR_RATED_MORE_SEVERE"

    else:

        disagreement_type = "DOCTOR_RATED_LESS_SEVERE"


    # --------------------------------------------------
    # HARD EXAMPLE FLAG
    # --------------------------------------------------

    hard_example = disagreement


    # --------------------------------------------------
    # AUDIT RECORD
    # --------------------------------------------------

    audit_record = {

        "audit_id":
            f"AUDIT-{datetime.now().strftime('%Y%m%d%H%M%S%f')}",

        "timestamp":
            datetime.now().isoformat(),

        "ai_prediction":
            ai_prediction,

        "ai_score":
            float(ai_score),

        "ai_reliability":
            ai_reliability,

        "doctor_grade":
            doctor_grade,

        "doctor_decision":
            doctor_decision.lower(),

        "doctor_note":
            doctor_note,

        "agreement":
            not disagreement,

        "audit_status":
            audit_status,

        "disagreement_type":
            disagreement_type,

        "severity_difference":
            severity_difference,

        "hard_example":
            hard_example
    }

    return audit_record


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("DOCTOR DISAGREEMENT AUDIT TEST")
    print("=" * 60)


    # --------------------------------------------------
    # TEST 1 — AGREEMENT
    # --------------------------------------------------

    print()
    print("TEST 1 — DOCTOR CONFIRMS AI")
    print("-" * 60)

    agreement_result = create_disagreement_audit(
        ai_prediction="Severe DR",
        ai_score=0.5808,
        ai_reliability="LOW",
        doctor_grade="Severe DR",
        doctor_decision="confirm",
        doctor_note="AI result reviewed and confirmed."
    )

    for key, value in agreement_result.items():
        print(f"{key:22s}: {value}")


    # --------------------------------------------------
    # TEST 2 — DISAGREEMENT
    # --------------------------------------------------

    print()
    print("TEST 2 — DOCTOR OVERRIDES AI")
    print("-" * 60)

    disagreement_result = create_disagreement_audit(
        ai_prediction="Severe DR",
        ai_score=0.5808,
        ai_reliability="LOW",
        doctor_grade="Moderate DR",
        doctor_decision="override",
        doctor_note="Doctor assessment indicates moderate DR."
    )

    for key, value in disagreement_result.items():
        print(f"{key:22s}: {value}")


    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("AUDIT SUMMARY")
    print("=" * 60)

    print(
        "Agreement test    :",
        agreement_result["audit_status"]
    )

    print(
        "Disagreement test :",
        disagreement_result["audit_status"]
    )

    print(
        "Hard example      :",
        disagreement_result["hard_example"]
    )

    print(
        "Severity difference:",
        disagreement_result["severity_difference"]
    )

    print()
    print("=" * 60)
    print("DISAGREEMENT AUDIT TEST COMPLETE")
    print("=" * 60)