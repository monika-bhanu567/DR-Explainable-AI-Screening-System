import sys
from pathlib import Path
from datetime import datetime

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))


from ai.decision_support.doctor_review.disagreement_audit import (
    create_disagreement_audit
)


# --------------------------------------------------
# DOCTOR REVIEW OPTIONS
# --------------------------------------------------

VALID_DECISIONS = {
    "confirm",
    "override",
    "recapture"
}


# --------------------------------------------------
# CREATE REVIEW RECORD
# --------------------------------------------------

def create_review_record(
    ai_prediction,
    ai_score,
    ai_reliability,
    doctor_decision,
    doctor_grade=None,
    doctor_note=""
):
    """
    Create an auditable doctor-review record.

    AI prediction and doctor decision are stored
    separately.

    Doctor-AI disagreement is automatically audited.

    This is a prototype workflow and not a
    clinical decision protocol.
    """

    doctor_decision = doctor_decision.lower().strip()

    if doctor_decision not in VALID_DECISIONS:
        raise ValueError(
            "Invalid doctor decision. "
            "Use: confirm, override, or recapture."
        )


    # --------------------------------------------------
    # CONFIRM AI RESULT
    # --------------------------------------------------

    if doctor_decision == "confirm":

        final_grade = ai_prediction


    # --------------------------------------------------
    # OVERRIDE AI RESULT
    # --------------------------------------------------

    elif doctor_decision == "override":

        if not doctor_grade:
            raise ValueError(
                "doctor_grade is required "
                "when overriding the AI result."
            )

        final_grade = doctor_grade


    # --------------------------------------------------
    # RECAPTURE IMAGE
    # --------------------------------------------------

    else:

        final_grade = None


    # --------------------------------------------------
    # DOCTOR-AI DISAGREEMENT AUDIT
    # --------------------------------------------------

    disagreement_audit = None

    if doctor_decision in {"confirm", "override"}:

        disagreement_audit = create_disagreement_audit(
            ai_prediction=ai_prediction,
            ai_score=ai_score,
            ai_reliability=ai_reliability,
            doctor_grade=final_grade,
            doctor_decision=doctor_decision,
            doctor_note=doctor_note
        )


    # --------------------------------------------------
    # REVIEW RECORD
    # --------------------------------------------------

    review_record = {

        "review_timestamp": (
            datetime.now().isoformat()
        ),

        "ai_prediction": ai_prediction,

        "ai_model_score": float(
            ai_score
        ),

        "ai_reliability": ai_reliability,

        "doctor_decision": doctor_decision,

        "doctor_grade": doctor_grade,

        "final_grade": final_grade,

        "doctor_note": doctor_note,

        "disagreement_audit": disagreement_audit
    }


    return review_record


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("DOCTOR-IN-THE-LOOP + DISAGREEMENT AUDIT TEST")
    print("=" * 60)


    # --------------------------------------------------
    # TEST 1 — DOCTOR CONFIRMS AI
    # --------------------------------------------------

    print()
    print("TEST 1 — DOCTOR CONFIRMS AI")
    print("-" * 60)

    review_confirm = create_review_record(

        ai_prediction="Severe DR",

        ai_score=0.5808,

        ai_reliability="LOW",

        doctor_decision="confirm",

        doctor_grade=None,

        doctor_note=(
            "AI result reviewed and confirmed."
        )
    )


    print(
        "AI prediction :",
        review_confirm["ai_prediction"]
    )

    print(
        "Doctor decision:",
        review_confirm["doctor_decision"]
    )

    print(
        "Final grade   :",
        review_confirm["final_grade"]
    )

    print(
        "Audit status  :",
        review_confirm["disagreement_audit"][
            "audit_status"
        ]
    )

    print(
        "Hard example  :",
        review_confirm["disagreement_audit"][
            "hard_example"
        ]
    )


    # --------------------------------------------------
    # TEST 2 — DOCTOR OVERRIDES AI
    # --------------------------------------------------

    print()
    print("TEST 2 — DOCTOR OVERRIDES AI")
    print("-" * 60)

    review_override = create_review_record(

        ai_prediction="Severe DR",

        ai_score=0.5808,

        ai_reliability="LOW",

        doctor_decision="override",

        doctor_grade="Moderate DR",

        doctor_note=(
            "Doctor assessment indicates "
            "moderate DR."
        )
    )


    print(
        "AI prediction :",
        review_override["ai_prediction"]
    )

    print(
        "Doctor decision:",
        review_override["doctor_decision"]
    )

    print(
        "Doctor grade  :",
        review_override["doctor_grade"]
    )

    print(
        "Final grade   :",
        review_override["final_grade"]
    )

    print(
        "Audit status  :",
        review_override["disagreement_audit"][
            "audit_status"
        ]
    )

    print(
        "Disagreement  :",
        review_override["disagreement_audit"][
            "disagreement_type"
        ]
    )

    print(
        "Severity diff :",
        review_override["disagreement_audit"][
            "severity_difference"
        ]
    )

    print(
        "Hard example  :",
        review_override["disagreement_audit"][
            "hard_example"
        ]
    )


    # --------------------------------------------------
    # TEST 3 — RECAPTURE
    # --------------------------------------------------

    print()
    print("TEST 3 — RECAPTURE")
    print("-" * 60)

    review_recapture = create_review_record(

        ai_prediction="Severe DR",

        ai_score=0.5808,

        ai_reliability="LOW",

        doctor_decision="recapture",

        doctor_grade=None,

        doctor_note=(
            "Image quality requires "
            "a new retinal image."
        )
    )


    print(
        "Doctor decision:",
        review_recapture["doctor_decision"]
    )

    print(
        "Final grade   :",
        review_recapture["final_grade"]
    )

    print(
        "Audit created :",
        review_recapture["disagreement_audit"]
        is not None
    )


    # --------------------------------------------------
    # COMPLETE
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("DOCTOR REVIEW + AUDIT TEST COMPLETE")
    print("=" * 60)