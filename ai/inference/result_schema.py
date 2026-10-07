import sys
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ==========================================================
# IMPORTS
# ==========================================================

from ai.decision_support.referral.referral import create_referral


# ==========================================================
# CREATE MASTER AI RESULT
# ==========================================================

def create_master_result(
    image_name: str,
    image_quality: Dict[str, Any],
    dr_prediction: Optional[Dict[str, Any]] = None,
    reliability: Optional[Dict[str, Any]] = None,
    safety_gate: Optional[Dict[str, Any]] = None,
    triage: Optional[Dict[str, Any]] = None,
    explanation: Optional[Dict[str, Any]] = None,
    progression: Optional[Dict[str, Any]] = None,
    bilateral_consistency: Optional[Dict[str, Any]] = None,
    macular_risk: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    return {

        "screening_id": (
            f"SCREEN-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        ),

        "timestamp": datetime.now().isoformat(),

        # --------------------------------------------------
        # IMAGE
        # --------------------------------------------------

        "image": {
            "image_name": image_name
        },

        # --------------------------------------------------
        # AI RESULTS
        # --------------------------------------------------

        "image_quality": image_quality,

        "dr_prediction": dr_prediction,

        "reliability": reliability,

        "safety_gate": safety_gate,

        "triage": triage,

        "explanation": explanation,

        "progression": progression,

        # --------------------------------------------------
        # BILATERAL ANALYSIS
        # --------------------------------------------------

        "bilateral_consistency": bilateral_consistency,

        # --------------------------------------------------
        # MACULAR / EXUDATE RISK
        # --------------------------------------------------

        "macular_risk": macular_risk,

        # --------------------------------------------------
        # DOCTOR WORKFLOW
        # --------------------------------------------------

        "workflow": {

            "doctor_review_required": (
                safety_gate.get(
                    "doctor_review_required",
                    False
                )
                if safety_gate
                else (
                    triage.get(
                        "doctor_review",
                        False
                    )
                    if triage
                    else False
                )
            ),

            "doctor_decision": None,

            "doctor_grade": None,

            "doctor_note": None,

            "final_grade": None,

            "review_timestamp": None,

            # --------------------------------------------------
            # DISAGREEMENT AUDIT
            # --------------------------------------------------

            "disagreement_audit": None,

            # --------------------------------------------------
            # REFERRAL
            # --------------------------------------------------

            "referral_status": "NOT_CREATED",

            "referral": None,

            "next_action": None
        }
    }


# ==========================================================
# APPLY DOCTOR REVIEW
# ==========================================================

def apply_doctor_review(
    master_result: Dict[str, Any],
    review_record: Dict[str, Any],
    patient_id: str = "DEMO-001"
):
    """
    Add the doctor's decision to the existing Master AI Result.

    If the doctor confirms or overrides the AI result,
    store the disagreement audit and create a referral
    using the final doctor-confirmed grade.

    If the doctor requests recapture, no referral or
    disagreement audit is created.

    This is a prototype workflow and NOT a clinical
    referral guideline.
    """

    # --------------------------------------------------
    # STORE DOCTOR REVIEW
    # --------------------------------------------------

    master_result["workflow"]["doctor_decision"] = (
        review_record["doctor_decision"]
    )

    master_result["workflow"]["doctor_grade"] = (
        review_record["doctor_grade"]
    )

    master_result["workflow"]["doctor_note"] = (
        review_record["doctor_note"]
    )

    master_result["workflow"]["final_grade"] = (
        review_record["final_grade"]
    )

    master_result["workflow"]["review_timestamp"] = (
        review_record["review_timestamp"]
    )


    # --------------------------------------------------
    # STORE DISAGREEMENT AUDIT
    # --------------------------------------------------

    master_result["workflow"][
        "disagreement_audit"
    ] = review_record.get(
        "disagreement_audit"
    )


    # --------------------------------------------------
    # RECAPTURE
    # --------------------------------------------------

    if (
        review_record["doctor_decision"].lower()
        == "recapture"
    ):

        master_result["workflow"][
            "referral_status"
        ] = "NOT_CREATED"

        master_result["workflow"][
            "referral"
        ] = None

        master_result["workflow"][
            "next_action"
        ] = "Recapture fundus image."

        return master_result


    # --------------------------------------------------
    # CONFIRM / OVERRIDE
    # --------------------------------------------------

    if (
        review_record["doctor_decision"].lower()
        in {"confirm", "override"}
    ):

        final_grade = (
            review_record["final_grade"]
        )

        if not final_grade:
            raise ValueError(
                "Final doctor-confirmed grade is required "
                "to create a referral."
            )


        # --------------------------------------------------
        # CREATE REFERRAL
        # --------------------------------------------------

        referral = create_referral(

            patient_id=patient_id,

            dr_grade=final_grade,

            doctor_decision=(
                review_record["doctor_decision"]
            ),

            doctor_note=(
                review_record["doctor_note"]
            )
        )


        master_result["workflow"][
            "referral_status"
        ] = referral.get(
            "status",
            "PENDING"
        )


        master_result["workflow"][
            "referral"
        ] = referral


        master_result["workflow"][
            "next_action"
        ] = (
            "Referral created based on final "
            "doctor-confirmed grade."
        )


        return master_result


    # --------------------------------------------------
    # INVALID DECISION
    # --------------------------------------------------

    raise ValueError(
        "Invalid doctor decision. Expected "
        "'confirm', 'override', or 'recapture'."
    )


# ==========================================================
# TEST
# ==========================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("MASTER RESULT — BILATERAL CONSISTENCY TEST")
    print("=" * 60)


    # --------------------------------------------------
    # LEFT EYE
    # --------------------------------------------------

    left_eye_result = {

        "image_name": "LEFT_EYE.jpg",

        "dr_prediction": {
            "predicted_class": 2,
            "dr_grade": "Moderate DR",
            "calibrated_score": 0.72
        }
    }


    # --------------------------------------------------
    # RIGHT EYE
    # --------------------------------------------------

    right_eye_result = {

        "image_name": "RIGHT_EYE.jpg",

        "dr_prediction": {
            "predicted_class": 3,
            "dr_grade": "Severe DR",
            "calibrated_score": 0.58
        }
    }


    # --------------------------------------------------
    # BILATERAL ANALYSIS
    # --------------------------------------------------

    from ai.decision_support.bilateral.consistency import (
        analyze_bilateral_consistency
    )


    bilateral_result = analyze_bilateral_consistency(

        left_eye_grade=(
            left_eye_result["dr_prediction"]["dr_grade"]
        ),

        right_eye_grade=(
            right_eye_result["dr_prediction"]["dr_grade"]
        ),

        left_eye_score=(
            left_eye_result["dr_prediction"][
                "calibrated_score"
            ]
        ),

        right_eye_score=(
            right_eye_result["dr_prediction"][
                "calibrated_score"
            ]
        )
    )


    # --------------------------------------------------
    # DEMO SINGLE-EYE INFORMATION
    # --------------------------------------------------

    image_quality = {
        "quality": "good"
    }

    dr_prediction = {
        "predicted_class": 2,
        "dr_grade": "Moderate DR",
        "calibrated_score": 0.72
    }

    reliability = {
        "predicted_class": "Moderate DR",
        "model_score": 0.72,
        "calibrated_score": 0.72,
        "entropy": 0.55,
        "reliability": "MEDIUM",
        "recommendation": (
            "Doctor review recommended."
        )
    }

    safety_gate = {
        "status": "REVIEW_REQUIRED",
        "safety_level": "MEDIUM_RISK",
        "action": "DOCTOR REVIEW",
        "reason": (
            "AI prediction has medium reliability."
        ),
        "ai_result_allowed": True,
        "doctor_review_required": True
    }

    triage = {
        "priority": "DOCTOR REVIEW",
        "level": "YELLOW",
        "action": (
            "Moderate DR detected. "
            "Doctor review recommended."
        ),
        "doctor_review": True
    }


    # --------------------------------------------------
    # CREATE MASTER RESULT
    # --------------------------------------------------

    result = create_master_result(

        image_name="LEFT_EYE.jpg",

        image_quality=image_quality,

        dr_prediction=dr_prediction,

        reliability=reliability,

        safety_gate=safety_gate,

        triage=triage,

        explanation=None,

        progression=None,

        bilateral_consistency=bilateral_result,

        macular_risk=None
    )


    # --------------------------------------------------
    # DISPLAY
    # --------------------------------------------------

    print()
    print("LEFT EYE")
    print("-" * 40)

    print(
        "Grade:",
        left_eye_result["dr_prediction"]["dr_grade"]
    )

    print(
        "Score:",
        left_eye_result["dr_prediction"][
            "calibrated_score"
        ]
    )


    print()
    print("RIGHT EYE")
    print("-" * 40)

    print(
        "Grade:",
        right_eye_result["dr_prediction"]["dr_grade"]
    )

    print(
        "Score:",
        right_eye_result["dr_prediction"][
            "calibrated_score"
        ]
    )


    print()
    print("BILATERAL CONSISTENCY")
    print("-" * 40)

    print(
        "Status:",
        result["bilateral_consistency"][
            "consistency_status"
        ]
    )

    print(
        "Severity difference:",
        result["bilateral_consistency"][
            "severity_difference"
        ]
    )

    print(
        "More severe eye:",
        result["bilateral_consistency"][
            "more_severe_eye"
        ]
    )

    print(
        "Review required:",
        result["bilateral_consistency"][
            "review_required"
        ]
    )

    print(
        "Recommendation:",
        result["bilateral_consistency"][
            "recommendation"
        ]
    )


    print()
    print("=" * 60)
    print("MASTER RESULT BILATERAL TEST COMPLETE")
    print("=" * 60)