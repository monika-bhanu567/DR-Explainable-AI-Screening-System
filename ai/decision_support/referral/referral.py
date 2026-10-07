from datetime import datetime, timedelta


# --------------------------------------------------
# REFERRAL PRIORITY
# --------------------------------------------------

REFERRAL_RULES = {
    "Proliferative DR": {
        "priority": "URGENT",
        "days": 1,
        "reason": "High-severity diabetic retinopathy detected."
    },

    "Severe DR": {
        "priority": "HIGH",
        "days": 3,
        "reason": "Severe diabetic retinopathy detected."
    },

    "Moderate DR": {
        "priority": "ROUTINE",
        "days": 14,
        "reason": "Moderate diabetic retinopathy detected."
    },

    "Mild DR": {
        "priority": "FOLLOW-UP",
        "days": 90,
        "reason": "Mild diabetic retinopathy detected."
    },

    "No DR": {
        "priority": "ROUTINE",
        "days": 180,
        "reason": "No diabetic retinopathy detected."
    }
}


# --------------------------------------------------
# CREATE REFERRAL
# --------------------------------------------------

def create_referral(
    patient_id,
    dr_grade,
    doctor_decision,
    doctor_note=""
):
    """
    Create a referral/follow-up record.

    This is a prototype workflow and NOT a
    clinical referral guideline.
    """

    if doctor_decision.lower() not in {
        "confirm",
        "override"
    }:

        return {
            "status": "not_created",
            "reason": (
                "Referral requires a confirmed "
                "or doctor-overridden result."
            )
        }


    if dr_grade not in REFERRAL_RULES:

        raise ValueError(
            f"Unknown DR grade: {dr_grade}"
        )


    rule = REFERRAL_RULES[
        dr_grade
    ]


    created_at = datetime.now()

    follow_up_date = (
        created_at
        + timedelta(days=rule["days"])
    )


    referral = {

        "referral_id": (
            f"REF-{created_at.strftime('%Y%m%d%H%M%S')}"
        ),

        "patient_id": patient_id,

        "dr_grade": dr_grade,

        "priority": rule["priority"],

        "reason": rule["reason"],

        "doctor_decision": doctor_decision,

        "doctor_note": doctor_note,

        "created_at": created_at.isoformat(),

        "recommended_follow_up": (
            follow_up_date.date().isoformat()
        ),

        "status": "PENDING"
    }


    return referral


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("REFERRAL / FOLLOW-UP TEST")
    print("=" * 60)


    referral = create_referral(

        patient_id="DEMO-001",

        dr_grade="Moderate DR",

        doctor_decision="confirm",

        doctor_note=(
            "AI result reviewed and confirmed."
        )
    )


    print()
    print("REFERRAL CREATED")
    print("-" * 60)

    for key, value in referral.items():

        print(
            f"{key:<25}: {value}"
        )


    print()
    print("=" * 60)
    print("REFERRAL TEST COMPLETE")
    print("=" * 60)