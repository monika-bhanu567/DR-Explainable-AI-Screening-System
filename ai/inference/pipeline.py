import sys
from pathlib import Path

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

from ai.preprocessing.image_quality import check_image_quality
from ai.inference.predict import predict_image
from ai.uncertainty.confidence import analyze_prediction
from ai.decision_support.triage import determine_triage
from ai.decision_support.safety_gate import evaluate_safety_gate
from ai.inference.result_schema import create_master_result
from ai.explainability.gradcam_plus_plus import generate_gradcam
from ai.progression.comparison import compare_scans
from ai.decision_support.bilateral.consistency import (
    analyze_bilateral_consistency
)
from ai.decision_support.macular_risk.proximity import (
    analyze_exudate_macular_proximity
)


# --------------------------------------------------
# SCREENING PIPELINE
# --------------------------------------------------

def run_screening(
    image_path,
    previous_image_path=None,
    previous_grade=None,
    bilateral_consistency=None,
    macular_image_path=None,
    exudate_mask_path=None
):

    image_path = Path(image_path)

    if macular_image_path is not None:
        macular_image_path = Path(macular_image_path)

    if exudate_mask_path is not None:
        exudate_mask_path = Path(exudate_mask_path)

    print()
    print("=" * 70)
    print("DIABETIC RETINOPATHY SCREENING PIPELINE")
    print("=" * 70)
    print("AI image:", image_path)

    # --------------------------------------------------
    # STEP 1 — IMAGE QUALITY
    # --------------------------------------------------

    print()
    print("[1/9] Checking image quality...")

    quality_result = check_image_quality(
        image_path
    )

    print(
        "Image quality:",
        quality_result["quality"]
    )

    # --------------------------------------------------
    # SAFETY: POOR IMAGE
    # --------------------------------------------------

    if quality_result["quality"].lower() != "good":

        print()
        print("Image quality is poor.")
        print("AI prediction blocked.")
        print("Recapture required.")

        triage_result = determine_triage(
            dr_grade=None,
            reliability=None,
            image_quality=quality_result["quality"]
        )

        safety_gate_result = evaluate_safety_gate(
            image_quality=quality_result["quality"],
            reliability=None,
            dr_grade=None,
            bilateral_consistency=bilateral_consistency
        )

        return create_master_result(
            image_name=image_path.name,
            image_quality=quality_result,
            dr_prediction=None,
            reliability=None,
            safety_gate=safety_gate_result,
            triage=triage_result,
            explanation=None,
            progression=None,
            bilateral_consistency=bilateral_consistency,
            macular_risk=None
        )

    # --------------------------------------------------
    # STEP 2 — DR PREDICTION
    # --------------------------------------------------

    print()
    print("[2/9] Running 512x512 DR model...")

    prediction_result = predict_image(
        image_path
    )

    current_grade = prediction_result[
        "dr_grade"
    ]

    print(
        "Predicted grade:",
        current_grade
    )

    print(
        "Calibrated score:",
        f"{prediction_result['calibrated_score']:.4f}"
    )

    # --------------------------------------------------
    # STEP 3 — CALIBRATED UNCERTAINTY
    # --------------------------------------------------

    print()
    print("[3/9] Calculating calibrated reliability...")

    reliability_result = analyze_prediction(
        prediction_result["probabilities"]
    )

    print(
        "Entropy:",
        f"{reliability_result['entropy']:.4f}"
    )

    print(
        "Reliability:",
        reliability_result["reliability"]
    )

    # --------------------------------------------------
    # STEP 4 — BILATERAL EYE CONSISTENCY
    # --------------------------------------------------

    print()
    print("[4/9] Checking bilateral eye consistency...")

    bilateral_result = bilateral_consistency

    if bilateral_result is not None:

        print(
            "Bilateral status:",
            bilateral_result["consistency_status"]
        )

        print(
            "Severity difference:",
            bilateral_result["severity_difference"]
        )

        print(
            "More severe eye:",
            bilateral_result["more_severe_eye"]
        )

        print(
            "Bilateral review required:",
            bilateral_result["review_required"]
        )

    else:

        print(
            "No paired-eye information supplied."
        )

    # --------------------------------------------------
    # STEP 5 — MACULAR / EXUDATE PROXIMITY RISK
    # --------------------------------------------------

    print()
    print("[5/9] Checking macular/exudate proximity risk...")

    macular_risk_result = None

    if (
        macular_image_path is not None
        and exudate_mask_path is not None
    ):

        print(
            "Macular analysis image:",
            macular_image_path
        )

        print(
            "Exudate mask:",
            exudate_mask_path
        )

        macular_risk_result = (
            analyze_exudate_macular_proximity(
                image_path=macular_image_path,
                exudate_mask_path=exudate_mask_path
            )
        )

        print(
            "Macular proximity:",
            macular_risk_result["macular_proximity"]
        )

        print(
            "Macular risk:",
            macular_risk_result["risk_level"]
        )

        print(
            "Minimum distance:",
            macular_risk_result[
                "minimum_distance_pixels"
            ]
        )

    else:

        print(
            "No matching high-resolution "
            "exudate mask supplied."
        )

        print(
            "Macular risk analysis skipped."
        )

    # --------------------------------------------------
    # STEP 6 — SAFETY GATE
    # --------------------------------------------------

    print()
    print("[6/9] Evaluating AI safety gate...")

    safety_gate_result = evaluate_safety_gate(
        image_quality=quality_result["quality"],
        reliability=reliability_result["reliability"],
        dr_grade=current_grade,
        bilateral_consistency=bilateral_result
    )

    print(
        "Safety status:",
        safety_gate_result["status"]
    )

    print(
        "Safety level:",
        safety_gate_result["safety_level"]
    )

    print(
        "Action:",
        safety_gate_result["action"]
    )

    # --------------------------------------------------
    # STEP 7 — SMART TRIAGE
    # --------------------------------------------------

    print()
    print("[7/9] Determining triage priority...")

    triage_result = determine_triage(
        dr_grade=current_grade,
        reliability=reliability_result["reliability"],
        image_quality=quality_result["quality"]
    )

    print(
        "Priority:",
        triage_result["priority"]
    )

    print(
        "Level:",
        triage_result["level"]
    )

    print(
        "Doctor review:",
        triage_result["doctor_review"]
    )

    # --------------------------------------------------
    # STEP 8 — EXPLAINABILITY
    # --------------------------------------------------

    print()
    print("[8/9] Generating Grad-CAM++ explanation...")

    heatmap_path = (
        PROJECT_ROOT
        / "ai"
        / "outputs"
        / "heatmaps"
        / f"{image_path.stem}_gradcam_plus_plus.jpg"
    )

    explanation_output = generate_gradcam(
        image_path,
        heatmap_path
    )

    explanation_result = {
        "method": "Grad-CAM++",
        "heatmap": explanation_output[
            "output_path"
        ],
        "raw_cam": explanation_output[
            "raw_cam_path"
        ]
    }

    print(
        "Heatmap:",
        explanation_output["output_path"]
    )

    # --------------------------------------------------
    # STEP 9 — PROGRESSION
    # --------------------------------------------------

    print()
    print("[9/9] Checking longitudinal progression...")

    progression_result = None

    if (
        previous_image_path is not None
        and previous_grade is not None
    ):

        progression_result = compare_scans(
            previous_path=previous_image_path,
            current_path=image_path,
            previous_grade=previous_grade,
            current_grade=current_grade
        )

        print(
            "Progression:",
            progression_result[
                "progression_status"
            ]
        )

    else:

        print(
            "No previous scan supplied."
        )

    # --------------------------------------------------
    # MASTER RESULT
    # --------------------------------------------------

    master_result = create_master_result(
        image_name=image_path.name,
        image_quality=quality_result,
        dr_prediction=prediction_result,
        reliability=reliability_result,
        safety_gate=safety_gate_result,
        triage=triage_result,
        explanation=explanation_result,
        progression=progression_result,
        bilateral_consistency=bilateral_result,
        macular_risk=macular_risk_result
    )

    # --------------------------------------------------
    # SUMMARY
    # --------------------------------------------------

    print()
    print("=" * 70)
    print("SCREENING SUMMARY")
    print("=" * 70)

    print(
        "DR grade        :",
        current_grade
    )

    print(
        "Calibrated score:",
        f"{prediction_result['calibrated_score']:.4f}"
    )

    print(
        "Entropy         :",
        f"{reliability_result['entropy']:.4f}"
    )

    print(
        "Reliability     :",
        reliability_result["reliability"]
    )

    print(
        "Safety status   :",
        safety_gate_result["status"]
    )

    print(
        "Triage priority :",
        triage_result["priority"]
    )

    print(
        "Doctor review   :",
        triage_result["doctor_review"]
    )

    if bilateral_result is not None:

        print(
            "Bilateral status:",
            bilateral_result[
                "consistency_status"
            ]
        )

        print(
            "Eye difference  :",
            bilateral_result[
                "severity_difference"
            ]
        )

    if macular_risk_result is not None:

        print(
            "Macular risk    :",
            macular_risk_result[
                "risk_level"
            ]
        )

        print(
            "Macular proximity:",
            macular_risk_result[
                "macular_proximity"
            ]
        )

    print()

    print("=" * 70)
    print("SCREENING PIPELINE COMPLETE")
    print("=" * 70)

    return master_result


# --------------------------------------------------
# TEST
# --------------------------------------------------

if __name__ == "__main__":

    test_image = (
        PROJECT_ROOT
        / "ai"
        / "datasets"
        / "processed"
        / "validation"
        / "IDRiD_006.jpg"
    )

    result = run_screening(
        test_image
    )

    print()
    print("MASTER RESULT KEYS:")
    print("-" * 70)

    for key in result:
        print("-", key)