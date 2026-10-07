import sys
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------
# PROJECT PATH
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

TEMPERATURE_PATH = (
    PROJECT_ROOT
    / "ai"
    / "checkpoints"
    / "temperature.pt"
)


CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


# ---------------------------------------------------------
# LOAD TEMPERATURE
# ---------------------------------------------------------

def load_temperature():

    if not TEMPERATURE_PATH.exists():
        raise FileNotFoundError(
            f"Temperature file not found: {TEMPERATURE_PATH}"
        )

    temperature_data = torch.load(
        TEMPERATURE_PATH,
        map_location="cpu"
    )

    temperature = float(
        temperature_data["temperature"]
    )

    if temperature <= 0:
        raise ValueError(
            "Temperature must be greater than zero."
        )

    return temperature


# ---------------------------------------------------------
# CALIBRATE PROBABILITIES
# ---------------------------------------------------------

def calibrate_logits(logits):

    temperature = load_temperature()

    logits = torch.as_tensor(
        logits,
        dtype=torch.float32
    )

    calibrated_logits = logits / temperature

    probabilities = torch.softmax(
        calibrated_logits,
        dim=-1
    )

    return probabilities.numpy()


# ---------------------------------------------------------
# ENTROPY
# ---------------------------------------------------------

def calculate_entropy(probabilities):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    probabilities = np.clip(
        probabilities,
        1e-12,
        1.0
    )

    entropy = -np.sum(
        probabilities * np.log(probabilities)
    )

    max_entropy = np.log(
        len(probabilities)
    )

    normalized_entropy = entropy / max_entropy

    return float(normalized_entropy)


# ---------------------------------------------------------
# RELIABILITY
# ---------------------------------------------------------

def calculate_reliability(probabilities):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    calibrated_score = float(
        probabilities[predicted_index]
    )

    entropy = calculate_entropy(
        probabilities
    )

    # Prototype thresholds.
    # These are NOT clinical thresholds.

    if entropy < 0.35:

        reliability = "HIGH"

        recommendation = (
            "AI result can be used for "
            "screening prioritization."
        )

    elif entropy < 0.65:

        reliability = "MEDIUM"

        recommendation = (
            "Doctor review recommended."
        )

    else:

        reliability = "LOW"

        recommendation = (
            "Manual review required. "
            "AI result should not be relied "
            "upon alone."
        )

    return {
        "predicted_class": predicted_class,
        "model_score": calibrated_score,
        "calibrated_score": calibrated_score,
        "entropy": entropy,
        "reliability": reliability,
        "recommendation": recommendation,
        "temperature": load_temperature()
    }


# ---------------------------------------------------------
# COMPLETE ANALYSIS
# ---------------------------------------------------------

def analyze_prediction(probabilities):

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64
    )

    if len(probabilities) != 5:

        raise ValueError(
            "Expected probabilities for exactly "
            "5 DR classes."
        )

    if not np.isclose(
        probabilities.sum(),
        1.0,
        atol=1e-3
    ):

        raise ValueError(
            "Probabilities must sum to approximately 1."
        )

    return calculate_reliability(
        probabilities
    )


# ---------------------------------------------------------
# TEST
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("CALIBRATED UNCERTAINTY TEST")
    print("=" * 60)

    # Example raw logits.
    # These are only for testing the calibration module.

    raw_logits = [
        0.0,
        -3.0,
        2.5,
        0.0,
        -0.2
    ]

    temperature = load_temperature()

    calibrated_probabilities = calibrate_logits(
        raw_logits
    )

    result = analyze_prediction(
        calibrated_probabilities
    )

    print()
    print("Temperature:", round(temperature, 4))

    print()
    print("Calibrated probabilities:")

    for class_name, probability in zip(
        CLASS_NAMES,
        calibrated_probabilities
    ):

        print(
            f"{class_name:20s}: "
            f"{probability:.4f}"
        )

    print()
    print("Predicted class :", result["predicted_class"])
    print(
        "Calibrated score:",
        round(result["calibrated_score"], 4)
    )
    print(
        "Entropy         :",
        round(result["entropy"], 4)
    )
    print(
        "Reliability     :",
        result["reliability"]
    )
    print(
        "Recommendation   :",
        result["recommendation"]
    )