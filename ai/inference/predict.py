import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import torch
import torch.nn.functional as F

from ai.models.dr_classifier import DRClassifier


# --------------------------------------------------
# PATHS
# --------------------------------------------------

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "checkpoints"
    / "dr_model_512.pth"
)

TEMPERATURE_PATH = (
    PROJECT_ROOT
    / "ai"
    / "checkpoints"
    / "temperature.pt"
)


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

IMAGE_SIZE = 512
NUM_CLASSES = 5

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


# --------------------------------------------------
# DEVICE
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

model = DRClassifier(
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()


# --------------------------------------------------
# LOAD TEMPERATURE
# --------------------------------------------------

temperature_data = torch.load(
    TEMPERATURE_PATH,
    map_location="cpu",
    weights_only=True
)

TEMPERATURE = float(
    temperature_data["temperature"]
)

if TEMPERATURE <= 0:
    raise ValueError(
        "Temperature must be greater than zero."
    )


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_image(image_path):

    image_path = Path(image_path)

    # --------------------------------------------------
    # Read image
    # --------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )


    # --------------------------------------------------
    # Convert BGR → RGB
    # --------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------
    # Resize to 512 × 512
    # --------------------------------------------------

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )


    # --------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------

    image_tensor = torch.tensor(
        image,
        dtype=torch.float32
    ).permute(2, 0, 1) / 255.0


    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)


    # Move to device
    image_tensor = image_tensor.to(device)


    # --------------------------------------------------
    # MODEL INFERENCE
    # --------------------------------------------------

    with torch.no_grad():

        # Raw model output
        logits = model(
            image_tensor
        )

        # Original probabilities
        original_probabilities = F.softmax(
            logits,
            dim=1
        )

        # Temperature scaling
        calibrated_logits = (
            logits / TEMPERATURE
        )

        # Calibrated probabilities
        calibrated_probabilities = F.softmax(
            calibrated_logits,
            dim=1
        )

        # Final prediction
        confidence, predicted_class = torch.max(
            calibrated_probabilities,
            dim=1
        )


    # --------------------------------------------------
    # CONVERT RESULTS
    # --------------------------------------------------

    predicted_class = (
        predicted_class.item()
    )

    confidence = (
        confidence.item()
    )

    raw_logits = (
        logits[0]
        .cpu()
        .tolist()
    )

    original_probabilities = (
        original_probabilities[0]
        .cpu()
        .tolist()
    )

    calibrated_probabilities = (
        calibrated_probabilities[0]
        .cpu()
        .tolist()
    )


    # --------------------------------------------------
    # RESULT
    # --------------------------------------------------

    return {

        "predicted_class":
            predicted_class,

        "dr_grade":
            CLASS_NAMES[predicted_class],

        "confidence":
            confidence,

        "calibrated_score":
            confidence,

        "temperature":
            TEMPERATURE,

        "raw_logits":
            raw_logits,

        "original_probabilities":
            original_probabilities,

        "probabilities":
            calibrated_probabilities
    }


# --------------------------------------------------
# TEST PREDICTION
# --------------------------------------------------

if __name__ == "__main__":

    # Test using one processed validation image

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
    print("CALIBRATED DR IMAGE INFERENCE")
    print("=" * 60)

    print(
        "Image:",
        test_image
    )

    print(
        "Using device:",
        device
    )

    print(
        "Input size:",
        f"{IMAGE_SIZE}x{IMAGE_SIZE}"
    )

    print(
        "Temperature:",
        f"{TEMPERATURE:.4f}"
    )


    # Run prediction
    result = predict_image(
        test_image
    )


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    print()
    print("PREDICTION")
    print("-" * 60)

    print(
        "Predicted class :",
        result["predicted_class"]
    )

    print(
        "DR grade        :",
        result["dr_grade"]
    )

    print(
        "Calibrated score:",
        f"{result['calibrated_score']:.4f}"
    )


    # --------------------------------------------------
    # ORIGINAL PROBABILITIES
    # --------------------------------------------------

    print()
    print("ORIGINAL PROBABILITIES")
    print("-" * 60)

    for class_id, probability in enumerate(
        result["original_probabilities"]
    ):

        print(
            f"{CLASS_NAMES[class_id]:20s}: "
            f"{probability:.4f}"
        )


    # --------------------------------------------------
    # CALIBRATED PROBABILITIES
    # --------------------------------------------------

    print()
    print("CALIBRATED PROBABILITIES")
    print("-" * 60)

    for class_id, probability in enumerate(
        result["probabilities"]
    ):

        print(
            f"{CLASS_NAMES[class_id]:20s}: "
            f"{probability:.4f}"
        )


    # --------------------------------------------------
    # RAW LOGITS
    # --------------------------------------------------

    print()
    print("RAW LOGITS")
    print("-" * 60)

    for class_id, logit in enumerate(
        result["raw_logits"]
    ):

        print(
            f"{CLASS_NAMES[class_id]:20s}: "
            f"{logit:.4f}"
        )


    print()
    print("=" * 60)
    print("CALIBRATED INFERENCE TEST COMPLETE")
    print("=" * 60)