import sys
from pathlib import Path

# --------------------------------------------------
# PROJECT ROOT
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


import cv2
import numpy as np
import torch

from ai.models.dr_classifier import DRClassifier


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

MODEL_PATH = (
    PROJECT_ROOT
    / "ai"
    / "checkpoints"
    / "dr_model_512.pth"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "ai"
    / "outputs"
    / "heatmaps"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


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
# GRAD-CAM++ CLASS
# --------------------------------------------------

class GradCAMPlusPlus:

    def __init__(self, model):

        self.model = model

        self.activations = None
        self.gradients = None

        # EfficientNet-B0 final convolution layer
        self.target_layer = (
            self.model.model.features[-1][0]
        )

        self.forward_handle = (
            self.target_layer.register_forward_hook(
                self.save_activations
            )
        )

        self.backward_handle = (
            self.target_layer.register_full_backward_hook(
                self.save_gradients
            )
        )

    def save_activations(
        self,
        module,
        input_data,
        output
    ):

        self.activations = output

    def save_gradients(
        self,
        module,
        grad_input,
        grad_output
    ):

        self.gradients = grad_output[0]

    def generate(
        self,
        input_tensor,
        target_class
    ):

        self.model.zero_grad()

        # Forward pass
        output = self.model(input_tensor)

        # Target class score
        target_score = output[
            0,
            target_class
        ]

        # Backward pass
        target_score.backward()

        activations = self.activations
        gradients = self.gradients

        # --------------------------------------------------
        # Grad-CAM++ WEIGHTS
        # --------------------------------------------------

        gradients_2 = gradients ** 2
        gradients_3 = gradients ** 3

        sum_activations = torch.sum(
            activations,
            dim=(2, 3),
            keepdim=True
        )

        denominator = (
            2.0 * gradients_2
            + sum_activations * gradients_3
        )

        denominator = torch.where(
            denominator != 0.0,
            denominator,
            torch.ones_like(denominator)
        )

        alpha = (
            gradients_2 / denominator
        )

        positive_gradients = torch.relu(
            gradients
        )

        weights = torch.sum(
            alpha * positive_gradients,
            dim=(2, 3)
        )

        # --------------------------------------------------
        # GENERATE RAW CAM
        # --------------------------------------------------

        cam = torch.sum(
            weights[:, :, None, None]
            * activations,
            dim=1
        )

        cam = torch.relu(cam)

        # Remove batch dimension
        cam = (
            cam[0]
            .detach()
            .cpu()
            .numpy()
        )

        # --------------------------------------------------
        # NORMALIZE RAW CAM
        # --------------------------------------------------

        cam -= cam.min()

        if cam.max() > 0:
            cam /= cam.max()

        return cam

    def close(self):

        self.forward_handle.remove()
        self.backward_handle.remove()


# --------------------------------------------------
# PREPARE IMAGE
# --------------------------------------------------

def prepare_image(image_path):

    image = cv2.imread(
        str(image_path)
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # Keep original for visualization
    original = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Same preprocessing used by inference
    image_resized = cv2.resize(
        original,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    # Convert to tensor
    tensor = torch.tensor(
        image_resized,
        dtype=torch.float32
    ).permute(2, 0, 1) / 255.0

    tensor = tensor.unsqueeze(0)

    return original, tensor


# --------------------------------------------------
# GENERATE GRAD-CAM++
# --------------------------------------------------

def generate_gradcam(
    image_path,
    output_path
):

    original_image, input_tensor = (
        prepare_image(image_path)
    )

    input_tensor = input_tensor.to(device)

    # --------------------------------------------------
    # FIND PREDICTION
    # --------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()

        model_score = probabilities[
            0,
            predicted_class
        ].item()

    # --------------------------------------------------
    # GENERATE GRAD-CAM++
    # --------------------------------------------------

    cam_generator = GradCAMPlusPlus(
        model
    )

    cam = cam_generator.generate(
        input_tensor,
        predicted_class
    )

    cam_generator.close()

    # --------------------------------------------------
    # SAVE RAW CAM
    # --------------------------------------------------

    raw_cam_path = (
        Path(output_path).with_suffix(".npy")
    )

    np.save(
        str(raw_cam_path),
        cam
    )

    # --------------------------------------------------
    # RESIZE CAM FOR VISUALIZATION
    # --------------------------------------------------

    cam_resized = cv2.resize(
        cam,
        (
            original_image.shape[1],
            original_image.shape[0]
        ),
        interpolation=cv2.INTER_LINEAR
    )

    # Convert to 0-255
    heatmap = np.uint8(
        255 * cam_resized
    )

    # Apply color map
    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    # --------------------------------------------------
    # OVERLAY
    # --------------------------------------------------

    overlay = cv2.addWeighted(
        original_image,
        0.60,
        heatmap,
        0.40,
        0
    )

    # --------------------------------------------------
    # ADD TEXT
    # --------------------------------------------------

    overlay_bgr = cv2.cvtColor(
        overlay,
        cv2.COLOR_RGB2BGR
    )

    text = (
        f"{CLASS_NAMES[predicted_class]} "
        f"({model_score:.2f})"
    )

    cv2.putText(
        overlay_bgr,
        text,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------
    # SAVE VISUALIZATION
    # --------------------------------------------------

    cv2.imwrite(
        str(output_path),
        overlay_bgr
    )

    return {
        "predicted_class": predicted_class,
        "dr_grade": CLASS_NAMES[predicted_class],
        "model_score": model_score,
        "output_path": str(output_path),
        "raw_cam_path": str(raw_cam_path)
    }


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

    output_file = (
        OUTPUT_DIR
        / "IDRiD_006_gradcam_plus_plus.jpg"
    )

    print()
    print("=" * 60)
    print("GRAD-CAM++ EXPLAINABILITY TEST")
    print("=" * 60)

    print(
        "Input image :",
        test_image
    )

    print(
        "Using device:",
        device
    )

    result = generate_gradcam(
        test_image,
        output_file
    )

    print()
    print("RESULT")
    print("-" * 60)

    print(
        "Predicted grade :",
        result["dr_grade"]
    )

    print(
        "Model score     :",
        f"{result['model_score']:.4f}"
    )

    print(
        "Heatmap saved   :",
        result["output_path"]
    )

    print(
        "Raw CAM saved   :",
        result["raw_cam_path"]
    )

    print()
    print("=" * 60)
    print("GRAD-CAM++ TEST COMPLETE")
    print("=" * 60)