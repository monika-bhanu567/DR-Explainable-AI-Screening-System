from pathlib import Path
import cv2


def preprocess_image(
    input_path: str,
    output_path: str,
    image_size: int = 224
):
    """
    Preprocess one retinal fundus image.

    Steps:
    1. Read image
    2. Remove unnecessary black background
    3. Crop to the retinal region
    4. Resize to 224 x 224
    5. Save the processed image
    """

    # --------------------------------------------------
    # Read image
    # --------------------------------------------------

    image = cv2.imread(input_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {input_path}"
        )

    # --------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------
    # Find retinal area
    # --------------------------------------------------

    # Pixels brighter than near-black
    # are considered part of the retinal image.

    _, mask = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )

    # --------------------------------------------------
    # Find largest contour
    # --------------------------------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        raise ValueError(
            f"Could not detect retinal region: {input_path}"
        )

    largest_contour = max(
        contours,
        key=cv2.contourArea
    )

    x, y, w, h = cv2.boundingRect(
        largest_contour
    )

    # --------------------------------------------------
    # Crop retinal region
    # --------------------------------------------------

    cropped = image[
        y:y + h,
        x:x + w
    ]

    # --------------------------------------------------
    # Resize
    # --------------------------------------------------

    resized = cv2.resize(
        cropped,
        (image_size, image_size),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Save processed image
    # --------------------------------------------------

    success = cv2.imwrite(
        str(output_path),
        resized
    )

    if not success:
        raise IOError(
            f"Could not save image: {output_path}"
        )

    print("Preprocessing successful!")
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Size  : {image_size} x {image_size}")


# ------------------------------------------------------
# TEST WITH ONE IMAGE
# ------------------------------------------------------

if __name__ == "__main__":

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    input_image = (
        project_root
        / "ai"
        / "datasets"
        / "raw"
        / "IDRiD"
        / "B_Disease_Grading"
        / "B. Disease Grading"
        / "1. Original Images"
        / "a. Training Set"
        / "IDRiD_001.jpg"
    )

    output_image = (
        project_root
        / "ai"
        / "datasets"
        / "processed"
        / "train"
        / "IDRiD_001.jpg"
    )

    preprocess_image(
        str(input_image),
        str(output_image)
    )