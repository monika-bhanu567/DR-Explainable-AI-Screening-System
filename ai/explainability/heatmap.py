import cv2
import numpy as np
from pathlib import Path


def normalize_cam(cam):
    """
    Normalize an activation map to the range 0-1.
    """

    cam = np.asarray(cam, dtype=np.float32)

    cam_min = cam.min()
    cam_max = cam.max()

    if cam_max - cam_min < 1e-8:
        return np.zeros_like(cam)

    return (cam - cam_min) / (
        cam_max - cam_min
    )


def create_heatmap(cam, output_size):
    """
    Convert a CAM/activation map into a color heatmap.

    Parameters
    ----------
    cam : numpy array
        2D activation map.

    output_size : tuple
        (width, height)

    Returns
    -------
    heatmap : numpy array
        RGB heatmap.
    """

    cam = normalize_cam(cam)

    width, height = output_size

    cam_resized = cv2.resize(
        cam,
        (width, height)
    )

    heatmap = np.uint8(
        cam_resized * 255
    )

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    heatmap = cv2.cvtColor(
        heatmap,
        cv2.COLOR_BGR2RGB
    )

    return heatmap


def overlay_heatmap(
    original_image,
    heatmap,
    alpha=0.4
):
    """
    Overlay a heatmap on the original fundus image.

    alpha controls the heatmap strength.
    """

    original_image = np.asarray(
        original_image,
        dtype=np.uint8
    )

    heatmap = np.asarray(
        heatmap,
        dtype=np.uint8
    )

    if original_image.shape[:2] != heatmap.shape[:2]:

        heatmap = cv2.resize(
            heatmap,
            (
                original_image.shape[1],
                original_image.shape[0]
            )
        )

    overlay = cv2.addWeighted(
        original_image,
        1.0 - alpha,
        heatmap,
        alpha,
        0
    )

    return overlay


def save_heatmap(
    image,
    output_path
):
    """
    Save an RGB image to disk.
    """

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    image_bgr = cv2.cvtColor(
        image,
        cv2.COLOR_RGB2BGR
    )

    success = cv2.imwrite(
        str(output_path),
        image_bgr
    )

    if not success:
        raise IOError(
            f"Could not save image: {output_path}"
        )

    return output_path


if __name__ == "__main__":

    print()
    print("=" * 60)
    print("HEATMAP UTILITY TEST")
    print("=" * 60)

    # Create a small artificial activation map
    test_cam = np.random.rand(
        7,
        7
    ).astype(np.float32)

    # Create a black test image
    test_image = np.zeros(
        (224, 224, 3),
        dtype=np.uint8
    )

    # Create heatmap
    heatmap = create_heatmap(
        test_cam,
        (224, 224)
    )

    # Overlay
    overlay = overlay_heatmap(
        test_image,
        heatmap
    )

    # Save
    PROJECT_ROOT = (
        Path(__file__).resolve().parents[2]
    )

    output_path = (
        PROJECT_ROOT
        / "ai"
        / "outputs"
        / "heatmaps"
        / "heatmap_utility_test.jpg"
    )

    save_heatmap(
        overlay,
        output_path
    )

    print()
    print("Heatmap created successfully!")
    print("Saved to:", output_path)

    print()
    print("=" * 60)
    print("HEATMAP TEST COMPLETE")
    print("=" * 60)