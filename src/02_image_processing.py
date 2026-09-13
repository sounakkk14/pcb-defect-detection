from pathlib import Path

import cv2


# -----------------------------
# PROJECT PATHS
# -----------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "pcb-defect-dataset"
    / "train"
    / "images"
    / "l_light_01_missing_hole_01_1_600.jpg"
)

OUTPUT_FOLDER = PROJECT_ROOT / "images"

OUTPUT_FOLDER.mkdir(exist_ok=True)


# -----------------------------
# MAIN PROGRAM
# -----------------------------

def main():

    print("Loading image...")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print("Image not found.")
        return

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge Detection
    edges = cv2.Canny(blur, 50, 150)

    # Save processed images
    cv2.imwrite(str(OUTPUT_FOLDER / "gray.jpg"), gray)
    cv2.imwrite(str(OUTPUT_FOLDER / "blur.jpg"), blur)
    cv2.imwrite(str(OUTPUT_FOLDER / "edges.jpg"), edges)

    # Show all windows
    cv2.imshow("Original", image)
    cv2.imshow("Gray", gray)
    cv2.imshow("Blur", blur)
    cv2.imshow("Edges", edges)

    print("Images saved successfully.")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()