from pathlib import Path

import cv2


# ---------------------------------------
# PROJECT PATHS
# ---------------------------------------

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


# ---------------------------------------
# MAIN PROGRAM
# ---------------------------------------

def main():

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print("Image not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    contours, hierarchy = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    print("Contours Found:", len(contours))

    largest_contour = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest_contour)

    result = image.copy()

    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (0, 0, 255),
        2,
    )

    cv2.drawContours(
        result,
        [largest_contour],
        -1,
        (255, 0, 0),
        2,
    )

    cv2.imwrite(
        str(OUTPUT_FOLDER / "pcb_detected.jpg"),
        result,
    )

    cv2.imshow("Original", image)
    cv2.imshow("Edges", edges)
    cv2.imshow("PCB Detection", result)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()