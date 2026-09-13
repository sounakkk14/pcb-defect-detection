from pathlib import Path

import cv2


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_PATH = (
    PROJECT_ROOT
    / "pcb-defect-dataset"
    / "train"
    / "images"
    / "l_light_01_missing_hole_01_1_600.jpg"
)


# --------------------------------------------------
# MAIN PROGRAM
# --------------------------------------------------

def main():

    print("Loading image...")

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        print("ERROR: Image could not be loaded.")
        return

    height, width, channels = image.shape

    print("\nImage Loaded Successfully!")
    print("----------------------------")
    print(f"Height   : {height}")
    print(f"Width    : {width}")
    print(f"Channels : {channels}")
    print(f"Total Pixels : {height * width}")

    cv2.imshow("AI PCB Inspection System", image)

    cv2.waitKey(0)

    cv2.destroyAllWindows()


# --------------------------------------------------
# PROGRAM ENTRY
# --------------------------------------------------

if __name__ == "__main__":
    main()