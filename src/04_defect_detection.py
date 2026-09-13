import cv2
from pathlib import Path
# PCB defect class names
class_names = {
    0: "mouse_bite",
    1: "spur",
    2: "missing_hole",
    3: "short",
    4: "open_circuit",
    5: "spurious_copper"
}

# Dataset folders
image_folder = Path("pcb-defect-dataset/train/images")
label_folder = Path("pcb-defect-dataset/train/labels")

# Find training images
images = list(image_folder.glob("*"))

print("Number of training images:", len(images))

# -------------------------------------------------
# Find the first image that has a matching YOLO label
# -------------------------------------------------

image_path = None
label_path = None

for candidate in images:

    possible_label = label_folder / (candidate.stem + ".txt")

    if possible_label.exists():
        image_path = candidate
        label_path = possible_label
        break


# Make sure we found a matching pair
if image_path is None:
    print("ERROR: No image-label pair found!")
    exit()


print("\nMatching pair found!")
print("Image:", image_path)
print("Label:", label_path)


# -------------------------------------------------
# Read YOLO label
# -------------------------------------------------

with open(label_path, "r") as file:
    labels = file.readlines()

print("\nYOLO labels:")

for label in labels:
    print(label.strip())


# -------------------------------------------------
# Read PCB image
# -------------------------------------------------

img = cv2.imread(str(image_path))

if img is None:
    print("ERROR: Image could not be loaded")

else:
    print("\nImage loaded successfully!")
    print("Image size:", img.shape)
        # Get image dimensions
    height, width, _ = img.shape

    # Read every defect label
    for label in labels:

        parts = label.strip().split()

        class_id = int(parts[0])

        x_center = float(parts[1])
        y_center = float(parts[2])
        box_width = float(parts[3])
        box_height = float(parts[4])

        # Convert YOLO normalized values into pixels
        x_center = int(x_center * width)
        y_center = int(y_center * height)

        box_width = int(box_width * width)
        box_height = int(box_height * height)

        # Calculate rectangle corners
        x1 = int(x_center - box_width / 2)
        y1 = int(y_center - box_height / 2)

        x2 = int(x_center + box_width / 2)
        y2 = int(y_center + box_height / 2)

        # Draw bounding box
        cv2.rectangle(
            img,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            2
        )

        # Write class ID
        cv2.putText(
            img,
           class_names[class_id],
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2
        )

    cv2.imshow("PCB Training Image", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()