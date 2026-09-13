from pathlib import Path

dataset = Path("pcb-defect-dataset")

splits = ["train", "val", "test"]

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in splits:

    print("\n" + "=" * 50)
    print(f"CHECKING: {split.upper()}")
    print("=" * 50)

    image_folder = dataset / split / "images"
    label_folder = dataset / split / "labels"

    images = [
        p for p in image_folder.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    labels = list(label_folder.glob("*.txt"))

    print("Images:", len(images))
    print("Labels:", len(labels))

    image_names = {p.stem for p in images}
    label_names = {p.stem for p in labels}

    # Find filename mismatches
    no_label = image_names - label_names
    no_image = label_names - image_names

    print("Images without label:", len(no_label))
    print("Labels without image:", len(no_image))

    # Show example mismatches
    print("\nFirst 10 images without matching labels:")

    for name in sorted(no_label)[:10]:
        print("IMAGE :", name)

    print("\nFirst 10 labels without matching images:")

    for name in sorted(no_image)[:10]:
        print("LABEL :", name)

    # Check YOLO annotations
    invalid_labels = []
    class_counts = {}

    for label_file in labels:

        with open(label_file, "r") as f:

            for line_number, line in enumerate(f, start=1):

                parts = line.strip().split()

                if not parts:
                    continue

                if len(parts) != 5:
                    invalid_labels.append(
                        (label_file.name, line_number, "Wrong number of values")
                    )
                    continue

                try:
                    class_id = int(parts[0])

                    x = float(parts[1])
                    y = float(parts[2])
                    w = float(parts[3])
                    h = float(parts[4])

                except ValueError:
                    invalid_labels.append(
                        (label_file.name, line_number, "Non-numeric value")
                    )
                    continue

                if class_id < 0 or class_id > 5:
                    invalid_labels.append(
                        (label_file.name, line_number, "Invalid class ID")
                    )

                if not (
                    0 <= x <= 1
                    and 0 <= y <= 1
                    and 0 < w <= 1
                    and 0 < h <= 1
                ):
                    invalid_labels.append(
                        (label_file.name, line_number, "Invalid coordinates")
                    )

                class_counts[class_id] = class_counts.get(class_id, 0) + 1

    print("\nDefects per class:")

    for class_id in sorted(class_counts):
        print(f"Class {class_id}: {class_counts[class_id]}")

    print("\nInvalid annotations:", len(invalid_labels))

    if invalid_labels:

        print("\nFirst few annotation problems:")

        for problem in invalid_labels[:10]:
            print(problem)


print("\nDataset check finished.")