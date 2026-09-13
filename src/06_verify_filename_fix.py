from pathlib import Path

dataset = Path("pcb-defect-dataset")
splits = ["train", "val", "test"]

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in splits:

    print("\n" + "=" * 55)
    print(f"VERIFYING {split.upper()}")
    print("=" * 55)

    image_folder = dataset / split / "images"
    label_folder = dataset / split / "labels"

    images = [
        p for p in image_folder.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    labels = list(label_folder.glob("*.txt"))

    image_names = {p.stem for p in images}
    label_names = {p.stem for p in labels}

    unmatched_images = image_names - label_names
    unmatched_labels = label_names - image_names

    mappable = []
    unmappable = []
    collisions = []

    for image_name in sorted(unmatched_images):

        # Convert image name ending _600 into expected _256 label name
        if image_name.endswith("_600"):
            expected_label = image_name[:-4] + "_256"
        else:
            unmappable.append(image_name)
            continue

        if expected_label in unmatched_labels:

            new_label_name = image_name

            # Check whether destination label already exists
            destination = label_folder / f"{new_label_name}.txt"

            if destination.exists():
                collisions.append(
                    (expected_label, new_label_name)
                )
            else:
                mappable.append(
                    (expected_label, new_label_name)
                )

        else:
            unmappable.append(image_name)

    print("Unmatched images :", len(unmatched_images))
    print("Unmatched labels :", len(unmatched_labels))

    print("\nCan safely map :", len(mappable))
    print("Cannot map     :", len(unmappable))
    print("Name collisions:", len(collisions))

    print("\nFirst 10 proposed fixes:")

    for old_name, new_name in mappable[:10]:
        print(f"{old_name}.txt")
        print("   ->")
        print(f"{new_name}.txt")
        print()