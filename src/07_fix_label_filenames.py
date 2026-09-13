from pathlib import Path

dataset = Path("pcb-defect-dataset")
splits = ["train", "val", "test"]

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

total_renamed = 0

for split in splits:

    print("\n" + "=" * 55)
    print(f"FIXING {split.upper()}")
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

    mappings = []
    unsafe = []

    for image_name in sorted(unmatched_images):

        if not image_name.endswith("_600"):
            unsafe.append(image_name)
            continue

        expected_old_label = image_name[:-4] + "_256"

        if expected_old_label not in unmatched_labels:
            unsafe.append(image_name)
            continue

        old_path = label_folder / f"{expected_old_label}.txt"
        new_path = label_folder / f"{image_name}.txt"

        if new_path.exists():
            unsafe.append(image_name)
            continue

        mappings.append((old_path, new_path))

    # Safety check before renaming
    if (
        len(unsafe) != 0
        or len(mappings) != len(unmatched_images)
        or len(mappings) != len(unmatched_labels)
    ):
        print("SAFETY CHECK FAILED!")
        print("Nothing renamed in this split.")
        print("Unsafe items:", len(unsafe))
        continue

    # Perform rename
    for old_path, new_path in mappings:
        old_path.rename(new_path)
        total_renamed += 1

    print("Renamed labels:", len(mappings))


print("\n" + "=" * 55)
print("REPAIR COMPLETE")
print("=" * 55)

print("Total labels renamed:", total_renamed)