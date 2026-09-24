from pathlib import Path
import uuid

# Change this to your dataset folder
ROOT = Path(r"W:\dataset\final_vehicledataset")

# Keep True first: it only previews changes.
DRY_RUN = True

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for split in ("train", "val", "test"):
    images_dir = ROOT / "images" / split
    labels_dir = ROOT / "labels" / split

    images = sorted(
        file for file in images_dir.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    pairs = []
    missing_labels = []

    for image in images:
        label = labels_dir / f"{image.stem}.txt"

        if label.exists():
            pairs.append((image, label))
        else:
            missing_labels.append(image.name)

    if missing_labels:
        print(f"\n{split}: STOPPED — these images have no matching label file:")
        for name in missing_labels[:20]:
            print(" ", name)
        continue

    print(f"\n{split}: {len(pairs)} image/label pairs found")

    renamed = []

    # Temporary names prevent filename collisions.
    for index, (image, label) in enumerate(pairs, start=1):
        new_stem = f"vehicle_{split}_{index:06d}"

        temp_image = image.with_name(f"temp_{uuid.uuid4().hex}{image.suffix}")
        temp_label = label.with_name(f"temp_{uuid.uuid4().hex}.txt")

        final_image = image.with_name(new_stem + image.suffix)
        final_label = label.with_name(new_stem + ".txt")

        print(f"{image.name}  ->  {final_image.name}")

        if not DRY_RUN:
            image.rename(temp_image)
            label.rename(temp_label)

        renamed.append((temp_image, temp_label, final_image, final_label))

    if not DRY_RUN:
        for temp_image, temp_label, final_image, final_label in renamed:
            temp_image.rename(final_image)
            temp_label.rename(final_label)

print("\nFinished.")
