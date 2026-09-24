from pathlib import Path

root = Path(r"W:\dataset\Final_vehicledataset")

print("=== DATASET CHECK ===")

for split in ("train", "val", "test"):
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split

    images = sorted(p for p in images_dir.iterdir() if p.is_file()) if images_dir.exists() else []
    labels = sorted(p for p in labels_dir.iterdir() if p.is_file()) if labels_dir.exists() else []

    print()
    print(split)
    print(f"Images: {len(images)}")
    print(f"Labels: {len(labels)}")