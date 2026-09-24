from ultralytics import YOLO

def main():
    model = YOLO(r"runs\detect\train-5\weights\best.pt")

    metrics = model.val(
        data=r"W:\dataset\final_vehicledataset\data.yaml",
        split="test",
        imgsz=640,
        conf=0.001,
        workers=0,
        plots=True,
        project="runs/evaluate",
        name="vehicle_test"
    )

    print("\n--- FINAL TEST RESULTS ---")
    precision = metrics.box.mp
    recall = metrics.box.mr
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    print(f"Precision:   {precision:.3f}")
    print(f"Recall:      {recall:.3f}")
    print(f"F1-score:    {f1:.3f}")
    print(f"mAP@50:      {metrics.box.map50:.3f}")
    print(f"mAP@50-95:   {metrics.box.map:.3f}")

if __name__ == "__main__":
    main()