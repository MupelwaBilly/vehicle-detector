from ultralytics import YOLO

def main():
    # Starts a new training run from the standard pretrained YOLO model,
    # not from your old vehicle model.
    model = YOLO("yolo11n.pt")

    model.train(
        data=r"W:\dataset\final_vehicledataset\data.yaml",
        epochs=100,
        imgsz=640,
        workers=0,
        project="runs",
        name="vehicle_clean_v1"
    )

if __name__ == "__main__":
    main()
    