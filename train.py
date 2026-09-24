from ultralytics import YOLO

def main():
    model = YOLO(r"C:\Users\Billy\Documents\vehicle detector\runs\detect\train-6\weights\last.pt")
    

    model.train(
        data=r"W:\dataset\final_vehicledataset\data.yaml",
        epochs=10,
        batch=24,
        imgsz=640,
        resume=True,
        workers=4,
    )

if __name__ == "__main__":
    main()