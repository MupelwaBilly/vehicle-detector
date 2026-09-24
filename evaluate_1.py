from ultralytics import YOLO

model = YOLO(r"runs\detect\train-6\weights\best.pt")  

model.predict(
    source=r"C:\Users\Billy\Pictures\exp",
    conf=0.7,
    save=True
)