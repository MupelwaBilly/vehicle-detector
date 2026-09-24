from ultralytics import YOLO

model = YOLO(r"runs\detect\train-5\weights\best.pt") 
print(model.names)