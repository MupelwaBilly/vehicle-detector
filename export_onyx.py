from ultralytics import YOLO

model = YOLO(r"runs\detect\train-6\weights\best.pt")  
model.export(format="onnx", imgsz=640, simplify=True, nms=True)