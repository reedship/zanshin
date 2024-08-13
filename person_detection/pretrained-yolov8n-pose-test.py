from ultralytics import YOLO
model = YOLO("yolov8n-pose.pt")
results = model(source="/Volumes/trainingdata/edited/seoi nage/17.mp4", show=True, conf=0.3, save=True)
