# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import os


model = YOLO("yolo26n-cls.pt")



results = model.train(data="mnist160", epochs=1, imgsz=64)