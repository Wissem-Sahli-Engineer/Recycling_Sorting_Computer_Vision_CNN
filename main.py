# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import os


model = YOLO("yolo26n-cls.pt")



model.train(data="dataset", epochs=25, imgsz=256)