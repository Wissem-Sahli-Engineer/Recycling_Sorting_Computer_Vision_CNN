# pyrefly: ignore [missing-import]
from ultralytics import YOLO
import os


model = YOLO("yolo26n-cls.pt")



# model.train(data="dataset", epochs=25, imgsz=256 , device="mps",)


model.train(
    data="dataset",
    epochs=25,
    imgsz=256, 
    batch=64,  
    device="mps", 
    workers=8,  # Faster CPU data loading
    cache=True,  # Cache dataset to RAM
)