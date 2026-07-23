# pyrefly: ignore [missing-import]
from ultralytics import YOLO
from pathlib import Path
import os

PATH = Path('runs/classify/train-3/weights/best.pt')

model = YOLO(PATH)

results = model.predict(source=0, show=True)