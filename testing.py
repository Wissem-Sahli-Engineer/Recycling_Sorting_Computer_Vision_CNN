# pyrefly: ignore [missing-import]
from ultralytics import YOLO
from pathlib import Path

PATH = Path('runs/classify/train-3/weights/best.pt')

TEST = Path('data_test/Garbage classification/trash/trash1.jpg')


model = YOLO(PATH)

results = model.predict(source = TEST ,save=True)

for res in results :
    top1_id = res.probs.top1
    
    top1_name = res.names[ top1_id ]

    top1_conf = res.probs.top1conf.item()

    print(
        f"[PREDICTION] Class: {top1_name} | Confidence: {top1_conf * 100:.2f}%"
    )