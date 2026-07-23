# pyrefly: ignore [missing-import]
from ultralytics import YOLO
from pathlib import Path
import os

PATH = Path('runs/classify/train-3/weights/best.pt')

model = YOLO(PATH)

"""
TEST = Path('data_test/Garbage classification/glass/glass1.jpg')

results = model.predict(source = TEST ,save=True)

for res in results :
    top1_id = res.probs.top1

    top1_name = res.names[ top1_id ]

    top1_conf = res.probs.top1conf.item()

    print(
        f"[PREDICTION] Class: {top1_name} | Confidence: {top1_conf * 100:.2f}%"
    )
"""


TEST_DATA = Path('data_test/Garbage classification')

dict = {}
error_details = []

for dir in os.listdir(TEST_DATA):

    error = 0
    img = 0 

    category_path = TEST_DATA / dir

    if not category_path.is_dir():
        continue

    for file in category_path.iterdir():

        if file.is_file():
            img += 1

        if file.is_file():

            results = model.predict(source = file , verbose=False)

            for res in results :

                top1_id = res.probs.top1

                top1_name = res.names[ top1_id ]

                top1_conf = res.probs.top1conf.item() * 100

                if top1_name != dir :

                    error += 1

                    error_details.append(
                        {
                            "file_path": str(file),
                            "true_class": dir,
                            "predicted_class": top1_name,
                            "confidence": top1_conf,
                        }
                    )
        
    dict[dir] = [img , error]

sum_img = 0
sum_error = 0

for dir , list in dict.items():
    print(f'{list[1]} errors of {list[0]} images in {dir}')

    sum_img += list[0]

    sum_error += list[1]

print(f"{sum_error} errors of {sum_img} images")

print(f'Accuracy = {((1-(sum_error/sum_img))*100):.2f}')

print("\n" + "=" * 60)
print("MISCLASSIFIED IMAGES LOG")
print("=" * 60)
for item in error_details:
    print(
        f"File: {item['file_path']} | True: {item['true_class']} | Pred: {item['predicted_class']} ({item['confidence']:.2f}%)"
    )