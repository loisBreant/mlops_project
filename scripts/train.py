from ultralytics import YOLO
import os

def train_model():
    model = YOLO('yolov8n.pt')

    results = model.train(
        data=os.path.join('../data', 'processed_yolo', 'data.yaml'),
        epochs=50,
        imgsz=320,
        batch=16,
        patience=10,
        project='../models',
        name='yolov8n_taco_320',
        exist_ok=True,
        pretrained=True,
        optimizer='auto',
        verbose=True
    )

    metrics = model.val()

    path = model.export(format='onnx', imgsz=320, opset=12)

if __name__ == '__main__':
    train_model()
