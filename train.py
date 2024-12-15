from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('yolov8n.pt')

    # Train the model
    model.train(data='./dataset/data.yaml', epochs=300, imgsz=320)
    # model.val()
