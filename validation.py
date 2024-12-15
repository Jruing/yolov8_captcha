from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO('./runs/detect/train2/weights/best.pt')

    model.val()
