from ultralytics import YOLO
import torch
from multiprocessing import freeze_support


def train_yolo_model(data_yaml_path='C:\Users\Hp 840 G4\OneDrive\Desktop\pcb_det\pcb_dataset\data.yaml', epochs=100, use_gpu=True, batch_size=8, image_size=(480,640)):
    # Check for GPU availability
    device = 'cuda' if use_gpu and torch.cuda.is_available() else 'cpu'

    # Initialize YOLO model (specify the model type, e.g., 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
    model = YOLO('yolo11n.pt')

    # Train the model with default augmentations enabled
    model.train(data=data_yaml_path,
                epochs=epochs,
                device=device,
                project='my_project',
                name='train_results_combined_data',
                batch=batch_size,
                imgsz=image_size,
                amp=True)

    model.save("pcb_det.pt")
    print("Training completed.")


if __name__ == '__main__':
    freeze_support()
    train_yolo_model()



