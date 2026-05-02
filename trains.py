from ultralytics import YOLO
import torch
import os
from pathlib import Path
import yaml


def validate_dataset(data_yaml_path):
    """Enhanced validation with path correction"""
    try:
        with open(data_yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        # Convert to absolute paths
        base_dir = Path(data_yaml_path).parent
        paths = {
            'train_images': Path(data['train']).resolve(),
            'val_images': Path(data['val']).resolve()
        }

        # Check paths
        for name, path in paths.items():
            # Handle OneDrive path variations
            if not path.exists() and "OneDrive" in str(path):
                alt_path = Path(str(path).replace("OneDrive", "OneDrive - Personal"))
                if alt_path.exists():
                    paths[name] = alt_path
                    print(f"⚠️ Fixed path: {path} → {alt_path}")
                    continue

            if not paths[name].exists():
                raise FileNotFoundError(f"Path not found: {paths[name]}")
            if not any(paths[name].iterdir()):
                raise ValueError(f"No images in: {paths[name]}")

        print("✅ Dataset validation passed")
        return True

    except Exception as e:
        print(f"❌ Validation failed: {str(e)}")
        print("\n💡 REQUIRED STRUCTURE:")
        print("pcb_dataset/")
        print("├── data.yaml")
        print("├── train/images/  # .jpg/.png files")
        print("├── train/labels/  # .txt files")
        print("├── val/images/")
        print("└── val/labels/")
        return False


def train_pcb_model():
    config = {
        'data_yaml': r'C:\Users\Hp 840 G4\OneDrive\Desktop\pcb_det\pcd_data\data.yaml',
        'model_type': 'yolov8n.pt',
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    if not validate_dataset(config['data_yaml']):
        print("\n🔧 Try these solutions:")
        print("1. Create missing directories")
        print("2. Add images to train/images/ and val/images/")
        print("3. Update data.yaml with EXACT paths shown above")
        return

    try:
        model = YOLO(config['model_type'])
        model.train(
            data=config['data_yaml'],
            epochs=config['epochs'],
            imgsz=config['imgsz'],
            batch=config['batch'],
            device=config['device'],
            augment=True,
            project='pcb_defect',
            name='training_v1'
        )
        print("✅ Training completed! Results saved to runs/detect/training_v1/")
    except Exception as e:
        print(f"🔥 Training failed: {str(e)}")


if __name__ == '__main__':
    train_pcb_model()