import os
import torch
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "yolov8n.pt")
DATA_YAML = os.path.join(BASE_DIR, "config", "data.yaml")


def main():
    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model = YOLO(MODEL_PATH)

    results = model.train(
        data=DATA_YAML,
        epochs=25,
        imgsz=640,
        batch=32,
        device=device,
        name="mask_detection_model",
    )

    print("Training complete. Results saved to: outputs/runs/detect/mask_detection_model")


if __name__ == "__main__":
    main()
