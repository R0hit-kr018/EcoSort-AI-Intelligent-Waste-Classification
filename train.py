# Training script for EcoSort AI
from ultralytics import YOLO
import os
import shutil
from pathlib import Path

# Get paths
script_dir = Path(__file__).parent
dataset_dir = script_dir / "dataset"
train_dir = dataset_dir / "train"
val_dir = dataset_dir / "val"

# Classes
classes = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Create train/val structure if needed
print("Organizing dataset into train/val split...")
for cls in classes:
    # Create class directories
    (train_dir / cls).mkdir(parents=True, exist_ok=True)
    (val_dir / cls).mkdir(parents=True, exist_ok=True)
    
    # Source directory with images
    src_cls_dir = dataset_dir / cls
    if src_cls_dir.exists():
        images = list(src_cls_dir.glob("*.jpg")) + list(src_cls_dir.glob("*.png"))
        
        # Split 80/20 for train/val
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        # Copy training images
        for img in train_images:
            dest = train_dir / cls / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
        
        # Copy validation images
        for img in val_images:
            dest = val_dir / cls / img.name
            if not dest.exists():
                shutil.copy2(img, dest)
        
        print(f"  {cls}: {len(train_images)} train, {len(val_images)} val")

# Load YOLOv8 Classification Model
print("\nLoading YOLOv8 model...")
model = YOLO("yolov8n-cls.pt")

# Train
# Epoch settings:
# - 5 epochs: enough for demo
# - 10 epochs: good performance
# - 20 epochs: optional (for better accuracy)
print("Starting training...")
print("Configuration: 5 epochs (demo) | Early stopping enabled")
model.train(
    data=str(dataset_dir),
    epochs=5,  # Change to 10 for good, 20 for optional
    imgsz=224,
    batch=16,
    patience=3  # Stop early if validation loss doesn't improve for 3 epochs
)