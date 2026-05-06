# Usage Guide - Vehicle Detection in Aerial Imagery

This guide provides comprehensive examples and instructions for using the multimodal vehicle detection project.

## 🎯 Overview

This project supports multiple use cases:
- **Single-class Detection**: Detect all vehicles as one class
- **Multi-class Detection**: Classify different vehicle types (8 or 9 classes)
- **Multi-modal Processing**: Handle both color and infrared imagery
- **Cross-validation**: Robust model evaluation with 10-fold validation

## 🚀 Quick Start Examples

### Basic Training

```bash
# Navigate to yolov5 directory
cd yolov5

# Train single-class vehicle detection
python train.py \
    --data ../data/vedai_car.yaml \
    --weights yolov5m.pt \
    --img 512 \
    --batch-size 16 \
    --epochs 50 \
    --device 0

# Train multi-class vehicle detection (8 classes)
python train.py \
    --data ../data/vedai8.yaml \
    --weights yolov5m.pt \
    --img 512 \
    --batch-size 16 \
    --epochs 50 \
    --device 0
```

### Using the Training Script

```bash
# Use the provided bash script
bash run_train.sh
```

The script automatically configures:
- Model: YOLOv5m
- Image size: 512×512
- Batch size: 16
- Epochs: 50
- Device: GPU 0

## 📊 Training Configurations

### Dataset Configurations

#### Single-Class Detection (`vedai_car.yaml`)
```yaml
path: /path/to/data/VEDAI
train: fold01_write_test_fixed.txt
test: fold03_write_test_fixed.txt
val: fold02_write_test_fixed.txt
nc: 1
names: ['car']
```

#### Multi-Class Detection (`vedai8.yaml`)
```yaml
path: /path/to/data/VEDAI
train: fold01_write_test_fixed.txt
test: fold03_write_test_fixed.txt
val: fold02_write_test_fixed.txt
nc: 8
names:
  - car
  - pickup
  - camping
  - truck
  - other
  - tractor
  - boat
  - van
```

#### Extended Multi-Class Detection (`vedai9.yaml`)
```yaml
path: /path/to/data/VEDAI
train: fold01_write_test_fixed.txt
test: fold03_write_test_fixed.txt
val: fold02_write_test_fixed.txt
nc: 9
names:
  - car
  - pick-up
  - camping car
  - truck
  - vehicle
  - tractor
  - boat
  - van
  - plane
```

### Model Configurations

#### YOLOv5 Model Sizes
```bash
# Small model (fast inference)
python train.py --weights yolov5s.pt --cfg yolov5s.yaml

# Medium model (balanced)
python train.py --weights yolov5m.pt --cfg yolov5m.yaml

# Large model (high accuracy)
python train.py --weights yolov5l.pt --cfg yolov5l.yaml

# Extra large model (maximum accuracy)
python train.py --weights yolov5x.pt --cfg yolov5x.yaml
```

## 🔍 Inference Examples

### Image Detection

```bash
# Detect vehicles in a single image
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/image.jpg \
    --img 512 \
    --conf 0.25 \
    --save-txt

# Detect vehicles in multiple images
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/images/ \
    --img 512 \
    --conf 0.25 \
    --save-txt

# Detect vehicles with custom confidence threshold
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/image.jpg \
    --img 512 \
    --conf 0.5 \
    --iou 0.45
```

### Video Detection

```bash
# Detect vehicles in video
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/video.mp4 \
    --img 512 \
    --conf 0.25 \
    --save-vid

# Real-time webcam detection
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source 0 \
    --img 512 \
    --conf 0.25
```

### Batch Processing

```bash
# Process entire dataset
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source ../data/VEDAI/images/ \
    --img 512 \
    --conf 0.25 \
    --save-txt \
    --save-conf
```

## 📈 Validation and Testing

### Model Validation

```bash
# Validate trained model
python val.py \
    --weights runs/train/exp/weights/best.pt \
    --data ../data/vedai8.yaml \
    --img 512 \
    --conf 0.001 \
    --iou 0.65 \
    --task test

# Validate with different confidence thresholds
python val.py \
    --weights runs/train/exp/weights/best.pt \
    --data ../data/vedai8.yaml \
    --img 512 \
    --conf 0.25 \
    --iou 0.5
```

### Cross-Validation

```bash
# Run validation on different folds
python val.py \
    --weights runs/train/exp/weights/best.pt \
    --data ../data/vedai8.yaml \
    --img 512 \
    --conf 0.25 \
    --iou 0.5 \
    --task val
```

## 🎛️ Advanced Training Options

### Hyperparameter Tuning

```bash
# Custom learning rate
python train.py \
    --data ../data/vedai8.yaml \
    --weights yolov5m.pt \
    --img 512 \
    --batch-size 16 \
    --epochs 50 \
    --lr0 0.01 \
    --lrf 0.1

# Custom augmentation
python train.py \
    --data ../data/vedai8.yaml \
    --weights yolov5m.pt \
    --img 512 \
    --batch-size 16 \
    --epochs 50 \
    --hsv-h 0.015 \
    --hsv-s 0.7 \
    --hsv-v 0.4 \
    --degrees 10.0 \
    --translate 0.1 \
    --scale 0.5
```

### Multi-GPU Training

```bash
# Train on multiple GPUs
python -m torch.distributed.run \
    --nproc_per_node 2 \
    train.py \
    --data ../data/vedai8.yaml \
    --weights yolov5m.pt \
    --img 512 \
    --batch-size 32 \
    --epochs 50 \
    --device 0,1
```

### Transfer Learning

```bash
# Fine-tune from pre-trained weights
python train.py \
    --data ../data/vedai8.yaml \
    --weights yolov5m.pt \
    --img 512 \
    --batch-size 16 \
    --epochs 30 \
    --freeze 10  # Freeze first 10 layers
```

## 📊 Data Processing

### Convert Annotations

```bash
# Convert polygon annotations to YOLO format
python convert_to_vehicle_yolo.py
```

This script:
- Processes both color (`co`) and infrared (`ir`) modalities
- Converts polygon coordinates to bounding boxes
- Normalizes coordinates for YOLO format
- Creates empty label files for images without annotations

### Dataset Splitting

```bash
# Create custom train/val/test splits
python -c "
import os
import random
from pathlib import Path

# Your custom splitting logic here
images = list(Path('data/VEDAI/images').glob('*.png'))
random.shuffle(images)

# Split into train/val/test (70/20/10)
train_split = int(0.7 * len(images))
val_split = int(0.9 * len(images))

train_files = images[:train_split]
val_files = images[train_split:val_split]
test_files = images[val_split:]

# Write to files
with open('data/train_custom.txt', 'w') as f:
    f.write('\n'.join([str(p) for p in train_files]))

with open('data/val_custom.txt', 'w') as f:
    f.write('\n'.join([str(p) for p in val_files]))

with open('data/test_custom.txt', 'w') as f:
    f.write('\n'.join([str(p) for p in test_files]))
"
```

## 🔧 Model Export and Deployment

### Export to Different Formats

```bash
# Export to ONNX
python export.py \
    --weights runs/train/exp/weights/best.pt \
    --include onnx \
    --img 512

# Export to TensorRT
python export.py \
    --weights runs/train/exp/weights/best.pt \
    --include engine \
    --img 512 \
    --device 0

# Export to CoreML (macOS)
python export.py \
    --weights runs/train/exp/weights/best.pt \
    --include coreml \
    --img 512
```

### Model Optimization

```bash
# Quantize model for faster inference
python export.py \
    --weights runs/train/exp/weights/best.pt \
    --include onnx \
    --img 512 \
    --int8
```

## 📊 Monitoring and Logging

### TensorBoard Integration

```bash
# Start TensorBoard
tensorboard --logdir runs/train

# View training metrics at http://localhost:6006
```

### Weights & Biases Integration

```bash
# Login to W&B
wandb login

# Training automatically logs to W&B
python train.py \
    --data ../data/vedai8.yaml \
    --weights yolov5m.pt \
    --img 512 \
    --batch-size 16 \
    --epochs 50
```

## 🧪 Testing and Evaluation

### Performance Testing

```bash
# Test inference speed
python val.py \
    --weights runs/train/exp/weights/best.pt \
    --data ../data/vedai8.yaml \
    --img 512 \
    --batch-size 1 \
    --task speed

# Test accuracy
python val.py \
    --weights runs/train/exp/weights/best.pt \
    --data ../data/vedai8.yaml \
    --img 512 \
    --conf 0.001 \
    --iou 0.65 \
    --task test
```

### Custom Evaluation

```python
# Custom evaluation script
import torch
from yolov5 import YOLOv5

# Load model
model = YOLOv5('runs/train/exp/weights/best.pt')

# Run inference
results = model.predict('path/to/image.jpg')

# Process results
for result in results:
    boxes = result.boxes
    print(f"Detected {len(boxes)} vehicles")
    for box in boxes:
        print(f"Class: {box.cls}, Confidence: {box.conf}")
```

## 🎯 Best Practices

### Training Tips
1. **Start Small**: Begin with YOLOv5s for quick experiments
2. **Monitor GPU Memory**: Adjust batch size based on available VRAM
3. **Use Validation**: Always validate on held-out data
4. **Save Checkpoints**: Enable automatic checkpoint saving
5. **Log Everything**: Use TensorBoard or W&B for monitoring

### Inference Tips
1. **Confidence Threshold**: Adjust based on your precision/recall needs
2. **Image Size**: Use consistent image sizes during training and inference
3. **Batch Processing**: Process multiple images together for efficiency
4. **Model Selection**: Choose model size based on speed/accuracy trade-offs

### Data Tips
1. **Quality Check**: Verify annotation quality before training
2. **Balanced Classes**: Ensure balanced representation across vehicle types
3. **Augmentation**: Use appropriate augmentations for aerial imagery
4. **Cross-validation**: Use multiple folds for robust evaluation

---

*This usage guide provides comprehensive examples for all aspects of the vehicle detection project.*
