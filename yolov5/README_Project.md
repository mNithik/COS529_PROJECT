# YOLOv5 for Multimodal Aerial Vehicle Detection

This directory contains the YOLOv5 implementation specifically configured for the multimodal aerial vehicle detection project.

## 🎯 Project-Specific Configuration

### Custom Dataset Configurations

#### Single-Class Vehicle Detection
```yaml
# vedai_car.yaml
path: /path/to/data/VEDAI
train: fold01_write_test_fixed.txt
test: fold03_write_test_fixed.txt
val: fold02_write_test_fixed.txt
nc: 1
names: ['car']
```

#### Multi-Class Vehicle Detection (8 Classes)
```yaml
# vedai8.yaml
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

#### Extended Multi-Class Vehicle Detection (9 Classes)
```yaml
# vedai9.yaml
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

## 🚀 Quick Start for COS529 Project

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python detect.py --weights yolov5s.pt --source data/images/bus.jpg
```

### Training
```bash
# Single-class vehicle detection
python train.py \
    --data ../data/vedai_car.yaml \
    --weights yolov5m.pt \
    --img 512 \
    --batch-size 16 \
    --epochs 50 \
    --device 0

# Multi-class vehicle detection
python train.py \
    --data ../data/vedai8.yaml \
    --weights yolov5m.pt \
    --img 512 \
    --batch-size 16 \
    --epochs 50 \
    --device 0

# Use the provided training script
bash run_train.sh
```

### Inference
```bash
# Detect vehicles in images
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/images/ \
    --img 512 \
    --conf 0.25

# Detect vehicles in video
python detect.py \
    --weights runs/train/exp/weights/best.pt \
    --source path/to/video.mp4 \
    --img 512 \
    --conf 0.25
```

### Validation
```bash
# Validate model performance
python val.py \
    --weights runs/train/exp/weights/best.pt \
    --data ../data/vedai8.yaml \
    --img 512 \
    --conf 0.001 \
    --iou 0.65
```

## 📊 Project-Specific Features

### Multi-Modal Processing
- **Color (co) Images**: Optical imagery for visual features
- **Infrared (ir) Images**: Thermal imagery for heat signatures
- **Dual Modality**: Process both modalities for comprehensive detection

### Aerial Imagery Optimization
- **Image Size**: 512×512 pixels (optimized for aerial vehicles)
- **Small Object Detection**: Specialized for detecting small vehicles in aerial views
- **Multi-scale Training**: Handles vehicles at different scales

### Cross-Validation Setup
- **10-Fold Validation**: Robust evaluation with multiple data splits
- **Stratified Splits**: Balanced representation across vehicle classes
- **Consistent Evaluation**: Standardized metrics across all folds

## 🛠️ Custom Modifications

### Data Processing
- **Annotation Conversion**: Polygon to YOLO format conversion
- **Multi-Modal Handling**: Support for both color and infrared images
- **Aerial-Specific Augmentations**: Custom augmentations for aerial imagery

### Model Architecture
- **YOLOv5 Variants**: Support for s, m, l, x model sizes
- **Custom Anchors**: Optimized anchor boxes for aerial vehicles
- **Multi-Scale Detection**: Enhanced detection at different scales

### Training Configuration
- **Batch Size**: Optimized for aerial imagery (16 for 512×512)
- **Learning Rate**: Auto-scaled based on batch size
- **Epochs**: 50 epochs for convergence
- **Device**: GPU-optimized training

## 📈 Performance Monitoring

### Training Metrics
- **Loss Tracking**: Training and validation loss curves
- **mAP Metrics**: Mean Average Precision at different IoU thresholds
- **Per-Class Performance**: Individual class detection metrics
- **Speed Analysis**: Inference time and FPS measurements

### Visualization Tools
- **TensorBoard**: Real-time training visualization
- **Weights & Biases**: Experiment tracking and comparison
- **Custom Plots**: Aerial-specific visualization tools

## 🔧 Advanced Usage

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

### Model Export
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
```

## 📁 Directory Structure

```
yolov5/
├── README_COS529.md          # This file
├── train.py                   # Training script
├── detect.py                  # Inference script
├── val.py                     # Validation script
├── export.py                  # Model export script
├── run_train.sh              # Training script for COS529
├── data/                      # Configuration files
│   ├── vedai_car.yaml        # Single-class config
│   ├── vedai8.yaml           # 8-class config
│   └── vedai9.yaml           # 9-class config
├── models/                    # Model architectures
│   ├── yolo.py               # YOLOv5 model definition
│   ├── common.py             # Common modules
│   └── experimental.py       # Experimental models
├── utils/                     # Utility functions
│   ├── general.py            # General utilities
│   ├── datasets.py           # Dataset handling
│   ├── loss.py               # Loss functions
│   ├── metrics.py            # Evaluation metrics
│   └── plots.py              # Visualization tools
├── requirements.txt          # Dependencies
└── yolov5m.pt               # Pre-trained weights
```

## 🎓 Academic Context

This YOLOv5 implementation is specifically designed for the COS529 Advanced Computer Vision course project, focusing on:

- **Multi-Modal Computer Vision**: Fusion of color and infrared imagery
- **Aerial Vehicle Detection**: Specialized for small object detection in aerial views
- **Deep Learning**: Implementation of state-of-the-art object detection
- **Research Methodology**: Comprehensive evaluation and analysis

## 🔬 Research Features

### Novel Contributions
- **Multi-Modal Fusion**: Integration of color and infrared modalities
- **Aerial Optimization**: Specialized for aerial vehicle detection
- **Cross-Validation**: Robust 10-fold evaluation methodology
- **Performance Analysis**: Comprehensive metrics and visualization

### Experimental Setup
- **Dataset**: VEDAI aerial vehicle detection dataset
- **Models**: YOLOv5s, YOLOv5m, YOLOv5l variants
- **Evaluation**: mAP, precision, recall, F1-score metrics
- **Visualization**: TensorBoard and custom plotting tools

## 📚 Technical Skills Demonstrated

- **Computer Vision**: Object detection, multi-modal fusion
- **Deep Learning**: PyTorch, YOLOv5, transfer learning
- **Data Processing**: Custom dataset handling, annotation conversion
- **Model Training**: Hyperparameter tuning, cross-validation
- **Software Engineering**: Modular design, configuration management

## 🤝 Usage Guidelines

### For Students
1. **Follow Setup Guide**: Use the comprehensive setup instructions
2. **Start Simple**: Begin with single-class detection
3. **Experiment**: Try different configurations and hyperparameters
4. **Document Results**: Keep track of experiments and results
5. **Ask Questions**: Use the provided documentation and examples

### For Researchers
1. **Customize Configurations**: Modify YAML files for your needs
2. **Extend Functionality**: Add new features and capabilities
3. **Compare Methods**: Use the evaluation framework for comparisons
4. **Publish Results**: Follow academic standards for reporting
5. **Contribute**: Share improvements and new features

## 📞 Support

For questions or issues specific to the COS529 project:

1. **Check Documentation**: Review all provided documentation
2. **Verify Setup**: Ensure proper installation and configuration
3. **Test Examples**: Run provided examples to verify functionality
4. **Check Logs**: Review training and inference logs for errors
5. **Seek Help**: Contact course instructors or TAs

---

*This YOLOv5 implementation is specifically configured for the COS529 Advanced Computer Vision project on multi-modal vehicle detection in aerial imagery.*
