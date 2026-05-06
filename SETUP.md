# Setup Guide - Vehicle Detection in Aerial Imagery

This guide provides detailed instructions for setting up and running the multimodal vehicle detection project for aerial imagery.

## 📋 Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+
- **Python**: 3.8 or higher
- **GPU**: NVIDIA GPU with CUDA support (recommended)
- **RAM**: 8GB minimum, 16GB recommended
- **Storage**: 10GB free space minimum

### Hardware Recommendations
- **GPU**: NVIDIA RTX 3060 or better (8GB+ VRAM)
- **CPU**: Intel i5 or AMD Ryzen 5 or better
- **RAM**: 16GB+ for optimal performance

## 🔧 Installation Steps

### Step 1: Clone the Repository

```bash
# Clone the project repository
git clone <your-repository-url>
cd multimodal-aerial-vehicle-detection

# Verify the project structure
ls -la
```

### Step 2: Set Up Python Environment

#### Option A: Using Conda (Recommended)
```bash
# Create a new conda environment
conda create -n aerial-vehicle-det python=3.8
conda activate aerial-vehicle-det

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Verify CUDA installation
python -c "import torch; print(torch.cuda.is_available())"
```

#### Option B: Using Virtual Environment
```bash
# Create virtual environment
python -m venv aerial_vehicle_env
source aerial_vehicle_env/bin/activate  # On Windows: aerial_vehicle_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
# Navigate to yolov5 directory
cd yolov5

# Install required packages
pip install -r requirements.txt

# Install additional dependencies for this project
pip install wandb  # For experiment tracking (optional)
pip install tensorboard  # For visualization (optional)
```

### Step 4: Verify Installation

```bash
# Test YOLOv5 installation
python detect.py --weights yolov5s.pt --source data/images/bus.jpg

# Test CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device count: {torch.cuda.device_count()}')"
```

## 📁 Dataset Setup

### Step 1: Verify Dataset Structure

Ensure your dataset is organized as follows:

```
project-root/
├── data/
│   ├── VEDAI/
│   │   ├── images/           # Original 512x512 images
│   │   │   ├── 00000000_co.png
│   │   │   ├── 00000000_ir.png
│   │   │   └── ...
│   │   ├── fold01_write_test_fixed.txt
│   │   ├── fold02_write_test_fixed.txt
│   │   └── fold03_write_test_fixed.txt
│   └── VEDAI_1024/
│       ├── images/           # Resized 1024x1024 images
│       └── labels/           # YOLO format annotations
```

### Step 2: Update Configuration Paths

Edit the configuration files to match your system paths:

```bash
# Update vedai_car.yaml
nano yolov5/data/vedai_car.yaml

# Update vedai8.yaml
nano yolov5/data/vedai8.yaml

# Update vedai9.yaml
nano yolov5/data/vedai9.yaml
```

**Example configuration update:**
```yaml
# Change this line in each config file
path: /your/path/to/project-root/data/VEDAI
```

### Step 3: Convert Annotations (If Needed)

If you have polygon annotations that need conversion:

```bash
# Run the conversion script
python convert_to_vehicle_yolo.py
```

## 🚀 Quick Start

### Test Installation

```bash
# Run a quick detection test
cd yolov5
python detect.py --weights yolov5s.pt --source data/images/bus.jpg --img 512
```

### Start Training

```bash
# Single-class vehicle detection
python train.py \
    --data ../data/vedai_car.yaml \
    --weights yolov5m.pt \
    --img 512 \
    --batch-size 16 \
    --epochs 10 \
    --device 0

# Or use the provided script
bash run_train.sh
```

## 🔍 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```bash
# Reduce batch size
python train.py --batch-size 8  # Instead of 16

# Use smaller model
python train.py --weights yolov5s.pt  # Instead of yolov5m.pt
```

#### Path Issues
```bash
# Check current directory
pwd

# Verify file paths
ls -la data/VEDAI/images/
ls -la yolov5/data/vedai_car.yaml
```

#### Dependency Conflicts
```bash
# Create fresh environment
conda create -n aerial-vehicle-fresh python=3.8
conda activate aerial-vehicle-fresh
pip install -r requirements.txt
```

### Performance Optimization

#### For Training Speed
```bash
# Use multiple GPUs (if available)
python -m torch.distributed.run --nproc_per_node 2 train.py --data ../data/vedai_car.yaml --weights yolov5m.pt --device 0,1

# Increase batch size (if GPU memory allows)
python train.py --batch-size 32
```

#### For Inference Speed
```bash
# Use TensorRT optimization
python export.py --weights best.pt --include engine --device 0

# Use smaller model for faster inference
python detect.py --weights yolov5s.pt
```

## 📊 Monitoring Training

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir runs/train

# View in browser: http://localhost:6006
```

### Weights & Biases (Optional)
```bash
# Login to W&B
wandb login

# Training will automatically log to W&B
python train.py --data ../data/vedai_car.yaml --weights yolov5m.pt
```

## 🧪 Testing Your Setup

### Run Complete Test Suite

```bash
# Test data loading
python -c "
import yaml
with open('data/vedai_car.yaml', 'r') as f:
    data = yaml.safe_load(f)
    print('Config loaded successfully:', data)
"

# Test model loading
python -c "
import torch
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
print('Model loaded successfully')
"

# Test inference
python detect.py --weights yolov5s.pt --source data/images/bus.jpg --save-txt
```

## 📝 Next Steps

After successful setup:

1. **Start Training**: Run your first training experiment
2. **Monitor Progress**: Use TensorBoard or W&B for visualization
3. **Validate Results**: Test your trained model on validation data
4. **Experiment**: Try different configurations and hyperparameters

## 🆘 Getting Help

If you encounter issues:

1. **Check Logs**: Look at terminal output for error messages
2. **Verify Paths**: Ensure all file paths are correct
3. **Check Dependencies**: Verify all packages are installed correctly
4. **GPU Memory**: Monitor GPU memory usage during training

## 📚 Additional Resources

- [YOLOv5 Documentation](https://docs.ultralytics.com/yolov5/)
- [PyTorch Installation Guide](https://pytorch.org/get-started/locally/)
- [CUDA Installation Guide](https://developer.nvidia.com/cuda-downloads)
- [VEDAI Dataset Information](https://downloads.greyc.fr/vedai/)

---

*This setup guide ensures you have everything needed to run the vehicle detection project successfully.*
