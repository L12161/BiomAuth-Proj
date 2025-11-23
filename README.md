# BiRealNet Transfer Learning for Fingerprint Recognition

This repository contains code for training BiRealNet on fingerprint datasets (FVC) using transfer learning from ImageNet-pretrained weights.

## Files

- `birealnet.py` - BiRealNet model architecture
- `train_fvc_transfer.py` - Main training script with transfer learning
- `extract_features.py` - Feature extraction script for trained models
- `train.py` - Original ImageNet training script (for reference)

## Dataset Setup

### FVC Dataset Structure

The code supports two dataset structures:

**Option 1: Subdirectory structure (recommended)**
```
fvc_data/
    subject_001/
        impression_1.tif
        impression_2.tif
        ...
    subject_002/
        impression_1.tif
        ...
```

**Option 2: Flat structure with naming convention**
```
fvc_data/
    001_1.tif  (subject 001, impression 1)
    001_2.tif  (subject 001, impression 2)
    002_1.tif  (subject 002, impression 1)
    ...
```

## Training

### Basic Usage

```bash
python train_fvc_transfer.py \
    --data /path/to/fvc/dataset \
    --pretrained /path/to/imagenet_checkpoint.pth.tar \
    --num_classes 100 \
    --unfreeze_layers 1 \
    --batch_size 32 \
    --epochs 100 \
    --learning_rate 0.0001
```

### Transfer Learning Options

Control which layers to train with `--unfreeze_layers`:

- `0`: Freeze all layers, only train FC layer (fastest, least flexible)
- `1`: Unfreeze layer4 + FC (recommended starting point)
- `2`: Unfreeze layer3 + layer4 + FC
- `3`: Unfreeze layer2 + layer3 + layer4 + FC
- `4`: Unfreeze all layers (slowest, most flexible)

### Full Training Example

```bash
# Stage 1: Train only FC layer (fast fine-tuning)
python train_fvc_transfer.py \
    --data ./fvc2002_db1 \
    --pretrained ./checkpoints/birealnet18_imagenet.pth.tar \
    --num_classes 100 \
    --unfreeze_layers 0 \
    --epochs 20 \
    --learning_rate 0.001 \
    --batch_size 64 \
    --save ./models_stage1

# Stage 2: Fine-tune layer4 + FC
python train_fvc_transfer.py \
    --data ./fvc2002_db1 \
    --pretrained ./checkpoints/birealnet18_imagenet.pth.tar \
    --resume ./models_stage1/model_best.pth.tar \
    --num_classes 100 \
    --unfreeze_layers 1 \
    --epochs 50 \
    --learning_rate 0.0001 \
    --batch_size 32 \
    --save ./models_stage2

# Stage 3 (optional): Fine-tune more layers if needed
python train_fvc_transfer.py \
    --data ./fvc2002_db1 \
    --pretrained ./checkpoints/birealnet18_imagenet.pth.tar \
    --resume ./models_stage2/model_best.pth.tar \
    --num_classes 100 \
    --unfreeze_layers 2 \
    --epochs 30 \
    --learning_rate 0.00001 \
    --batch_size 32 \
    --save ./models_stage3
```

### All Command-Line Arguments

```
--data              Path to FVC dataset root directory
--pretrained        Path to ImageNet pretrained checkpoint (.pth.tar)
--num_classes       Number of unique subjects/identities in dataset
--unfreeze_layers   Number of layers to unfreeze (0-4)
--batch_size        Batch size (default: 32)
--epochs            Number of training epochs (default: 100)
--learning_rate     Initial learning rate (default: 0.0001)
--weight_decay      Weight decay for regularization (default: 1e-4)
--save              Directory to save checkpoints (default: ./models_fvc)
--resume            Path to checkpoint to resume training from
--model             Model architecture: birealnet18 or birealnet34
--image_size        Input image size (default: 224)
--workers           Number of data loading workers (default: 4)
--train_split       Train/val split ratio (default: 0.8)
--seed              Random seed (default: 42)
```

## Feature Extraction

After training, extract 512-dimensional feature embeddings:

### Extract from Directory

```bash
python extract_features.py \
    --model_path ./models_fvc/model_best.pth.tar \
    --data /path/to/fingerprints \
    --output features.npy \
    --model birealnet18 \
    --num_classes 100 \
    --batch_size 32
```

### Extract from Single Image

```bash
python extract_features.py \
    --model_path ./models_fvc/model_best.pth.tar \
    --data /path/to/single_fingerprint.tif \
    --output single_feature.npy \
    --model birealnet18 \
    --num_classes 100
```

### Use Features for Matching

```python
import numpy as np

# Load features
data = np.load('features.npy', allow_pickle=True).item()
features = data['features']  # Shape: (N, 512)
image_paths = data['image_paths']

# Compare two fingerprints using cosine similarity
def cosine_similarity(f1, f2):
    return np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))

# Example: Compare first two fingerprints
similarity = cosine_similarity(features[0], features[1])
print(f"Similarity: {similarity:.4f}")

# Threshold for matching (tune based on validation set)
threshold = 0.7
if similarity > threshold:
    print("Match!")
else:
    print("No match")
```

## Important Notes

### 1. Grayscale vs RGB Conversion

Fingerprint images are typically grayscale, but BiRealNet expects 3-channel RGB input. The code automatically converts grayscale images to RGB by replicating the channel 3 times.

### 2. ImageNet Normalization

The code uses ImageNet normalization values by default:
- Mean: [0.485, 0.456, 0.406]
- Std: [0.229, 0.224, 0.225]

For better results, you may want to compute dataset-specific statistics:

```python
# Compute mean and std for your fingerprint dataset
from torchvision import transforms
import torch

# Load your dataset
dataset = FVCDataset(data_root, transform=transforms.ToTensor())
loader = DataLoader(dataset, batch_size=64)

mean = torch.zeros(3)
std = torch.zeros(3)
for images, _ in loader:
    for i in range(3):
        mean[i] += images[:, i, :, :].mean()
        std[i] += images[:, i, :, :].std()

mean /= len(loader)
std /= len(loader)

print(f"Mean: {mean}")
print(f"Std: {std}")
```

### 3. Binary Network Considerations

BiRealNet is a binarized network designed for efficiency. While great for deployment, it may have reduced capacity compared to full-precision networks. If accuracy is critical, consider:

- Using BiRealNet34 (larger model)
- Unfreezing more layers (--unfreeze_layers 3 or 4)
- Comparing with a standard ResNet baseline

### 4. Data Augmentation

The training script applies fingerprint-specific augmentations:
- Random rotation (±15°)
- Random horizontal flip
- Color jitter (brightness/contrast)
- Random crops

These help with robustness to varying capture conditions.

### 5. Checkpoint Format

The code handles checkpoints in two formats:
- With 'state_dict' key: `checkpoint['state_dict']`
- Direct state dict: `checkpoint` itself

It also handles DataParallel 'module.' prefix automatically.

## Expected Performance

Performance depends on:
- Dataset quality and size
- Number of subjects
- Image quality variation
- Transfer learning configuration

Typical results on FVC datasets:
- **Subject identification (closed-set)**: 85-95% accuracy
- **Verification (EER)**: 1-5% depending on dataset
- **Feature extraction time**: ~10ms per image on GPU

## Troubleshooting

### CUDA Out of Memory

Reduce `--batch_size` to 16 or 8.

### Poor Accuracy

1. Try unfreezing more layers: `--unfreeze_layers 2`
2. Increase training epochs: `--epochs 150`
3. Lower learning rate: `--learning_rate 0.00005`
4. Check dataset structure and labels

### Model Not Loading

Ensure:
- Checkpoint path is correct
- `--num_classes` matches training
- `--model` matches (birealnet18 or birealnet34)

## Citation

If you use this code, please cite the BiRealNet paper:

```bibtex
@inproceedings{liu2018bi,
  title={Bi-real net: Enhancing the performance of 1-bit cnns with improved representational capability and fault tolerance},
  author={Liu, Zechun and Wu, Baoyuan and Luo, Wenhan and Yang, Xin and Liu, Wei and Cheng, Kwang-Ting},
  booktitle={Proceedings of the European conference on computer vision (ECCV)},
  pages={722--737},
  year={2018}
}
```

## License

Please check the original BiRealNet repository for license information.
