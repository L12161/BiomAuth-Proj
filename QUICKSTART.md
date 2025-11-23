# Quick Start Guide - BiRealNet Fingerprint Recognition

## Overview

This package provides everything you need to train BiRealNet on fingerprint datasets using transfer learning from ImageNet weights. The key advantage is leveraging pre-trained representations while adapting to fingerprint-specific features.

## Key Problems Identified in Original Code

1. **ImageNet-specific**: Original code was hardcoded for 1000 ImageNet classes
2. **Wrong data format**: Expects ImageFolder structure, not FVC format
3. **No transfer learning controls**: Couldn't selectively unfreeze layers
4. **RGB vs Grayscale**: No handling for grayscale fingerprints
5. **Knowledge distillation**: Used teacher model which may not help for fingerprints

## Quick Start

### Step 1: Organize Your Data

Organize your FVC dataset like this:
```
fvc_data/
    subject_001/
        impression_1.tif
        impression_2.tif
    subject_002/
        impression_1.tif
        impression_2.tif
```

### Step 2: Train with Transfer Learning

```bash
# Basic training (recommended for most cases)
python train_fvc_transfer.py \
    --data ./fvc_data \
    --pretrained ./imagenet_checkpoint.pth.tar \
    --num_classes 100 \
    --unfreeze_layers 1 \
    --epochs 100 \
    --batch_size 32
```

### Step 3: Extract Features

```bash
python extract_features.py \
    --model_path ./models_fvc/model_best.pth.tar \
    --data ./test_fingerprints \
    --output features.npy \
    --num_classes 100
```

### Step 4: Verify Performance

```bash
python verify_fingerprints.py \
    --features features.npy \
    --output_dir ./results
```

## Transfer Learning Strategy

### Conservative Approach (Faster, Less Data Needed)
```bash
# Stage 1: Train only FC layer (20 epochs)
--unfreeze_layers 0 --epochs 20 --learning_rate 0.001

# Stage 2: Fine-tune layer4 (50 epochs)
--unfreeze_layers 1 --epochs 50 --learning_rate 0.0001
```

### Aggressive Approach (More Flexible, Needs More Data)
```bash
# Stage 1: Train FC + layer4 (30 epochs)
--unfreeze_layers 1 --epochs 30 --learning_rate 0.0001

# Stage 2: Fine-tune layer3+4 (50 epochs)
--unfreeze_layers 2 --epochs 50 --learning_rate 0.00005

# Stage 3: Fine-tune all (30 epochs) - if you have lots of data
--unfreeze_layers 4 --epochs 30 --learning_rate 0.00001
```

## Important Parameters

### Learning Rate Guidelines
- `unfreeze_layers=0`: LR = 0.001 (only FC, can be higher)
- `unfreeze_layers=1`: LR = 0.0001 (layer4 + FC)
- `unfreeze_layers=2+`: LR = 0.00005 or lower (more layers)

### Batch Size Guidelines
- Small dataset (<1000 samples): batch_size = 16-32
- Medium dataset (1000-5000 samples): batch_size = 32-64
- Large dataset (>5000 samples): batch_size = 64-128

### Number of Classes
Count unique subjects/identities in your dataset. For example:
- FVC2002 DB1: 100 subjects
- FVC2004 DB1: 100 subjects
- Custom dataset: Count your unique subject folders

## Expected Results

With proper training, you should see:
- **Training accuracy**: 90-98% (depending on dataset difficulty)
- **Validation accuracy**: 85-95%
- **Verification EER**: 1-5% (lower is better)
- **Feature extraction**: ~512D embeddings

## Troubleshooting

### Problem: Low accuracy (<70%)
**Solutions:**
1. Unfreeze more layers: `--unfreeze_layers 2`
2. Train longer: `--epochs 150`
3. Check dataset labels are correct
4. Try lower learning rate: `--learning_rate 0.00005`

### Problem: CUDA Out of Memory
**Solutions:**
1. Reduce batch size: `--batch_size 16` or `--batch_size 8`
2. Use smaller images: `--image_size 192` (default is 224)

### Problem: Overfitting (train acc >> val acc)
**Solutions:**
1. Increase weight decay: `--weight_decay 0.0001` or `0.001`
2. Freeze more layers: `--unfreeze_layers 0` or `1`
3. Use more data augmentation (already included)
4. Train for fewer epochs

### Problem: Features don't separate well
**Solutions:**
1. Train longer with `--unfreeze_layers 2`
2. Check if normalization is appropriate for your data
3. Consider computing dataset-specific mean/std
4. Try both cosine and euclidean metrics

## Files Included

1. **train_fvc_transfer.py** - Main training script
2. **extract_features.py** - Feature extraction for matching
3. **verify_fingerprints.py** - Verification evaluation (EER, ROC)
4. **birealnet.py** - Model architecture
5. **README.md** - Comprehensive documentation

## Next Steps

1. **Optimize threshold**: Use `verify_fingerprints.py` to find best threshold
2. **Ensemble models**: Train multiple models with different seeds
3. **Test on different databases**: Evaluate cross-database performance
4. **Compare with baseline**: Try training standard ResNet18 for comparison

## Common Command Examples

### Training on small dataset
```bash
python train_fvc_transfer.py \
    --data ./fvc2002_db1 \
    --pretrained ./weights.pth.tar \
    --num_classes 100 \
    --unfreeze_layers 0 \
    --epochs 50 \
    --batch_size 64 \
    --learning_rate 0.001
```

### Training on large dataset
```bash
python train_fvc_transfer.py \
    --data ./large_fvc_dataset \
    --pretrained ./weights.pth.tar \
    --num_classes 500 \
    --unfreeze_layers 2 \
    --epochs 100 \
    --batch_size 32 \
    --learning_rate 0.00005
```

### Resume training
```bash
python train_fvc_transfer.py \
    --data ./fvc_data \
    --pretrained ./weights.pth.tar \
    --resume ./models_fvc/checkpoint.pth.tar \
    --num_classes 100 \
    --unfreeze_layers 1 \
    --epochs 100
```

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review troubleshooting section above
3. Verify your dataset structure matches expected format
4. Check PyTorch and CUDA compatibility

Good luck with your fingerprint recognition project!
