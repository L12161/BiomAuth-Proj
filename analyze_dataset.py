"""
Dataset Analysis Utility

This script analyzes your FVC dataset and computes:
- Dataset statistics (number of subjects, impressions per subject)
- Image dimensions and formats
- Dataset-specific normalization parameters (mean, std)
"""

import os
import numpy as np
import argparse
from PIL import Image
import glob
from collections import defaultdict


parser = argparse.ArgumentParser("FVC Dataset Analysis")
parser.add_argument('--data', type=str, required=True, help='path to FVC dataset root')
parser.add_argument('--compute_normalization', action='store_true', 
                    help='compute mean and std for normalization (slower)')
args = parser.parse_args()


def parse_subject_id(filename):
    """Extract subject ID from filename"""
    basename = os.path.basename(filename)
    basename = os.path.splitext(basename)[0]
    
    parts = basename.split('_')
    if len(parts) >= 1:
        subject_str = parts[0].lstrip('s').lstrip('subject')
        try:
            return int(subject_str)
        except ValueError:
            pass
    
    dirname = os.path.basename(os.path.dirname(filename))
    if dirname:
        subject_str = dirname.lstrip('s').lstrip('subject').split('_')[0]
        try:
            return int(subject_str)
        except ValueError:
            pass
    
    return None


def analyze_dataset(data_root):
    """Analyze dataset structure and statistics"""
    
    print("="*70)
    print("DATASET ANALYSIS")
    print("="*70)
    print(f"\nDataset root: {data_root}\n")
    
    # Check if data is in subdirectories or flat
    subdirs = [d for d in os.listdir(data_root) 
               if os.path.isdir(os.path.join(data_root, d))]
    
    subject_to_images = defaultdict(list)
    all_images = []
    image_formats = defaultdict(int)
    image_modes = defaultdict(int)
    image_dimensions = []
    
    if len(subdirs) > 0:
        print(f"Dataset structure: SUBDIRECTORY (found {len(subdirs)} directories)")
        print("\nScanning subdirectories...")
        
        for subject_dir in subdirs:
            subject_path = os.path.join(data_root, subject_dir)
            image_files = glob.glob(os.path.join(subject_path, '*.*'))
            image_files = [f for f in image_files 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
            
            subject_id = parse_subject_id(subject_dir)
            if subject_id is not None:
                subject_to_images[subject_id].extend(image_files)
                all_images.extend(image_files)
    else:
        print("Dataset structure: FLAT")
        print("\nScanning files...")
        
        all_files = [f for f in os.listdir(data_root) 
                    if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
        
        for img_file in all_files:
            img_path = os.path.join(data_root, img_file)
            subject_id = parse_subject_id(img_file)
            
            if subject_id is not None:
                subject_to_images[subject_id].append(img_path)
                all_images.append(img_path)
    
    # Analyze images
    print(f"\nAnalyzing {len(all_images)} images...")
    
    for img_path in all_images:
        try:
            img = Image.open(img_path)
            
            # Track format
            ext = os.path.splitext(img_path)[1].lower()
            image_formats[ext] += 1
            
            # Track mode (RGB, L, etc.)
            image_modes[img.mode] += 1
            
            # Track dimensions
            image_dimensions.append(img.size)
            
        except Exception as e:
            print(f"Warning: Could not open {img_path}: {e}")
    
    # Print statistics
    print("\n" + "-"*70)
    print("DATASET STATISTICS")
    print("-"*70)
    
    print(f"\nTotal images: {len(all_images)}")
    print(f"Total subjects: {len(subject_to_images)}")
    
    if len(subject_to_images) > 0:
        impressions_per_subject = [len(imgs) for imgs in subject_to_images.values()]
        print(f"\nImpressions per subject:")
        print(f"  Min: {min(impressions_per_subject)}")
        print(f"  Max: {max(impressions_per_subject)}")
        print(f"  Mean: {np.mean(impressions_per_subject):.2f}")
        print(f"  Median: {np.median(impressions_per_subject):.0f}")
        
        # Show distribution
        impression_counts = defaultdict(int)
        for count in impressions_per_subject:
            impression_counts[count] += 1
        
        print(f"\nDistribution:")
        for count in sorted(impression_counts.keys()):
            print(f"  {count} impressions: {impression_counts[count]} subjects")
    
    print(f"\nImage formats:")
    for fmt, count in sorted(image_formats.items()):
        print(f"  {fmt}: {count} images ({100*count/len(all_images):.1f}%)")
    
    print(f"\nImage modes:")
    for mode, count in sorted(image_modes.items()):
        print(f"  {mode}: {count} images ({100*count/len(all_images):.1f}%)")
        if mode == 'L':
            print(f"       (Grayscale - will be converted to RGB)")
        elif mode == 'RGB':
            print(f"       (RGB - native format)")
    
    if len(image_dimensions) > 0:
        widths = [w for w, h in image_dimensions]
        heights = [h for w, h in image_dimensions]
        
        print(f"\nImage dimensions:")
        print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={np.mean(widths):.0f}")
        print(f"  Height: min={min(heights)}, max={max(heights)}, mean={np.mean(heights):.0f}")
        
        # Show common dimensions
        dimension_counts = defaultdict(int)
        for dim in image_dimensions:
            dimension_counts[dim] += 1
        
        print(f"\nMost common dimensions:")
        for dim, count in sorted(dimension_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"  {dim[0]}x{dim[1]}: {count} images ({100*count/len(image_dimensions):.1f}%)")
    
    # Training recommendations
    print("\n" + "-"*70)
    print("TRAINING RECOMMENDATIONS")
    print("-"*70)
    
    num_subjects = len(subject_to_images)
    num_images = len(all_images)
    
    print(f"\n--num_classes {num_subjects}")
    
    if num_images < 500:
        print("\nDataset size: SMALL")
        print("Recommended settings:")
        print("  --unfreeze_layers 0")
        print("  --epochs 50")
        print("  --batch_size 16")
        print("  --learning_rate 0.001")
    elif num_images < 2000:
        print("\nDataset size: MEDIUM")
        print("Recommended settings:")
        print("  --unfreeze_layers 1")
        print("  --epochs 100")
        print("  --batch_size 32")
        print("  --learning_rate 0.0001")
    else:
        print("\nDataset size: LARGE")
        print("Recommended settings:")
        print("  --unfreeze_layers 2")
        print("  --epochs 100")
        print("  --batch_size 64")
        print("  --learning_rate 0.00005")
    
    # Compute normalization if requested
    if args.compute_normalization:
        print("\n" + "-"*70)
        print("COMPUTING NORMALIZATION PARAMETERS")
        print("-"*70)
        print("\nThis may take a while...")
        
        means = []
        stds = []
        
        for i, img_path in enumerate(all_images):
            try:
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                img_array = np.array(img).astype(np.float32) / 255.0
                
                # Compute mean and std for each channel
                means.append(img_array.mean(axis=(0, 1)))
                stds.append(img_array.std(axis=(0, 1)))
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i+1}/{len(all_images)} images...")
                    
            except Exception as e:
                print(f"Warning: Could not process {img_path}: {e}")
        
        if len(means) > 0:
            mean = np.mean(means, axis=0)
            std = np.mean(stds, axis=0)
            
            print(f"\nDataset-specific normalization parameters:")
            print(f"  Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
            print(f"  Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")
            
            print(f"\nPyTorch code:")
            print(f"  normalize = transforms.Normalize(")
            print(f"      mean=[{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}],")
            print(f"      std=[{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}])")
            
            # Compare with ImageNet
            imagenet_mean = np.array([0.485, 0.456, 0.406])
            imagenet_std = np.array([0.229, 0.224, 0.225])
            
            mean_diff = np.abs(mean - imagenet_mean).mean()
            std_diff = np.abs(std - imagenet_std).mean()
            
            print(f"\nComparison with ImageNet normalization:")
            print(f"  Mean difference: {mean_diff:.4f}")
            print(f"  Std difference:  {std_diff:.4f}")
            
            if mean_diff < 0.05 and std_diff < 0.05:
                print("  → ImageNet normalization is likely fine")
            else:
                print("  → Consider using dataset-specific normalization above")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)


if __name__ == '__main__':
    analyze_dataset(args.data)
