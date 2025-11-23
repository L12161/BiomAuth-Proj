"""
Feature Extraction Script for Fingerprint Matching

This script loads a trained BiRealNet model and extracts feature embeddings
from fingerprint images. These embeddings can be used for:
- Fingerprint verification (1:1 matching)
- Fingerprint identification (1:N matching)
- Building a fingerprint database
"""

import os
import torch
import torch.nn as nn
import numpy as np
import argparse
from torchvision import transforms
from PIL import Image
import glob
from birealnet import birealnet18, birealnet34


parser = argparse.ArgumentParser("Extract Features from Fingerprints")
parser.add_argument('--model_path', type=str, required=True, help='path to trained model checkpoint')
parser.add_argument('--data', type=str, required=True, help='path to fingerprint images or directory')
parser.add_argument('--output', type=str, default='features.npy', help='output file for features')
parser.add_argument('--model', type=str, default='birealnet18', choices=['birealnet18', 'birealnet34'])
parser.add_argument('--num_classes', type=int, default=100, help='number of classes in trained model')
parser.add_argument('--image_size', type=int, default=224, help='input image size')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for extraction')
args = parser.parse_args()


class FeatureExtractor(nn.Module):
    """Wrapper to extract features before the FC layer"""
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.conv1 = model.conv1
        self.bn1 = model.bn1
        self.maxpool = model.maxpool
        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
        self.avgpool = model.avgpool
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 512-dimensional features
        
        return x


def load_model(model_path, model_type='birealnet18', num_classes=100):
    """Load trained model and wrap with feature extractor"""
    
    # Create model
    if model_type == 'birealnet18':
        model = birealnet18(num_classes=num_classes)
    else:
        model = birealnet34(num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    model.load_state_dict(new_state_dict)
    
    # Wrap with feature extractor
    feature_extractor = FeatureExtractor(model)
    feature_extractor.eval()
    
    return feature_extractor


def extract_features_from_image(image_path, model, transform):
    """Extract features from a single image"""
    
    # Load and preprocess image
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
        model = model.cuda()
    
    # Extract features
    with torch.no_grad():
        features = model(image_tensor)
        features = features.cpu().numpy()
    
    return features


def extract_features_batch(image_paths, model, transform, batch_size=32):
    """Extract features from multiple images in batches"""
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    all_features = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        
        for img_path in batch_paths:
            try:
                image = Image.open(img_path)
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if len(batch_images) == 0:
            continue
        
        # Stack into batch
        batch_tensor = torch.stack(batch_images)
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()
        
        # Extract features
        with torch.no_grad():
            features = model(batch_tensor)
            features = features.cpu().numpy()
            all_features.append(features)
        
        print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")
    
    # Concatenate all features
    all_features = np.concatenate(all_features, axis=0)
    
    return all_features


def compute_similarity(features1, features2, metric='cosine'):
    """
    Compute similarity between two feature vectors
    
    Args:
        features1: First feature vector (1D array)
        features2: Second feature vector (1D array)
        metric: 'cosine' or 'euclidean'
    
    Returns:
        Similarity score (higher = more similar for cosine, lower = more similar for euclidean)
    """
    if metric == 'cosine':
        # Cosine similarity
        norm1 = np.linalg.norm(features1)
        norm2 = np.linalg.norm(features2)
        if norm1 == 0 or norm2 == 0:
            return 0
        similarity = np.dot(features1, features2) / (norm1 * norm2)
        return similarity
    elif metric == 'euclidean':
        # Euclidean distance
        distance = np.linalg.norm(features1 - features2)
        return distance
    else:
        raise ValueError(f"Unknown metric: {metric}")


def main():
    print("Loading model...")
    model = load_model(args.model_path, args.model, args.num_classes)
    
    # Define transforms (same as validation transforms)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Check if input is a directory or single file
    if os.path.isdir(args.data):
        print(f"Extracting features from directory: {args.data}")
        
        # Get all image files
        image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']:
            image_paths.extend(glob.glob(os.path.join(args.data, ext)))
            # Also check subdirectories
            image_paths.extend(glob.glob(os.path.join(args.data, '**', ext), recursive=True))
        
        image_paths = sorted(list(set(image_paths)))  # Remove duplicates
        print(f"Found {len(image_paths)} images")
        
        # Extract features in batches
        features = extract_features_batch(image_paths, model, transform, args.batch_size)
        
        # Save features with corresponding image paths
        output_data = {
            'features': features,
            'image_paths': image_paths
        }
        np.save(args.output, output_data)
        print(f"Saved features to {args.output}")
        print(f"Feature shape: {features.shape}")
        
    elif os.path.isfile(args.data):
        print(f"Extracting features from single image: {args.data}")
        
        features = extract_features_from_image(args.data, model, transform)
        
        # Save features
        np.save(args.output, features)
        print(f"Saved features to {args.output}")
        print(f"Feature shape: {features.shape}")
    
    else:
        print(f"Error: {args.data} is not a valid file or directory")


if __name__ == '__main__':
    main()
