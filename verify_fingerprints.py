"""
Fingerprint Verification Script

This script performs 1:1 fingerprint matching using extracted features.
It can be used for:
- Genuine vs impostor testing
- Computing EER (Equal Error Rate)
- Testing matching thresholds
"""

import os
import numpy as np
import argparse
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser("Fingerprint Verification")
parser.add_argument('--features', type=str, required=True, help='path to features.npy file')
parser.add_argument('--output_dir', type=str, default='./verification_results', help='directory to save results')
parser.add_argument('--metric', type=str, default='cosine', choices=['cosine', 'euclidean'], 
                    help='distance metric')
args = parser.parse_args()


def compute_similarity(features1, features2, metric='cosine'):
    """
    Compute similarity/distance between feature vectors
    
    Returns:
        For cosine: similarity (higher = more similar)
        For euclidean: distance (lower = more similar)
    """
    if metric == 'cosine':
        # Cosine similarity
        dot_product = np.sum(features1 * features2, axis=1)
        norm1 = np.linalg.norm(features1, axis=1)
        norm2 = np.linalg.norm(features2, axis=1)
        similarity = dot_product / (norm1 * norm2 + 1e-8)
        return similarity
    elif metric == 'euclidean':
        # Euclidean distance
        distance = np.linalg.norm(features1 - features2, axis=1)
        return -distance  # Negative so higher = more similar
    else:
        raise ValueError(f"Unknown metric: {metric}")


def parse_subject_id(filename):
    """
    Extract subject ID from filename
    Examples:
        - '001_1.tif' -> 1
        - 's001_2.tif' -> 1
        - 'subject_001/impression_1.tif' -> 1
    """
    basename = os.path.basename(filename)
    # Remove extension
    basename = os.path.splitext(basename)[0]
    
    # Try to extract subject number
    parts = basename.split('_')
    if len(parts) >= 1:
        subject_str = parts[0].lstrip('s').lstrip('subject')
        try:
            return int(subject_str)
        except ValueError:
            pass
    
    # Try to extract from directory name
    dirname = os.path.basename(os.path.dirname(filename))
    if dirname:
        subject_str = dirname.lstrip('s').lstrip('subject').split('_')[0]
        try:
            return int(subject_str)
        except ValueError:
            pass
    
    return None


def generate_pairs(features, image_paths):
    """
    Generate genuine and impostor pairs
    
    Returns:
        genuine_scores: List of similarity scores for genuine pairs (same subject)
        impostor_scores: List of similarity scores for impostor pairs (different subjects)
    """
    
    # Parse subject IDs from filenames
    subject_ids = []
    valid_indices = []
    for i, path in enumerate(image_paths):
        subject_id = parse_subject_id(path)
        if subject_id is not None:
            subject_ids.append(subject_id)
            valid_indices.append(i)
    
    # Filter features to valid indices
    features = features[valid_indices]
    subject_ids = np.array(subject_ids)
    image_paths = [image_paths[i] for i in valid_indices]
    
    print(f"Found {len(set(subject_ids))} unique subjects")
    print(f"Valid samples: {len(subject_ids)}")
    
    # Organize by subject
    subject_to_indices = {}
    for i, subject_id in enumerate(subject_ids):
        if subject_id not in subject_to_indices:
            subject_to_indices[subject_id] = []
        subject_to_indices[subject_id].append(i)
    
    genuine_scores = []
    impostor_scores = []
    
    # Generate genuine pairs (same subject, different impressions)
    print("Generating genuine pairs...")
    for subject_id, indices in subject_to_indices.items():
        if len(indices) < 2:
            continue
        
        # All pairs within this subject
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                idx1, idx2 = indices[i], indices[j]
                score = compute_similarity(
                    features[idx1:idx1+1], 
                    features[idx2:idx2+1], 
                    args.metric
                )[0]
                genuine_scores.append(score)
    
    print(f"Generated {len(genuine_scores)} genuine pairs")
    
    # Generate impostor pairs (different subjects)
    print("Generating impostor pairs...")
    subjects = list(subject_to_indices.keys())
    num_impostor_pairs = min(len(genuine_scores) * 3, 10000)  # Limit to avoid too many pairs
    
    for _ in range(num_impostor_pairs):
        # Random pair from different subjects
        subj1, subj2 = np.random.choice(subjects, 2, replace=False)
        idx1 = np.random.choice(subject_to_indices[subj1])
        idx2 = np.random.choice(subject_to_indices[subj2])
        
        score = compute_similarity(
            features[idx1:idx1+1], 
            features[idx2:idx2+1], 
            args.metric
        )[0]
        impostor_scores.append(score)
    
    print(f"Generated {len(impostor_scores)} impostor pairs")
    
    return np.array(genuine_scores), np.array(impostor_scores)


def compute_eer(genuine_scores, impostor_scores):
    """
    Compute Equal Error Rate (EER)
    
    Returns:
        eer: Equal Error Rate (percentage)
        eer_threshold: Threshold at EER
    """
    
    # Create labels (1 for genuine, 0 for impostor)
    y_true = np.concatenate([
        np.ones(len(genuine_scores)),
        np.zeros(len(impostor_scores))
    ])
    
    # Combine scores
    y_scores = np.concatenate([genuine_scores, impostor_scores])
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    
    # Find EER (where FPR = 1 - TPR, i.e., FPR = FNR)
    fnr = 1 - tpr
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    
    return eer * 100, eer_threshold, fpr, tpr, thresholds


def plot_score_distributions(genuine_scores, impostor_scores, output_dir):
    """Plot genuine and impostor score distributions"""
    
    plt.figure(figsize=(10, 6))
    plt.hist(genuine_scores, bins=50, alpha=0.5, label='Genuine', color='green', density=True)
    plt.hist(impostor_scores, bins=50, alpha=0.5, label='Impostor', color='red', density=True)
    plt.xlabel('Similarity Score')
    plt.ylabel('Density')
    plt.title('Score Distributions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'score_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved score distributions to {output_path}")


def plot_roc_curve(fpr, tpr, eer, output_dir):
    """Plot ROC curve"""
    
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    
    # Mark EER point
    eer_fpr = eer / 100
    eer_tpr = 1 - eer_fpr
    plt.plot(eer_fpr, eer_tpr, 'ro', markersize=10, 
             label=f'EER = {eer:.2f}%')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    output_path = os.path.join(output_dir, 'roc_curve.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved ROC curve to {output_path}")


def find_threshold_for_far(fpr, tpr, thresholds, target_far):
    """Find threshold for a target False Accept Rate (FAR)"""
    
    # Find index where FAR is closest to target
    idx = np.nanargmin(np.abs(fpr - target_far))
    
    threshold = thresholds[idx]
    actual_far = fpr[idx]
    frr = 1 - tpr[idx]  # False Reject Rate
    
    return threshold, actual_far, frr


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load features
    print(f"Loading features from {args.features}")
    data = np.load(args.features, allow_pickle=True).item()
    features = data['features']
    image_paths = data['image_paths']
    
    print(f"Loaded {len(features)} feature vectors of dimension {features.shape[1]}")
    
    # Generate genuine and impostor pairs
    genuine_scores, impostor_scores = generate_pairs(features, image_paths)
    
    # Compute statistics
    print("\n" + "="*60)
    print("VERIFICATION RESULTS")
    print("="*60)
    
    print(f"\nGenuine pairs:")
    print(f"  Count: {len(genuine_scores)}")
    print(f"  Mean score: {genuine_scores.mean():.4f}")
    print(f"  Std: {genuine_scores.std():.4f}")
    print(f"  Min: {genuine_scores.min():.4f}")
    print(f"  Max: {genuine_scores.max():.4f}")
    
    print(f"\nImpostor pairs:")
    print(f"  Count: {len(impostor_scores)}")
    print(f"  Mean score: {impostor_scores.mean():.4f}")
    print(f"  Std: {impostor_scores.std():.4f}")
    print(f"  Min: {impostor_scores.min():.4f}")
    print(f"  Max: {impostor_scores.max():.4f}")
    
    # Compute EER
    eer, eer_threshold, fpr, tpr, thresholds = compute_eer(genuine_scores, impostor_scores)
    
    print(f"\n{'='*60}")
    print(f"Equal Error Rate (EER): {eer:.2f}%")
    print(f"EER Threshold: {eer_threshold:.4f}")
    print(f"{'='*60}")
    
    # Find operating points for different FARs
    print(f"\nOperating Points:")
    print(f"{'FAR':<10} {'Threshold':<12} {'FRR':<10} {'Accuracy'}")
    print("-" * 50)
    
    for target_far in [0.1, 0.01, 0.001, 0.0001]:
        threshold, actual_far, frr = find_threshold_for_far(fpr, tpr, thresholds, target_far)
        accuracy = (1 - (actual_far + frr) / 2) * 100
        print(f"{target_far:<10.4f} {threshold:<12.4f} {frr:<10.4f} {accuracy:.2f}%")
    
    # Plot distributions
    plot_score_distributions(genuine_scores, impostor_scores, args.output_dir)
    
    # Plot ROC curve
    plot_roc_curve(fpr, tpr, eer, args.output_dir)
    
    # Save results to file
    results_file = os.path.join(args.output_dir, 'verification_results.txt')
    with open(results_file, 'w') as f:
        f.write("Fingerprint Verification Results\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Metric: {args.metric}\n")
        f.write(f"Features file: {args.features}\n\n")
        
        f.write(f"Genuine pairs: {len(genuine_scores)}\n")
        f.write(f"Impostor pairs: {len(impostor_scores)}\n\n")
        
        f.write(f"Equal Error Rate (EER): {eer:.2f}%\n")
        f.write(f"EER Threshold: {eer_threshold:.4f}\n\n")
        
        f.write("Operating Points:\n")
        f.write(f"{'FAR':<10} {'Threshold':<12} {'FRR':<10} {'Accuracy'}\n")
        f.write("-" * 50 + "\n")
        for target_far in [0.1, 0.01, 0.001, 0.0001]:
            threshold, actual_far, frr = find_threshold_for_far(fpr, tpr, thresholds, target_far)
            accuracy = (1 - (actual_far + frr) / 2) * 100
            f.write(f"{target_far:<10.4f} {threshold:<12.4f} {frr:<10.4f} {accuracy:.2f}%\n")
    
    print(f"\nResults saved to {results_file}")
    print(f"All outputs saved to {args.output_dir}")


if __name__ == '__main__':
    main()
