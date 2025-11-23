import os
import sys
import numpy as np
import time
import torch
import logging
import argparse
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import glob
from birealnet import birealnet18, birealnet34

parser = argparse.ArgumentParser("BiRealNet Transfer Learning for FVC Fingerprints")
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight decay')
parser.add_argument('--save', type=str, default='./models_fvc', help='path for saving trained models')
parser.add_argument('--data', type=str, required=True, help='path to FVC dataset root')
parser.add_argument('--pretrained', type=str, required=True, help='path to ImageNet pretrained checkpoint')
parser.add_argument('--num_classes', type=int, default=100, help='number of identities in dataset')
parser.add_argument('--unfreeze_layers', type=int, default=1, 
                    help='layers to unfreeze: 0=only FC, 1=layer4+FC, 2=layer3+layer4+FC, 3=layer2+3+4+FC, 4=all')
parser.add_argument('--model', type=str, default='birealnet18', choices=['birealnet18', 'birealnet34'])
parser.add_argument('--image_size', type=int, default=224, help='input image size')
parser.add_argument('--resume', type=str, default='', help='path to checkpoint to resume from')
parser.add_argument('--workers', default=4, type=int, help='number of data loading workers')
parser.add_argument('--train_split', type=float, default=0.8, help='train/val split ratio')
parser.add_argument('--seed', type=int, default=42, help='random seed')
args = parser.parse_args()

# Set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

if not os.path.exists('log'):
    os.mkdir('log')

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join('log/fvc_transfer_log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


class FVCDataset(Dataset):
    """
    FVC Dataset Loader
    Expected structure:
    data_root/
        001_1.tif (subject 001, impression 1)
        001_2.tif (subject 001, impression 2)
        ...
        002_1.tif (subject 002, impression 1)
        ...
    OR:
    data_root/
        subject_001/
            impression_1.tif
            impression_2.tif
        subject_002/
            ...
    """
    def __init__(self, data_root, transform=None, split='train', train_ratio=0.8, seed=42):
        self.data_root = data_root
        self.transform = transform
        self.samples = []
        
        # Check if data is in subdirectories or flat
        subdirs = [d for d in os.listdir(data_root) 
                   if os.path.isdir(os.path.join(data_root, d))]
        
        if len(subdirs) > 0:
            # Data organized in subdirectories
            subject_to_images = {}
            for subject_idx, subject_dir in enumerate(sorted(subdirs)):
                subject_path = os.path.join(data_root, subject_dir)
                image_files = sorted(glob.glob(os.path.join(subject_path, '*.*')))
                image_files = [f for f in image_files 
                              if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
                
                for img_path in image_files:
                    subject_to_images.setdefault(subject_idx, []).append(img_path)
        else:
            # Flat structure - parse filenames (e.g., 001_1.tif, 001_2.tif)
            subject_to_images = {}
            all_files = [f for f in os.listdir(data_root) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'))]
            
            for img_file in sorted(all_files):
                # Try to extract subject ID from filename
                # Common patterns: 001_1.tif, s001_1.tif, 1_1.tif
                basename = os.path.splitext(img_file)[0]
                parts = basename.split('_')
                if len(parts) >= 2:
                    subject_id = parts[0].lstrip('s')  # Remove 's' prefix if exists
                    try:
                        subject_idx = int(subject_id)
                        img_path = os.path.join(data_root, img_file)
                        subject_to_images.setdefault(subject_idx, []).append(img_path)
                    except ValueError:
                        logging.warning(f"Could not parse subject ID from {img_file}")
        
        # Split data into train/val
        np.random.seed(seed)
        subject_ids = sorted(subject_to_images.keys())
        
        for subject_idx in subject_ids:
            images = subject_to_images[subject_idx]
            if len(images) < 2:
                logging.warning(f"Subject {subject_idx} has only {len(images)} image(s), skipping")
                continue
            
            # Split images of this subject into train/val
            n_train = max(1, int(len(images) * train_ratio))
            np.random.shuffle(images)
            
            if split == 'train':
                selected_images = images[:n_train]
            else:  # val
                selected_images = images[n_train:]
            
            for img_path in selected_images:
                self.samples.append((img_path, subject_idx))
        
        logging.info(f'{split.upper()}: Loaded {len(self.samples)} samples from {len(subject_ids)} subjects')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            image = Image.open(img_path)
        except Exception as e:
            logging.error(f"Error loading image {img_path}: {e}")
            # Return a blank image
            image = Image.new('RGB', (224, 224))
        
        # Convert to RGB (BiRealNet expects 3 channels)
        if image.mode != 'RGB':
            if image.mode == 'L':  # Grayscale
                # Convert grayscale to RGB by replicating channel
                image = image.convert('RGB')
            else:
                image = image.convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def freeze_layers(model, num_layers_to_unfreeze):
    """
    Freeze layers for transfer learning
    num_layers_to_unfreeze: 
        0 = freeze all (only train FC)
        1 = unfreeze layer4 + FC
        2 = unfreeze layer3 + layer4 + FC
        3 = unfreeze layer2 + layer3 + layer4 + FC
        4 = unfreeze all layers
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Always train the new FC layer
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Unfreeze requested layers
    if num_layers_to_unfreeze >= 1:
        for param in model.layer4.parameters():
            param.requires_grad = True
    if num_layers_to_unfreeze >= 2:
        for param in model.layer3.parameters():
            param.requires_grad = True
    if num_layers_to_unfreeze >= 3:
        for param in model.layer2.parameters():
            param.requires_grad = True
    if num_layers_to_unfreeze >= 4:
        for param in model.layer1.parameters():
            param.requires_grad = True
        for param in model.conv1.parameters():
            param.requires_grad = True
        for param in model.bn1.parameters():
            param.requires_grad = True
    
    # Log trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f'Trainable parameters: {trainable_params:,}/{total_params:,} '
                f'({100*trainable_params/total_params:.2f}%)')
    
    # Log which layers are trainable
    logging.info(f"Unfrozen layers: ", end="")
    unfrozen = []
    if num_layers_to_unfreeze >= 4:
        unfrozen.append("conv1+bn1+layer1")
    if num_layers_to_unfreeze >= 3:
        unfrozen.append("layer2")
    if num_layers_to_unfreeze >= 2:
        unfrozen.append("layer3")
    if num_layers_to_unfreeze >= 1:
        unfrozen.append("layer4")
    unfrozen.append("fc")
    logging.info(" + ".join(unfrozen))


class AverageMeter:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train(train_loader, model, criterion, optimizer, epoch):
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.train()
    
    for i, (images, targets) in enumerate(train_loader):
        images = images.cuda()
        targets = targets.cuda()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Compute accuracy
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        
        # Update meters
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0:
            logging.info(f'Epoch: [{epoch}][{i}/{len(train_loader)}] '
                        f'Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}%')
    
    return losses.avg, top1.avg


def validate(val_loader, model, criterion):
    losses = AverageMeter()
    top1 = AverageMeter()
    
    model.eval()
    
    with torch.no_grad():
        for i, (images, targets) in enumerate(val_loader):
            images = images.cuda()
            targets = targets.cuda()
            
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
    
    logging.info(f'Validation: Loss: {losses.avg:.4f} Acc@1: {top1.avg:.2f}%')
    
    return losses.avg, top1.avg


def save_checkpoint(state, is_best, save_dir):
    filename = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save_dir, 'model_best.pth.tar')
        torch.save(state, best_filename)
        logging.info(f'Saved best model to {best_filename}')


def main():
    if not torch.cuda.is_available():
        logging.error('No CUDA device available!')
        sys.exit(1)
    
    start_t = time.time()
    cudnn.benchmark = True
    
    logging.info("Arguments:")
    for arg in vars(args):
        logging.info(f"  {arg}: {getattr(args, arg)}")
    
    # Data preprocessing for fingerprints
    # Note: Using ImageNet normalization as starting point, but you may want to 
    # compute statistics specific to your fingerprint dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    # Training transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((args.image_size + 32, args.image_size + 32)),  # Slightly larger
        transforms.RandomCrop(args.image_size),
        transforms.RandomRotation(15),  # Fingerprints can be rotated
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Varying image quality
        transforms.ToTensor(),
        normalize
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Create datasets
    logging.info(f'Loading FVC dataset from {args.data}')
    train_dataset = FVCDataset(
        args.data, 
        transform=train_transforms, 
        split='train',
        train_ratio=args.train_split,
        seed=args.seed
    )
    
    val_dataset = FVCDataset(
        args.data,
        transform=val_transforms,
        split='val',
        train_ratio=args.train_split,
        seed=args.seed
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )
    
    logging.info(f'Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}')
    
    # Create model
    logging.info(f'Creating model: {args.model}')
    if args.model == 'birealnet18':
        model = birealnet18(num_classes=1000)  # Start with ImageNet classes
    else:
        model = birealnet34(num_classes=1000)
    
    # Load ImageNet pretrained weights
    logging.info(f'Loading pretrained weights from {args.pretrained}')
    checkpoint = torch.load(args.pretrained, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Remove 'module.' prefix if present (from DataParallel)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    # Load weights (except FC layer which has wrong dimensions)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() 
                      if k in model_dict and v.shape == model_dict[k].shape}
    
    logging.info(f'Loaded {len(pretrained_dict)}/{len(model_dict)} layers from pretrained model')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    
    # Replace FC layer for fingerprint dataset
    logging.info(f'Replacing FC layer: 512 -> {args.num_classes}')
    model.fc = nn.Linear(512, args.num_classes)
    
    # Freeze layers according to transfer learning strategy
    freeze_layers(model, args.unfreeze_layers)
    
    model = model.cuda()
    
    # Loss function
    criterion = nn.CrossEntropyLoss().cuda()
    
    # Optimizer - only optimize parameters that require gradients
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs,
        eta_min=args.learning_rate * 0.01
    )
    
    start_epoch = 0
    best_acc = 0.0
    
    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        logging.info(f'Resuming from checkpoint {args.resume}')
        checkpoint = torch.load(args.resume)
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        logging.info(f'Resumed from epoch {start_epoch}, best acc: {best_acc:.2f}%')
    
    # Training loop
    logging.info('Starting training...')
    for epoch in range(start_epoch, args.epochs):
        logging.info(f'\nEpoch: {epoch}/{args.epochs} - LR: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch)
        
        # Validate
        val_loss, val_acc = validate(val_loader, model, criterion)
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
        
        save_checkpoint({
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'args': args,
        }, is_best, args.save)
        
        logging.info(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                    f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Best Acc: {best_acc:.2f}%')
    
    training_time = (time.time() - start_t) / 3600
    logging.info(f'\nTraining completed in {training_time:.2f} hours')
    logging.info(f'Best validation accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
