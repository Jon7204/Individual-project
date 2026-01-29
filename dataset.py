import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageFile
import torchvision.transforms as transforms
import config

# Allow loading of truncated images
ImageFile.LOAD_TRUNCATED_IMAGES = True


class VehicleDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform
        self.class_to_idx = {cls: idx for idx, cls in enumerate(config.CLASSES)}
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        try:
            # Load image
            img_path = os.path.join(self.images_dir, row['filename'])
            image = Image.open(img_path).convert('RGB')
            
            # Crop bounding box
            bbox = (int(row['xmin']), int(row['ymin']), 
                    int(row['xmax']), int(row['ymax']))
            cropped_image = image.crop(bbox)
            
            # Apply transforms
            if self.transform:
                cropped_image = self.transform(cropped_image)
            
            # Get label
            label = self.class_to_idx[row['class']]
            
            return cropped_image, label
            
        except Exception as e:
            print(f"Warning: Error loading image {row['filename']}: {e}")
            # Return a black image and label 0 as fallback
            black_image = Image.new('RGB', (config.IMAGE_SIZE, config.IMAGE_SIZE), (0, 0, 0))
            if self.transform:
                black_image = self.transform(black_image)
            return black_image, 0


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])


def create_data_loaders():
    # Load training dataset
    train_dataset = VehicleDataset(
        csv_file=config.TRAIN_CSV,
        images_dir=config.IMAGES_DIR,
        transform=get_transforms(train=True)
    )
    
    # Split training into train and validation
    train_size = int(config.TRAIN_VAL_SPLIT * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED)
    )
    
    # Create validation dataset with validation transforms
    val_dataset = VehicleDataset(
        csv_file=config.TRAIN_CSV,
        images_dir=config.IMAGES_DIR,
        transform=get_transforms(train=False)
    )
    
    # Update validation subset to use validation transforms
    val_subset.dataset = val_dataset
    
    # Load test dataset
    test_dataset = VehicleDataset(
        csv_file=config.TEST_CSV,
        images_dir=config.IMAGES_DIR,
        transform=get_transforms(train=False)
    )
    
    # Determine if pin_memory should be used (not supported on MPS)
    use_pin_memory = config.DEVICE.type == 'cuda'
    
    # Create data loaders with fewer workers to avoid multiprocessing issues
    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues with corrupted images
        pin_memory=use_pin_memory
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_subset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory
    )
    
    return train_loader, val_loader, test_loader