import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import argparse
import os
import config
from sklearn.utils.class_weight import compute_class_weight
from dataset import create_data_loaders
from model import get_model
from utils import (create_directories, save_checkpoint, 
                   plot_training_history, calculate_per_class_accuracy)


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                         'acc': f'{100 * correct / total:.2f}%'})
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 
                            'acc': f'{100 * correct / total:.2f}%'})
    
    avg_loss = running_loss / total
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_preds, all_labels


def train_model(model_type='simple'):
    # Set random seed
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Create directories
    create_directories()
    
    # Determine model-specific settings
    if model_type == 'simple':
        model_name = "SimpleCNN (Baseline)"
        learning_rate = config.LR_SIMPLE_CNN
        output_prefix = "baseline"
    elif model_type == 'resnet50':
        model_name = "ResNet-50 (Transfer Learning)"
        learning_rate = config.LR_RESNET_TRANSFER
        output_prefix = "resnet50"
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    print("="*60)
    print(f"{model_name} Training")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Number of classes: {config.NUM_CLASSES}")
    print(f"Image size: {config.IMAGE_SIZE}x{config.IMAGE_SIZE}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {learning_rate}")
    print(f"Number of epochs: {config.NUM_EPOCHS}")
    print("="*60)
    
    # Create data loaders
    print("\nLoading datasets")
    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = create_data_loaders()
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\nInitializing {model_name}")
    if model_type == 'resnet50':
        print("Downloading pre-trained ImageNet weights")
    model = get_model(model_type=model_type)
    
    # Calculate class weights to address imbalance
    print("\nCalculating class weights")
    
    # Extract all training labels
    all_train_labels = []
    for _, label in train_loader.dataset:
        all_train_labels.append(label)
    
    # Compute balanced class weights
    class_weights = compute_class_weight(
        'balanced',
        classes=np.arange(config.NUM_CLASSES),
        y=all_train_labels
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(config.DEVICE)
    
    print("Class weights:")
    for idx, class_name in enumerate(config.CLASSES):
        print(f"  {class_name}: {class_weights[idx]:.4f}")
    
    # Loss function with class weights and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    # Only optimize unfrozen parameters (important for ResNet-50)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=learning_rate
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', 
                                                      factor=0.5, patience=5)
    
    # Training loop
    print("\nStarting training")
    print("="*60)
    
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(config.NUM_EPOCHS):
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}]")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, config.DEVICE)
        
        # Validate
        val_loss, val_acc, val_preds, val_labels = validate(model, val_loader, 
                                                            criterion, config.DEVICE)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save checkpoint
        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
        
        save_checkpoint(model, optimizer, epoch, val_acc, is_best, model_type=model_type)
        
        # Early stopping
        if patience_counter >= config.PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Plot training history
    print("\nGenerating training plots")
    plot_training_history(train_losses, val_losses, train_accs, val_accs)
    
    # Save training metrics to file
    print("\nSaving training metrics")
    history_path = os.path.join(config.RESULTS_DIR, f'{output_prefix}_training_metrics.txt')
    with open(history_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write(f"{model_name.upper()} TRAINING METRICS SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        if model_type == 'resnet50':
            f.write("Pre-training: ImageNet\n")
            f.write("Backbone: Frozen (only final layers trained)\n\n")
        
        f.write("Final Results:\n")
        f.write("-" * 70 + "\n")
        f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        f.write(f"Final Training Accuracy: {train_accs[-1]:.2f}%\n")
        f.write(f"Final Validation Accuracy: {val_accs[-1]:.2f}%\n")
        f.write(f"Final Training Loss: {train_losses[-1]:.4f}\n")
        f.write(f"Final Validation Loss: {val_losses[-1]:.4f}\n")
        f.write(f"Total Epochs Trained: {len(train_accs)}\n\n")
        
        f.write("="*70 + "\n")
        f.write("EPOCH-BY-EPOCH METRICS\n")
        f.write("="*70 + "\n")
        f.write(f"{'Epoch':<8} {'Train Loss':<14} {'Train Acc (%)':<14} {'Val Loss':<14} {'Val Acc (%)':<14}\n")
        f.write("-"*70 + "\n")
        
        for i in range(len(train_losses)):
            f.write(f"{i+1:<8} {train_losses[i]:<14.4f} {train_accs[i]:<14.2f} "
                    f"{val_losses[i]:<14.4f} {val_accs[i]:<14.2f}\n")
    
    print(f"Training metrics saved to {history_path}")
    
    # Print dataset loading statistics
    print("\n" + "="*60)
    print("Dataset Loading Statistics")
    print("="*60)
    
    train_stats = train_dataset.get_statistics()
    val_stats = val_dataset.get_statistics()
    test_stats = test_dataset.get_statistics()
    
    print(f"\nTraining Set:")
    print(f"  Total images attempted: {train_stats['total']}")
    print(f"  Successfully loaded: {train_stats['success']}")
    print(f"  Failed to load: {train_stats['failure']}")
    if train_stats['failed_files']:
        print(f"  Failed files: {', '.join(train_stats['failed_files'][:5])}")
        if len(train_stats['failed_files']) > 5:
            print(f"  ... and {len(train_stats['failed_files']) - 5} more")
    
    print(f"\nValidation Set:")
    print(f"  Total images attempted: {val_stats['total']}")
    print(f"  Successfully loaded: {val_stats['success']}")
    print(f"  Failed to load: {val_stats['failure']}")
    
    print(f"\nTest Set:")
    print(f"  Total images attempted: {test_stats['total']}")
    print(f"  Successfully loaded: {test_stats['success']}")
    print(f"  Failed to load: {test_stats['failure']}")
    
    # Final validation metrics
    print("\nFinal validation metrics:")
    per_class_acc = calculate_per_class_accuracy(val_labels, val_preds)
    for class_name, acc in per_class_acc.items():
        print(f"  {class_name}: {acc:.2f}%")
    
    print("\n" + "="*60)
    print(f"Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print("="*60)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train vehicle classification model')
    parser.add_argument('--model', type=str, default='simple', 
                       choices=['simple', 'resnet50'],
                       help='Model architecture to train (default: simple)')
    
    args = parser.parse_args()
    
    # Train the specified model
    train_model(model_type=args.model)