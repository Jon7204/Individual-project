import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import config


def create_directories():
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, val_acc, is_best=False):
   
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc
    }
    
    # Save latest checkpoint
    latest_path = os.path.join(config.CHECKPOINT_DIR, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ Best model saved with validation accuracy: {val_acc:.4f}")


def load_checkpoint(model, optimizer=None, checkpoint_path=None):
   
    if checkpoint_path is None:
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, 'best_model.pth')
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch']


def plot_training_history(train_losses, val_losses, train_accs, val_accs):
    
    (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies
    ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(config.PLOTS_DIR, 'training_history.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix'):
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=config.CLASSES, 
                yticklabels=config.CLASSES,
                cbar_kws={'label': 'Count'})
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    save_path = os.path.join(config.PLOTS_DIR, f'{title.lower().replace(" ", "_")}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to {save_path}")
    plt.close()


def print_classification_report(y_true, y_pred, dataset_name='Test'):

    print(f"\n{'='*60}")
    print(f"{dataset_name} Set Classification Report")
    print('='*60)
    
    report = classification_report(y_true, y_pred, target_names=config.CLASSES, digits=4)
    print(report)
    
    # Save report to file
    report_path = os.path.join(config.RESULTS_DIR, 
                               f'{dataset_name.lower()}_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write(f"{dataset_name} Set Classification Report\n")
        f.write('='*60 + '\n')
        f.write(report)
    
    # Debug
    print(f"Classification report saved to {report_path}")


def calculate_per_class_accuracy(y_true, y_pred):

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    per_class_acc = {}
    for idx, class_name in enumerate(config.CLASSES):
        mask = y_true == idx
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean() * 100
            per_class_acc[class_name] = acc
        else:
            per_class_acc[class_name] = 0.0
    
    return per_class_acc