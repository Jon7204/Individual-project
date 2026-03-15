import torch
import numpy as np
from tqdm import tqdm
import sys
import config
from dataset import create_data_loaders
from model import get_model
from utils import (load_checkpoint, plot_confusion_matrix, 
                   print_classification_report, calculate_per_class_accuracy)


def evaluate_model(model, test_loader, device):

    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            # Statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Store predictions and labels
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({'acc': f'{100 * correct / total:.2f}%'})
    
    accuracy = 100 * correct / total
    
    return accuracy, all_preds, all_labels


def main():
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py [model_type]")
        print("model_type options: simple, resnet50, efficientnet")
        sys.exit(1)

    if sys.argv[1].lower() not in ['simple', 'resnet50', 'efficientnet']:
        print(f"Error: Unknown model type '{sys.argv[1]}'")
        print("Valid options: simple, resnet50, efficientnet")
        sys.exit(1)
    model_type = sys.argv[1].lower()
    
    print("="*60)
    print("Vehicle Classification Evaluation")
    print("="*60)
    print(f"Device: {config.DEVICE}")
    print(f"Model: {model_type.upper()}")
    
    # Load test data
    print("\nLoading test dataset")
    _, _, test_loader, _, _, _ = create_data_loaders()
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nInitializing model")
    model = get_model(model_type=model_type)
    
    # Load best checkpoint
    print("\nLoading best model checkpoint")
    load_checkpoint(model, model_type=model_type)
    
    # Evaluate on test set
    print("\nEvaluating on test set")
    test_acc, test_preds, test_labels = evaluate_model(model, test_loader, 
                                                       config.DEVICE)
    
    print("\n" + "="*60)
    print(f"Test Accuracy: {test_acc:.2f}%")
    print("="*60)
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    per_class_acc = calculate_per_class_accuracy(test_labels, test_preds)
    for class_name, acc in per_class_acc.items():
        print(f"  {class_name}: {acc:.2f}%")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix")
    plot_confusion_matrix(test_labels, test_preds, title='Test Confusion Matrix')
    
    # Print classification report
    print_classification_report(test_labels, test_preds, dataset_name='Test')
    
    # Save test metrics summary
    print("\nSaving test metrics summary")
    test_summary_path = os.path.join(config.RESULTS_DIR, 'test_metrics_summary.txt')
    with open(test_summary_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("TEST SET METRICS SUMMARY\n")
        f.write("="*60 + "\n\n")
        
        f.write(f"Overall Test Accuracy: {test_acc:.2f}%\n\n")
        
        f.write("Per-Class Accuracy:\n")
        f.write("-"*60 + "\n")
        for class_name, acc in per_class_acc.items():
            f.write(f"  {class_name:<25} {acc:>6.2f}%\n")
    
    print(f"Test metrics summary saved to {test_summary_path}")
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print("="*60)


if __name__ == "__main__":
    main()