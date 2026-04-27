import matplotlib.pyplot as plt
import pandas as pd
import re
import os

files = {
    'Baseline CNN': 'past_results/baseline_training_metrics.txt',
    'ResNet-50': 'past_results/resnet50_training_metrics.txt',
    'ResNet-50 + Class Weights': 'past_results/resnet50_classWeights_training_metrics.txt',
    'EfficientNet-B3 Frozen': 'past_results/efficientnet_frozen_training_metrics.txt',
    'EfficientNet-B3 Unfrozen': 'past_results/efficientnet_unfrozen_training_metrics.txt',
}

def parse_metrics(filepath):
    epochs, val_loss, val_acc = [], [], []
    with open(filepath, 'r') as f:
        lines = f.readlines()
    for line in lines:
        parts = line.strip().split()
        if len(parts) == 5:
            try:
                epoch = int(parts[0])
                val_l = float(parts[3])
                val_a = float(parts[4])
                epochs.append(epoch)
                val_loss.append(val_l)
                val_acc.append(val_a)
            except ValueError:
                continue
    return epochs, val_loss, val_acc

colors = [
    '#1f77b4', 
    '#ff7f0e', 
    '#2ca02c', 
    '#d62728', 
    '#9467bd', 
]

# Validation Accuracy Plot
fig, ax = plt.subplots(figsize=(10, 6))
for (label, filepath), color in zip(files.items(), colors):
    epochs, val_loss, val_acc = parse_metrics(filepath)
    ax.plot(epochs, val_acc, label=label, color=color, linewidth=2)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Accuracy (%)', fontsize=12)
ax.set_title('Validation Accuracy Comparison Across Models', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_accuracy_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Validation accuracy plot saved")

# Validation Loss Plot
fig, ax = plt.subplots(figsize=(10, 6))
for (label, filepath), color in zip(files.items(), colors):
    epochs, val_loss, val_acc = parse_metrics(filepath)
    ax.plot(epochs, val_loss, label=label, color=color, linewidth=2)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Validation Loss', fontsize=12)
ax.set_title('Validation Loss Comparison Across Models', fontsize=14, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('validation_loss_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Validation loss plot saved")