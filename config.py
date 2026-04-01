import torch

# Dataset paths
IMAGES_DIR = "Original-Dataset/Images"
LABELS_DIR = "Original-Dataset/Labels/CSV Format"
TRAIN_CSV = f"{LABELS_DIR}/train_labels.csv"
TEST_CSV = f"{LABELS_DIR}/test_labels.csv"

# Classes
CLASSES = [
    "military truck",
    "military tank",
    "military aircraft",
    "military helicopter",
    "civilian car",
    "civilian aircraft"
]

NUM_CLASSES = len(CLASSES)

# Model hyperparameters
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50
PATIENCE = 10  # Early stopping patience

# Learning rates for different models
LR_SIMPLE_CNN = 0.001      # For training SimpleCNN from scratch
LR_RESNET_TRANSFER = 0.0001  # For fine-tuning ResNet-50 
LR_EFFICIENTNET_TRANSFER = 0.00001  # For fine-tuning EfficientNet-B3 

# Data split
TRAIN_VAL_SPLIT = 0.8  # 80% train, 20% validation from training set

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Output directories
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
PLOTS_DIR = "plots"

# Random seed for reproducibility
RANDOM_SEED = 42