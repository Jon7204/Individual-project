import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import config

# Simple baseline CNN model
class SimpleCNN(nn.Module):
    
    def __init__(self, num_classes=config.NUM_CLASSES):
        super(SimpleCNN, self).__init__()
        
        # Convolutional Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)  
        
        # Convolutional Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Convolutional Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2) 
        
        # Convolutional Block 4
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(2, 2)  
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # Conv Block 1
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        
        # Conv Block 2
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        
        # Conv Block 3
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Conv Block 4
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x

# ResNet50 with transfer learning
class ResNet50Transfer(nn.Module):
 
    def __init__(self, num_classes=config.NUM_CLASSES, freeze_backbone=True):
        super(ResNet50Transfer, self).__init__()
        
        # Load pre-trained ResNet-50
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        if freeze_backbone:
            for param in self.resnet.parameters():
                param.requires_grad = False
        
        # Get number of features from the last layer
        num_features = self.resnet.fc.in_features
        
        # Replace final fully connected layer for our 6 classes
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.resnet(x)

    # Unfreeze the last num_layers for fine tuning
    # Not used in final training, could be used in future expansion of the project
    def unfreeze_layers(self, num_layers=10):

        # Get all children modules
        children = list(self.resnet.children())
        
        # Unfreeze last num_layers
        for child in children[-num_layers:]:
            for param in child.parameters():
                param.requires_grad = True

class EfficientNetTransfer(nn.Module):
    
    def __init__(self, num_classes=config.NUM_CLASSES, freeze_backbone=True):
        super(EfficientNetTransfer, self).__init__()
        
        # Load pre-trained EfficientNet-B3
        self.efficientnet = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        
        if freeze_backbone:
            for param in self.efficientnet.parameters():
                param.requires_grad = False
        
        # Get number of features from the last layer
        num_features = self.efficientnet.classifier[1].in_features
        
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )
        
    def forward(self, x):
        return self.efficientnet(x)


    # Not used in final training, could be used in future expansion of the project 
    def unfreeze_layers(self, num_blocks=2):
        blocks = list(self.efficientnet.features.children())
        
        # Unfreeze last num_blocks
        for block in blocks[-num_blocks:]:
            for param in block.parameters():
                param.requires_grad = True


def get_model(model_type):
    
    if model_type.lower() == 'simple':
        model = SimpleCNN(num_classes=config.NUM_CLASSES)
        model_name = "SimpleCNN"
    elif model_type.lower() == 'resnet50':
        model = ResNet50Transfer(num_classes=config.NUM_CLASSES, freeze_backbone=True)
        model_name = "ResNet-50 (Transfer Learning)"
    elif model_type.lower() == 'efficientnet':
        model = EfficientNetTransfer(num_classes=config.NUM_CLASSES, freeze_backbone=False)
        model_name = "EfficientNet-B3 (Transfer Learning)"
    
    model = model.to(config.DEVICE)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Frozen parameters: {total_params - trainable_params:,}")
    
    return model