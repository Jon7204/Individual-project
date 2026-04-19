import torchvision.models as models
import torch

model_old = models.efficientnet_b3(pretrained=True)
model_new = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)

for (name1, p1), (name2, p2) in zip(model_old.named_parameters(), model_new.named_parameters()):
    if not torch.equal(p1, p2):
        print(f"Difference found: {name1}")
        
print("All weights identical")