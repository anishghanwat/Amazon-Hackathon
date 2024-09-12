import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights

class ProductInfoExtractor(nn.Module):
    def __init__(self, num_classes):
        super(ProductInfoExtractor, self).__init__()
        self.resnet = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

def create_model(num_classes):
    """
    Create and return an instance of the ImageClassifier model.
    
    Args:
    num_classes (int): The number of classes for classification.
    
    Returns:
    ImageClassifier: An instance of the ImageClassifier model.
    """
    return ImageClassifier(num_classes)

if __name__ == "__main__":
    # Example usage
    num_classes = 10  # Replace with your actual number of classes
    model = create_model(num_classes)
    print(model)

    # Test with a random input
    sample_input = torch.randn(1, 3, 224, 224)  # Batch size 1, 3 color channels, 224x224 image
    output = model(sample_input)
    print(f"Output shape: {output.shape}")