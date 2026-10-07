import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


class DRClassifier(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()

        # Load pretrained EfficientNet-B0
        self.model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT
        )

        # Get the number of input features
        input_features = self.model.classifier[1].in_features

        # Replace the original classifier
        self.model.classifier[1] = nn.Linear(
            input_features,
            num_classes
        )

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":

    model = DRClassifier(num_classes=5)

    # Test with a fake batch of 2 images
    test_input = torch.randn(2, 3, 224, 224)

    output = model(test_input)

    print("MODEL TEST")
    print("------------------------------")
    print("Input shape :", test_input.shape)
    print("Output shape:", output.shape)
    print("Model test successful!")