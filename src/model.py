import torch
import torch.nn as nn

class MedNISTCNN(nn.Module):
    """
    A simple CNN for MedNIST image classification.

    Input:
        batch_size x 1 x 64 x 64

    Output:
        batch_size x num_classes
    """

    def __init__(self, num_classes: int):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1,
        ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    model = MedNISTCNN(num_classes=6)

    dummy_batch = torch.randn(64,1,64,64)
    output = model(dummy_batch)

    print("Input Shape: ", dummy_batch.shape)
    print("Output Shape: ", output.shape)