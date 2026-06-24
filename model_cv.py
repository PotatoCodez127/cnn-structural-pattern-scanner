# model_cv.py
import torch.nn as nn


class PatternScanner(nn.Module):
    def __init__(self):
        super(PatternScanner, self).__init__()

        # Convolutional Block
        # in_channels=1 (just price), out_channels=16 (16 different sliding kernels),
        #   kernel_size=5 (looks at 5 candles at a time)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5)
        self.relu = nn.ReLU()

        # Max Pooling (Shrinks the sequence length by half, keeping the strongest pattern signals)
        self.pool = nn.MaxPool1d(kernel_size=2)

        # Fully Connected Block (Translates the visual features into classifications)
        # After convolution and pooling of a length-50 sequence, we end up with 16 channels
        #   of length 23
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 23, 32)
        self.fc2 = nn.Linear(32, 3)  # 3 outputs: Noise, Double Bottom, Double Top

    def forward(self, x):
        # 1. Visual Feature Extraction
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)

        # 2. Classification
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        # Note: In PyTorch, CrossEntropyLoss applies Softmax automatically,
        # so we output the raw scores (logits) here.
        return x
