"""The compact research network; importing this module requires PyTorch."""

import torch
from torch import nn


class ResearchCNN(nn.Module):
    """Accept raw log-mel features and apply training-only normalization."""

    def __init__(self, mean, std, classes: int = 50):
        super().__init__()
        mean = torch.as_tensor(mean, dtype=torch.float32).reshape(1, 1, 64, 1)
        std = torch.as_tensor(std, dtype=torch.float32).reshape(1, 1, 64, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std.clamp_min(1e-6))
        blocks = []
        channels = 1
        for width in (16, 32, 64):
            blocks.extend(
                [
                    nn.Conv2d(channels, width, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(width),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                ]
            )
            channels = width
        self.features = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(0.25)
        self.classifier = nn.Linear(128, classes)

    def forward(self, audio_features):
        normalized = (audio_features - self.mean) / self.std
        x = self.features(normalized)
        pooled = torch.cat((x.mean(dim=(2, 3)), x.amax(dim=(2, 3))), dim=1)
        return self.classifier(self.dropout(pooled))
