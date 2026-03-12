import torchvision
import torch.nn as nn
from schemas import (
    NeuralNetConfig,
    ResNet18Config,
    ResNet34Config,
    ResNet50Config
)

def build_neural_net(cfg):
    """
    Builds a neural network model based on the provided configuration.
    """
    weights = "DEFAULT" if cfg.pretrained else None

    if isinstance(cfg, ResNet18Config):
        model = torchvision.models.resnet18(weights=weights)
    elif isinstance(cfg, ResNet34Config):
        model = torchvision.models.resnet34(weights=weights)
    elif isinstance(cfg, ResNet50Config):
        model = torchvision.models.resnet50(weights=weights)
    
    if cfg.in_channels != 3:
        # Modify the first convolutional layer to accept the specified number of input channels
        model.conv1 = nn.Conv2d(cfg.in_channels,
                                model.conv1.out_channels,
                                kernel_size=model.conv1.kernel_size,
                                stride=model.conv1.stride,
                                padding=model.conv1.padding,
                                bias=model.conv1.bias)
    
    # Freeze weights if specified
    if cfg.freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(cfg.dropout),
        nn.Linear(in_features, cfg.num_classes)
    )

    return model