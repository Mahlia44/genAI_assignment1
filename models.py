import torch.nn as nn
from torchvision.models import resnet18

class ImageEncoder(nn.Module):
    """
    A simple neural network template for self-supervised learning.

    Structure:
    1. Encoder: Maps an input image of shape 
       (input_channels, input_dim, input_dim) 
       into a lower-dimensional feature representation.
    2. Projector: Transforms the encoder output into the final 
       embedding space of size `proj_dim`.

    Notes:
    - DO NOT modify the fixed class variables: 
      `input_dim`, `input_channels`, and `feature_dim`.
    - You may freely modify the architecture of the encoder 
      and projector (layers, activations, normalization, etc.).
    - You may add additional helper functions or class variables if needed.
    """

    ####### DO NOT MODIFY THE CLASS VARIABLES #######
    input_dim: int = 64
    input_channels: int = 3
    feature_dim: int = 1000
    proj_dim: int = 128
    #################################################

    def __init__(self):
        super().__init__()

        enc = resnet18(weights=None)          # initialize from scratch
        enc.conv1 = nn.Conv2d(
            self.input_channels, self.input_dim, kernel_size=3, stride=1, padding=1, bias=False
        )                                     # adapt to small 64x64 input
        enc.maxpool = nn.Identity()           # remove initial downsampling
        enc.fc = nn.Identity()                # remove classification head
        self.encoder = enc                    # encoder outputs 512-dim features


        self.projector = nn.Sequential(
            nn.Linear(512, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.proj_dim)
        )
        
    

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor of shape 
                              (batch_size, input_channels, input_dim, input_dim).

        Returns:
            torch.Tensor: Output embedding of shape (batch_size, proj_dim).
        """
        features = self.encoder(x)   # (batch_size, ...)
        projected_features = self.projector(features)  # (batch_size, proj_dim)
        #return features, projected_features
        return nn.functional.normalize(projected_features, dim=1)  # return ONLY projections

    
    def get_features(self, x):
        """
        Get the features from the encoder.

        Args:
            x (torch.Tensor): Input tensor of shape 
                              (batch_size, input_channels, input_dim, input_dim).

        Returns:
            torch.Tensor: Output features of shape (batch_size, feature_dim).
        """
        features = self.encoder(x)
        return features