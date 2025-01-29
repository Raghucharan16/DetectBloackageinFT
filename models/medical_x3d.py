import torch
import torch.nn as nn
from pytorchvideo.models.hub import x3d_m
from torchvision.ops import StochasticDepth

class MedicalX3D(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        # Kinetics-pretrained backbone
        self.backbone = x3d_m(pretrained=pretrained)
        
        # Medical adaptations
        self.backbone.blocks[0].conv.conv_t = nn.Conv3d(
            1, 24, kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1), bias=False
        )
        
        # Attention mechanisms
        self.temporal_attn = nn.MultiheadAttention(2048, 8, dropout=0.2)
        self.spatial_attn = nn.Sequential(
            nn.Conv3d(2048, 512, 1),
            nn.GroupNorm(32, 512),
            nn.SiLU()
        )
        
        # Classifier
        self.head = nn.Sequential(
            StochasticDepth(0.2, "row"),
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        # x: (B,1,16,224,224)
        features = self.backbone(x)
        
        # Temporal attention
        temp_attn, _ = self.temporal_attn(
            features.unsqueeze(1), features.unsqueeze(1), features.unsqueeze(1)
        )
        
        # Spatial attention
        spatial_attn = self.spatial_attn(features.unsqueeze(2)).squeeze()
        
        # Combine features
        combined = temp_attn + spatial_attn
        return self.head(combined.mean(dim=[2,3,4]))
