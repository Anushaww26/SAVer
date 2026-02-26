import torch 
import torch.nn as nn
import torch.optim as optim 
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import os
import random
import cv2
from PIL import Image
from segment_anything import sam_model_registry
from config.config import Config



def load_medsam():
    sam_model = sam_model_registry["vit_b"](checkpoint=Config.MEDSAM_CHECKPOINT)
    print("MedSAM model loaded successfully")
    return sam_model


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # Channel attention
        x = x * self.channel_att(x)

        # Spatial attention
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        x = x * self.spatial_att(torch.cat([avg_pool, max_pool], dim=1))

        return x
    
class MedSAMFineTune(nn.Module):
    def __init__(
        self,
        medsam_model,
        num_classes=17,
        num_planes=3,
        freeze_encoder=True
    ):
        super().__init__()

        # 🔥 Use ONLY the image encoder
        self.encoder = medsam_model.image_encoder

        # Freeze logic
        self.freeze_encoder = freeze_encoder
        if freeze_encoder:
            self._freeze_encoder()

        # Encoder output: (B, 256, 64, 64) for ViT-B
        self.cbam = CBAM(256)

        # Lightweight decoder (upsamples ×8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),  # 64 → 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),   # 128 → 256
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),    # 256 → 512
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        # 🔥 One binary head per structure
        self.heads = nn.ModuleList([
            nn.Conv2d(32, 1, kernel_size=1)
            for _ in range(num_classes)
        ])

        self.plane_head=nn.Sequential(
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128,num_planes)
        )

    # ------------------------
    # Forward
    # ------------------------
    def forward(self, x):
        """
        x: (B, 3, H, W)  where H,W ≈ 512
        returns: (B, 17, H, W)
        """

        features = self.encoder(x)     # (B,256,64,64)
        feats = torch.mean(features, dim=(2, 3))  # (B, 256)
        plane_logits=self.plane_head(feats) #(B,3)
        features = self.cbam(features)

        features = self.decoder(features)  # (B,32,H/1,W/1)

        logits = torch.cat(
            [head(features) for head in self.heads],
            dim=1
        )

        # Upsample to original resolution if needed
        if logits.shape[-2:] != x.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        
        
           
        
        return logits,plane_logits


    def _freeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = False
        print("✅ MedSAM encoder frozen")
        
    def unfreeze_encoder(self):
        for param in self.encoder.parameters():
            param.requires_grad = True
        print("🔥 MedSAM encoder unfrozen")

       


