import torch 
import torch.nn as nn
import torchvision.transforms.functional as TF
import torchvision.transforms as  T
import numpy as np
import os
import random
import cv2
from PIL import Image
from config.config import Config

from torch.utils.data import Dataset

def speckle_dropout(image,p=0.1):
        mask=np.random.rand(*image.shape)>p
        return image*mask
    

class CerebellumDataset(Dataset):
    def __init__(self, image_paths, mask_paths, train=True):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.train = train
        self.img_size = Config.IMG_SIZE

    def __len__(self):
        
        return len(self.image_paths)
    
    def _get_plane_label(self,img_path):
        path= img_path.lower()
        if "transthalamic" in path or "tt" in path:
            return 0
        
        elif "transventricular" in path or "tv" in path:
            return 1
        
        elif "transcerebellar" in path or "tc" in path:
            return 2
        
        else: 
            raise ValueError(f"Unknown plane for image:{img_path}")


    def __getitem__(self, idx):
        # --- Load ---
        image = Image.open(self.image_paths[idx]).convert("L")  # grayscale
        

        data = np.load(self.mask_paths[idx])
        mask = data["mask"].astype(np.uint8)  # (H, W, C)

        plane_label=self._get_plane_label(self.image_paths[idx])
        plane_label=torch.tensor(plane_label,dtype=torch.long)
        

        # --- Optional fixed crop (dataset-specific) ---
        if self.image_paths[idx].endswith(".png"):
            CROP_X = 393
            CROP_Y = 68
            CROP_W = 1141
            CROP_H = 793
            crop_box=(CROP_X,CROP_Y,CROP_X+CROP_W,CROP_Y+CROP_H)
            image=image.crop(crop_box)
        original_size = image.size 
        image = np.array(image)
        image_np=np.array(image)
        # --- Augmentation ---
        if self.train:
            if random.random() < 0.5:
                image = np.fliplr(image)
                mask = np.fliplr(mask)

            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                M = cv2.getRotationMatrix2D(
                    (image.shape[1] // 2, image.shape[0] // 2),
                    angle, 1.0
                )
                image = cv2.warpAffine(
                    image, M, (image.shape[1], image.shape[0]),
                    flags=cv2.INTER_LINEAR
                )
                mask = np.stack([
                    cv2.warpAffine(
                        mask[:, :, c], M,
                        (mask.shape[1], mask.shape[0]),
                        flags=cv2.INTER_NEAREST
                    ) for c in range(mask.shape[2])
                ], axis=2)

            # Ultrasound-specific
            if random.random() < 0.3:
                image = cv2.GaussianBlur(image, (5, 5), 0)

            if random.random() < 0.3:
                noise = np.random.normal(0, 10, image.shape)
                image = np.clip(image + noise, 0, 255)

            if random.random() < 0.3:
                gamma = random.uniform(0.7, 1.5)
                image = np.clip(255 * (image / 255) ** gamma, 0, 255)

            #if random.random() <0.2 and plane_label==2:
            #    image=speckle_dropout(image)  

            
            

        # --- Resize ONCE ---
        image = cv2.resize(image, (self.img_size, self.img_size))
        mask = np.stack([
            cv2.resize(mask[:, :, c],
                       (self.img_size, self.img_size),
                       interpolation=cv2.INTER_NEAREST)
            for c in range(mask.shape[2])
        ], axis=2)
        image = torch.from_numpy(image).float().unsqueeze(0) / 255.0
        image = image.repeat(3, 1, 1)  # MedSAM expects 3 channels , uncomment this when not training with gradients

        mask = torch.from_numpy(mask).permute(2, 0, 1).float()
        mask = (mask > 0).float()  # SAFE binarization

        return image,mask,plane_label