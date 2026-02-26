import torch
torch.cuda.empty_cache()
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
import os
import random
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config.config import Config
from dataset.load_img_mask_plane import CerebellumDataset
from model.model_with_classifier import MedSAMFineTune,load_medsam
#from .loss import DiceFocalLoss
from .plane_aware_anatomical_loss import PlaneAwareDiceFocalPresenceLoss
from collections import Counter
import wandb


def collect_plane_data(image_dir,mask_dir):
    images,masks=[],[]
    for fname in os.listdir(image_dir):
        if not fname.endswith(('.png','.jpeg','.jpg')):
             continue
        
        img_path=os.path.join(image_dir,fname)
        mask_name=fname.replace('.jpeg','_mask.npz').replace('.png','_mask.npz').replace('.jpg','_mask.npz')
        mask_path=os.path.join(mask_dir,mask_name)
        if not os.path.exists(mask_path):
            continue

        images.append(img_path)
        masks.append(mask_path)
    return images,masks


def prepare_data_splits():
    all_images=[]
    all_masks=[]

    plane_configs=[(Config.TT_IMAGES,Config.TT_MASKS),(Config.TV_IMAGES,Config.TV_MASKS),(Config.TC_IMAGES,Config.TC_MASKS)]

    for img_dir,mask_dir in plane_configs:
        imgs,masks=collect_plane_data(img_dir,mask_dir)
        all_images.extend(imgs)
        all_masks.extend(masks)
    assert len(all_images)== len(all_masks )

    X_train,X_test,Y_train,Y_test= train_test_split(all_images,all_masks,test_size=Config.VAL_SPLIT,random_state=Config.RANDOM_SEED,shuffle=True)
    return X_train,X_test,Y_train,Y_test
        
     



def dice_coefficient(pred, target, threshold=0.5, smooth=1e-6):
    pred = (torch.sigmoid(pred) > threshold).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))

    dice = (2 * intersection + smooth) / (union + smooth)

    valid = target.sum(dim=(2, 3)) > 0
    dice = dice[valid]

    return dice.mean() if dice.numel() > 0 else torch.tensor(0.0, device=pred.device)


def iou_score(pred, target, threshold=0.5, smooth=1e-6):
    # Convert logits → binary masks
    pred = (torch.sigmoid(pred) > threshold).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

    iou = (intersection + smooth) / (union + smooth)

    # Ignore empty GT classes
    valid = target.sum(dim=(2, 3)) > 0
    iou = iou[valid]

    return iou.mean() if iou.numel() > 0 else torch.tensor(0.0, device=pred.device)

def train_epoch(model, dataloader,seg_criterion,plane_criterion, optimizer, device, epoch,lambda_plane=0.3):
    model.train()
    total_loss, total_dice ,total_plane_acc= 0, 0,0
    
    optimizer.zero_grad()
    for batch_idx, (images, masks,plane_label) in enumerate(tqdm(dataloader, desc=f'Training {epoch}')):
        #print(images.size())
        images, masks,plane_label = images.to(device), masks.to(device),plane_label.to(Config.DEVICE)
        
        outputs,plane_logits = model(images)
        #outputs=apply_plane_mask(outputs,plane_label)
        
        seg_loss = seg_criterion(outputs, masks, plane_label)
        
        plane_loss=plane_criterion(plane_logits,plane_label)
        loss=seg_loss+lambda_plane*plane_loss
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        

        total_loss += loss.item()
        total_plane_acc+=(plane_logits.argmax(1)==plane_label).float().mean().item()
        total_dice += dice_coefficient(outputs, masks).mean().item()
    return total_loss / len(dataloader), total_dice / len(dataloader),total_plane_acc/len(dataloader)

def validate_epoch(model, dataloader, seg_criterion,plane_criterion, device):
    model.eval()
    total_loss, total_dice, total_iou ,total_plane_acc= 0, 0, 0,0
    with torch.no_grad():
        for images, masks ,plane_label in tqdm(dataloader, desc='Validation'):
            images, masks,plane_label = images.to(device), masks.to(device),plane_label.to(device)
            
            outputs,plane_logits = model(images)
            #outputs=apply_plane_mask(outputs,plane_label)
            
            #loss = criterion(outputs,masks)
            seg_loss = seg_criterion(outputs, masks, plane_label)

            plane_loss=plane_criterion(plane_logits,plane_label)
            loss= seg_loss +0.3*plane_loss

            total_loss += loss.item()
            total_dice += dice_coefficient(outputs, masks).mean().item()
            total_iou += iou_score(outputs, masks).mean().item()
            total_plane_acc+=(plane_logits.argmax(1)==plane_label).float().mean().item()
    torch.cuda.empty_cache()
    return total_loss / len(dataloader), total_dice / len(dataloader), total_iou / len(dataloader),total_plane_acc/len(dataloader)


def main():
    os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
    print(f"Device: {Config.DEVICE}")
    print(f"Image Size: {Config.IMG_SIZE}")
    print(f"Batch Size: {Config.BATCH_SIZE}")
    wandb.init(
    project="MedSAM-PlaneAware-Segmentation",
    name=f"run_{Config.NUM_EPOCHS}ep_bs{Config.BATCH_SIZE}_lr{Config.LEARNING_RATE}",
    config={
        "epochs": Config.NUM_EPOCHS,
        "batch_size": Config.BATCH_SIZE,
        "lr": Config.LEARNING_RATE,
        "img_size": Config.IMG_SIZE,
        "num_classes": Config.NUM_CLASSES,
        "freeze_encoder": Config.FREEZE_IMAGE_ENCODER,
        "lambda_plane": 0.3,
        "loss": "PlaneAwareDiceFocalPresence + CE",
        "optimizer": "AdamW"
    }
)
    
    # Load MedSAM
    medsam = load_medsam().to(Config.DEVICE)
    model = MedSAMFineTune(medsam, num_classes=Config.NUM_CLASSES, freeze_encoder=Config.FREEZE_IMAGE_ENCODER).to(Config.DEVICE)
    torch.cuda.empty_cache()
    wandb.watch(model, log="gradients", log_freq=100)
    # Prepare data
    train_imgs, val_imgs, train_masks, val_masks = prepare_data_splits()
    
    train_dataset = CerebellumDataset(train_imgs, train_masks, train=True)
    val_dataset = CerebellumDataset(val_imgs, val_masks, train=False)
    plane_labels = [train_dataset._get_plane_label(p) for p in train_imgs]
    counts = Counter(plane_labels)

    weights = [1.0 / counts[label] for label in plane_labels]

    sampler = torch.utils.data.WeightedRandomSampler(
    weights,
    num_samples=len(weights),
    replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)

    # Loss and optimizer
    seg_criterion= PlaneAwareDiceFocalPresenceLoss()
    plane_criterion=nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-4)
    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}

    best_dice = max(history['val_dice']) if history['val_dice'] else 0
    start_epoch=0
    lrs=[]
    try :
        for epoch in range(start_epoch, Config.NUM_EPOCHS):
            print(f"\n{'='*60}\nEpoch {epoch + 1}/{Config.NUM_EPOCHS}\n{'='*60}")

            if epoch == Config.UNFREEZE_AFTER_EPOCH:
                model.unfreeze_encoder()

            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            train_loss, train_dice ,train_plane_acc= train_epoch(model, train_loader,seg_criterion,plane_criterion, optimizer, Config.DEVICE, epoch + 1)
            val_loss, val_dice, val_iou ,val_plane_acc= validate_epoch(model, val_loader, seg_criterion,plane_criterion, Config.DEVICE)
            #scheduler.step()

            history['train_loss'].append(train_loss)
            history['train_dice'].append(train_dice)
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_dice)
            history['val_iou'].append(val_iou)
            wandb.log({
                "epoch": epoch + 1,

                # ---- Losses ----
                "train/loss": train_loss,
                "val/loss": val_loss,

                # ---- Segmentation Metrics ----
                "train/dice": train_dice,
                "val/dice": val_dice,
                "val/iou": val_iou,

                # ---- Plane Classification ----
                "train/plane_acc": train_plane_acc,
                "val/plane_acc": val_plane_acc,

                # ---- Optimization ----
                "lr": current_lr
            })

            print(f"Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f}, Plane_acc {train_plane_acc}")
            print(f"Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Val IoU: {val_iou:.4f} ,Plane_acc {val_plane_acc}")
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}, Validation Loss: {val_loss:.4f}, Current LR: {current_lr}")

            # Save best model
            # if val_dice > best_dice:
            #     best_dice = val_dice
            #     torch.save({
            #         'model_state_dict': model.state_dict(),
            #         }, os.path.join(Config.CHECKPOINT_DIR, f'best_model.pth'))
                
            #     print(f"Saved best model with Dice: {best_dice:.4f}")
            if val_dice > best_dice:
                best_dice = val_dice

                ckpt_path = os.path.join(Config.CHECKPOINT_DIR, "best_model.pth")
                torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

                wandb.run.summary["best_val_dice"] = best_dice
                wandb.save(ckpt_path)

                print(f"Saved best model with Dice: {best_dice:.4f}")

    except KeyboardInterrupt:
        print("\nTraining interrupted by user!")
        
    finally:
        print("\nSaving training plots...")

        # Determine max epoch to plot
        max_plot_epoch = min(60, len(history['train_loss']))

        # Slice the history dictionaries
        sliced_history = {k: v[:max_plot_epoch] for k, v in history.items()}
        sliced_lrs = lrs[:max_plot_epoch]

        # Plot history
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        axes[0].plot(sliced_history['train_loss'], label='Train Loss')
        axes[0].plot(sliced_history['val_loss'], label='Val Loss')
        axes[1].plot(sliced_history['train_dice'], label='Train Dice')
        axes[1].plot(sliced_history['val_dice'], label='Val Dice')
        axes[1].plot(sliced_history['val_iou'], label='Val IoU')
        for ax in axes:
            ax.set_xlabel('Epoch')
            ax.grid(True)
            ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(Config.CHECKPOINT_DIR, 'training_history_epochs.jpeg'))

        
if __name__ == '__main__':
    main()