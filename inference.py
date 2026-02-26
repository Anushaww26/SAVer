
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import numpy as np
import os
import cv2
from PIL import Image
import pandas as pd
from scipy import stats as scipy_stats
from segment_anything import sam_model_registry
import torch.nn as nn
import open_clip
import json
from tqdm import tqdm


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from model.model_with_classifier import MedSAMFineTune
from config.config import Config
from dataset.load_test_data import CerebellumDataset
from utils.utils import postprocess_masks



def apply_plane_mask_probs(probs, plane_label):
    """Apply plane-specific class masking to probabilities"""
    masked = probs.clone()
    valid_classes = Config.PLANE_CLASS_MAP.get(plane_label, [])

    for c in range(masked.shape[1]):
        if c not in valid_classes:
            masked[:, c] = 0.0
    return masked


def compute_structure_uniformity(pred_masks):
    """
    Standard planes should have consistent, well-balanced structure sizes.
    Non-standard often have one dominant structure or very unbalanced sizes.
    """
    structure_sizes = []
    for c in range(pred_masks.shape[0]):
        size = pred_masks[c].sum()
        if size > 0:
            structure_sizes.append(size)
    
    if len(structure_sizes) < 2:
        return 0.0
    
    # Calculate coefficient of variation (lower = more uniform)
    cv = np.std(structure_sizes) / (np.mean(structure_sizes) + 1e-6)
    
    # Convert to score (0-1, where 1 is uniform)
    uniformity = 1.0 / (1.0 + cv)
    
    return uniformity


def compute_spatial_distribution(pred_masks):
    """
    Standard planes have well-distributed structures across the image.
    Non-standard often have clustered or edge-heavy structures.
    """
    h, w = pred_masks.shape[1], pred_masks.shape[2]
    
    # Divide image into quadrants
    h_mid, w_mid = h // 2, w // 2
    quadrants = [
        pred_masks[:, :h_mid, :w_mid],      # Top-left
        pred_masks[:, :h_mid, w_mid:],      # Top-right
        pred_masks[:, h_mid:, :w_mid],      # Bottom-left
        pred_masks[:, h_mid:, w_mid:]       # Bottom-right
    ]
    
    # Count structures in each quadrant
    quadrant_counts = [np.sum(q.sum(axis=(1, 2)) > 0) for q in quadrants]
    
    # Standard planes should have structures in multiple quadrants
    non_zero_quadrants = sum(1 for count in quadrant_counts if count > 0)
    
    # Uniformity across quadrants
    if non_zero_quadrants == 0:
        return 0.0
    
    quadrant_uniformity = 1.0 - (np.std(quadrant_counts) / (np.mean(quadrant_counts) + 1e-6))
    
    # Combine: prefer images with structures in 3-4 quadrants
    distribution_score = (non_zero_quadrants / 4.0) * 0.6 + quadrant_uniformity * 0.4
    
    return distribution_score


def compute_edge_penalty(pred_masks):
    """
    Structures touching image edges often indicate non-standard/partial views.
    Standard planes typically show complete structures centered in the image.
    """
    h, w = pred_masks.shape[1], pred_masks.shape[2]
    edge_pixels = 0
    total_pixels = 0
    
    for c in range(pred_masks.shape[0]):
        if pred_masks[c].sum() > 0:
            # Check edges (5 pixel border)
            edge_mask = np.zeros_like(pred_masks[c])
            edge_mask[:5, :] = 1      # Top
            edge_mask[-5:, :] = 1     # Bottom
            edge_mask[:, :5] = 1      # Left
            edge_mask[:, -5:] = 1     # Right
            
            edge_pixels += (pred_masks[c] * edge_mask).sum()
            total_pixels += pred_masks[c].sum()
    
    if total_pixels == 0:
        return 1.0
    
    edge_ratio = edge_pixels / total_pixels
    
    # Return penalty (1.0 = no edge touching, 0.0 = lots of edge touching)
    return max(0.0, 1.0 - edge_ratio * 2.0)


def compute_structure_completeness(pred_masks, plane_pred):
    """
    Structure-based anatomical completeness score.
    Standard plane requires ALL expected structures to be present.
    """

    expected_structs = Config.PLANE_STRUCTURE_MAP.get(plane_pred, [])

    if len(expected_structs) == 0:
        return 0.0

    present = 0
    quality = []

    H, W = pred_masks.shape[1], pred_masks.shape[2]
    image_area = H * W
    MIN_CSP_PIXELS = 300
    MIN_STRUCT_PIXELS = 150

    # --- CSP HARD RULE ---
    csp_name = "csp"

    if csp_name in Config.STRUCTURE_IDX:
        csp_idx = Config.STRUCTURE_IDX[csp_name]

        if csp_idx >= pred_masks.shape[0]:
            return 0.0

        if pred_masks[csp_idx].sum() < MIN_CSP_PIXELS:
            return 0.0

    # --- Structure completeness ---
    for struct_name in expected_structs:

        if struct_name not in Config.STRUCTURE_IDX:
            continue

        c = Config.STRUCTURE_IDX[struct_name]

        if c >= pred_masks.shape[0]:
            continue

        area = pred_masks[c].sum()

        if area >= MIN_STRUCT_PIXELS:
            present += 1

            size_ratio = area / image_area

            if size_ratio > 0.005:
                quality.append(1.0)
            elif size_ratio > 0.001:
                quality.append(0.5)
            else:
                quality.append(0.0)

        for struct_name in expected_structs:
            if struct_name not in Config.STRUCTURE_IDX:
                continue

            c = Config.STRUCTURE_IDX[struct_name]

            if c >= pred_masks.shape[0]:
                continue

            area = pred_masks[c].sum()

            if area > 0:
                present += 1

                size_ratio = area / image_area

    return float(np.mean(quality)) if len(quality) > 0 else 0.0




def predict(model, image_tensor, debug=False):
    """Run inference on a single image"""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(Config.DEVICE)

        seg_logits, plane_logits = model(image_tensor)

        # ---- plane prediction ----
        plane_probs = F.softmax(plane_logits, dim=1)
        plane_conf, plane_pred = torch.max(plane_probs, dim=1)
        
        # Get top 2 predictions to check ambiguity
        top2_probs, top2_preds = torch.topk(plane_probs, k=2, dim=1)
        plane_margin = (top2_probs[0, 0] - top2_probs[0, 1]).item()
        
        plane_pred = plane_pred.item()
        plane_conf = plane_conf.item()

        # ---- segmentation ----
        probs = torch.sigmoid(seg_logits)
        probs = apply_plane_mask_probs(probs, plane_pred)

        pred_masks = (probs > Config.THRESHOLD).cpu().numpy()[0] #binary masks
        
        # Keep raw logits for accurate pixel-wise heatmap generation
        seg_logits_numpy = seg_logits.cpu().numpy()[0]  # Raw logits before sigmoid
        
        # Keep probabilities for confidence computation
        probs_numpy = probs.cpu().numpy()[0]

        # ---- DISCRIMINATIVE quality metrics ----
        #struct_conf = compute_structure_confidence(probs, pred_masks)
        completeness = compute_structure_completeness(pred_masks, plane_pred)
        uniformity = compute_structure_uniformity(pred_masks)
        distribution = compute_spatial_distribution(pred_masks)
        edge_penalty = compute_edge_penalty(pred_masks)
        
        num_structures = (pred_masks.sum(axis=(1, 2)) > 0).sum()
        total_coverage = pred_masks.sum() / (pred_masks.shape[1] * pred_masks.shape[2])

        # ---- combined confidence with discriminative features ----
        standard_conf = (
            0.15 * plane_conf +
            0.30 * completeness +      # MOST IMPORTANT: all structures present
            0.15 * uniformity +
            0.10 * distribution +
            0.10 * edge_penalty
        )

        # ---- STRICT standard plane determination ----
        # The KEY discriminator: ALL expected structures must be present
        is_standard = (
            plane_conf >= 0.99 and           # Very confident about plane
            plane_margin >= 0.70 and         # Clear winner
            completeness >= 0.30 and         # ALL structures present (KEY!)
            uniformity >= 0.01 and           # Balanced structure sizes
            num_structures >= 10 and          # Multiple structures
            0.05 <= total_coverage <= 1   # Reasonable coverage
        )

        if debug:
            print(f"  Plane: {Config.PLANE_ID_TO_NAME.get(plane_pred, 'Unknown')}")
            print(f"  Plane conf: {plane_conf:.4f} (≥0.98) {'✓' if plane_conf >= 0.90 else '✗'}")
            print(f"  Plane margin: {plane_margin:.4f} (≥0.90) {'✓' if plane_margin >= 0.70 else '✗'}")
            print(f"  Completeness: {completeness:.4f} (≥0.95) {'✓' if completeness >= 0.30 else '✗'} ← KEY")
            print(f"  Uniformity: {uniformity:.4f} (≥0.40) {'✓' if uniformity >= 0.01 else '✗'}")
            print(f"  Distribution: {distribution:.4f} (≥0.50) {'✓' if distribution >= 0.40 else '✗'}")
            print(f"  Edge penalty: {edge_penalty:.4f} (≥0.70) {'✓' if edge_penalty >= 0.60 else '✗'}")
            print(f"  Num structures: {num_structures} (≥3) {'✓' if num_structures >= 10 else '✗'}")
            print(f"  Coverage: {total_coverage:.4f} (0.08-0.45) {'✓' if 0.05 <= total_coverage <= 0.85 else '✗'}")
            print(f"  Combined: {standard_conf:.4f}")
            print(f"  → {'STANDARD' if is_standard else 'NON-STANDARD'}")

    return (
        pred_masks,
        probs_numpy,
        seg_logits_numpy,
        plane_pred,
        plane_conf,
        
        standard_conf,
        is_standard,
        plane_margin,
        completeness,
        uniformity,
        distribution,
        edge_penalty,
        num_structures,
        total_coverage
    )




def draw_color_legend(image, masks):
    """Draw legend showing structure colors and names"""
    y = 120
    for c in range(masks.shape[0]):
        if masks[c].sum() == 0:
            continue

        cv2.rectangle(image, (50, y), (70, y + 20),
                      Config.structure_colors[c], -1)
        cv2.putText(
            image,
            Config.class_names[c],
            (80, y + 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2
        )
        y += 28
    return image


def draw_contours_on_image(
    image,
    masks,
    original_size,
    plane_pred,
    plane_conf,
    completeness,
    uniformity,
    is_standard,
    standard_conf
):
    """Draw segmentation contours and labels on image"""
    h, w = masks.shape[1], masks.shape[2]
    image_resized = cv2.resize(image, (w, h))
    output = image_resized.copy()

    # ---- draw contours ----
    for c in range(masks.shape[0]):
        if masks[c].sum() == 0:
            continue

        mask_uint8 = (masks[c] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(
            mask_uint8,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(
            output,
            contours,
            -1,
            Config.structure_colors[c],
            Config.CONTOUR_THICKNESS
        )

    output = cv2.resize(output, (original_size[1], original_size[0]))

    # ---- text overlay ----
    plane_name = Config.PLANE_ID_TO_NAME.get(plane_pred, "Unknown")
    label = "STANDARD" if is_standard else "NON-STANDARD"
    color = (0, 255, 0) if is_standard else (0, 0, 255)

    cv2.putText(
        output,
        f"{plane_name} | Comp:{completeness:.2f} Unif:{uniformity:.2f}",
        (30, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        output,
        f"{label} | Confidence:{standard_conf:.3f}",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        color,
        2
    )

    output = draw_color_legend(output, masks)
    return output


class CerebellumInferDataset(torch.utils.data.Dataset):
    """Dataset for inference on test images"""
    def __init__(self, image_dir):
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        # --- Load image ONCE ---
        image = Image.open(img_path).convert("RGB")

        # --- Optional crop (PNG only) ---
        if img_path.endswith(".png"):
            CROP_X = 393
            CROP_Y = 68
            CROP_W = 1141
            CROP_H = 793
            crop_box = (CROP_X, CROP_Y, CROP_X + CROP_W, CROP_Y + CROP_H)
            image = image.crop(crop_box)

        # --- Save original (for visualization) ---
        original_image = np.array(image)
        h, w = original_image.shape[:2]
        size = torch.tensor([h, w], dtype=torch.int32)


        # --- Preprocess for model ---
        image = TF.resize(image, (Config.IMG_SIZE, Config.IMG_SIZE))
        image = TF.to_tensor(image)

        return image, original_image, img_path, size
    

CLASSNAMES = [
    "transcerebellar",
    "transthalamic",
    "transventricular"
]

# loading fetal_clip model
with open(PATH_FETALCLIP_CONFIG, "r") as file:
    config_fetalclip = json.load(file)
open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

model, preprocess_train, preprocess_test = open_clip.create_model_and_transforms("FetalCLIP", pretrained=PATH_FETALCLIP_WEIGHT)
tokenizer = open_clip.get_tokenizer("FetalCLIP")
model.eval()
model.to(device)

# clip_adapter
class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.fc(x)
        return x


class CustomFetalCLIP(nn.Module):

    def __init__(self, model, device, text_prompts, preprocess):
        super().__init__()
        self.image_encoder = model.encode_image
        self.text_encoder = model.encode_text
        self.logit_scale = model.logit_scale
        self.adapter = Adapter(768, 4)
        self.device = device
        self.text_prompts = text_prompts
        self.preprocess = preprocess
        
        # Dynamic ratio learning module
        self.ratio_mlp = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Keeps ratio between 0 and 1
        )
        
        # Learnable temperature per class
        num_classes = len(text_prompts)
        self.temperature = nn.Parameter(torch.ones(num_classes))
        
        # Uncertainty estimation head
        self.uncertainty_head = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output between 0 and 1 (0=certain, 1=uncertain)
        )

        # Precompute text features once to save time
        text_tokens = tokenizer(text_prompts).to(device)
        with torch.no_grad():
            text_features = self.text_encoder(text_tokens)
            self.register_buffer('text_features', text_features / text_features.norm(dim=-1, keepdim=True))

    def forward(self, images, return_uncertainty=False, return_ratio=False):
        # Extract image features
        image_features = self.image_encoder(images)
        
        # Adapt features
        x = self.adapter(image_features)
        
        # Compute dynamic ratio per sample
        dynamic_ratio = self.ratio_mlp(image_features)  # Shape: [batch_size, 1]
        
        # Blend with dynamic ratio
        image_features = dynamic_ratio * x + (1 - dynamic_ratio) * image_features
        
        # Normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        # Estimate uncertainty for each sample
        uncertainty = self.uncertainty_head(image_features)  # Shape: [batch_size, 1]
        
        # Compute base logits
        logit_scale = self.logit_scale.exp()
        logits = logit_scale * image_features @ self.text_features.t()
        
        # Apply class-specific temperature scaling
        # Temperature > 1: softer predictions, Temperature < 1: sharper predictions
        logits = logits / self.temperature.unsqueeze(0)  # Shape: [batch_size, num_classes]
        
        # Apply uncertainty modulation
        # High uncertainty reduces confidence across all classes
        certainty = 1 - uncertainty  # Convert to certainty (1=certain, 0=uncertain)
        logits = logits * certainty  # Dampens logits for uncertain samples

        # Return based on flags
        if return_uncertainty and return_ratio:
            return logits, uncertainty, self.temperature, dynamic_ratio
        elif return_uncertainty:
            return logits, uncertainty, self.temperature
        elif return_ratio:
            return logits, dynamic_ratio
        return logits


# ============================================================
# ---------------- ADAPTER FUNCTIONS -------------------------
# ============================================================

def load_adapter_model():
    """
    Load and initialize the adapter model once.
    Returns the custom model and preprocessing function.
    """
    with open(PATH_FETALCLIP_CONFIG, "r") as f:
        config_fetalclip = json.load(f)

    open_clip.factory._MODEL_CONFIGS["FetalCLIP"] = config_fetalclip

    model, _, preprocess = open_clip.create_model_and_transforms(
        "FetalCLIP",
        pretrained=PATH_FETALCLIP_WEIGHT
    )

    model.eval().to(DEVICE)

    # Create Custom Model
    text_prompts = [f"an image of {cls}" for cls in CLASSNAMES]

    custom_model = CustomFetalCLIP(
        model,
        DEVICE,
        text_prompts,
        preprocess
    ).to(DEVICE)

    checkpoint = torch.load(ADAPTER_CHECKPOINT_PATH, map_location=DEVICE)
    custom_model.load_state_dict(checkpoint["model_state_dict"])
    custom_model.eval()

    print("Adapter model loaded successfully.\n")
    
    return custom_model, preprocess


def adapter_predict(custom_model, preprocess, image_pil):
    """
    Run adapter prediction on a single PIL image.
    
    Args:
        custom_model: The loaded CustomFetalCLIP model
        preprocess: Preprocessing function from open_clip
        image_pil: PIL Image object
    
    Returns:
        dict with predicted_class, confidence_score, uncertainty_value
    """
    with torch.no_grad():
        input_tensor = preprocess(image_pil).unsqueeze(0).to(DEVICE)

        # Request uncertainty
        logits, uncertainty, _ = custom_model(
            input_tensor,
            return_uncertainty=True
        )

        probs = torch.softmax(logits, dim=-1)
        confidence, prediction = torch.max(probs, dim=1)

        predicted_class = CLASSNAMES[prediction.item()]
        confidence_score = confidence.item()
        uncertainty_value = uncertainty.item()

        return {
            "predicted_class": predicted_class,
            "confidence_score": round(confidence_score, 4),
            "uncertainty_value": round(uncertainty_value, 4)
        }



def main():
    print("Starting inference with DISCRIMINATIVE features and HEATMAP generation...")
    print("=" * 70)
    
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    # Create subdirectory for heatmaps
    heatmap_dir = os.path.join(Config.OUTPUT_DIR, "heatmaps")
    os.makedirs(heatmap_dir, exist_ok=True)

    structure_csv_path = os.path.join(
        Config.OUTPUT_DIR,
        "structure_pixel_confidence.csv"
    )

    import csv
    with open(structure_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "image_name",
            "structure_name",
            "mean_confidence",
            "num_pixels"
        ])

    print(f"Structure CSV will be saved progressively to:")
    print(structure_csv_path)
    print("=" * 70)

    print("\nLoading segmentation model...")
    sam_model = sam_model_registry["vit_b"](checkpoint=None)
    seg_model = MedSAMFineTune(sam_model, num_classes=Config.NUM_CLASSES)

    checkpoint = torch.load(Config.BEST_CHECKPOINT, map_location=Config.DEVICE)
    seg_model.load_state_dict(checkpoint["model_state_dict"])
    seg_model = seg_model.to(Config.DEVICE).eval()

    print(f"Segmentation model loaded from {Config.BEST_CHECKPOINT}")

    print("\nLoading adapter model...")
    adapter_model, adapter_preprocess = load_adapter_model()


    test_data = CerebellumInferDataset(Config.TEST_IMAGES)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False)

    print(f"Found {len(test_data)} test images")
    print("=" * 70)

    results = []
    adapter_results = []
    stats = {"standard": 0, "non-standard": 0}

    # ============================================================
    # Inference Loop
    # ============================================================
    for idx, (images, original_image, img_path, size) in enumerate(test_loader):

        image_filename = os.path.basename(img_path[0])
        fname = os.path.splitext(image_filename)[0]

        print(f"\n[{idx + 1}/{len(test_data)}] {image_filename}")

        original_image = original_image.squeeze(0).numpy()
        size = size.squeeze(0).numpy()

        # ============================================================
        # Run Segmentation Prediction
        # ============================================================
        (
            pred_masks,
            probs_numpy,
            seg_logits_numpy,
            plane_pred,
            plane_conf,
            struct_conf,
            standard_conf,
            is_standard,
            plane_margin,
            #completeness,
            uniformity,
            distribution,
            edge_penalty,
            num_structures,
            total_coverage
        ) = predict(seg_model, images, debug=True)

        # Update stats
        if is_standard:
            stats["standard"] += 1
        else:
            stats["non-standard"] += 1

        pred_masks = postprocess_masks(pred_masks)

        # ============================================================
        # Run Adapter Prediction
        # ============================================================
        # Convert numpy array back to PIL Image for adapter
        image_pil = Image.fromarray(original_image)
        
        adapter_result = adapter_predict(adapter_model, adapter_preprocess, image_pil)
        
        print(f"  Adapter → Pred: {adapter_result['predicted_class']} | "
              f"Conf: {adapter_result['confidence_score']:.4f} | "
              f"Unc: {adapter_result['uncertainty_value']:.4f}")

        # ============================================================
        # Save contour overlay
        # ============================================================
        overlay = draw_contours_on_image(
            original_image,
            pred_masks,
            size,
            plane_pred,
            plane_conf,
            #completeness,
            uniformity,
            is_standard,
            standard_conf
        )





        # ============================================================
        # Image-level summary (includes both models)
        # ============================================================
        results.append({
            "image_path": img_path[0],
            "predicted_plane": Config.PLANE_ID_TO_NAME.get(plane_pred, "Unknown"),
            "plane_confidence": round(plane_conf, 4),
            "structure_confidence": round(struct_conf, 4),
            "uniformity": round(uniformity, 4),
            "distribution": round(distribution, 4),
            "edge_penalty": round(edge_penalty, 4),
            "num_structures": int(num_structures),
            "coverage": round(total_coverage, 4),
            "standard_confidence": round(standard_conf, 4),
            "standard_plane": "standard" if is_standard else "non-standard",
            # Adapter results
            # "adapter_predicted_class": adapter_result['predicted_class'],
            "adapter_confidence": adapter_result['confidence_score'],
            "adapter_uncertainty": adapter_result['uncertainty_value']
        })


    df = pd.DataFrame(results)

    csv_results_path = os.path.join(
        Config.OUTPUT_DIR,
        "standard_plane_predictions_with_adapter.csv"
    )

    df.to_csv(csv_results_path, index=False)


    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Standard planes: {stats['standard']}")
    print(f"Non-standard planes: {stats['non-standard']}")
    print(f"Total images: {len(test_data)}")
    print("=" * 70)

    print(f"\n✅ Image-level results (with adapter) saved to: {csv_results_path}")
    print(f"✅ Structure-level CSV updated progressively at: {structure_csv_path}")
    print(f"✅ Heatmaps saved to: {heatmap_dir}\n")



if __name__ == "__main__":
    main()