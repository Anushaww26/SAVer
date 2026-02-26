import torch
import torch.nn as nn
import torch.nn.functional as F
from config.config import Config

class PlaneAwareDiceFocalPresenceLoss(nn.Module):
    def __init__(
        self,
        dice_weight=1.0,
        focal_weight=1.0,
        hausdorf_weight=0.01,
        presence_weight=0.2,
        gamma=2.0,
        area_priors=Config.AREA_PRIORS,
        plane_structure_map=Config.PLANE_STRUCTURE_MAP,
        structure_idx=Config.STRUCTURE_IDX,
        beta=10.0
    ):
        super().__init__()

        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.hausdorf_weight = hausdorf_weight
        self.presence_weight = presence_weight
        self.gamma = gamma
        self.beta = beta

        self.bce = nn.BCEWithLogitsLoss(reduction="none")

        self.area_priors = area_priors
        self.plane_structure_map = plane_structure_map
        self.structure_idx = structure_idx

        # learnable slack per structure
        self.delta = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(1))
            for k in area_priors.keys()
        })

    def soft_hausdorf(self, prob, gt, eps=1e-6):
        prob = torch.clamp(prob, eps, 1 - eps)
        gx_p = torch.abs(prob[:, 1:] - prob[:, :-1])
        gy_p = torch.abs(prob[1:, :] - prob[:-1, :])
        gx_g = torch.abs(gt[:, 1:] - gt[:, :-1])
        gy_g = torch.abs(gt[1:, :] - gt[:-1, :])
        return torch.abs((gx_p.mean() + gy_p.mean()) -
                         (gx_g.mean() + gy_g.mean()))

    def forward(self, seg_logits, seg_targets, plane_labels):
        B, C, H, W = seg_logits.shape
        device = seg_logits.device

        total_dice = total_focal = total_haus = total_presence = 0.0
        valid_batches = 0

        prob_maps = torch.sigmoid(seg_logits)

        for b in range(B):
            plane = plane_labels[b].item()
            valid_classes = Config.PLANE_CLASS_MAP[plane]

            if len(valid_classes) == 0:
                continue

            pred = seg_logits[b, valid_classes]
            gt = seg_targets[b, valid_classes]
            prob = prob_maps[b, valid_classes]

            # -------- Dice --------
            intersection = (prob * gt).sum(dim=(1, 2))
            union = prob.sum(dim=(1, 2)) + gt.sum(dim=(1, 2))
            dice = (2 * intersection + 1e-6) / (union + 1e-6)
            total_dice += (1 - dice).mean()

            # -------- Focal --------
            bce = self.bce(pred, gt)
            p_t = torch.exp(-bce)
            focal = ((1 - p_t) ** self.gamma) * bce
            total_focal += focal.mean()

            # -------- Hausdorff --------
            haus_c, count = 0.0, 0
            for c in range(prob.shape[0]):
                if gt[c].sum() == 0:
                    continue
                haus_c += self.soft_hausdorf(prob[c], gt[c])
                count += 1
            if count > 0:
                total_haus += haus_c / count

            # -------- Presence loss --------
            required_structures = self.plane_structure_map[plane]

            for s in required_structures:
                idx = self.structure_idx[s]
                area = prob_maps[b, idx].mean()
                #tau = self.area_priors[s] + F.softplus(self.delta[s])
                area = prob_maps[:, idx].mean(dim=(1, 2))
                device = prob_maps.device

                tau = torch.tensor(
                    self.area_priors[s],
                    device=device,
                    dtype=prob_maps.dtype
                ) + self.delta[s].to(device)

                total_presence += F.softplus(self.beta * (tau - area)) / self.beta

            valid_batches += 1

        if valid_batches == 0:
            return torch.tensor(0.0, device=device)

        return (
            self.dice_weight * (total_dice / valid_batches) +
            self.focal_weight * (total_focal / valid_batches) +
            self.hausdorf_weight * (total_haus / valid_batches) +
            self.presence_weight * (total_presence / valid_batches)
        )
