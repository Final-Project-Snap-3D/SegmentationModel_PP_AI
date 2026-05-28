import torch
import torch.nn as nn


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5, eps=1e-7):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.eps = eps

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)

        prob = torch.sigmoid(pred)
        intersection = (prob * target).sum(dim=(1, 2, 3))
        dice = (2 * intersection + self.eps) / (
            prob.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + self.eps
        )
        dice_loss = (1 - dice).mean()

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
