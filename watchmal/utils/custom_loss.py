import torch
import torch.nn as nn
import torch.nn.functional as F

import math

class PositionHuberLoss(nn.Module):
    def __init__(self, delta=1.0, reduction='mean', weight=None):
        super().__init__()
        self.delta = delta
        self.reduction = reduction
        self.weight = weight

    def forward(self, input, target):
        # Calculate the element-wise absolute error
        abs_error = torch.abs(input - target)
        above_z = math.fabs(input[:,2])-1810
        if above_z < 0:
            above_z = 0

        # Apply the Huber loss logic
        loss = torch.where(abs_error <= self.delta,
                           0.5 * (abs_error ** 2),
                           self.delta * abs_error - 0.5 * (self.delta ** 2))

        # Apply optional weighting
        if self.weight is not None:
            loss = loss * self.weight

        # Apply reduction
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        elif self.reduction == 'none':
            return loss
        else:
            raise ValueError("Reduction type not supported.")