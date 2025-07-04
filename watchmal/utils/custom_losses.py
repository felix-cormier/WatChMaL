import torch
import torch.nn as nn
import torch.nn.functional as F

class OutsidePenaltyHuberLoss(nn.Module):
    def __init__(self, delta=1.0, penalty_weight_3rd=1.0, penalty_weight_radius=1.0):
        super().__init__()
        self.delta = delta
        self.penalty_weight_3rd = penalty_weight_3rd
        self.penalty_weight_radius = penalty_weight_radius
        self.huber_loss = nn.HuberLoss(delta=delta)

    def forward(self, input, target):
        # Compute the base Huber loss
        base_loss = self.huber_loss(input, target)

        # Penalty for 3rd element (index 2) exceeding abs(1810)
        third_element = input[:, 2]
        penalty_3rd = torch.abs(torch.abs(third_element) - 1810)
        penalty_3rd = penalty_3rd * (torch.abs(third_element) > 1810)

        # Penalty for radius formed by 0th and 1st elements exceeding 1690
        radius = torch.sqrt(input[:, 0]**2 + input[:, 1]**2)
        penalty_radius = torch.abs(radius - 1690)
        penalty_radius = penalty_radius * (radius > 1690)

        # Weighted average penalties
        penalty_term_3rd = self.penalty_weight_3rd * penalty_3rd.mean()
        penalty_term_radius = self.penalty_weight_radius * penalty_radius.mean()
        # Total loss
        total_loss = base_loss + penalty_term_3rd + penalty_term_radius
        if penalty_term_3rd>0. or penalty_term_radius > 0.:
            print(f"total loss: {total_loss}, base_loss: {base_loss}, penalty z: {penalty_term_3rd}, penalty r: {penalty_term_radius}")
        return total_loss