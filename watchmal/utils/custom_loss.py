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

        # Penalty for |input[:, 2]| > 1810 (quadratic)
        third_element = input[:, 2]
        excess_3rd = torch.abs(third_element) - 1810
        penalty_3rd = (excess_3rd.clamp(min=0))**2  # square the positive excess

        # Penalty for sqrt(x^2 + y^2) > 1690 (quadratic)
        radius = torch.sqrt(input[:, 0]**2 + input[:, 1]**2)
        excess_radius = radius - 1690
        penalty_radius = (excess_radius.clamp(min=0))**2  # square the positive excess

        # Mean penalties
        penalty_term_3rd = self.penalty_weight_3rd * penalty_3rd.mean()
        penalty_term_radius = self.penalty_weight_radius * penalty_radius.mean()

        total_loss = base_loss + penalty_term_3rd + penalty_term_radius
        if penalty_term_3rd>0. or penalty_term_radius > 0.:
            print(f"excess 3rd: {penalty_3rd}, mask radius: {torch.abs(third_element) > 1810}")
            print(f"excess radius: {penalty_radius}, mask 3rd: {radius > 1690}")
            print(f"total loss: {total_loss}, base_loss: {base_loss}, penalty z: {penalty_term_3rd}, penalty r: {penalty_term_radius}")
        return total_loss


class RelativeHuberLoss(nn.Module):
    def __init__(self, delta=1.0, epsilon=1e-8):
        super().__init__()
        self.delta = delta
        self.epsilon = epsilon

    def forward(self, input, target):
        relative_error = (input - target) / (torch.abs(target) + self.epsilon)
        abs_err = torch.abs(relative_error)
        loss = torch.where(
            abs_err <= self.delta,
            0.5 * relative_error ** 2,
            self.delta * (abs_err - 0.5 * self.delta)
        )
        return loss.mean()


class ThresholdScaledHuberLoss(nn.Module):
    def __init__(self, delta=0.2, threshold=100.0, k=5.0):
        """
        Huber loss that scales by k for samples whose energy (or target, if
        energy is not provided) falls below a fixed threshold value.

        Args:
            delta:     Huber loss transition point.
            threshold: Fixed value below which the loss is scaled. Compared
                       against energy when provided, otherwise against target.
            k:         Scale factor applied to losses where the value < threshold.
        """
        super().__init__()
        self.delta = delta
        self.threshold = threshold
        self.k = k

    def forward(self, input, target, energy=None):
        abs_err = torch.abs(input - target)
        loss = torch.where(
            abs_err <= self.delta,
            0.5 * (input - target) ** 2,
            self.delta * (abs_err - 0.5 * self.delta)
        )
        # Use energy for threshold comparison when provided, otherwise fall back to target
        if energy is not None:
            reference = energy.squeeze(-1) if energy.dim() > 1 and energy.shape[-1] == 1 else energy
            # broadcast to match loss shape if needed
            if reference.shape != loss.shape:
                reference = reference.unsqueeze(-1).expand_as(loss)
        else:
            reference = target
        scale = torch.where(reference < self.threshold, torch.full_like(loss, self.k), torch.ones_like(loss))
        return (scale * loss).mean()