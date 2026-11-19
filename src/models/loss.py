"""
Quantile (Pinball) Loss Function with High-Pollution Sample Weighting.
Implements PRD Section 7.2:
- Pinball loss across 10th, 50th (median), and 90th percentiles
- Double penalty weight on hours with Poor or worse pollution levels
"""

from typing import List, Optional
import torch
import torch.nn as nn


class WeightedPinballLoss(nn.Module):
    """
    Computes Asymmetric Pinball Loss for multi-quantile regression:
        L_q(y, y_hat) = max(q * (y - y_hat), (q - 1) * (y - y_hat))
    With customizable penalty multipliers for severe pollution episodes.
    """

    def __init__(
        self,
        quantiles: List[float] = [0.10, 0.50, 0.90],
        high_pollution_weight: float = 2.0,
        high_pollution_threshold_norm: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("quantiles", torch.tensor(quantiles, dtype=torch.float32))
        self.high_pollution_weight = high_pollution_weight
        self.high_pollution_threshold_norm = high_pollution_threshold_norm

    def forward(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            y_pred: Predicted quantiles of shape (N, horizon, n_targets, n_quantiles)
            y_true: Ground truth tensor of shape (N, horizon, n_targets)
        Returns:
            Scalar weighted pinball loss
        """
        # Expand y_true to match quantiles: (N, horizon, n_targets, 1)
        y_true_expanded = y_true.unsqueeze(-1)
        # Residuals: (N, horizon, n_targets, n_quantiles)
        diff = y_true_expanded - y_pred

        # Quantile tensor: shape (1, 1, 1, n_quantiles)
        q = self.quantiles.view(1, 1, 1, -1)

        # Pinball loss per element: max(q * diff, (q - 1) * diff)
        pinball = torch.maximum(q * diff, (q - 1.0) * diff)

        # Apply high pollution sample weighting
        if self.high_pollution_weight > 1.0:
            # Check where ground truth exceeds threshold (normalized concentration)
            high_mask = (y_true_expanded > self.high_pollution_threshold_norm).float()
            sample_weights = 1.0 + (self.high_pollution_weight - 1.0) * high_mask
            pinball = pinball * sample_weights

        return torch.mean(pinball)
