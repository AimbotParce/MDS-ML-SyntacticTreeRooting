from typing import Dict

import torch
import torch.nn.functional as F


class WeightedMSE:
    def __init__(self, class_weights: Dict[int, float]):
        """
        Weighted Mean Squared Error Loss
        Args:
            class_weights (list): List of class weights.
        """
        self.class_weights = class_weights

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Weighted Mean Squared Error Loss
        """
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        weights = torch.tensor([self.class_weights[int(y)] for y in y_true], dtype=torch.float32).to(y_pred.device)
        loss = F.mse_loss(y_pred, y_true, reduction="none")
        return (loss * weights).mean()


class WeightedCrossEntropy:
    def __init__(self, class_weights: Dict[int, float]):
        """
        Weighted Categorical Cross Entropy Loss
        Args:
            class_weights (list): List of class weights.
        """
        self.class_weights = class_weights

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Weighted Categorical Cross Entropy Loss
        """
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        weights = torch.tensor([self.class_weights[int(y)] for y in y_true], dtype=torch.float32).to(y_pred.device)
        loss = F.cross_entropy(y_pred, y_true, reduction="none")
        return (loss * weights).mean()
