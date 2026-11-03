"""AlphaPulse v2.0 - Base Abstract Deep Learning Architecture."""

from __future__ import annotations

import abc
import torch
import torch.nn as nn


class BaseStockModel(nn.Module, abc.ABC):
    """Abstract base class for all neural forecasting architectures in the Model Zoo."""

    @abc.abstractmethod
    def model_type(self) -> str:
        """Returns the canonical string identifier of the model architecture."""
        pass

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
