from abc import ABC, abstractmethod
from typing import List

import torch
from torch import nn


class Optimiser(ABC):
    @abstractmethod
    def step(self, models: List[nn.Module], X, y_true) -> List[nn.Module]:
        pass

    @abstractmethod
    def get_losses(self) -> List[torch.Tensor]:
        pass
