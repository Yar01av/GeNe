from abc import ABC, abstractmethod
from typing import List, Callable

import torch
from torch import nn


class Optimiser(ABC):
    @abstractmethod
    def step(self, models: List[nn.Module], loss_function: Callable[[nn.Module], float]) -> List[nn.Module]:
        """

        :type models: List[nn.Module]. A list of models to be optimised.
        :type loss_function: Callable[[nn.Module], float]. A function that takes a model and returns a loss value.
        """
        pass
