from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from gene.selections.base import BaseSelection


# TODO: measure inequality with GINI index
class LossProportionalSelection(BaseSelection):
    """
    Select the models with a probability proportional to the loss.
    """

    def __init__(self, n, allow_duplicates=False):
        self._n = n
        self._allow_duplicates = allow_duplicates

    def __call__(self, population: List[Tuple[nn.Module, float]]) -> List[nn.Module]:
        n_models = len(population)
        models, losses = zip(*population)
        loss_total = torch.sum(torch.tensor(losses))
        normalised_losses = torch.tensor(losses) / loss_total

        # Sample proportionally to the loss n times using NumPy.
        sampled_indices = np.random.choice(
            n_models,
            self._n,
            p=normalised_losses.detach().numpy(),
            replace=self._allow_duplicates or n_models < self._n
        )

        return [models[i] for i in sampled_indices]
