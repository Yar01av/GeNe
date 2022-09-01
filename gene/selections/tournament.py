from typing import List, Tuple

import numpy as np
import torch
from torch import nn

from gene.selections.base import BaseSelection


class TournamentSelection(BaseSelection):
    """
    Selects the models that are the best in their respective sample of the population.
    """

    def __init__(self, n_samples, samples_size, allow_duplicates=False):
        self._n_samples = n_samples
        self._samples_size = samples_size
        self._allow_duplicates = allow_duplicates

    def __call__(self, population: List[Tuple[nn.Module, float]]) -> List[nn.Module]:
        n_models = len(population)
        models, losses = zip(*population)
        losses = torch.tensor(losses)

        samples = np.random.choice(
            n_models,
            (self._n_samples, self._samples_size),
            replace=self._allow_duplicates or n_models < self._samples_size
        )
        losses_samples = losses.gather(samples)
        best_indices = torch.argmin(losses_samples, dim=1)

        return [models[i] for i in best_indices]
