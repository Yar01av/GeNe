from copy import deepcopy
from typing import List

import numpy as np
import torch

from gene.optimisers.base import Optimiser
from torch import nn, no_grad


class DivisionOptimiser(Optimiser):
    def __init__(self, loss, random_function, selection_limit=10, division_factor=2):
        """
        :param loss: A function that takes an outputs of the model and true values.
        :param random_function: A function that takes produces a tensor of the given shape (as tuple) filled with random
                                values.
        :param selection_limit: Maximum number of models that remains after removing the worst-performing ones.
        :param division_factor: How many offsprings does a model have.
        """

        self._loss = loss
        self._selection_limit = selection_limit
        self._division_factor = division_factor
        self._random_function = random_function
        self._scores = None

    def get_scores(self) -> List[torch.Tensor]:
        return deepcopy(self._scores)

    @no_grad()
    def step(self, models: List[nn.Module], X, y_true) -> List[nn.Module]:
        # Mutate the models
        models_to_mutate = models * self._division_factor
        models = [self._mutate(model) for model in models_to_mutate] + models

        # Keep only the best models
        models_with_score = [(model, self._loss(model(X), y_true)) for model in models]
        models_with_score = sorted(models_with_score, key=lambda x: x[1])
        models_with_score = models_with_score[:self._selection_limit]

        self._scores = [score for model, score in models_with_score]

        return [model for model, score in models_with_score]

    def _mutate(self, model: nn.Module) -> nn.Module:
        original_model = deepcopy(model)
        params = original_model.parameters()

        for param in params:
            param.data = param.data + self._random_function(param.data.shape)

        # TODO: implement mutation

        return original_model
