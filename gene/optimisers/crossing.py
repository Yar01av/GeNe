from copy import deepcopy
from typing import List

import numpy as np
import random
from torch import nn
from itertools import combinations
from gene.optimisers.base import Optimiser
from gene.optimisers.division import DivisionOptimiser
from gene.selections.top_n import TopNSelection


class CrossingOptimiser(Optimiser):
    def __init__(self,
                 target_func,
                 random_function,
                 selection=TopNSelection(10),
                 max_couples=1,
                 n_children_per_couple=1,
                 keep_parents=True,
                 device="cpu"):
        """
        :param target_func: A function that takes an outputs of the model and true values and returns a target value.
                            The smaller is assumed to be better.
        :param random_function: A function that takes produces a tensor of the given shape (as tuple) filled with random
                                values.
        :param selection: A selection instance that takes a list of models with targets and returns a list of models.
        :param n_children_per_couple: How many offsprings does a model have.
        :param max_couples: At most how many couples should be formed. The values is between 1 and
        len(random_functions) choose 2.
        :param keep_parents: Should the parents compete with the children for survival.
        :param device: Which torch device should be used to generate the new models. This effects what models can be
        passed to .step().
        """

        self._keep_parents = keep_parents
        self._n_children_per_couple = n_children_per_couple
        self._max_couples = max_couples
        self._target_func = target_func
        self._selection = selection
        self._random_function = random_function
        self._device = device

    def step(self, models: List[nn.Module], X, y_true) -> List[nn.Module]:
        assert len(models) != 0
        if len(models) == 1:
            models = 2*models

        possible_pairs = list(combinations(models, r=2))[:self._max_couples]
        children = [self._make_child(*couple) for couple in possible_pairs for _ in range(self._n_children_per_couple)]

        # Mutate the models
        children = [self._mutate(self._random_function, model, self._device) for model in children]
        new_models = children + models if self._keep_parents else children

        # Keep only the best models
        models_with_targets = [(model, self._target_func(model(X), y_true)) for model in new_models]

        return self._selection(models_with_targets)

    @staticmethod
    def _mutate(random_function, model: nn.Module, device) -> nn.Module:
        original_model = deepcopy(model)
        params = original_model.parameters()

        for param in params:
            param.data = param.data + random_function(param.data.shape).to(device)

        return original_model

    @staticmethod
    def _make_child(parent0, parent1) -> nn.Module:
        new_model = deepcopy(parent0)

        for parent0_param, parent1_param in zip(new_model.parameters(), parent1.parameters()):
            parent0_param.data = random.choice([parent0_param.data, parent1_param.data])

        return new_model
