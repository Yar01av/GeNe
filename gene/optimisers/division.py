from copy import deepcopy
from typing import List, Union

import numpy as np
import torch
import ray
from gene.optimisers.base import Optimiser
from torch import nn, no_grad

from gene.util import split_into_batchs, flatten_2d_list


class ParallelDivisionOptimiser(Optimiser):
    def __init__(self, target_func, random_function, selection_limit=10, division_factor=2, device="cpu"):
        """
        :param target_func: A function that takes an outputs of the model and true values.
        :param random_function: A function that takes produces a tensor of the given shape (as tuple) filled with random
                                values.
        :param selection_limit: Maximum number of models that remains after removing the worst-performing ones.
        :param division_factor: How many offsprings does a model have.
        """

        self._target_func = target_func
        self._selection_limit = selection_limit
        self._division_factor = division_factor
        self._random_function = ray.put(random_function)
        self._targets = None
        self._device = device

        ray.init(ignore_reinit_error=True, num_gpus=1)
        self._remote_mutate_batch = ray.remote(num_gpus=1 if device == "cuda" else 0)\
            (lambda ms: [self._mutate(ray.get(self._random_function), m, device) for m in ms])
        self._remote_apply_model_batch = ray.remote(num_gpus=1 if device == "cuda" else 0)\
            (lambda ms, x: [m(x) for m in ms])
        self._remote_compute_loss = ray.remote(num_gpus=1 if device == "cuda" else 0)\
            (lambda pred_batch, gt: [self._target_func(pred, gt) for pred in pred_batch])

    def get_last_targets(self) -> List[torch.Tensor]:
        return deepcopy(self._targets)

    @no_grad()
    def step(self, models: List[nn.Module], X, y_true) -> List[nn.Module]:
        # Mutate the models
        models_to_mutate = models * self._division_factor
        remote_models = [self._remote_mutate_batch.remote(models)
                         for models in split_into_batchs(models_to_mutate, 16)]
        local_nested_models = ray.get(remote_models)
        local_models = flatten_2d_list(local_nested_models)
        new_models = local_models + models

        # Apply the models to the input
        remote_nested_predictions = [self._remote_apply_model_batch.remote(ms, X) for ms in split_into_batchs(new_models, 16)]
        local_nested_predictions = ray.get(remote_nested_predictions)
        local_predictions = flatten_2d_list(local_nested_predictions)

        # Compute the targets
        remote_y_true = ray.put(y_true)
        remote_nested_targets = [self._remote_compute_loss.remote(pred_batch, remote_y_true)
                                for pred_batch in split_into_batchs(local_predictions, 16)]
        local_nested_targets = ray.get(remote_nested_targets)
        local_targets = flatten_2d_list(local_nested_targets)

        # Prune the worst models away
        models_with_targets = zip(new_models, local_targets)
        models_with_targets = sorted(models_with_targets, key=lambda x: x[1])
        models_with_targets = models_with_targets[:self._selection_limit]

        self._targets = [targets.cpu().numpy() for model, targets in models_with_targets]

        return [model for model, losses in models_with_targets]

    # TODO: rewrite as a dynamic method
    @staticmethod
    def _mutate(random_function, model: nn.Module, device: str) -> nn.Module:
        original_model = deepcopy(model)
        params = original_model.parameters()

        for param in params:
            param.data = param.data + random_function(param.data.shape).to(device)

        return original_model


class DivisionOptimiser(Optimiser):
    def __init__(self,
                 target_func,
                 random_function,
                 selection_limit=10,
                 division_factor=2,
                 device="cpu",
                 direction="min"):
        """
        :param target_func: A function that takes an outputs of the model and true values and returns a target value.
                            The smaller is assumed to be better.
        :param random_function: A function that takes produces a tensor of the given shape (as tuple) filled with random
                                values.
        :param selection_limit: Maximum number of models that remains after removing the worst-performing ones.
        :param division_factor: How many offsprings does a model have.
        """

        self._target = target_func
        self._selection_limit = selection_limit
        self._division_factor = division_factor
        self._random_function = random_function
        self._last_targets = None
        self._device = device

    def get_last_targets(self) -> List[torch.Tensor]:
        return deepcopy(self._last_targets)

    @no_grad()
    def step(self, models: List[nn.Module], X, y_true) -> List[nn.Module]:
        # Mutate the models
        models_to_mutate = models * self._division_factor
        mutated_models = [self._mutate(self._random_function, model, self._device) for model in models_to_mutate]
        new_models = mutated_models + models

        # Keep only the best models
        # remote_loss = ray.remote(self._target)
        models_with_targets = [(model, self._target(model(X), y_true)) for model in new_models]
        models_with_targets = sorted(models_with_targets, key=lambda x: x[1])
        models_with_targets = models_with_targets[:self._selection_limit]

        self._last_targets = [targets.cpu().numpy() for model, targets in models_with_targets]

        return [model for model, targets in models_with_targets]

    @staticmethod
    def _mutate(random_function, model: nn.Module, device) -> nn.Module:
        original_model = deepcopy(model)
        params = original_model.parameters()

        for param in params:
            param.data = param.data + random_function(param.data.shape).to(device)

        return original_model
