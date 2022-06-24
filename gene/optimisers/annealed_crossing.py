from typing import List

import torch
from torch import no_grad, nn

from gene.optimisers.base import Optimiser
from gene.optimisers.crossing import CrossingOptimiser


class AnnealedCrossingOptimiser(Optimiser):
    def __init__(self,
                 target_func,
                 init_std,
                 std_updater,
                 selection_limit=10,
                 max_couples=1,
                 n_children_per_couple=1,
                 keep_parents=True,
                 device="cpu"):
        """
        Optimiser that relies on simulated annealing, where the new models are created through crossing,
        to improve the weights of the networks.

        :param init_std: The initial standard deviation of the mutations.
        :param std_updater: A function that takes the current standard deviation and returns a new one. This is done
        after a new generation of models is produced.
        :param target_func: A function that takes an outputs of the model and true values and returns a target value.
                            The smaller is assumed to be better.
        :param selection_limit: Maximum number of models that remains after removing the worst-performing ones.
        :param n_children_per_couple: How many offsprings does a model have.
        :param max_couples: At most how many couples should be formed. The values is between 1 and
        len(random_functions) choose 2.
        :param keep_parents: Should the parents compete with the children for survival.
        :param device: Which torch device should be used to generate the new models. This effects what models can be
        passed to .step().
        """

        self._std = init_std
        self._std_updater = std_updater
        self._div_optimiser = CrossingOptimiser(target_func=target_func,
                                                random_function=lambda shape: torch.normal(0, self._std, shape),
                                                selection_limit=selection_limit,
                                                max_couples=max_couples,
                                                n_children_per_couple=n_children_per_couple,
                                                keep_parents=keep_parents,
                                                device=device)

    def step(self, models: List[nn.Module], X, y_true) -> List[nn.Module]:
        new_models = self._div_optimiser.step(models, X, y_true)
        self._std = self._std_updater(self._std)

        return new_models
