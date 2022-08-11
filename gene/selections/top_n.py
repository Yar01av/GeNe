from typing import List, Tuple

from torch import nn

from gene.selections.base import BaseSelection


class TopNSelection(BaseSelection):
    def __init__(self, n):
        self._n = n

    def __call__(self, population: List[Tuple[nn.Module, float]]) -> List[nn.Module]:
        best_models = sorted(population, key=lambda x: x[1])[:self._n]

        return [model for model, _ in best_models]
