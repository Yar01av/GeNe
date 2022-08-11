from abc import abstractmethod, ABC
from typing import List, Tuple

from torch import nn


class BaseSelection(ABC):
    """
    Base class for all selection algorithms.
    """

    @abstractmethod
    def __call__(self, population: List[Tuple[nn.Module, float]]) -> List[nn.Module]:
        pass
