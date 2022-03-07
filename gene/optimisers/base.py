from abc import ABC, abstractmethod
from typing import List


class Optimiser(ABC):
    @abstractmethod
    def evolve(self, models: List) -> List:
        pass
