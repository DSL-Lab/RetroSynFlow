from abc import ABC, abstractmethod
from typing import List, Tuple
from torch import Tensor


class SamplingMetric(ABC):
    @abstractmethod
    def update(self, molecules: List[Tuple[Tensor, Tensor]], real: bool):
        pass

    @abstractmethod
    def compute(self, ddp: bool = False):
        pass

    @abstractmethod
    def reset(self):
        pass
