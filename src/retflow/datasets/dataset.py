from abc import ABC, abstractmethod
from typing import Tuple, List
from dataclasses import dataclass

from torch.nn import Module
from torch.utils.data import DataLoader
from retflow.metrics.metric import SamplingMetric
from retflow.datasets.info import RetrosynthesisInfo
from retflow.runner import DistributedHelper


@dataclass
class Dataset(ABC):
    name: str
    batch_size: int

    @abstractmethod
    def load(
        self, dist_helper: DistributedHelper | None = None
    ) -> Tuple[DataLoader, DataLoader, RetrosynthesisInfo]:
        pass

    @abstractmethod
    def get_metrics(self) -> List[SamplingMetric]:
        pass

    @abstractmethod
    def download(self) -> None:
        pass

    @abstractmethod
    def load_eval(self, load_valid=False) -> Tuple[DataLoader, RetrosynthesisInfo]:
        pass
