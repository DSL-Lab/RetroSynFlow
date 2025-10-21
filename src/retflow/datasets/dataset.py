from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple

from torch.utils.data import DataLoader

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
    def download(self) -> None:
        pass

    @abstractmethod
    def load_eval(self, load_valid=False) -> Tuple[DataLoader, RetrosynthesisInfo]:
        pass
