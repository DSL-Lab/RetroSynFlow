from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict
from retflow.models import Model
from retflow.datasets import Dataset
from retflow.optimizers import Optimizer
from retflow.methods import Method
from retflow.runner.distributed_helper import DistributedHelper

import torch


@dataclass
class Problem(ABC):
    model: Model
    dataset: Dataset
    method: Method

    @abstractmethod
    def setup_problem(self, dist_helper: DistributedHelper | None):
        pass

    @abstractmethod
    def get_optimizer(self, optimizer: Optimizer) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def one_epoch(
        self,
        optim: torch.optim.Optimizer,
        sched: torch.optim.lr_scheduler._LRScheduler,
        grad_clip: float | None,
        dist_helper: DistributedHelper | None,
    ) -> Dict[str, float]:
        pass

    @abstractmethod
    def validation(self, dist_helper: DistributedHelper | None):
        pass

    @abstractmethod
    def sample_generation(
        self,
        num_samples: int,
        examples_per_sample: int,
        dist_helper: DistributedHelper | None = None,
    ):
        pass

    @abstractmethod
    def setup_problem_eval(self, model_checkpoint, on_valid: bool = False):
        pass

    @abstractmethod
    def sample_generation_eval(self, examples_per_sample: int):
        pass
