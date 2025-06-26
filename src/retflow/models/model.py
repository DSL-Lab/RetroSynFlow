from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import torch

from retflow.retro_utils import GraphDimensions


@dataclass()
class Model(ABC):
    """Abstract base class for defining models"""

    @abstractmethod
    def load_model(
        self,
        input_shape: GraphDimensions,
        output_shape: GraphDimensions,
        reduce_output: bool,
        checkpoint: Path | None = None,
    ) -> torch.nn.Module:
        pass
