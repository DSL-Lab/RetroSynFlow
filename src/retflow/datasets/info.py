from dataclasses import dataclass, field
from typing import ClassVar, Dict, List

import torch
from rdkit import Chem

from retflow.utils import GraphDimensions

SYNTHON_NAMES = [
    "MultiSynthonUSPTO",
    "MultiSynthonProductUSPTO",
    "ReactionCenterUSPTO",
    "GRSynthonUSPTO"
]

NAMES = ["USPTO", "MIT"]
DOWNLOAD_URL_TEMPLATE = "https://zenodo.org/record/8114657/files/{fname}?download=1"
USPTO_MIT_DOWNLOAD_URL = (
    "https://github.com/wengong-jin/nips17-rexgen/raw/master/USPTO/data.zip"
)

BONDS = {
    Chem.BondType.SINGLE: 0,
    Chem.BondType.DOUBLE: 1,
    Chem.BondType.TRIPLE: 2,
    Chem.BondType.AROMATIC: 3,
}

USPTO_TYPES = {
    "N": 0,
    "C": 1,
    "O": 2,
    "S": 3,
    "Cl": 4,
    "F": 5,
    "B": 6,
    "Br": 7,
    "P": 8,
    "Si": 9,
    "I": 10,
    "Sn": 11,
    "Mg": 12,
    "Cu": 13,
    "Zn": 14,
    "Se": 15,
    "*": 16,
}


USPTO_ATOM_ENCODER = {
    "N": 0,
    "C": 1,
    "O": 2,
    "S": 3,
    "Cl": 4,
    "F": 5,
    "B": 6,
    "Br": 7,
    "P": 8,
    "Si": 9,
    "I": 10,
    "Sn": 11,
    "Mg": 12,
    "Cu": 13,
    "Zn": 14,
    "Se": 15,
    "*": 16,
}

USPTO_ATOM_DECODER = [
    "N",
    "C",
    "O",
    "S",
    "Cl",
    "F",
    "B",
    "Br",
    "P",
    "Si",
    "I",
    "Sn",
    "Mg",
    "Cu",
    "Zn",
    "Se",
    "*",
]

USPTO_VALENCIES: List[int] = [5, 4, 6, 6, 7, 1, 3, 7, 5, 4, 7, 4, 2, 4, 2, 6, 0]

USPTO_ATOM_WEIGHTS: Dict[int, float] = {
    1: 14.01,
    2: 12.01,
    3: 16.0,
    4: 32.06,
    5: 35.45,
    6: 19.0,
    7: 10.81,
    8: 79.91,
    9: 30.98,
    10: 28.01,
    11: 126.9,
    12: 118.71,
    13: 24.31,
    14: 63.55,
    15: 65.38,
    16: 78.97,
    17: 0.0,
}


@dataclass
class RetrosynthesisInfo:
    input_dim: GraphDimensions
    output_dim: GraphDimensions
    n_nodes_dist: torch.Tensor
    node_types_dist: torch.Tensor
    edge_types_dist: torch.Tensor
    valency_dist: torch.Tensor
    dummy_nodes_dist: torch.Tensor
    max_n_nodes: int
    remove_h: bool = True
    max_weight: int = 1000
    num_dummy_nodes: List[int] = field(
        default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    )
    max_n_dummy_nodes = 10
    atom_encoder: ClassVar[Dict[str, int]] = USPTO_ATOM_ENCODER
    atom_decoder: ClassVar[List[str]] = USPTO_ATOM_DECODER
    valencies: ClassVar[List[int]] = USPTO_VALENCIES
    atom_weights: ClassVar[Dict[int, float]] = USPTO_ATOM_WEIGHTS
    bonds: ClassVar[Chem.BondType] = BONDS
