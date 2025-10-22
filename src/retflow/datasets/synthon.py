import copy
from dataclasses import dataclass
from typing import Tuple

from torch_geometric.loader import DataLoader

from retflow import config
from retflow.datasets.data.uspto_synthon import SynthonUSPTO
                                            
from retflow.datasets.info import SYNTHON_NAMES, RetrosynthesisInfo
from retflow.datasets.retro import RetroDataset
from retflow.runner import DistributedHelper


@dataclass
class SynthonDataset(RetroDataset):
    def load(self, dist_helper: DistributedHelper | None = None):
        if self.name not in SYNTHON_NAMES:
            raise ValueError(f"{self.name} is not any of the RetroSynthesis Datasets.")

        save_dir = config.get_dataset_directory() / self.name

        if self.name == "SynthonUSPTO":
            train_dataset = SynthonUSPTO(split="train", root=str(save_dir))
            val_dataset = SynthonUSPTO(split="val", root=str(save_dir))
        else:
            raise NotImplementedError

        train_loader, val_loader = self._get_train_and_val_loaders(
            train_dataset, val_dataset, dist_helper
        )
        self.info = self._get_info(train_dataset, val_dataset)

        self.train_smiles = train_dataset.r_smiles
        return train_loader, val_loader, self.info

    def download(self) -> None:
        save_dir = config.get_dataset_directory() / self.name
        for split in ["train", "val", "test"]:
            if self.name == "SynthonUSPTO":
                SynthonUSPTO(
                    split=split, root=str(save_dir), download_and_process=True
                )

    def load_eval(self, load_valid=False) -> Tuple[DataLoader | RetrosynthesisInfo]:
        if self.name not in SYNTHON_NAMES:
            raise ValueError(f"{self.name} is not any of the RetroSynthesis Datasets.")

        save_dir = config.get_dataset_directory() / self.name

        if self.name == "SynthonUSPTO":
            test_dataset = SynthonUSPTO(
                split="val" if load_valid else "test", root=str(save_dir)
            )
            self.info = self._get_info(
                SynthonUSPTO(split="train", root=str(save_dir)),
                SynthonUSPTO(split="val", root=str(save_dir)),
            )
        else:
            raise NotImplementedError

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        return test_loader, self.info

    def _compute_input_output_dims(self, max_n_nodes, train_dataset):
        input_dim, output_dim = super()._compute_input_output_dims(
            max_n_nodes, train_dataset
        )

        ex = train_dataset[0]

        # for the extra context product molecule (it has the same dimension as the reactant molecule)
        input_dim.node_dim += ex.x.shape[1]
        input_dim.edge_dim += ex.edge_attr.shape[1]

        return input_dim, output_dim


@dataclass
class ToySynthonDataset(RetroDataset):
    num_molecules: int = 2

    def load(self, dist_helper: DistributedHelper | None = None):
        if self.name not in SYNTHON_NAMES:
            raise ValueError(f"{self.name} is not any of the RetroSynthesis Datasets.")

        save_dir = config.get_dataset_directory() / self.name

        if self.name == "SynthonUSPTO":
            train_dataset = SynthonUSPTO(split="train", root=str(save_dir))[0: self]
            val_dataset = copy.deepcopy(train_dataset)
        else:
            raise NotImplementedError

        train_loader, val_loader = self._get_train_and_val_loaders(
            train_dataset, val_dataset, dist_helper=dist_helper
        )
        self.info = self._get_info(train_dataset, val_dataset)
        self.train_smiles = train_dataset.r_smiles
        return train_loader, val_loader, self.info
