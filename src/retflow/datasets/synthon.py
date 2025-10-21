import copy
from dataclasses import dataclass
from typing import Tuple

from torch_geometric.loader import DataLoader

from retflow import config
from retflow.datasets.data.uspto_synthon import (MultiSynthonProductUSPTO,
                                                 MultiSynthonUSPTO)
from retflow.datasets.info import SYNTHON_NAMES, RetrosynthesisInfo
from retflow.datasets.retro import RetroDataset
from retflow.runner import DistributedHelper
from retflow.utils import ExtraMolecularFeatures, to_dense


@dataclass
class SynthonDataset(RetroDataset):
    def load(self, dist_helper: DistributedHelper | None = None):
        if self.name not in SYNTHON_NAMES:
            raise ValueError(f"{self.name} is not any of the RetroSynthesis Datasets.")

        save_dir = config.get_dataset_directory() / self.name

        if self.name == "MultiSynthonUSPTO":
            train_dataset = MultiSynthonUSPTO(split="train", root=str(save_dir))
            val_dataset = MultiSynthonUSPTO(split="val", root=str(save_dir))
        elif self.name == "MultiSynthonProductUSPTO":
            train_dataset = MultiSynthonProductUSPTO(split="train", root=str(save_dir))
            val_dataset = MultiSynthonProductUSPTO(split="val", root=str(save_dir))
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
            if self.name == "MultiSynthonUSPTO":
                MultiSynthonUSPTO(
                    split=split, root=str(save_dir), download_and_process=True
                )
            elif self.name == "MultiSynthonProductUSPTO":
                MultiSynthonProductUSPTO(
                    split=split, root=str(save_dir), download_and_process=True
                )

    def load_eval(self, load_valid=False) -> Tuple[DataLoader | RetrosynthesisInfo]:
        if self.name not in SYNTHON_NAMES:
            raise ValueError(f"{self.name} is not any of the RetroSynthesis Datasets.")

        save_dir = config.get_dataset_directory() / self.name

        if self.name == "MultiSynthonUSPTO":
            test_dataset = MultiSynthonUSPTO(
                split="val" if load_valid else "test", root=str(save_dir)
            )

            self.info = self._get_info(
                MultiSynthonUSPTO(split="train", root=str(save_dir)),
                MultiSynthonUSPTO(split="val", root=str(save_dir)),
            )
        elif self.name == "MultiSynthonProductUSPTO":
            test_dataset = MultiSynthonProductUSPTO(
                split="val" if load_valid else "test", root=str(save_dir)
            )
            self.info = self._get_info(
                MultiSynthonProductUSPTO(split="train", root=str(save_dir)),
                MultiSynthonProductUSPTO(split="val", root=str(save_dir)),
            )
        else:
            raise NotImplementedError

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size)
        return test_loader, self.info

    def _compute_input_output_dims(self, max_n_nodes, train_dataset):
        input_dim, output_dim = super()._compute_input_output_dims(
            max_n_nodes, train_dataset
        )
        if self.name in ["SynthonUSPTO", "MultiSynthonUSPTO"]:
            return input_dim, output_dim

        ex = train_dataset[0]

        if self.name in ["ReactionCenterUSPTO"]:
            r_ex_dense, r_node_mask = to_dense(ex.x, ex.edge_index, ex.edge_attr, ex.batch)
            extra_mol_f = ExtraMolecularFeatures()
            extra_mol_data = extra_mol_f(r_ex_dense.X, r_ex_dense.E, r_node_mask)
            input_dim.node_dim += extra_mol_data.X.shape[-1]
            input_dim.y_dim += extra_mol_data.y.shape[-1]
            return input_dim, output_dim
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

        train_dataset = MultiSynthonUSPTO(split="train", root=str(save_dir))[
            0 : self.num_molecules
        ]
        val_dataset = copy.deepcopy(train_dataset)

        train_loader, val_loader = self._get_train_and_val_loaders(
            train_dataset, val_dataset, dist_helper=dist_helper
        )
        self.info = self._get_info(train_dataset, val_dataset)
        self.train_smiles = train_dataset.r_smiles
        return train_loader, val_loader, self.info
