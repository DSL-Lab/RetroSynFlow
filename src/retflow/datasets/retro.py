from dataclasses import dataclass
from typing import List, Tuple
import copy
import os

import torch
from torch.nn import Module
from torch_geometric.loader import DataLoader
from torch.utils.data.distributed import DistributedSampler

from retflow import config
from retflow.datasets.dataset import Dataset

from retflow.retro_utils import (
    ExtraFeatures,
    GraphDimensions,
    to_dense,
)
from retflow.datasets.data.uspto import USPTO
from retflow.datasets.info import (
    NAMES,
    RetrosynthesisInfo,
)
from retflow.runner import DistributedHelper


@dataclass
class RetroDataset(Dataset):
    def load(self, dist_helper: DistributedHelper | None = None):
        if self.name not in NAMES:
            raise ValueError(f"{self.name} is not any of the RetroSynthesis Datasets.")

        save_dir = config.get_dataset_directory() / self.name

        train_dataset = USPTO(split="train", root=str(save_dir))
        val_dataset = USPTO(split="val", root=str(save_dir))

        train_loader, val_loader = self._get_train_and_val_loaders(
            train_dataset, val_dataset, dist_helper
        )
        self.info = self._get_info(train_dataset, val_dataset)

        self.train_smiles = train_dataset.r_smiles
        return train_loader, val_loader, self.info

    def download(self) -> None:
        save_dir = config.get_dataset_directory() / self.name
        for split in ["train", "val", "test"]:
            USPTO(split=split, root=str(save_dir), download_and_process=True)

    def load_eval(self, load_valid=False) -> Tuple[DataLoader | RetrosynthesisInfo]:
        if self.name not in NAMES:
            raise ValueError(f"{self.name} is not any of the RetroSynthesis Datasets.")

        save_dir = config.get_dataset_directory() / self.name

        eval_dataset = USPTO(split="val" if load_valid else "test", root=str(save_dir))

        self.info = self._get_info(
            USPTO(split="train", root=str(save_dir)),
            USPTO(split="val", root=str(save_dir)),
        )

        test_loader = DataLoader(eval_dataset, batch_size=self.batch_size)
        return test_loader, self.info


    def _get_info(self, train_dataset, val_dataset):
        dummy_nodes_dist = torch.zeros(RetrosynthesisInfo.max_n_dummy_nodes + 1).to(
            config.get_device()
        )
        n_nodes_dist = self._node_counts(train_dataset, val_dataset)
        max_n_nodes = len(n_nodes_dist) - 1

        edge_dist = self._edge_counts(train_dataset)

        node_type_dist = self._node_types(train_dataset)
        valency_dist = self._valency_count(max_n_nodes, train_dataset, val_dataset)

        input_dim, output_dim = self._compute_input_output_dims(
            max_n_nodes, train_dataset
        )

        return RetrosynthesisInfo(
            input_dim=input_dim,
            output_dim=output_dim,
            n_nodes_dist=n_nodes_dist,
            node_types_dist=node_type_dist,
            edge_types_dist=edge_dist,
            valency_dist=valency_dist,
            dummy_nodes_dist=dummy_nodes_dist,
            max_n_nodes=max_n_nodes,
        )

    def _get_train_and_val_loaders(
        self, train_dataset, val_dataset, dist_helper: DistributedHelper | None = None
    ):
        if dist_helper is not None:
            batch_size_per_gpu = max(
                1, self.batch_size // dist_helper.get_ddp_status()[1]["WORLD_SIZE"]
            )
            train_loader = DataLoader(
                dataset=train_dataset,
                sampler=DistributedSampler(train_dataset, shuffle=True),
                batch_size=batch_size_per_gpu,
                pin_memory=True,
                num_workers=min(6, os.cpu_count()),
            )

            val_loader = DataLoader(
                dataset=val_dataset,
                sampler=DistributedSampler(val_dataset, shuffle=False),
                batch_size=batch_size_per_gpu,
                pin_memory=True,
                num_workers=min(6, os.cpu_count()),
            )
        else:
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=min(6, os.cpu_count()),
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=min(6, os.cpu_count()),
            )

        return train_loader, val_loader

    @staticmethod
    def _compute_input_output_dims(max_n_nodes, train_dataset):
        extra_features = ExtraFeatures(max_n_nodes)

        ex = train_dataset[0]

        r_ex_dense, r_node_mask = to_dense(ex.x, ex.edge_index, ex.edge_attr, ex.batch)

        input_dim = GraphDimensions(
            node_dim=ex.x.shape[1], edge_dim=ex.edge_attr.shape[1], y_dim=ex.y.shape[1] + 1  # type: ignore
        )

        # doesn't matter if we use the product or reactant molecule to compute
        # the extra features to determine the dimension
        ex_extra_feat = extra_features(r_ex_dense.X, r_ex_dense.E, r_node_mask)

        input_dim.node_dim += ex_extra_feat.X.size(-1)
        input_dim.edge_dim += ex_extra_feat.E.size(-1)
        input_dim.y_dim += ex_extra_feat.y.size(-1)

        input_dim.node_dim += ex.x.shape[1]
        input_dim.edge_dim += ex.edge_attr.shape[1]

        output_dim = GraphDimensions(
            node_dim=ex.x.shape[1], edge_dim=ex.edge_attr.shape[1], y_dim=0  # type: ignore
        )

        return input_dim, output_dim

    @staticmethod
    def _edge_counts(train_dataset):
        num_classes = train_dataset[0].edge_attr.shape[1]  # type: ignore

        d = torch.zeros(num_classes, dtype=torch.float)

        for graph in train_dataset:
            num_nodes: int = graph.num_nodes  # type: ignore
            num_edges = graph.edge_index.shape[1]  # type: ignore
            edge_type_counts = torch.sum(graph.edge_attr, dim=0)  # type: ignore

            all_pairs = num_nodes * (num_nodes - 1)
            num_non_edges = all_pairs - num_edges

            d[0] += num_non_edges
            d[1:] += edge_type_counts[1:]

        d = d / d.sum()
        d = d.to(config.get_device())
        return d

    @staticmethod
    def _node_counts(train_dataset, val_dataset, max_nodes_possible=300):
        all_counts = torch.zeros(max_nodes_possible)

        for dataset in [train_dataset, val_dataset]:
            num_nodes_list = [graph.num_nodes for graph in dataset]
            for num_node in num_nodes_list:
                all_counts[num_node] += 1

        max_index = max(all_counts.nonzero())  # type: ignore
        all_counts = all_counts[: max_index + 1]
        all_counts = all_counts / all_counts.sum()

        return all_counts.to(config.get_device())

    @staticmethod
    def _node_types(train_dataset):
        num_classes = train_dataset[0].x.shape[1]
        counts = torch.zeros(num_classes)

        for graph in train_dataset:
            counts += torch.sum(graph.x, dim=0)

        counts = counts / counts.sum()
        return counts.to(config.get_device())

    @staticmethod
    def _valency_count(max_n_nodes, train_dataset, val_dataset):
        valencies = torch.zeros(
            3 * max_n_nodes - 2
        )  # Max valency possible if everything is connected

        # No bond, single bond, double bond, triple bond, aromatic bond
        multiplier = torch.tensor([0, 1, 2, 3, 1.5])

        for dataset in [train_dataset, val_dataset]:
            for atom, data in enumerate(dataset):
                edges = data.edge_attr[data.edge_index[0] == atom]
                edges_total = edges.sum(dim=0)
                valency = (edges_total * multiplier).sum()
                valencies[valency.long().item()] += 1
        valencies = valencies / valencies.sum()
        return valencies.to(config.get_device())

    # @staticmethod
    # def _dummy_atoms_counts(max_n_dummy_nodes, train_dataset):
    #     dummy_atoms = torch.zeros(max_n_dummy_nodes + 1).to(config.get_device())
    #     for data in train_dataset:
    #         cnt = torch.sum(data.p_x[:, -1]).long()
    #         dummy_atoms[cnt] += 1
    #     out = dummy_atoms / dummy_atoms.sum()
    #     return out.to(config.get_device())


@dataclass
class SmallRetroDataset(RetroDataset):
    full_test: bool = False

    def load(self, dist_helper: DistributedHelper | None = None):
        if self.name not in NAMES:
            raise ValueError(f"{self.name} is not any of the RetroSynthesis datasets.")

        save_dir = config.get_dataset_directory() / self.name

        train_dataset = USPTO(split="train", root=str(save_dir))
        val_dataset = USPTO(split="val", root=str(save_dir))

        train_dataset = train_dataset[0 : int(0.2 * len(train_dataset))]
        val_dataset = val_dataset[0 : int(0.2 * len(val_dataset))]

        train_loader, val_loader = self._get_train_and_val_loaders(
            train_dataset, val_dataset, dist_helper=dist_helper
        )
        self.info = self._get_info(train_dataset, val_dataset)
        self.train_smiles = train_dataset.r_smiles
        return train_loader, val_loader, self.info

    def load_eval(self, load_valid=False) -> Tuple[DataLoader | RetrosynthesisInfo]:
        if self.name not in NAMES:
            raise ValueError(f"{self.name} is not any of the RetroSynthesis Datasets.")

        save_dir = config.get_dataset_directory() / self.name

        eval_dataset = USPTO(split="val" if load_valid else "test", root=str(save_dir))
        if not self.full_test:
            eval_dataset = eval_dataset[0 : int(0.2 * len(eval_dataset))]

        self.info = self._get_info(
            USPTO(split="train", root=str(save_dir)),
            USPTO(split="val", root=str(save_dir)),
        )

        test_loader = DataLoader(eval_dataset, batch_size=self.batch_size)
        return test_loader, self.info


@dataclass
class ToyRetroDataset(RetroDataset):
    num_molecules: int = 2

    def load(self, dist_helper: DistributedHelper | None = None):
        if self.name not in NAMES:
            raise ValueError(f"{self.name} is not any of the RetroSynthesis Datasets.")

        save_dir = config.get_dataset_directory() / self.name

        train_dataset = USPTO(split="train", root=str(save_dir))[0 : self.num_molecules]
        val_dataset = copy.deepcopy(train_dataset)

        train_loader, val_loader, self.info = self._get_loaders_and_info(
            train_dataset, val_dataset, dist_helper=dist_helper
        )
        self.train_smiles = train_dataset.r_smiles
        return train_loader, val_loader, self.info
