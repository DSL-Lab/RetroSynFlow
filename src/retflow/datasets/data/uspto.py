from typing import Dict, Sequence, Any
import os
import subprocess
import pandas as pd

from rdkit import Chem
import torch
from torch_geometric.data import Data, InMemoryDataset
from retflow.retro_utils.data import (
    build_graph_from_mol_with_mapping,
    compute_nodes_order_mapping,
)
from retflow.datasets.info import (
    RetrosynthesisInfo,
    DOWNLOAD_URL_TEMPLATE,
)


class USPTO(InMemoryDataset):
    def __init__(self, split, root, download_and_process=False, swap=False):
        self.split = split

        if self.split == "train":
            self.file_idx = 0
        elif self.split == "val":
            self.file_idx = 1
        elif self.split == "test":
            self.file_idx = 2
        else:
            raise NotImplementedError
        self.download_and_process = download_and_process
        super().__init__(root=root)

        self.slices: Dict
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

        if swap:
            self.data = Data(
                x=self.data.p_x,
                edge_index=self.data.p_edge_index,
                edge_attr=self.data.p_edge_attr,
                p_x=self.data.x,
                p_edge_index=self.data.edge_index,
                p_edge_attr=self.data.edge_attr,
                y=self.data.y,
                idx=self.data.idx,
                r_smiles=self.data.p_smiles,
                p_smiles=self.data.r_smiles,
            )
            self.slices = {
                "x": self.slices["p_x"],
                "edge_index": self.slices["p_edge_index"],
                "edge_attr": self.slices["p_edge_attr"],
                "y": self.slices["y"],
                "idx": self.slices["idx"],
                "p_x": self.slices["x"],
                "p_edge_index": self.slices["edge_index"],
                "p_edge_attr": self.slices["edge_attr"],
                "r_smiles": self.slices["p_smiles"],
                "p_smiles": self.slices["r_smiles"],
            }

    @property
    def raw_file_names(self):
        return ["uspto50k_train.csv", "uspto50k_val.csv", "uspto50k_test.csv"]

    @property
    def split_file_name(self):
        return ["uspto50k_train.csv", "uspto50k_val.csv", "uspto50k_test.csv"]

    @property
    def split_paths(self):
        files = to_list(self.split_file_name)
        return [os.path.join(self.raw_dir, f) for f in files]

    @property
    def processed_file_names(self):
        return [f"train.pt", f"val.pt", f"test.pt"]

    def download(self):
        if self.download_and_process:
            os.makedirs(self.raw_dir, exist_ok=True)
            for fname in self.raw_file_names:
                url = DOWNLOAD_URL_TEMPLATE.format(fname=fname)
                path = os.path.join(self.raw_dir, fname)
                subprocess.run(f"wget {url} -O {path}", shell=True)

    def process(self):
        if self.download_and_process:
            self._process_data()

    def _process_data(self):
        table = pd.read_csv(self.split_paths[self.file_idx])
        data_list = []
        for i, reaction_smiles in enumerate(
            table["reactants>reagents>production"].values
        ):
            reactants_smi, _, product_smi = reaction_smiles.split(">")
            rmol = Chem.MolFromSmiles(reactants_smi)
            pmol = Chem.MolFromSmiles(product_smi)
            r_num_nodes = rmol.GetNumAtoms()
            p_num_nodes = pmol.GetNumAtoms()
            assert p_num_nodes <= r_num_nodes

            new_r_num_nodes = p_num_nodes + RetrosynthesisInfo.max_n_dummy_nodes
            if r_num_nodes > new_r_num_nodes:
                if self.split in ["train", "val"]:
                    continue
                else:
                    reactants_smi, product_smi = "C", "C"
                    rmol = Chem.MolFromSmiles(reactants_smi)
                    pmol = Chem.MolFromSmiles(product_smi)
                    p_num_nodes = pmol.GetNumAtoms()
                    new_r_num_nodes = p_num_nodes + RetrosynthesisInfo.max_n_dummy_nodes

            r_num_nodes = new_r_num_nodes

            try:
                mapping = compute_nodes_order_mapping(rmol)
                r_x, r_edge_index, r_edge_attr = build_graph_from_mol_with_mapping(
                    rmol,
                    mapping,
                    r_num_nodes,
                    types=RetrosynthesisInfo.atom_encoder,
                    bonds=RetrosynthesisInfo.bonds,
                )
                p_x, p_edge_index, p_edge_attr = build_graph_from_mol_with_mapping(
                    pmol,
                    mapping,
                    r_num_nodes,
                    types=RetrosynthesisInfo.atom_encoder,
                    bonds=RetrosynthesisInfo.bonds,
                )
            except Exception as e:
                continue

            if self.split in ["train", "val"]:
                assert len(p_x) == len(r_x)

            product_mask = ~(p_x[:, -1].bool()).squeeze()
            if len(r_x) == len(p_x) and not torch.allclose(
                r_x[product_mask], p_x[product_mask]
            ):
                continue

            if self.split == "train" and len(p_edge_attr) == 0:
                continue

            # Shuffle nodes to avoid leaking
            if len(p_x) == len(r_x):
                new2old_idx = torch.randperm(r_num_nodes).long()
                old2new_idx = torch.empty_like(new2old_idx)
                old2new_idx[new2old_idx] = torch.arange(r_num_nodes)

                r_x = r_x[new2old_idx]
                r_edge_index = torch.stack(
                    [old2new_idx[r_edge_index[0]], old2new_idx[r_edge_index[1]]], dim=0
                )
                r_edge_index, r_edge_attr = self.sort_edges(
                    r_edge_index, r_edge_attr, r_num_nodes
                )

                p_x = p_x[new2old_idx]
                p_edge_index = torch.stack(
                    [old2new_idx[p_edge_index[0]], old2new_idx[p_edge_index[1]]], dim=0
                )
                p_edge_index, p_edge_attr = self.sort_edges(
                    p_edge_index, p_edge_attr, r_num_nodes
                )

                product_mask = ~(p_x[:, -1].bool()).squeeze()
                assert torch.allclose(r_x[product_mask], p_x[product_mask])

            y = torch.zeros(size=(1, 0), dtype=torch.float)
            data = Data(
                x=r_x,
                edge_index=r_edge_index,
                edge_attr=r_edge_attr,
                y=y,
                idx=i,
                p_x=p_x,
                p_edge_index=p_edge_index,
                p_edge_attr=p_edge_attr,
                r_smiles=reactants_smi,
                p_smiles=product_smi,
            )

            data_list.append(data)
        if self.split == "test":
            data_list.sort(key=lambda data: len(data.x), reverse=True)
            for i, data in enumerate(data_list):
                data.idx = i
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])

    @staticmethod
    def sort_edges(edge_index, edge_attr, max_num_nodes):
        if len(edge_attr) != 0:
            perm = (edge_index[0] * max_num_nodes + edge_index[1]).argsort()
            edge_index = edge_index[:, perm]
            edge_attr = edge_attr[perm]

        return edge_index, edge_attr


def to_list(value: Any) -> Sequence:
    if isinstance(value, Sequence) and not isinstance(value, str):
        return value
    else:
        return [value]
