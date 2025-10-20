from typing import Dict
import os
import subprocess
import pandas as pd
from tqdm import tqdm
from pathlib import Path

from rdkit import Chem
import torch
from torch_geometric.data import Data, InMemoryDataset

from retflow.retro_utils.data import (
    compute_nodes_mapping,
    build_graph_from_mol,
    build_graph_from_mol_with_mapping,
)

from seq_graph_retro.utils.parse import get_reaction_info
from seq_graph_retro.utils.chem import apply_edits_to_mol, get_mol

from retflow.datasets.data.uspto import to_list
from retflow.datasets.info import RetrosynthesisInfo, DOWNLOAD_URL_TEMPLATE


class ReactionCenterUSPTO(InMemoryDataset):
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

    @property
    def raw_file_names(self):
        return ["uspto50k_train_can.csv", "uspto50k_val_can.csv", "uspto50k_test_can.csv"]

    @property
    def split_file_name(self):
        return ["uspto50k_train_can.csv", "uspto50k_val_can.csv", "uspto50k_test_can.csv"]

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
                file = Path(self.raw_dir) / fname
                if file.exists():
                    continue
                url = DOWNLOAD_URL_TEMPLATE.format(fname=fname)
                path = os.path.join(self.raw_dir, fname)
                subprocess.run(f"wget {url} -O {path}", shell=True)

    def process(self):
        if self.download_and_process:
            self._process_data()

    def _process_data(self):
        table = pd.read_csv(self.split_paths[self.file_idx])
        data_list = []
        for i, reaction_smiles in tqdm(
            enumerate(table["reactants>reagents>production"].values)
        ):
            reactants_smi, _, product_smi = reaction_smiles.split(">")
            rmol = Chem.MolFromSmiles(reactants_smi)
            pmol = Chem.MolFromSmiles(product_smi)

            try:
                reaction_info = get_reaction_info(reaction_smiles, kekulize=True,
                                        use_h_labels=True,
                                        rxn_class=int(table.loc[i, "class"]))
            except:
                print(f"Failed to extract reaction info. Skipping reaction {i}")
                continue
        

            products = get_mol(product_smi)

            synthons_mol = apply_edits_to_mol(Chem.Mol(products), reaction_info.core_edits)

            p_num_nodes = pmol.GetNumAtoms()
            s_num_nodes = synthons_mol.GetNumAtoms()
            assert p_num_nodes == s_num_nodes

            try:
                mapping = compute_nodes_mapping(pmol)
                p_x, p_edge_index, p_edge_attr = build_graph_from_mol_with_mapping(
                    pmol,
                    mapping,
                    p_num_nodes,
                    types=RetrosynthesisInfo.atom_encoder,
                    bonds=RetrosynthesisInfo.bonds,
                )
                s_x, s_edge_index, s_edge_attr = build_graph_from_mol_with_mapping(
                    synthons_mol,
                    mapping,
                    p_num_nodes,
                    types=RetrosynthesisInfo.atom_encoder,
                    bonds=RetrosynthesisInfo.bonds,
                )
            except Exception as e:
                continue

            assert len(s_x) == len(p_x)

            product_mask = ~(p_x[:, -1].bool()).squeeze()
            if len(p_x) == len(s_x) and not torch.allclose(
                p_x[product_mask], s_x[product_mask]
            ):
                continue

            if len(s_edge_attr) == 0:
                continue

            # Shuffle nodes to avoid leaking
            if len(s_x) == len(p_x):
                new2old_idx = torch.randperm(p_num_nodes).long()
                old2new_idx = torch.empty_like(new2old_idx)
                old2new_idx[new2old_idx] = torch.arange(p_num_nodes)

                p_x = p_x[new2old_idx]
                p_edge_index = torch.stack(
                    [old2new_idx[p_edge_index[0]], old2new_idx[p_edge_index[1]]],
                    dim=0,
                )
                p_edge_index, p_edge_attr = self.sort_edges(
                    p_edge_index, p_edge_attr, p_num_nodes
                )

                s_x = s_x[new2old_idx]
                s_edge_index = torch.stack(
                    [old2new_idx[s_edge_index[0]], old2new_idx[s_edge_index[1]]],
                    dim=0,
                )
                s_edge_index, s_edge_attr = self.sort_edges(
                    s_edge_index, s_edge_attr, p_num_nodes
                )

                synthon_mask = ~(s_x[:, -1].bool()).squeeze()
                assert torch.allclose(p_x[synthon_mask], s_x[synthon_mask])

            y = torch.zeros(size=(1, 0), dtype=torch.float)
            data = Data(
                x=p_x,
                edge_index=p_edge_index,
                edge_attr=p_edge_attr,
                y=y,
                idx=i,
                s_x=s_x,
                s_edge_index=s_edge_index,
                s_edge_attr=s_edge_attr,
                p_smiles=product_smi,
                s_smiles=Chem.MolToSmiles(synthons_mol, canonical=True),
                r_smiles=reactants_smi,
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
