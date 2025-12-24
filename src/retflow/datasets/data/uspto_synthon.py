import os
import subprocess
from typing import Dict

import pandas as pd
import torch
from rdkit import Chem
from torch_geometric.data import Data
from torchdrug.data import Molecule
from tqdm import tqdm

from retflow.datasets.data.uspto import USPTO, to_list
from retflow.datasets.info import DOWNLOAD_URL_TEMPLATE, RetrosynthesisInfo
from retflow.utils.data import (build_graph_from_mol,
                                build_graph_from_mol_with_mapping,
                                compute_nodes_mapping, get_synthons,
                                reactants_with_partial_atom_mapping)


class SynthonUSPTO(USPTO):
    def __init__(self, split, root, download_and_process=False, swap=False):
        super().__init__(split, root, download_and_process, swap)
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
        

        self.slices: Dict
        self.data, self.slices = torch.load(self.processed_paths[self.file_idx])

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
        for i, reaction_smiles in tqdm(
            enumerate(table["reactants>reagents>production"].values)
        ):
            reactants_smi, _, product_smi = reaction_smiles.split(">")
            rmol = Chem.MolFromSmiles(reactants_smi)
            pmol = Chem.MolFromSmiles(product_smi)

            reactants = Molecule.from_molecule(rmol, atom_feature="synthon_completion")
            product = Molecule.from_molecule(pmol, atom_feature="synthon_completion")

            reactants.bond_stereo[:] = 0
            product.bond_stereo[:] = 0
            reactants = reactants_with_partial_atom_mapping(
                reactants, product, atom_feature="synthon_completion"
            )
            _, synthons = get_synthons(reactants, product)

            full_synthons_smi = ".".join([s.to_smiles() for s in synthons])
            synthons_mol = Chem.MolFromSmiles(full_synthons_smi, sanitize=False)

            r_num_nodes = rmol.GetNumAtoms()
            s_num_nodes = synthons_mol.GetNumAtoms()
            assert s_num_nodes <= r_num_nodes
            assert s_num_nodes <= pmol.GetNumAtoms()
            new_r_num_nodes = s_num_nodes + RetrosynthesisInfo.max_n_dummy_nodes

            if r_num_nodes > new_r_num_nodes:
                if self.split in ["train", "val"]:
                    continue
                else:
                    reactants_smi, synthons_smi, product_smi = "C", "C", "C"
                    rmol = Chem.MolFromSmiles(reactants_smi)
                    synthons_mol = Chem.MolFromSmiles(synthons_smi)
                    pmol = Chem.MolFromSmiles(product_smi)
                    s_num_nodes = synthons_mol.GetNumAtoms()
                    new_r_num_nodes = s_num_nodes + RetrosynthesisInfo.max_n_dummy_nodes

            r_num_nodes = new_r_num_nodes
            try:
                mapping = compute_nodes_mapping(rmol)
                r_x, r_edge_index, r_edge_attr = build_graph_from_mol(
                    rmol,
                    r_num_nodes,
                    types=RetrosynthesisInfo.atom_encoder,
                    bonds=RetrosynthesisInfo.bonds,
                )
                s_x, s_edge_index, s_edge_attr = build_graph_from_mol_with_mapping(
                    synthons_mol,
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
                assert len(s_x) == len(r_x)

            synthon_mask = ~(s_x[:, -1].bool()).squeeze()
            if len(r_x) == len(s_x) and not torch.allclose(
                r_x[synthon_mask], s_x[synthon_mask]
            ):
                continue

            if self.split == "train" and len(s_edge_attr) == 0:
                continue

            # Shuffle nodes to avoid leaking
            if len(s_x) == len(r_x):
                new2old_idx = torch.randperm(r_num_nodes).long()
                old2new_idx = torch.empty_like(new2old_idx)
                old2new_idx[new2old_idx] = torch.arange(r_num_nodes)

                r_x = r_x[new2old_idx]
                r_edge_index = torch.stack(
                    [old2new_idx[r_edge_index[0]], old2new_idx[r_edge_index[1]]],
                    dim=0,
                )
                r_edge_index, r_edge_attr = self.sort_edges(
                    r_edge_index, r_edge_attr, r_num_nodes
                )

                p_x = p_x[new2old_idx]
                p_edge_index = torch.stack(
                    [old2new_idx[p_edge_index[0]], old2new_idx[p_edge_index[1]]],
                    dim=0,
                )
                p_edge_index, p_edge_attr = self.sort_edges(
                    p_edge_index, p_edge_attr, r_num_nodes
                )

                s_x = s_x[new2old_idx]
                s_edge_index = torch.stack(
                    [old2new_idx[s_edge_index[0]], old2new_idx[s_edge_index[1]]],
                    dim=0,
                )
                s_edge_index, s_edge_attr = self.sort_edges(
                    s_edge_index, s_edge_attr, r_num_nodes
                )

                synthon_mask = ~(s_x[:, -1].bool()).squeeze()
                assert torch.allclose(r_x[synthon_mask], s_x[synthon_mask])

            y = torch.zeros(size=(1, 0), dtype=torch.float)
            data = Data(
                x=r_x,
                edge_index=r_edge_index,
                edge_attr=r_edge_attr,
                y=y,
                idx=i,
                s_x=s_x,
                s_edge_index=s_edge_index,
                s_edge_attr=s_edge_attr,
                p_x=p_x,
                p_edge_index=p_edge_index,
                p_edge_attr=p_edge_attr,
                r_smiles=reactants_smi,
                p_smiles=product_smi,
            )

            data_list.append(data)
        if self.split == "test":
            data_list.sort(key=lambda data: len(data.x))
            for i, data in enumerate(data_list):
                data.idx = i
        torch.save(self.collate(data_list), self.processed_paths[self.file_idx])
