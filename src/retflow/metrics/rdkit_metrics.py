from typing import List, Tuple

import numpy as np
import copy
from rdkit import Chem
from torch import Tensor
from torch import distributed as dist

from retflow.metrics.metric import SamplingMetric
from retflow.retro_utils.data import (
    build_molecule,
    build_molecule_with_partial_charges,
    check_valency,
)

allowed_bonds = {
    "H": 1,
    "C": 4,
    "N": 3,
    "O": 2,
    "F": 1,
    "B": 3,
    "Al": 3,
    "Si": 4,
    "P": [3, 5],
    "S": 4,
    "Cl": 1,
    "As": 3,
    "Br": 1,
    "I": 1,
    "Hg": [1, 2],
    "Bi": [3, 5],
    "Se": [2, 4, 6],
}

bond_dict = [
    None,
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
ATOM_VALENCY = {6: 4, 7: 3, 8: 2, 9: 1, 15: 3, 16: 2, 17: 1, 35: 1, 53: 1}


class BasicMolecularMetrics(SamplingMetric):
    def __init__(self, dataset_info, train_smiles=None):
        self.atom_decoder = dataset_info.atom_decoder
        self.dataset_info = dataset_info
        self.dataset_smiles_list = train_smiles if train_smiles is not None else []
        self.molecules = []

    def update(self, molecules: List[Tuple[Tensor]], real: bool):
        if real:
            return
        self.molecules.extend(molecules)

    def compute_validity(self, generated):
        """generated: list of couples (positions, atom_types)"""
        if len(generated) == 0:
            return [], 0, [], []

        num_components = []
        valid_smiles = []
        all_smiles = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule(atom_types, edge_types, self.dataset_info.atom_decoder)
            smiles = mol2smiles(mol)

            if smiles is not None:
                try:
                    mol_frags = Chem.GetMolFrags(mol, asMols=True)
                except Exception as e:
                    print(f"Problem with fragmenting the molecule: {e}")
                    continue
                num_components.append(len(mol_frags))
                valid_smiles.append(smiles)
                all_smiles.append(smiles)
            else:
                all_smiles.append(None)

        return (
            valid_smiles,
            len(valid_smiles) / len(generated),
            np.array(num_components),
            all_smiles,
        )

    @staticmethod
    def compute_uniqueness(valid):
        if len(valid) == 0:
            return [], 0

        return list(set(valid)), len(set(valid)) / len(valid)

    def compute_novelty(self, unique):
        if len(unique) == 0:
            return [], 0

        num_novel = 0
        novel = []

        for smiles in unique:
            if smiles not in self.dataset_smiles_list:
                novel.append(smiles)
                num_novel += 1

        return novel, num_novel / len(unique)

    def compute_relaxed_validity(self, generated):
        valid = []
        for graph in generated:
            atom_types, edge_types = graph
            mol = build_molecule_with_partial_charges(
                atom_types, edge_types, self.dataset_info.atom_decoder
            )
            smiles = mol2smiles(mol)
            if smiles is not None:
                valid.append(smiles)

        return valid, len(valid) / len(generated)

    def compute(self, ddp: bool = False):
        """
        generated: list of pairs (positions: n x 3, atom_types: n [int])
        the positions and atom types should already be masked.
        """
        if ddp:
            dist.barrier()
            gathered_molecule_list = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gathered_molecule_list, self.molecules)
            all_molecules = []
            for proc_list in gathered_molecule_list:
                all_molecules.extend(proc_list)
            self.molecules = all_molecules

        valid, validity, num_components, all_smiles = self.compute_validity(
            copy.deepcopy(self.molecules)
        )

        nc_mu = num_components.mean() if len(num_components) > 0 else 0
        nc_min = num_components.min() if len(num_components) > 0 else 0
        nc_max = num_components.max() if len(num_components) > 0 else 0

        relaxed_valid, relaxed_validity = self.compute_relaxed_validity(self.molecules)
        unique, uniqueness = self.compute_uniqueness(valid)
        _, novelty = self.compute_novelty(unique)

        return {
            "rdkit/validity": validity,
            "rdkit/relaxed_validity": relaxed_validity,
            "rdkit/uniqueness": uniqueness,
            "rdkit/novelty": novelty,
            "rdkit/nc_max": nc_max,
            "rdkit/nc_mu": nc_mu,
            "rdkit/nc_min": nc_min,
        }

    def reset(self):
        self.molecules = []


def mol2smiles(mol):
    try:
        Chem.SanitizeMol(mol)
    except ValueError:
        return None
    return Chem.MolToSmiles(mol)


def correct_mol(m):
    # xsm = Chem.MolToSmiles(x, isomericSmiles=True)
    mol = m

    #####
    no_correct = False
    flag, _ = check_valency(mol)
    if flag:
        no_correct = True

    while True:
        flag, atomid_valence = check_valency(mol)
        if flag:
            break
        else:
            assert len(atomid_valence) == 2
            idx = atomid_valence[0]
            v = atomid_valence[1]
            queue = []
            check_idx = 0
            for b in mol.GetAtomWithIdx(idx).GetBonds():
                type = int(b.GetBondType())
                queue.append((b.GetIdx(), type, b.GetBeginAtomIdx(), b.GetEndAtomIdx()))
                if type == 12:
                    check_idx += 1
            queue.sort(key=lambda tup: tup[1], reverse=True)

            if queue[-1][1] == 12:
                return None, no_correct
            elif len(queue) > 0:
                start = queue[check_idx][2]
                end = queue[check_idx][3]
                t = queue[check_idx][1] - 1
                mol.RemoveBond(start, end)
                if t >= 1:
                    mol.AddBond(start, end, bond_dict[t])
    return mol, no_correct


def valid_mol_can_with_seg(m, largest_connected_comp=True):
    if m is None:
        return None
    sm = Chem.MolToSmiles(m, isomericSmiles=True)
    if largest_connected_comp and "." in sm:
        vsm = [
            (s, len(s)) for s in sm.split(".")
        ]  # 'C.CC.CCc1ccc(N)cc1CCC=O'.split('.')
        vsm.sort(key=lambda tup: tup[1], reverse=True)
        mol = Chem.MolFromSmiles(vsm[0][0])
    else:
        mol = Chem.MolFromSmiles(sm)
    return mol


def check_stability(
    atom_types, edge_types, dataset_info, debug=False, atom_decoder=None
):
    if atom_decoder is None:
        atom_decoder = dataset_info.atom_decoder

    n_bonds = np.zeros(len(atom_types), dtype="int")

    for i in range(len(atom_types)):
        for j in range(i + 1, len(atom_types)):
            n_bonds[i] += abs((edge_types[i, j] + edge_types[j, i]) / 2)
            n_bonds[j] += abs((edge_types[i, j] + edge_types[j, i]) / 2)
    n_stable_bonds = 0
    for atom_type, atom_n_bond in zip(atom_types, n_bonds):
        possible_bonds = allowed_bonds[atom_decoder[atom_type]]
        if type(possible_bonds) == int:
            is_stable = possible_bonds == atom_n_bond
        else:
            is_stable = atom_n_bond in possible_bonds
        if not is_stable and debug:
            print(
                "Invalid bonds for molecule %s with %d bonds"
                % (atom_decoder[atom_type], atom_n_bond)
            )
        n_stable_bonds += int(is_stable)

    molecule_stable = n_stable_bonds == len(atom_types)
    return molecule_stable, n_stable_bonds, len(atom_types)
