from pathlib import Path
from torchdrug.datasets import USPTO50k
from torchdrug import utils
from torchdrug.data import Molecule
from collections import defaultdict
from tqdm import tqdm
import os
import math
import csv

from rdkit import Chem

from retflow.utils import reactants_with_partial_atom_mapping
from retflow.datasets.info import DOWNLOAD_URL_TEMPLATE
from retflow.datasets.info import RetrosynthesisInfo


class _TorchDrugUSPTO(USPTO50k):
    def __init__(self, data_split, path, as_synthon=False, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        self.data_split = data_split
        self.as_synthon = as_synthon

        if self.data_split == "train":
            self.file_idx = 0
        elif self.data_split == "val":
            self.file_idx = 1
        elif self.data_split == "test":
            self.file_idx = 2
        else:
            raise NotImplementedError

        self.load_csv(
            str(Path(self.path) / self.raw_file_names[self.file_idx]),
            smiles_field="reactants>reagents>production",
            target_fields=self.target_fields,
            verbose=verbose,
            **kwargs
        )

        if as_synthon:
            prefix = "Computing synthons"
            process_fn = self._get_synthon
        else:
            prefix = "Computing reaction centers"
            process_fn = self._get_reaction_center

        data = self.data
        targets = self.targets
        ids = self.ids
        data, tgts, ids = zip(
            *sorted(zip(data, targets["class"], ids), key=lambda x: x[0][0].num_atom)
        )

        targets = {"class": tgts}
        self.data = []
        self.targets = defaultdict(list)
        self.ids = []

        indexes = range(len(data))
        if verbose:
            indexes = tqdm(indexes, prefix)
        invalid = 0
        for i in indexes:
            reactant, product = data[i]

            rmol = reactant.to_molecule()
            r_num_nodes = rmol.GetNumAtoms()

            pmol = product.to_molecule()

            p_num_nodes = pmol.GetNumAtoms()

            new_r_num_nodes = p_num_nodes + RetrosynthesisInfo.max_n_dummy_nodes
            if r_num_nodes > new_r_num_nodes:
                reactants_smi, product_smi = "C", "C"
                rmol = Chem.MolFromSmiles(reactants_smi)
                pmol = Chem.MolFromSmiles(product_smi)
                reactant = Molecule.from_molecule(
                    rmol, atom_feature="center_identification"
                )
                product = Molecule.from_molecule(
                    pmol, atom_feature="center_identification"
                )

            reactant.bond_stereo[:] = 0
            product.bond_stereo[:] = 0

            reactant_converted = reactants_with_partial_atom_mapping(
                reactant, product, kwargs["atom_feature"]
            )
            reactants, products = process_fn(reactant_converted, product)
            if not reactants:
                invalid += 1
                continue
            if r_num_nodes > new_r_num_nodes:
                assert product.num_atom == 1
            self.data += zip([reactant], products)
            for k in targets:
                new_k = self.target_alias.get(k, k)
                self.targets[new_k] += [targets[k][i] - 1] * len([reactant])
            self.targets["sample id"] += [i] * len([reactant])
            self.ids += [ids[i]] * len([reactant])
        self.valid_rate = 1 - invalid / len(data)

    def load_csv(
        self, csv_file, smiles_field="smiles", target_fields=None, verbose=0, **kwargs
    ):
        """
        Load the dataset from a csv file.

        Parameters:
            csv_file (str): file name
            smiles_field (str, optional): name of the SMILES column in the table.
                Use ``None`` if there is no SMILES column.
            target_fields (list of str, optional): name of target columns in the table.
                Default is all columns other than the SMILES column.
            verbose (int, optional): output verbose level
            **kwargs
        """
        if target_fields is not None:
            target_fields = set(target_fields)
        self.ids = []
        with open(csv_file, "r") as fin:
            reader = csv.reader(fin)
            if verbose:
                reader = iter(
                    tqdm(
                        reader, "Loading %s" % csv_file, utils.get_line_count(csv_file)
                    )
                )
            fields = next(reader)
            smiles = []
            targets = defaultdict(list)
            for values in reader:
                if not any(values):
                    continue
                if smiles_field is None:
                    smiles.append("")
                for field, value in zip(fields, values):
                    if field == smiles_field:
                        smiles.append(value)
                    elif target_fields is None or field in target_fields:
                        value = utils.literal_eval(value)
                        if value == "":
                            value = math.nan
                        targets[field].append(value)
                    if field == "id":
                        self.ids.append(value)
        self.load_smiles(smiles, targets, verbose=verbose, **kwargs)

    @property
    def raw_file_names(self):
        return ["uspto50k_train.csv", "uspto50k_val.csv", "uspto50k_test.csv"]

    @staticmethod
    def download(split, path):
        raw_file_names = ["uspto50k_train.csv", "uspto50k_val.csv", "uspto50k_test.csv"]
        if split == "train":
            file_idx = 0
        elif split == "val":
            file_idx = 1
        elif split == "test":
            file_idx = 2
        url = DOWNLOAD_URL_TEMPLATE.format(fname=raw_file_names[file_idx])
        utils.download(url, path)
