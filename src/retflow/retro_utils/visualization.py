import os

from rdkit.Chem import Draw, AllChem
from rdkit.Geometry import Point3D
import rdkit.Chem


def create_dummy_conformer(molecule):
    conformer = AllChem.Conformer()
    for i in range(molecule.GetNumAtoms()):
        conformer.SetAtomPosition(i, Point3D(0, 0, 0))

    molecule.AddConformer(conformer, assignId=True)
    return molecule


class MolecularVisualization:
    def __init__(self, dataset_infos, path: str):
        self.dataset_infos = dataset_infos
        self.path = path

        if not os.path.exists(path):
            os.makedirs(path)

    def visualize(
        self,
        molecules,
        num_molecules_to_visualize: int,
        prefix="",
        suffix="",
    ):
        # visualize the final molecules
        print(f"Visualizing {num_molecules_to_visualize} of {len(molecules)}")
        if num_molecules_to_visualize > len(molecules):
            print(f"Shortening to {len(molecules)}")
            num_molecules_to_visualize = len(molecules)

        for i in range(num_molecules_to_visualize):
            file_path = os.path.join(
                self.path, "{}molecule_{}{}.png".format(prefix, i, suffix)
            )
            try:
                Draw.MolToFile(molecules[i], file_path)
            except rdkit.Chem.KekulizeException:
                print("Can't kekulize molecule")
            except ValueError:
                print("Maximum BFS search size exceeded")
