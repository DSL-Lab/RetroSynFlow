from torch import Tensor

import torch.nn as nn

from retflow.datasets.info import RetrosynthesisInfo


class ClassCrossEntropy(nn.Module):
    def __init__(self, class_id, bond=False) -> None:
        super().__init__()
        self.class_id = class_id
        self.bond = bond

    def forward(
        self,
        masked_pred_X: Tensor,
        masked_pred_E: Tensor,
        true_X: Tensor,
        true_E: Tensor,
    ) -> None:
        """Update state with predictions and targets.
        Args:
            preds: Predictions from model   (bs, n, d) or (bs, n, n, d)
            target: Ground truth values     (bs, n, d) or (bs, n, n, d)
        """
        if self.bond:
            preds = masked_pred_E
            target = true_E
        else:
            preds = masked_pred_X
            target = true_X

        target = target.reshape(-1, target.shape[-1])
        mask = (target != 0.0).any(dim=-1)

        prob = nn.Softmax(dim=-1)(preds)[..., self.class_id]
        prob = prob.flatten()[mask]

        target = target[:, self.class_id]
        target = target[mask]

        return nn.BCELoss(reduction="mean")(prob, target)


class HydrogenCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class CarbonCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class NitroCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class OxyCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class FluorCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class BoronCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class BrCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class ClCE(ClassCrossEntropy):
    def __init__(
        self,
        i,
    ):
        super().__init__(i)


class IodineCE(ClassCrossEntropy):
    def __init__(
        self,
        i,
    ):
        super().__init__(i)


class PhosphorusCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class SulfurCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class SeCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class SiCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class SnCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class CuCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class MgCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class ZnCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class DummyCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i)


class NoBondCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i, bond=True)


class SingleCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i, bond=True)


class DoubleCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i, bond=True)


class TripleCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i, bond=True)


class AromaticCE(ClassCrossEntropy):
    def __init__(self, i):
        super().__init__(i, bond=True)


class_dict = {
    "H": HydrogenCE,
    "C": CarbonCE,
    "N": NitroCE,
    "O": OxyCE,
    "F": FluorCE,
    "B": BoronCE,
    "Br": BrCE,
    "Cl": ClCE,
    "I": IodineCE,
    "P": PhosphorusCE,
    "S": SulfurCE,
    "Se": SeCE,
    "Si": SiCE,
    "Sn": SnCE,
    "Mg": MgCE,
    "Cu": CuCE,
    "Zn": ZnCE,
    "*": DummyCE,
}


def get_molecule_train_metrics(dataset_info: RetrosynthesisInfo):
    return [
        class_dict[atom_type](i)
        for i, atom_type in enumerate(dataset_info.atom_decoder)
    ] + [
        NoBondCE(0),
        SingleCE(1),
        DoubleCE(2),
        TripleCE(3),
        AromaticCE(4),
    ]
