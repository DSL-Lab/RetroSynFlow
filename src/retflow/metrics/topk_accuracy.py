import torch
from rdkit import Chem
from torch import distributed as dist

from retflow.metrics.rdkit_metrics import build_molecule

K_VALUES = [1, 3, 5, 10, 50]


def top_k_accuracy(
    grouped_samples, ground_truth, atom_decoder, grouped_scores=None, ddp: bool = False
):
    """
    Compute top-N accuracy. Inputs are matrices of atom types and bonds.
    """
    top_k_success = {k: 0 for k in K_VALUES}
    top_k_success_scoring = {k: 0 for k in K_VALUES}
    total = 0

    for i, sampled_reactants in enumerate(grouped_samples):
        true_reactants = ground_truth[i]
        true_mol = build_molecule(true_reactants[0], true_reactants[1], atom_decoder)
        true_smi = Chem.MolToSmiles(true_mol)
        if true_smi is None:
            continue

        sampled_smis = []
        for sample in sampled_reactants:
            sampled_mol = build_molecule(sample[0], sample[1], atom_decoder)
            sampled_smi = Chem.MolToSmiles(sampled_mol)
            if sampled_smi is None:
                continue
            sampled_smis.append(sampled_smi)

        total += 1
        for k in top_k_success.keys():
            top_k_success[k] += true_smi in set(sampled_smis[:k])

        if grouped_scores is not None:
            scores = grouped_scores[i]
            sorted_sampled_smis = [
                sampled_smis[j]
                for j, _ in sorted(enumerate(scores), key=lambda t: t[1], reverse=True)
            ]

            for k in top_k_success_scoring.keys():
                top_k_success_scoring[k] += true_smi in set(sorted_sampled_smis[:k])

    if ddp:
        dist.barrier()
        total = torch.tensor(total).cuda()
        dist.all_reduce(total)
        total = total.item()

        for k in top_k_success.keys():
            tk = torch.tensor(top_k_success[k]).cuda()
            dist.all_reduce(tk)
            top_k_success[k] = tk.item()

        for k in top_k_success_scoring.keys():
            tks = torch.tensor(top_k_success_scoring[k]).cuda()
            dist.all_reduce(tks)
            top_k_success_scoring[k] = tks.item()

    metrics = {}
    for k in top_k_success.keys():
        metrics[f"top_{k}_accuracy"] = top_k_success[k] / total

    for k in top_k_success_scoring.keys():
        metrics[f"top_{k}_scoring"] = top_k_success_scoring[k] / total

    return metrics
